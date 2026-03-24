#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/ChooseObject/choose_obj_from_prepared.py

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = _THIS_FILE.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from LLMModule.ollama_client import chat_json
from LLMModule.retry_llm_json import RetryResult, ValidationResult, run_retry_loop


PREPARED_INFO_DEFAULT = "data/sourse/3D-FRONT/prepared_model_info.json"
FUTURE_ROOT_DEFAULT = "data/sourse/3D-FRONT/3D-FUTURE-model"
DEFAULT_OUT = "data/input/objects.json"


_word_re = re.compile(r"[А-Яа-яA-Za-z0-9/\-+]+")


def norm(s: str) -> str:
    return " ".join(_word_re.findall((s or "").lower().replace("ё", "е")))


def token_set(s: str) -> set[str]:
    return set(norm(s).split())


def char_multiset(s: str) -> dict[str, int]:
    d: dict[str, int] = {}
    for ch in norm(s).replace(" ", ""):
        d[ch] = d.get(ch, 0) + 1
    return d


def jaccard_tokens(a: str, b: str) -> float:
    A, B = token_set(a), token_set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def overlap_chars(a: str, b: str) -> float:
    A, B = char_multiset(a), char_multiset(b)
    inter = sum(min(A.get(k, 0), B.get(k, 0)) for k in set(A) | set(B))
    denom = max(sum(A.values()), sum(B.values()), 1)
    return inter / denom


def fuzzy_score(a: str, b: str) -> float:
    return 0.7 * jaccard_tokens(a, b) + 0.3 * overlap_chars(a, b)


CATEGORY_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "King-size Bed": {"touch_wall": {"sides": ["back", "left", "right"]}},
    "Single bed": {"touch_wall": {"sides": ["back", "left", "right"]}},
    "Kids Bed": {"touch_wall": {"sides": ["back", "left", "right"]}},
    "Wardrobe": {"touch_wall": {"sides": ["back"]}},
    "Bookcase / jewelry Armoire": {"touch_wall": {"sides": ["back"]}},
    "Sideboard / Side Cabinet / Console Table": {"touch_wall": {"sides": ["back"]}},
    "Wine Cabinet": {"touch_wall": {"sides": ["back"]}},
    "Children Cabinet": {"touch_wall": {"sides": ["back"]}},
    "TV Stand": {"touch_wall": {"sides": ["back"]}},
    "Drawer Chest / Corner cabinet": {"touch_wall": {"sides": ["back"]}},
    "Shelf": {"mount_type": "wall", "mount_height_m": 1.5, "mount_anchor": "center"},
    "Pendant Lamp": {"mount_type": "ceiling"},
    "Ceiling Lamp": {"mount_type": "ceiling"},
    "Floor Lamp": {"mount_type": "floor"},
    "Wall Lamp": {"mount_type": "wall", "mount_height_m": 1.6, "mount_anchor": "center"},
    "Coffee Table": {"mount_type": "floor"},
    "Corner/Side Table": {"mount_type": "floor"},
    "Round End Table": {"mount_type": "floor"},
    "Dining Table": {"mount_type": "floor"},
    "Desk": {"mount_type": "floor"},
    "Dressing Table": {"mount_type": "floor"},
    "Dining Chair": {"mount_type": "floor"},
    "Lounge Chair / Cafe Chair / Office Chair": {"mount_type": "floor"},
    "Dressing Chair": {"mount_type": "floor"},
    "Classic Chinese Chair": {"mount_type": "floor"},
    "Barstool": {"mount_type": "floor"},
    "armchair": {"mount_type": "floor"},
    "Three-Seat / Multi-seat Sofa": {"touch_wall": {"sides": ["back"]}},
    "Loveseat Sofa": {"touch_wall": {"sides": ["back"]}},
    "L-shaped Sofa": {"touch_wall": {"sides": ["back"]}},
    "Lazy Sofa": {"mount_type": "floor"},
    "Chaise Longue Sofa": {"touch_wall": {"sides": ["back"]}},
    "Footstool / Sofastool / Bed End Stool / Stool": {"mount_type": "floor"},
    "Nightstand": {"mount_type": "floor"},
}


ROOM_TYPE_PRIORS: dict[str, list[tuple[str, int]]] = {
    "bedroom": [
        ("King-size Bed", 1),
        ("Nightstand", 2),
        ("Wardrobe", 1),
        ("Drawer Chest / Corner cabinet", 1),
        ("Dressing Table", 1),
        ("Dressing Chair", 1),
        ("Ceiling Lamp", 1),
    ],
    "kids": [
        ("Kids Bed", 1),
        ("Children Cabinet", 1),
        ("Desk", 1),
        ("Lounge Chair / Cafe Chair / Office Chair", 1),
        ("Ceiling Lamp", 1),
    ],
    "living": [
        ("Three-Seat / Multi-seat Sofa", 1),
        ("armchair", 1),
        ("Coffee Table", 1),
        ("TV Stand", 1),
        ("Sideboard / Side Cabinet / Console Table", 1),
        ("Ceiling Lamp", 1),
    ],
    "dining": [
        ("Dining Table", 1),
        ("Dining Chair", 4),
        ("Sideboard / Side Cabinet / Console Table", 1),
        ("Ceiling Lamp", 1),
    ],
    "office": [
        ("Desk", 1),
        ("Lounge Chair / Cafe Chair / Office Chair", 1),
        ("Bookcase / jewelry Armoire", 1),
        ("Ceiling Lamp", 1),
    ],
}


CATEGORY_ALIASES: dict[str, list[str]] = {
    "King-size Bed": [
        "двуспаль", "кроват", "king size bed", "king-size bed", "double bed", "bed"
    ],
    "Single bed": [
        "односпаль", "single bed"
    ],
    "Kids Bed": [
        "детск", "kids bed", "child bed"
    ],
    "Nightstand": [
        "тумб", "nightstand", "bedside table", "bedside", "side table near bed"
    ],
    "Wardrobe": [
        "шкаф", "гардероб", "wardrobe", "closet"
    ],
    "Drawer Chest / Corner cabinet": [
        "комод", "drawer chest", "chest of drawers", "dresser"
    ],
    "Bookcase / jewelry Armoire": [
        "стеллаж", "книжн", "bookcase", "armoire"
    ],
    "Shelf": [
        "полк", "shelf", "shelves"
    ],
    "Desk": [
        "письмен", "рабоч", "desk", "work desk", "стол"
    ],
    "Dressing Table": [
        "туалетн", "vanity", "dressing table", "makeup table"
    ],
    "Dressing Chair": [
        "vanity chair", "dressing chair"
    ],
    "Dining Table": [
        "обеден", "dining table"
    ],
    "Dining Chair": [
        "обеден", "dining chair"
    ],
    "Lounge Chair / Cafe Chair / Office Chair": [
        "офисн кресл", "рабоч кресл", "office chair", "lounge chair", "chair", "стул"
    ],
    "armchair": [
        "кресл", "armchair", "easy chair"
    ],
    "Three-Seat / Multi-seat Sofa": [
        "диван", "sofa", "couch"
    ],
    "Loveseat Sofa": [
        "loveseat"
    ],
    "L-shaped Sofa": [
        "углов", "l shaped sofa", "l-shaped sofa"
    ],
    "Coffee Table": [
        "журналь", "кофейн", "coffee table"
    ],
    "Corner/Side Table": [
        "приставн", "боков", "side table", "corner table"
    ],
    "Round End Table": [
        "кругл", "round end table"
    ],
    "TV Stand": [
        "тв тумб", "тумб под телевиз", "tv stand", "television stand"
    ],
    "Sideboard / Side Cabinet / Console Table": [
        "буфет", "консол", "sideboard", "console table", "side cabinet"
    ],
    "Pendant Lamp": [
        "подвесн", "pendant lamp", "pendant light"
    ],
    "Ceiling Lamp": [
        "светиль", "люстр", "ceiling lamp", "ceiling light", "lamp", "light", "лампа"
    ],
    "Floor Lamp": [
        "торшер", "floor lamp"
    ],
    "Wall Lamp": [
        "бра", "настенн", "wall lamp", "wall light"
    ],
    "Barstool": [
        "барн", "barstool", "bar stool"
    ],
    "Children Cabinet": [
        "детск шкаф", "children cabinet"
    ],
    "Wine Cabinet": [
        "винн шкаф", "wine cabinet"
    ],
    "Classic Chinese Chair": [
        "classic chinese chair"
    ],
    "Chaise Longue Sofa": [
        "chaise longue"
    ],
    "Lazy Sofa": [
        "lazy sofa"
    ],
    "Footstool / Sofastool / Bed End Stool / Stool": [
        "пуфик", "банкетк", "табурет", "ottoman", "stool", "footstool"
    ],
}


NUMBER_WORDS: dict[str, int] = {
    "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "pair": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "один": 1, "одна": 1, "одно": 1,
    "два": 2, "две": 2, "пара": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}


LIGHT_PRIORITY = ["Ceiling Lamp", "Pendant Lamp", "Floor Lamp", "Wall Lamp"]


FORCED_CATEGORY_PREFERENCES: dict[str, list[str]] = {
    "Chair": [
        "Lounge Chair / Cafe Chair / Office Chair",
        "Dining Chair",
        "Dressing Chair",
        "Classic Chinese Chair",
        "armchair",
        "Barstool",
        "Footstool / Sofastool / Bed End Stool / Stool",
    ],
    "Lamp": [
        "Ceiling Lamp",
        "Floor Lamp",
        "Wall Lamp",
        "Pendant Lamp",
    ],
}


@dataclass
class LLMSettings:
    enabled: bool
    provider: str
    ollama_url: str
    ollama_models: list[str]
    timeout_sec: int
    temperature: float
    max_attempts: int
    think: str
    debug_dir: Optional[str]


def _as_xy_point(pt: Any) -> tuple[float, float]:
    if isinstance(pt, dict):
        return float(pt["x"]), float(pt["y"])
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    raise RuntimeError(f"Invalid floor_polygon point: {pt!r}")


def load_room_metrics(room_json_path: str) -> dict[str, Any]:
    p = Path(room_json_path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))

    room = data.get("room")
    if not isinstance(room, dict):
        raise RuntimeError("room.json: missing field 'room'")

    fp = room.get("floor_polygon")
    if not isinstance(fp, list) or len(fp) < 3:
        raise RuntimeError("room.json: floor_polygon must contain at least 3 points")

    poly = [_as_xy_point(x) for x in fp]
    xs = [x for x, _ in poly]
    ys = [y for _, y in poly]

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    ceiling_h = float(room.get("ceiling_height", room.get("ceiling_height_m", room.get("height_m", 2.7))))

    area = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        area += x1 * y2 - x2 * y1
    area = abs(area) * 0.5

    return {
        "room_json_path": str(p),
        "span_x_m": float(span_x),
        "span_y_m": float(span_y),
        "ceiling_height_m": float(ceiling_h),
        "area_m2": float(area),
        "polygon": poly,
    }


def load_prepared_info(path: str) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("prepared_model_info.json must be a list")
    return data


def model_obj_path(future_root: str, model_id: str) -> Optional[Path]:
    d = Path(future_root).expanduser().resolve() / model_id
    p1 = d / "normalized_model.obj"
    p2 = d / "raw_model.obj"
    if p1.is_file():
        return p1
    if p2.is_file():
        return p2
    return None


def fits_room(entry: dict[str, Any], room_metrics: dict[str, Any], margin: float = 0.10) -> bool:
    sx = float(entry.get("size_x", 0.0))
    sy = float(entry.get("size_y", 0.0))
    sz = float(entry.get("size_z", 0.0))

    room_x = max(0.0, room_metrics["span_x_m"] - margin)
    room_y = max(0.0, room_metrics["span_y_m"] - margin)
    room_h = room_metrics["ceiling_height_m"]

    fit_xy = (sx <= room_x and sy <= room_y) or (sy <= room_x and sx <= room_y)
    fit_h = sz <= room_h + 1e-9
    return fit_xy and fit_h


def maybe_parse_prompt_as_structured_json(prompt_text: str) -> Optional[dict[str, Any]]:
    s = prompt_text.strip()
    if not s:
        return None
    if not (s.startswith("{") or s.startswith("[")):
        return None

    try:
        obj = json.loads(s)
    except Exception:
        return None

    if isinstance(obj, list):
        items = []
        for x in obj:
            if isinstance(x, str):
                items.append({"category": x, "count": 1})
            elif isinstance(x, dict) and isinstance(x.get("category"), str):
                items.append({"category": x["category"], "count": int(x.get("count", 1))})
        return {"items": items}

    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        items = []
        for x in obj["items"]:
            if isinstance(x, str):
                items.append({"category": x, "count": 1})
            elif isinstance(x, dict) and isinstance(x.get("category"), str):
                items.append({"category": x["category"], "count": int(x.get("count", 1))})
        return {"items": items}

    return None


_REQUIRED_ITEM_RE = re.compile(r"^\s*-\s*(.+?)\s*x\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_required_items_block(prompt_text: str) -> Optional[dict[str, Any]]:
    if "REQUIRED_ITEMS:" not in prompt_text:
        return None

    matches = list(_REQUIRED_ITEM_RE.finditer(prompt_text))
    if not matches:
        return None

    items: list[dict[str, Any]] = []
    for m in matches:
        category = m.group(1).strip()
        count = int(m.group(2))
        if category and count > 0:
            items.append({"category": category, "count": count})

    if not items:
        return None
    return {"items": items}


def infer_room_type(prompt_text: str) -> Optional[str]:
    p = norm(prompt_text)

    if any(x in p for x in ["детск", "kids", "child", "children"]):
        return "kids"
    if any(x in p for x in ["спаль", "bedroom"]):
        return "bedroom"
    if any(x in p for x in ["гостин", "living room", "livingroom", "living"]):
        return "living"
    if any(x in p for x in ["столов", "dining room", "diningroom", "dining"]):
        return "dining"
    if any(x in p for x in ["кабинет", "office", "workspace", "рабоч"]):
        return "office"

    return None


def _contains_alias(prompt_n: str, alias_n: str) -> bool:
    return alias_n in prompt_n


def _find_number_near_alias(prompt_n: str, alias_n: str) -> Optional[int]:
    idx = prompt_n.find(alias_n)
    if idx == -1:
        return None

    left = prompt_n[:idx].strip()
    if not left:
        return None

    tokens = left.split()
    tail = tokens[-5:] if len(tokens) >= 5 else tokens

    for t in reversed(tail):
        if t.isdigit():
            return max(1, int(t))
        if t in NUMBER_WORDS:
            return NUMBER_WORDS[t]

    return None


def _default_count_for_category(category: str, prompt_n: str) -> int:
    if category == "Nightstand":
        if "двуспаль" in prompt_n or "king" in prompt_n or "double bed" in prompt_n:
            return 2
        return 1
    return 1


def _first_available_from_preferences(names: list[str], available_categories: list[str]) -> Optional[str]:
    available_set = set(available_categories)
    for x in names:
        if x in available_set:
            return x
    return None


def _map_requested_category_to_available(requested_category: str, available_categories: list[str]) -> Optional[str]:
    requested_category = requested_category.strip()

    if requested_category in available_categories:
        return requested_category

    preferred = FORCED_CATEGORY_PREFERENCES.get(requested_category)
    if preferred:
        hit = _first_available_from_preferences(preferred, available_categories)
        if hit is not None:
            return hit

    requested_n = norm(requested_category)
    if requested_n in {"chair", "стул"}:
        hit = _first_available_from_preferences(FORCED_CATEGORY_PREFERENCES["Chair"], available_categories)
        if hit is not None:
            return hit

    if requested_n in {"lamp", "light", "лампа", "светильник", "люстра"}:
        hit = _first_available_from_preferences(FORCED_CATEGORY_PREFERENCES["Lamp"], available_categories)
        if hit is not None:
            return hit

    best_cat = None
    best_score = -1.0
    for cat in available_categories:
        s = fuzzy_score(requested_category, cat)
        if s > best_score:
            best_score = s
            best_cat = cat

    if best_score >= 0.45:
        return best_cat
    return None


def _match_category_from_prompt(prompt_text: str, available_categories: list[str]) -> dict[str, int]:
    prompt_n = norm(prompt_text)
    result: dict[str, int] = defaultdict(int)

    for category, aliases in CATEGORY_ALIASES.items():
        mapped_category = _map_requested_category_to_available(category, available_categories)
        if mapped_category is None:
            continue

        matched = False
        best_count = 0

        for alias in aliases:
            alias_n = norm(alias)
            if not alias_n:
                continue

            if _contains_alias(prompt_n, alias_n):
                matched = True
                cnt = _find_number_near_alias(prompt_n, alias_n)
                if cnt is None:
                    cnt = _default_count_for_category(mapped_category, prompt_n)
                best_count = max(best_count, cnt)

        if matched:
            result[mapped_category] = max(result[mapped_category], best_count or 1)

    if not any(k in result for k in LIGHT_PRIORITY):
        if any(x in prompt_n for x in ["светиль", "люстр", "lamp", "light", "lighting", "лампа"]):
            for k in LIGHT_PRIORITY:
                mapped = _map_requested_category_to_available(k, available_categories)
                if mapped is not None:
                    result[mapped] = max(result[mapped], 1)
                    break

    if not result:
        scored = []
        for cat in available_categories:
            score = fuzzy_score(prompt_text, cat)
            alias_bonus = 0.0

            for aliases_cat, aliases in CATEGORY_ALIASES.items():
                mapped = _map_requested_category_to_available(aliases_cat, available_categories)
                if mapped != cat:
                    continue
                for alias in aliases:
                    alias_bonus = max(alias_bonus, fuzzy_score(prompt_text, alias))

            total = max(score, alias_bonus)
            if total >= 0.34:
                scored.append((total, cat))

        scored.sort(reverse=True)

        for _, cat in scored[:5]:
            result[cat] += 1

    return dict(result)


def _estimate_max_items_by_area(area_m2: float) -> int:
    if area_m2 < 10.0:
        return 4
    if area_m2 < 18.0:
        return 6
    return 8


def _available_categories_compact(available_categories: list[str], max_items: int = 80) -> list[str]:
    if len(available_categories) <= max_items:
        return available_categories
    return available_categories[:max_items]


def _build_llm_system_prompt() -> str:
    return (
        "You convert an interior-design user prompt into a compact furniture request JSON.\n"
        "Return ONLY a JSON object with shape:\n"
        "{\n"
        '  "items": [\n'
        '    {"category": "<one of allowed categories>", "count": <integer >= 1>}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1. Use ONLY categories from the allowed categories list.\n"
        "2. Do not invent categories.\n"
        "3. Keep output compact and realistic for the room size.\n"
        "4. Prefer essential objects explicitly requested by the user.\n"
        "5. If the prompt is vague, infer a reasonable minimal set for the room type.\n"
        "6. Counts must be positive integers.\n"
        "7. No explanations. No markdown. JSON only."
    )


def _build_llm_user_prompt(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
) -> str:
    compact_categories = _available_categories_compact(available_categories)
    room_type = infer_room_type(prompt_text)
    max_items = _estimate_max_items_by_area(float(room_metrics["area_m2"]))

    payload = {
        "task": "Convert a user room prompt into furniture category request JSON.",
        "user_prompt": prompt_text,
        "room_metrics": {
            "span_x_m": round(float(room_metrics["span_x_m"]), 3),
            "span_y_m": round(float(room_metrics["span_y_m"]), 3),
            "ceiling_height_m": round(float(room_metrics["ceiling_height_m"]), 3),
            "area_m2": round(float(room_metrics["area_m2"]), 3),
        },
        "inferred_room_type": room_type,
        "soft_constraints": {
            "max_total_items": max_items,
            "prefer_minimal_layout": True,
        },
        "allowed_categories": compact_categories,
        "examples": [
            {
                "prompt": "small bedroom with a double bed, wardrobe and two nightstands",
                "json": {
                    "items": [
                        {"category": "King-size Bed", "count": 1},
                        {"category": "Nightstand", "count": 2},
                        {"category": "Wardrobe", "count": 1},
                    ]
                },
            },
            {
                "prompt": "cozy living room with a sofa, an armchair, a coffee table and lighting",
                "json": {
                    "items": [
                        {"category": "Three-Seat / Multi-seat Sofa", "count": 1},
                        {"category": "armchair", "count": 1},
                        {"category": "Coffee Table", "count": 1},
                        {"category": "Ceiling Lamp", "count": 1},
                    ]
                },
            },
        ],
        "required_output": {
            "items": [
                {"category": "string-from-allowed-list", "count": "positive-integer"}
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_ollama_text(resp: dict[str, Any]) -> str:
    message = resp.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

    response_text = resp.get("response")
    if isinstance(response_text, str):
        return response_text.strip()

    raise RuntimeError(f"Could not extract text from Ollama response: keys={list(resp.keys())}")


def _merge_same_categories(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agg: dict[str, int] = defaultdict(int)
    order: list[str] = []

    for x in items:
        category = x["category"]
        count = int(x["count"])
        if category not in agg:
            order.append(category)
        agg[category] += count

    return [{"category": cat, "count": agg[cat]} for cat in order if agg[cat] > 0]


def _normalize_layout_request_from_obj(
    obj: Any,
    available_categories: list[str],
    room_metrics: dict[str, Any],
) -> ValidationResult[dict[str, Any]]:
    if not isinstance(obj, dict):
        return ValidationResult(ok=False, feedback="Root JSON must be an object.")

    items = obj.get("items")
    if not isinstance(items, list):
        return ValidationResult(ok=False, feedback='Field "items" must be an array.')

    max_total_items = _estimate_max_items_by_area(float(room_metrics["area_m2"]))
    normalized_items: list[dict[str, Any]] = []
    feedbacks: list[str] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            feedbacks.append(f"items[{idx}] must be an object.")
            continue

        raw_category = item.get("category")
        raw_count = item.get("count", 1)

        if not isinstance(raw_category, str) or not raw_category.strip():
            feedbacks.append(f'items[{idx}].category must be a non-empty string.')
            continue

        try:
            count = int(raw_count)
        except Exception:
            feedbacks.append(f"items[{idx}].count must be an integer.")
            continue

        if count <= 0:
            feedbacks.append(f"items[{idx}].count must be > 0.")
            continue

        mapped = _map_requested_category_to_available(raw_category, available_categories)
        if mapped is None:
            feedbacks.append(
                f'items[{idx}].category="{raw_category}" does not match any available category.'
            )
            continue

        normalized_items.append({"category": mapped, "count": count})

    if feedbacks:
        return ValidationResult(ok=False, feedback="\n".join(feedbacks))

    if not normalized_items:
        return ValidationResult(ok=False, feedback="LLM returned an empty items list.")

    normalized_items = _merge_same_categories(normalized_items)

    total_count = sum(int(x["count"]) for x in normalized_items)
    if total_count > max_total_items:
        normalized_items = _truncate_items_to_limit(normalized_items, max_total_items)

    if not normalized_items:
        return ValidationResult(
            ok=False,
            feedback=f"After normalization, items became empty. Room limit: {max_total_items}.",
        )

    return ValidationResult(
        ok=True,
        normalized={"items": normalized_items},
        feedback="",
    )


def _truncate_items_to_limit(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    out: list[dict[str, Any]] = []
    used = 0
    for item in items:
        if used >= limit:
            break
        category = str(item["category"])
        count = int(item["count"])
        can_take = min(count, limit - used)
        if can_take > 0:
            out.append({"category": category, "count": can_take})
            used += can_take
    return out


def _validate_layout_request_text(
    raw_text: str,
    available_categories: list[str],
    room_metrics: dict[str, Any],
) -> ValidationResult[dict[str, Any]]:
    raw_text = raw_text.strip()
    if not raw_text:
        return ValidationResult(ok=False, feedback="Model returned an empty response.")

    try:
        obj = json.loads(raw_text)
    except Exception as e:
        return ValidationResult(ok=False, feedback=f"Response is not valid JSON: {e}")

    return _normalize_layout_request_from_obj(obj, available_categories, room_metrics)


def _debug_dir_for_model(base_debug_dir: Optional[str], model_name: str) -> Optional[str]:
    if base_debug_dir is None:
        return None
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return str((Path(base_debug_dir).expanduser().resolve() / safe_model).resolve())


def _generate_layout_request_via_ollama(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
    llm: LLMSettings,
) -> tuple[RetryResult[dict[str, Any]], str]:
    system_prompt = _build_llm_system_prompt()
    initial_prompt = _build_llm_user_prompt(
        prompt_text=prompt_text,
        room_metrics=room_metrics,
        available_categories=available_categories,
    )

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["category", "count"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    model_list = [m for m in llm.ollama_models if isinstance(m, str) and m.strip()]
    if not model_list:
        model_list = ["qwen3:30b", "gpt-oss:20b"]

    last_error: Optional[Exception] = None

    for model_name in model_list:
        try:
            model_debug_dir = _debug_dir_for_model(llm.debug_dir, model_name)

            def generate_fn(cur_prompt: str) -> str:
                resp = chat_json(
                    base_url=llm.ollama_url,
                    model=model_name,
                    system_prompt=system_prompt,
                    user_prompt=cur_prompt,
                    json_schema=schema,
                    timeout_sec=llm.timeout_sec,
                    temperature=llm.temperature,
                    think=llm.think,
                    extra_options={
                        "num_predict": 512,
                        "num_ctx": 8192,
                    },
                )
                return _extract_ollama_text(resp)

            def validate_fn(raw_text: str) -> ValidationResult[dict[str, Any]]:
                return _validate_layout_request_text(
                    raw_text=raw_text,
                    available_categories=available_categories,
                    room_metrics=room_metrics,
                )

            result = run_retry_loop(
                generate_fn=generate_fn,
                validate_fn=validate_fn,
                initial_prompt=initial_prompt,
                max_attempts=llm.max_attempts,
                debug_dir=model_debug_dir,
            )

            print(f"llm_model_success = {model_name}")
            return result, model_name

        except Exception as e:
            print(f"llm_model_failed = {model_name}: {e}")
            last_error = e
            continue

    raise RuntimeError(f"All Ollama models failed. Last error: {last_error}")


def heuristic_layout_request(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
) -> dict[str, Any]:
    area = float(room_metrics["area_m2"])

    explicit = _match_category_from_prompt(prompt_text, available_categories)
    if explicit:
        items = [{"category": cat, "count": cnt} for cat, cnt in explicit.items() if cnt > 0]
        return {"items": items}

    room_type = infer_room_type(prompt_text)
    priors = ROOM_TYPE_PRIORS.get(room_type or "", [])

    if not priors:
        priors = [
            ("Desk", 1),
            ("Lounge Chair / Cafe Chair / Office Chair", 1),
            ("Ceiling Lamp", 1),
        ]

    max_items = 4 if area < 10 else 6 if area < 18 else 8

    out: list[dict[str, Any]] = []
    current = 0

    for cat, cnt in priors:
        mapped = _map_requested_category_to_available(cat, available_categories)
        if mapped is None:
            continue

        use_cnt = cnt
        if current + use_cnt > max_items:
            use_cnt = max(0, max_items - current)
        if use_cnt <= 0:
            break

        out.append({"category": mapped, "count": use_cnt})
        current += use_cnt

    return {"items": out}


def draft_layout_request_via_llm(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
    llm_settings: LLMSettings,
) -> Optional[dict[str, Any]]:
    if not llm_settings.enabled:
        return None

    if llm_settings.provider != "ollama":
        print(f"llm_debug = unsupported_provider:{llm_settings.provider}")
        return None

    try:
        retry_result, selected_model = _generate_layout_request_via_ollama(
            prompt_text=prompt_text,
            room_metrics=room_metrics,
            available_categories=available_categories,
            llm=llm_settings,
        )

        print(
            "llm_debug =",
            json.dumps(
                {
                    "provider": llm_settings.provider,
                    "selected_model": selected_model,
                    "models_tried": llm_settings.ollama_models,
                    "attempts_used": retry_result.attempts_used,
                    "normalized": retry_result.normalized,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return retry_result.normalized

    except Exception as e:
        print(f"llm_debug = failed: {e}")
        return None


def build_layout_request(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
    llm_settings: LLMSettings,
) -> dict[str, Any]:
    required = parse_required_items_block(prompt_text)
    if required is not None:
        normalized = _normalize_layout_request_from_obj(required, available_categories, room_metrics)
        if normalized.ok and normalized.normalized is not None:
            return normalized.normalized
        raise RuntimeError(f"Could not normalize REQUIRED_ITEMS:\n{normalized.feedback}")

    structured = maybe_parse_prompt_as_structured_json(prompt_text)
    if structured is not None:
        normalized = _normalize_layout_request_from_obj(structured, available_categories, room_metrics)
        if normalized.ok and normalized.normalized is not None:
            return normalized.normalized

    llm_result = draft_layout_request_via_llm(
        prompt_text=prompt_text,
        room_metrics=room_metrics,
        available_categories=available_categories,
        llm_settings=llm_settings,
    )
    if llm_result is not None:
        return llm_result

    return heuristic_layout_request(prompt_text, room_metrics, available_categories)


def category_score(prompt_text: str, entry: dict[str, Any], target_category: str) -> float:
    text = " ".join([
        str(entry.get("category") or ""),
        str(entry.get("super-category") or ""),
        str(entry.get("style") or ""),
        str(entry.get("theme") or ""),
        str(entry.get("material") or ""),
    ])

    score = 0.0
    if entry.get("category") == target_category:
        score += 10.0

    score += 2.5 * fuzzy_score(prompt_text, text)
    score += 1.0 * fuzzy_score(target_category, str(entry.get("category") or ""))
    return score


def weighted_pick(scored_pool: list[tuple[float, dict[str, Any]]], rng: random.Random) -> dict[str, Any]:
    if not scored_pool:
        raise RuntimeError("weighted_pick: empty pool")

    weights = [max(1e-6, score) for score, _ in scored_pool]
    total = sum(weights)
    r = rng.random() * total

    acc = 0.0
    for w, (_, rec) in zip(weights, scored_pool):
        acc += w
        if r <= acc:
            return rec

    return scored_pool[-1][1]


def select_best_models(
    prepared: list[dict[str, Any]],
    layout_request: dict[str, Any],
    prompt_text: str,
    room_metrics: dict[str, Any],
    future_root: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_objects: list[dict[str, Any]] = []
    raw_selected: list[dict[str, Any]] = []

    by_category: dict[str, list[dict[str, Any]]] = {}
    for rec in prepared:
        cat = rec.get("category")
        if not isinstance(cat, str):
            continue
        if not fits_room(rec, room_metrics):
            continue
        by_category.setdefault(cat, []).append(rec)

    debug_stats: list[dict[str, Any]] = []

    for item_req in layout_request.get("items", []):
        category = item_req.get("category")
        count = int(item_req.get("count", 1))
        if not isinstance(category, str) or count <= 0:
            continue

        initial_pool = by_category.get(category, [])
        if not initial_pool:
            debug_stats.append({
                "category": category,
                "requested": count,
                "reason": "no_candidates_in_prepared",
            })
            continue

        existing_pool: list[dict[str, Any]] = []
        for rec in initial_pool:
            model_id = str(rec.get("model_id") or "")
            if not model_id:
                continue
            obj_path = model_obj_path(future_root, model_id)
            if obj_path is not None:
                existing_pool.append(rec)

        if not existing_pool:
            debug_stats.append({
                "category": category,
                "requested": count,
                "reason": "no_existing_obj_for_category",
                "prepared_candidates": len(initial_pool),
            })
            continue

        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in existing_pool:
            s = category_score(prompt_text, rec, category)
            scored.append((s, rec))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_k = min(10, len(scored))
        top_scored = scored[:top_k]

        added_for_category = 0
        used_model_ids: set[str] = set()

        while added_for_category < count:
            available_unique = [(s, rec) for s, rec in top_scored if str(rec["model_id"]) not in used_model_ids]
            if not available_unique:
                break

            rec = weighted_pick(available_unique, rng)
            model_id = str(rec["model_id"])
            obj_path = model_obj_path(future_root, model_id)
            if obj_path is None:
                used_model_ids.add(model_id)
                continue

            sx_mm = max(1, int(round(float(rec["size_x"]) * 2.0 * 1000.0)))
            sy_mm = max(1, int(round(float(rec["size_z"]) * 2.0 * 1000.0)))
            sz_mm = max(1, int(round(float(rec["size_y"]) * 2.0 * 1000.0)))

            constraints = dict(
                CATEGORY_CONSTRAINTS.get(str(rec.get("category") or ""), {"mount_type": "floor"})
            )

            item_obj = {
                "name": str(rec.get("category") or model_id),
                "min_size_mm": [sx_mm, sy_mm, sz_mm],
                "max_size_mm": [sx_mm, sy_mm, sz_mm],
                "color": [0.7, 0.7, 0.7],
                "constraints": constraints,
                "mesh_path": str(obj_path.resolve()),
                "mesh_fit_mode": "uniform",
                "mesh_texture_dirs": [str(obj_path.parent.resolve())],
                "asset_source": "3dfuture_prepared",
                "asset_meta": {
                    "model_id": model_id,
                    "super_category": rec.get("super-category"),
                    "category": rec.get("category"),
                    "style": rec.get("style"),
                    "theme": rec.get("theme"),
                    "material": rec.get("material"),
                    "dir": str(obj_path.parent.resolve()),
                },
            }

            selected_objects.append(item_obj)
            raw_selected.append(rec)
            used_model_ids.add(model_id)
            added_for_category += 1

        while added_for_category < count and top_scored:
            rec = weighted_pick(top_scored, rng)
            model_id = str(rec["model_id"])
            obj_path = model_obj_path(future_root, model_id)
            if obj_path is None:
                break

            sx_mm = max(1, int(round(float(rec["size_x"]) * 2.0 * 1000.0)))
            sy_mm = max(1, int(round(float(rec["size_z"]) * 2.0 * 1000.0)))
            sz_mm = max(1, int(round(float(rec["size_y"]) * 2.0 * 1000.0)))

            constraints = dict(
                CATEGORY_CONSTRAINTS.get(str(rec.get("category") or ""), {"mount_type": "floor"})
            )

            item_obj = {
                "name": str(rec.get("category") or model_id),
                "min_size_mm": [sx_mm, sy_mm, sz_mm],
                "max_size_mm": [sx_mm, sy_mm, sz_mm],
                "color": [0.7, 0.7, 0.7],
                "constraints": constraints,
                "mesh_path": str(obj_path.resolve()),
                "mesh_fit_mode": "uniform",
                "mesh_texture_dirs": [str(obj_path.parent.resolve())],
                "asset_source": "3dfuture_prepared",
                "asset_meta": {
                    "model_id": model_id,
                    "super_category": rec.get("super-category"),
                    "category": rec.get("category"),
                    "style": rec.get("style"),
                    "theme": rec.get("theme"),
                    "material": rec.get("material"),
                    "dir": str(obj_path.parent.resolve()),
                },
            }

            selected_objects.append(item_obj)
            raw_selected.append(rec)
            added_for_category += 1

        debug_stats.append({
            "category": category,
            "requested": count,
            "prepared_candidates": len(initial_pool),
            "existing_candidates": len(existing_pool),
            "added": added_for_category,
            "top_model_ids": [x[1].get("model_id") for x in scored[:5]],
        })

    print("select_debug =", json.dumps(debug_stats, ensure_ascii=False, indent=2))
    return selected_objects, raw_selected


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Choose furniture objects from prepared_model_info.json with optional LLM prompt parsing")

    p.add_argument("--room-json", required=True, help="Path to room.json")
    p.add_argument("--prompt", default=None, help="Prompt text")
    p.add_argument("--prompt-file", default=None, help="Path to prompt file")

    p.add_argument("--prepared-info", default=PREPARED_INFO_DEFAULT)
    p.add_argument("--future-root", default=FUTURE_ROOT_DEFAULT)
    p.add_argument("--out", default=DEFAULT_OUT, help="Output objects.json path")
    p.add_argument("--run-dir", default=None, help="Run directory for chooser logs")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--disable-llm", action="store_true")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ollama-model", default="qwen3:30b")
    p.add_argument("--ollama-models", nargs="*", default=None)
    p.add_argument("--ollama-timeout", type=int, default=600)
    p.add_argument("--ollama-temperature", type=float, default=0.0)
    p.add_argument("--llm-max-attempts", type=int, default=6)
    p.add_argument("--llm-think", choices=["low", "medium", "high"], default="low")
    p.add_argument("--llm-debug-dir", default=None)

    return p


def read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt is not None:
        return str(prompt)
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    raise RuntimeError("You must provide either --prompt or --prompt-file")


def build_llm_settings(args: argparse.Namespace, run_dir: Optional[str]) -> LLMSettings:
    debug_dir = args.llm_debug_dir
    if debug_dir is None and run_dir is not None:
        debug_dir = str((Path(run_dir).expanduser().resolve() / "llm_choose_debug").resolve())

    models: list[str] = []
    if args.ollama_models:
        models = [str(x).strip() for x in args.ollama_models if str(x).strip()]
    elif args.ollama_model:
        models = [str(args.ollama_model).strip()]

    if not models:
        models = ["qwen3:30b", "gpt-oss:20b"]

    return LLMSettings(
        enabled=(not bool(args.disable_llm)) and args.llm_provider != "none",
        provider=str(args.llm_provider),
        ollama_url=str(args.ollama_url),
        ollama_models=models,
        timeout_sec=int(args.ollama_timeout),
        temperature=float(args.ollama_temperature),
        max_attempts=int(args.llm_max_attempts),
        think=str(args.llm_think),
        debug_dir=debug_dir,
    )


def main() -> None:
    args = build_cli().parse_args()

    prompt_text = read_prompt(args.prompt, args.prompt_file)
    rng = random.Random(int(args.seed))

    room_metrics = load_room_metrics(args.room_json)
    prepared = load_prepared_info(args.prepared_info)
    available_categories = sorted({
        str(x.get("category"))
        for x in prepared
        if isinstance(x, dict) and isinstance(x.get("category"), str)
    })

    llm_settings = build_llm_settings(args, args.run_dir)

    layout_request = build_layout_request(
        prompt_text=prompt_text,
        room_metrics=room_metrics,
        available_categories=available_categories,
        llm_settings=llm_settings,
    )
    print("layout_request =", json.dumps(layout_request, ensure_ascii=False, indent=2))

    objects_items, selected_raw = select_best_models(
        prepared=prepared,
        layout_request=layout_request,
        prompt_text=prompt_text,
        room_metrics=room_metrics,
        future_root=args.future_root,
        rng=rng,
    )

    out_obj = {
        "seed": int(args.seed),
        "items": objects_items,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        chooser_meta = {
            "room_metrics": room_metrics,
            "layout_request": layout_request,
            "seed": int(args.seed),
            "llm": {
                "enabled": llm_settings.enabled,
                "provider": llm_settings.provider,
                "ollama_url": llm_settings.ollama_url,
                "ollama_models": llm_settings.ollama_models,
                "timeout_sec": llm_settings.timeout_sec,
                "temperature": llm_settings.temperature,
                "max_attempts": llm_settings.max_attempts,
                "think": llm_settings.think,
                "debug_dir": llm_settings.debug_dir,
            },
        }

        (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        (run_dir / "chooser_request.json").write_text(
            json.dumps(chooser_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "chooser_selected_raw.json").write_text(
            json.dumps(selected_raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"✅ objects.json saved: {out_path}")
    print(f"selected items: {len(objects_items)}")


if __name__ == "__main__":
    main()