from __future__ import annotations

import json
import math
import re
from typing import Any

from .kitchen_constants import COLOR_SYNONYMS, PATTERN_SYNONYMS, STYLE_KEYWORDS

_WORD_RE = re.compile(r"[a-zа-яё0-9_+.-]+", re.IGNORECASE)
_SIZE_RE = re.compile(r"(?P<a>\d{2,5})\s*[xх×*]\s*(?P<b>\d{2,5})(?:\s*[xх×*]\s*(?P<c>\d{1,5}))?", re.IGNORECASE)
_FLOAT_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower().replace("ё", "е").replace("×", "x").replace("х", "x")


def tokenize(value: Any) -> set[str]:
    return {m.group(0) for m in _WORD_RE.finditer(normalize_text(value))}


def parse_json_maybe(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    match = _FLOAT_RE.search(str(value).replace("\xa0", " ").replace(" ", "").replace(",", "."))
    if not match:
        return default
    try:
        return float(match.group(0))
    except Exception:
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    number = safe_float(value, None)
    return int(round(number)) if number is not None else default


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def extract_size_triplet_mm(*texts: Any) -> tuple[int | None, int | None, int | None]:
    for value in texts:
        match = _SIZE_RE.search(normalize_text(value))
        if not match:
            continue
        return safe_int(match.group("a")), safe_int(match.group("b")), safe_int(match.group("c")) if match.group("c") else None
    return None, None, None


def contains_any(text: Any, needles: list[str] | tuple[str, ...]) -> bool:
    haystack = normalize_text(text)
    return any(normalize_text(needle) in haystack for needle in needles)


def score_keyword_overlap(text: Any, desired: list[str] | tuple[str, ...] | set[str]) -> float:
    if not desired:
        return 0.5
    haystack = normalize_text(text)
    haystack_tokens = tokenize(haystack)
    hits = 0.0
    for item in desired:
        item_text = normalize_text(item)
        if not item_text:
            continue
        if item_text in haystack:
            hits += 1.0
            continue
        words = tokenize(item_text)
        if words and words.intersection(haystack_tokens):
            hits += 0.5
    return clamp01(hits / max(1.0, len(desired)))


def detect_color_families(*texts: Any) -> set[str]:
    joined = " ".join(normalize_text(text) for text in texts if text is not None)
    found: set[str] = set()
    for family, synonyms in COLOR_SYNONYMS.items():
        if any(normalize_text(synonym) in joined for synonym in synonyms):
            found.add(family)
    if "light_wood" in found or "dark_wood" in found:
        found.add("wood")
    if "stone" in found:
        found.add("gray")
    return found


def detect_pattern(*texts: Any) -> str:
    joined = " ".join(normalize_text(text) for text in texts if text is not None)
    for pattern, synonyms in PATTERN_SYNONYMS.items():
        if any(normalize_text(synonym) in joined for synonym in synonyms):
            return pattern
    return "decor"


def detect_finish(*texts: Any) -> str:
    joined = " ".join(normalize_text(text) for text in texts if text is not None)
    if any(x in joined for x in ("глянец", "gloss", "high gloss")):
        return "gloss"
    if any(x in joined for x in ("мат", "matt", "matte", "ms", "pe", "st9")):
        return "matte"
    if any(x in joined for x in ("структур", "texture", "woodgrain", "синхрон")):
        return "textured"
    return "unknown"


def detect_tone(colors: set[str], *texts: Any) -> str:
    joined = " ".join(normalize_text(text) for text in texts if text is not None)
    if colors.intersection({"white", "beige", "light_wood"}):
        return "light"
    if colors.intersection({"black", "dark_wood", "brown"}) or any(x in joined for x in ("антрацит", "графит", "темн", "тёмн")):
        return "dark"
    return "neutral"


def detect_style_tags(*texts: Any) -> list[str]:
    joined = " ".join(normalize_text(text) for text in texts if text is not None)
    tags: set[str] = set()
    for style, keywords in STYLE_KEYWORDS.items():
        if any(normalize_text(keyword) in joined for keyword in keywords):
            tags.add(style)
    colors = detect_color_families(joined)
    pattern = detect_pattern(joined)
    if colors.intersection({"white", "beige", "light_wood"}) and pattern in {"wood", "plain"}:
        tags.add("scandinavian")
    if pattern in {"marble", "stone", "concrete"} or colors.intersection({"gray", "black"}):
        tags.add("modern")
    return sorted(tags or {"modern"})


def normalize_color_request(colors: Any) -> list[str]:
    if not colors:
        return []
    if isinstance(colors, str):
        colors = [colors]
    result: list[str] = []
    for color in colors:
        families = detect_color_families(color)
        result.extend(sorted(families) if families else [normalize_text(color)])
    seen: set[str] = set()
    unique: list[str] = []
    for item in result:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
