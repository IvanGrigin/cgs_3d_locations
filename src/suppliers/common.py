# -*- coding: utf-8 -*-
"""
This module contains legacy standalone parsing helpers used by old scrapers.
It provides a lightweight asset schema and generic extraction utilities.
These functions support one-off catalog parsers outside the main adapter flow.
The code is intentionally generic and site-agnostic.
Keep it stable for backward compatibility with older scripts.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from src.suppliers.utils import DEFAULT_HEADERS


@dataclass
class ParsedAsset:
    supplier: str
    source_url: str

    title: Optional[str] = None
    brand: Optional[str] = None
    collection: Optional[str] = None
    designer: Optional[str] = None

    category_raw: Optional[str] = None
    category_norm: Optional[str] = None
    description: Optional[str] = None

    style: list[str] | None = None
    materials: list[str] | None = None
    colors: list[str] | None = None
    room_tags: list[str] | None = None

    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    height_m: Optional[float] = None

    price_value: Optional[float] = None
    price_currency: Optional[str] = None
    price_type: Optional[str] = None

    download_formats: list[str] | None = None
    blender_ready_score: Optional[float] = None

    preview_images: list[str] | None = None
    download_url: Optional[str] = None

    raw_meta: dict[str, Any] | None = None


def ensure_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def uniq_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_html(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def extract_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tag in soup.select('script[type="application/ld+json"]'):
        raw = tag.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue

        if isinstance(parsed, dict):
            out.append(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    out.append(item)
    return out


def first_jsonld_of_type(items: list[dict[str, Any]], expected_type: str) -> Optional[dict[str, Any]]:
    expected_type = expected_type.lower()
    for item in items:
        value = item.get("@type")
        if isinstance(value, str) and value.lower() == expected_type:
            return item
        if isinstance(value, list):
            for v in value:
                if isinstance(v, str) and v.lower() == expected_type:
                    return item
    return None


def text_or_none(node) -> Optional[str]:
    if node is None:
        return None
    text = node.get_text(" ", strip=True)
    return text or None


DIM_PATTERNS = [
    re.compile(
        r"(?:dimensions?|size)\s*(?:of\s+the\s+\w+)?\s*:\s*"
        r"(?:H\s*)?(?P<h>\d+(?:[.,]\d+)?)\s*x\s*"
        r"(?:L|W)\s*(?P<w>\d+(?:[.,]\d+)?)\s*x\s*"
        r"(?:D)\s*(?P<d>\d+(?:[.,]\d+)?)\s*cm",
        re.IGNORECASE,
    ),
    re.compile(
        r"width\s*[:=]\s*(?P<w>\d+(?:[.,]\d+)?)\s*cm.*?"
        r"depth\s*[:=]\s*(?P<d>\d+(?:[.,]\d+)?)\s*cm.*?"
        r"height\s*[:=]\s*(?P<h>\d+(?:[.,]\d+)?)\s*cm",
        re.IGNORECASE | re.DOTALL,
    ),
]


def parse_dimensions_from_text(text: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not text:
        return None, None, None

    normalized = text.replace("‎", " ").replace(",", ".")
    for pattern in DIM_PATTERNS:
        match = pattern.search(normalized)
        if match:
            h = float(match.group("h")) / 100.0
            w = float(match.group("w")) / 100.0
            d = float(match.group("d")) / 100.0
            return w, d, h

    return None, None, None


PRICE_PAT = re.compile(r"(?P<cur>[$€£])\s?(?P<val>\d[\d,\.]*)")


def parse_first_price(text: str) -> tuple[Optional[float], Optional[str]]:
    if not text:
        return None, None
    match = PRICE_PAT.search(text)
    if not match:
        return None, None
    raw_value = match.group("val").replace(",", "")
    try:
        value = float(raw_value)
    except Exception:
        return None, None
    return value, match.group("cur")


def normalize_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.lower()

    mapping = [
        ("nightstand", ["nightstand", "bedside table", "bedside cabinet", "прикроват"]),
        ("bed", ["bed", "кровать"]),
        ("wardrobe", ["wardrobe", "closet", "cupboard", "шкаф"]),
        ("sofa", ["sofa", "диван"]),
        ("armchair", ["armchair", "кресло"]),
        ("chair", ["chair", "стул"]),
        ("desk", ["desk", "table desk", "письменный стол", "рабочий стол"]),
        ("table", ["table", "стол"]),
        ("tv_stand", ["tv stand", "tv unit", "tv cabinet", "тв тумба", "tv тумба"]),
        ("bookcase", ["bookcase", "shelf", "shelving", "стеллаж", "полка", "буфет"]),
        ("rug", ["rug", "carpet", "ковер"]),
        ("ceiling_light", ["pendant lamp", "ceiling light", "suspension light", "люстра", "светильник"]),
        ("mirror", ["mirror", "зеркало"]),
        ("wall_art", ["wall art", "painting", "панно", "картина"]),
        ("plant", ["plant container", "vase", "pot", "кашпо"]),
        ("cabinet", ["cabinet", "тумба"]),
    ]

    for norm, keys in mapping:
        if any(key in s for key in keys):
            return norm
    return s.strip()


def compute_blender_ready_score(formats: list[str]) -> float:
    formats_l = {x.lower() for x in formats}
    score = 0.0
    if "glb" in formats_l or "gltf" in formats_l:
        score += 1.0
    if "fbx" in formats_l:
        score += 0.8
    if "blend" in formats_l:
        score += 0.8
    if "obj" in formats_l:
        score += 0.6
    if "3ds" in formats_l:
        score += 0.2
    if "max" in formats_l:
        score += 0.1
    return round(score, 3)


def save_assets_json(path: str | Path, assets: list[ParsedAsset]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(asset) for asset in assets]
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def polite_sleep(seconds: float = 1.0) -> None:
    time.sleep(seconds)
