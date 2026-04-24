#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_REFS: dict[str, dict[str, str]] = {
    "ad_popular_styles": {
        "title": "Architectural Digest - 10 Most Popular Interior Design Styles to Know Now",
        "url": "https://www.architecturaldigest.com/story/most-popular-interior-design-styles",
        "note": "Core styles such as Scandinavian, Japandi, Boho, Mediterranean, Country House, Midcentury, Industrial, Bauhaus, Minimalism, Traditional.",
    },
    "housebeautiful_30_styles": {
        "title": "House Beautiful - 30 Types of Interior Design Styles You Need to Know in 2026",
        "url": "https://www.housebeautiful.com/design-inspiration/a41613197/types-of-interior-design-styles/",
        "note": "Broad style list including Traditional, Transitional, Farmhouse, Americana, Rustic, Minimalist, Bohemian and many more.",
    },
    "houzz_12_styles": {
        "title": "Houzz - Your Guide to 12 Popular Decorating Styles",
        "url": "https://www.houzz.com/magazine/your-guide-to-12-popular-decorating-styles-stsetivw-vs~123846244",
        "note": "Practical definitions and hallmarks for Contemporary, Modern, Traditional, Midcentury, Farmhouse, Transitional, Industrial, Scandinavian, Rustic, Coastal, Eclectic and related styles.",
    },
    "spruce_27_styles": {
        "title": "The Spruce - 27 Interior Design Styles You Should Know to Help You Decorate Like a Pro",
        "url": "https://www.thespruce.com/interior-design-styles-guide-8606237",
        "note": "Expanded set with hybrids and newer styles such as Modern Farmhouse, Industrial Farmhouse, Coastal Farmhouse, Modern Cottage, Modern Organic, Maximalist, Moroccan, French Country.",
    },
    "housebeautiful_biophilic": {
        "title": "House Beautiful - What Is Biophilic Design?",
        "url": "https://www.housebeautiful.com/home-remodeling/interior-designers/a38425977/what-is-biophilic-design/",
        "note": "Biophilic design as a distinct nature-first style.",
    },
    "housebeautiful_english_country": {
        "title": "House Beautiful - The Enduring Appeal of English Country Style",
        "url": "https://www.housebeautiful.com/design-inspiration/a36814511/english-cottages/",
        "note": "Used as support for an English-country modifier tag.",
    },
    "housebeautiful_french_country": {
        "title": "House Beautiful - Everything You Need To Know About French Country Design",
        "url": "https://www.housebeautiful.com/design-inspiration/a24563993/french-country-design-style/",
        "note": "Used to map Provence-like signals to French Country.",
    },
    "spruce_modern_organic": {
        "title": "The Spruce - What Is Modern Organic Style?",
        "url": "https://www.thespruce.com/what-is-modern-organic-design-style-8412131",
        "note": "Used to normalize eco/natural/organic signals to Modern Organic.",
    },
}


STYLE_TAXONOMY: list[dict[str, Any]] = [
    {"id": "contemporary", "family": "modern_clean", "sources": ["houzz_12_styles", "housebeautiful_30_styles"]},
    {"id": "modern", "family": "modern_clean", "sources": ["houzz_12_styles", "spruce_27_styles"]},
    {"id": "midcentury_modern", "family": "modern_clean", "sources": ["ad_popular_styles", "houzz_12_styles", "housebeautiful_30_styles"]},
    {"id": "minimalist", "family": "modern_clean", "sources": ["ad_popular_styles", "housebeautiful_30_styles", "spruce_27_styles"]},
    {"id": "scandinavian", "family": "modern_clean", "sources": ["ad_popular_styles", "houzz_12_styles", "spruce_27_styles"]},
    {"id": "japandi", "family": "modern_clean", "sources": ["ad_popular_styles", "spruce_27_styles"]},
    {"id": "bauhaus", "family": "modern_clean", "sources": ["ad_popular_styles"]},
    {"id": "modern_organic", "family": "modern_clean", "sources": ["spruce_27_styles", "spruce_modern_organic"]},
    {"id": "biophilic", "family": "modern_clean", "sources": ["housebeautiful_biophilic"]},
    {"id": "industrial", "family": "urban_raw", "sources": ["ad_popular_styles", "houzz_12_styles", "spruce_27_styles"]},
    {"id": "bohemian", "family": "global_eclectic", "sources": ["ad_popular_styles", "housebeautiful_30_styles", "spruce_27_styles"]},
    {"id": "eclectic", "family": "global_eclectic", "sources": ["houzz_12_styles", "spruce_27_styles"]},
    {"id": "maximalist", "family": "global_eclectic", "sources": ["housebeautiful_30_styles", "spruce_27_styles"]},
    {"id": "moroccan", "family": "global_eclectic", "sources": ["spruce_27_styles"]},
    {"id": "coastal", "family": "casual_natural", "sources": ["houzz_12_styles", "spruce_27_styles"]},
    {"id": "farmhouse", "family": "casual_natural", "sources": ["housebeautiful_30_styles", "houzz_12_styles", "spruce_27_styles"]},
    {"id": "modern_farmhouse", "family": "casual_natural", "sources": ["housebeautiful_30_styles", "spruce_27_styles"]},
    {"id": "industrial_farmhouse", "family": "casual_natural", "sources": ["spruce_27_styles"]},
    {"id": "coastal_farmhouse", "family": "casual_natural", "sources": ["spruce_27_styles"]},
    {"id": "rustic", "family": "casual_natural", "sources": ["housebeautiful_30_styles", "houzz_12_styles", "spruce_27_styles"]},
    {"id": "country_house", "family": "casual_natural", "sources": ["ad_popular_styles"]},
    {"id": "modern_cottage", "family": "casual_natural", "sources": ["spruce_27_styles"]},
    {"id": "traditional", "family": "classic_decorative", "sources": ["ad_popular_styles", "houzz_12_styles", "spruce_27_styles", "housebeautiful_30_styles"]},
    {"id": "transitional", "family": "classic_decorative", "sources": ["houzz_12_styles", "spruce_27_styles", "housebeautiful_30_styles"]},
    {"id": "french_country", "family": "classic_decorative", "sources": ["houzz_12_styles", "spruce_27_styles", "housebeautiful_french_country"]},
    {"id": "americana", "family": "classic_decorative", "sources": ["housebeautiful_30_styles"]},
    {"id": "victorian", "family": "classic_decorative", "sources": ["housebeautiful_30_styles"]},
    {"id": "art_deco", "family": "classic_decorative", "sources": ["houzz_12_styles", "housebeautiful_30_styles"]},
    {"id": "hollywood_regency", "family": "classic_decorative", "sources": ["houzz_12_styles", "spruce_27_styles"]},
    {"id": "shabby_chic", "family": "classic_decorative", "sources": ["spruce_27_styles"]},
    {"id": "mediterranean", "family": "classic_decorative", "sources": ["ad_popular_styles", "houzz_12_styles", "spruce_27_styles"]},
    {"id": "retro", "family": "classic_decorative", "sources": ["housebeautiful_30_styles"]},
]


STYLE_PRIORITY = [
    "japandi",
    "midcentury_modern",
    "modern_organic",
    "biophilic",
    "modern_farmhouse",
    "industrial_farmhouse",
    "coastal_farmhouse",
    "french_country",
    "art_deco",
    "hollywood_regency",
    "scandinavian",
    "industrial",
    "bohemian",
    "moroccan",
    "mediterranean",
    "farmhouse",
    "rustic",
    "modern_cottage",
    "bauhaus",
    "victorian",
    "shabby_chic",
    "americana",
    "eclectic",
    "maximalist",
    "retro",
    "coastal",
    "country_house",
    "transitional",
    "traditional",
    "modern",
    "contemporary",
    "minimalist",
]


STYLE_FIELDS = [
    ("style_family_web", "TEXT"),
    ("style_primary_web", "TEXT"),
    ("style_tags_web_json", "TEXT"),
    ("style_confidence_web", "REAL"),
    ("style_method_web", "TEXT"),
    ("style_signals_web_json", "TEXT"),
]


STYLE_FAMILIES = {entry["id"]: entry["family"] for entry in STYLE_TAXONOMY}
STYLE_SET = {entry["id"] for entry in STYLE_TAXONOMY} | {"retro"}

RAW_STYLE_STYLE_MAP: dict[str, tuple[str, list[str]]] = {
    "современный": ("contemporary", []),
    "модерн": ("modern", []),
    "сканди": ("scandinavian", ["scandi"]),
    "скандинавский": ("scandinavian", ["scandi"]),
    "лофт": ("industrial", ["loft"]),
    "классический": ("traditional", []),
    "арт деко": ("art_deco", []),
    "ар-деко": ("art_deco", []),
    "арт-деко": ("art_deco", []),
    "прованс": ("french_country", ["provence"]),
    "эко": ("modern_organic", ["eco", "biophilic"]),
    "ретро": ("retro", ["vintage"]),
    "честерфилд": ("traditional", ["chesterfield"]),
    "английский": ("traditional", ["english_country"]),
    "американский": ("americana", ["americana"]),
    "итальянский": ("contemporary", ["italian"]),
    "этнический": ("eclectic", ["global"]),
}


TEXT_PATTERN_RULES: list[dict[str, Any]] = [
    {"style": "japandi", "patterns": ["japandi", "ваби саби", "wabi sabi"], "weight": 7, "method": "explicit_text"},
    {"style": "midcentury_modern", "patterns": ["midcentury", "mid-century", "mid century", "мидсенчури"], "weight": 7, "method": "explicit_text"},
    {"style": "modern_organic", "patterns": ["modern organic", "organic modern", "modern-organic", "органик"], "weight": 7, "method": "explicit_text"},
    {"style": "modern_farmhouse", "patterns": ["modern farmhouse"], "weight": 7, "method": "explicit_text"},
    {"style": "industrial_farmhouse", "patterns": ["industrial farmhouse"], "weight": 7, "method": "explicit_text"},
    {"style": "coastal_farmhouse", "patterns": ["coastal farmhouse"], "weight": 7, "method": "explicit_text"},
    {"style": "french_country", "patterns": ["french country", "прованс", "provence", "provencal"], "weight": 6, "method": "explicit_text"},
    {"style": "art_deco", "patterns": ["art deco", "арт деко", "арт-деко", "ар-деко"], "weight": 6, "method": "explicit_text"},
    {"style": "hollywood_regency", "patterns": ["hollywood regency", "regency"], "weight": 6, "method": "explicit_text"},
    {"style": "scandinavian", "patterns": ["scandinavian", "сканди", "скандинав", "nordic", "датск", "danish", "hygge"], "weight": 5, "method": "explicit_text"},
    {"style": "industrial", "patterns": ["industrial", "лофт", "warehouse", "raw steel"], "weight": 5, "method": "explicit_text"},
    {"style": "bohemian", "patterns": ["bohemian", "boho", "богем"], "weight": 5, "method": "explicit_text"},
    {"style": "moroccan", "patterns": ["moroccan", "марок"], "weight": 5, "method": "explicit_text"},
    {"style": "mediterranean", "patterns": ["mediterranean", "средизем"], "weight": 5, "method": "explicit_text"},
    {"style": "farmhouse", "patterns": ["farmhouse"], "weight": 5, "method": "explicit_text"},
    {"style": "rustic", "patterns": ["rustic", "рустик"], "weight": 5, "method": "explicit_text"},
    {"style": "modern_cottage", "patterns": ["modern cottage", "english cottage"], "weight": 5, "method": "explicit_text"},
    {"style": "bauhaus", "patterns": ["bauhaus", "баухаус"], "weight": 5, "method": "explicit_text"},
    {"style": "victorian", "patterns": ["victorian", "викториан"], "weight": 5, "method": "explicit_text"},
    {"style": "shabby_chic", "patterns": ["shabby chic", "шэбби"], "weight": 5, "method": "explicit_text"},
    {"style": "americana", "patterns": ["americana", "американский"], "weight": 5, "method": "explicit_text"},
    {"style": "biophilic", "patterns": ["biophilic", "биофил"], "weight": 5, "method": "explicit_text"},
    {"style": "eclectic", "patterns": ["eclectic", "эклек"], "weight": 5, "method": "explicit_text"},
    {"style": "maximalist", "patterns": ["maximal", "максимал"], "weight": 5, "method": "explicit_text"},
    {"style": "retro", "patterns": ["retro", "ретро", "винтаж", "vintage"], "weight": 5, "method": "explicit_text"},
    {"style": "coastal", "patterns": ["coastal", "морск", "beach house", "beach"], "weight": 5, "method": "explicit_text"},
    {"style": "country_house", "patterns": ["country house"], "weight": 5, "method": "explicit_text"},
    {"style": "transitional", "patterns": ["transitional"], "weight": 5, "method": "explicit_text"},
    {"style": "traditional", "patterns": ["traditional", "классичес"], "weight": 4, "method": "explicit_text"},
    {"style": "modern", "patterns": [" modern ", "модерн", "modernist"], "weight": 4, "method": "explicit_text"},
    {"style": "contemporary", "patterns": ["contemporary", "современ"], "weight": 4, "method": "explicit_text"},
    {"style": "minimalist", "patterns": ["minimalist", "minimalism", "минимал"], "weight": 4, "method": "explicit_text"},
]


MODIFIER_PATTERNS: list[tuple[str, list[str]]] = [
    ("provence", ["прованс", "provence", "provencal"]),
    ("chesterfield", ["chesterfield", "честерфилд"]),
    ("english_country", ["english", "англий", "british", "britain"]),
    ("italian", ["italian", "итал"]),
    ("scandi", ["scandi", "сканди", "скандинав"]),
    ("danish", ["danish", "датск"]),
    ("eco", [" eco ", " эко ", "sustainable", "биофил"]),
    ("biophilic", ["biophilic", "биофил"]),
    ("global", ["ethnic", "этничес", "global"]),
    ("loft", ["loft", "лофт"]),
    ("vintage", ["vintage", "винтаж"]),
]


def normalize(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).lower().replace("ё", "е")
    value = re.sub(r"[\n\r\t/|;:()\\[\\]{}!?,]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return f" {value} "


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def ensure_columns(con: sqlite3.Connection, table: str) -> None:
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in STYLE_FIELDS:
        if name in existing:
            continue
        con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
    con.commit()


def split_raw_style(style: str | None) -> list[str]:
    if not style:
        return []
    chunks = re.split(r"[,/|;]+", style)
    out: list[str] = []
    for chunk in chunks:
        value = normalize(chunk).strip()
        if value:
            out.append(value)
    return out


def add_score(
    scores: dict[str, float],
    style: str,
    weight: float,
    reason: str,
    reasons: list[dict[str, Any]],
) -> None:
    scores[style] += weight
    reasons.append({"style": style, "weight": weight, "reason": reason})


def map_raw_style(style: str | None) -> tuple[dict[str, float], set[str], list[dict[str, Any]]]:
    scores: dict[str, float] = defaultdict(float)
    tags: set[str] = set()
    reasons: list[dict[str, Any]] = []
    for token in split_raw_style(style):
        for key, (core_style, token_tags) in RAW_STYLE_STYLE_MAP.items():
            if key in token:
                add_score(scores, core_style, 6.0, f"raw_style:{key}", reasons)
                tags.update(token_tags)
    return scores, tags, reasons


def infer_from_text(text: str) -> tuple[dict[str, float], set[str], list[dict[str, Any]]]:
    scores: dict[str, float] = defaultdict(float)
    tags: set[str] = set()
    reasons: list[dict[str, Any]] = []
    for rule in TEXT_PATTERN_RULES:
        for pattern in rule["patterns"]:
            probe = normalize(pattern)
            if probe.strip() and probe in text:
                add_score(scores, rule["style"], rule["weight"], f"{rule['method']}:{pattern.strip()}", reasons)
                break
    for tag, patterns in MODIFIER_PATTERNS:
        for pattern in patterns:
            probe = normalize(pattern)
            if probe.strip() and probe in text:
                tags.add(tag)
                break
    return scores, tags, reasons


def infer_from_materials_and_category(row: sqlite3.Row) -> tuple[dict[str, float], set[str], list[dict[str, Any]]]:
    scores: dict[str, float] = defaultdict(float)
    tags: set[str] = set()
    reasons: list[dict[str, Any]] = []
    materials = normalize(row["materials"])
    category = normalize(row["category_raw"])
    title = normalize(row["title"])
    desc = normalize(row["description"])
    site = row["source_site"] or ""

    if (" leather " in materials or "кожа" in materials) and (" metal " in materials or "металл" in materials):
        add_score(scores, "industrial", 3.0, "materials:leather+metal", reasons)
    if (" raw wood " in materials or " дерево " in materials or " дуб " in materials or " oak " in materials) and (
        " linen " in materials or "лен" in materials or " cotton " in materials or "хлопок" in materials
    ):
        add_score(scores, "modern_organic", 2.0, "materials:wood+linen", reasons)
    if (" walnut " in materials or "орех" in materials or " teak " in materials or "тик" in materials) and (
        " leg " in desc or "ножк" in desc or "шпон" in desc or "veneer" in desc
    ):
        add_score(scores, "midcentury_modern", 2.0, "materials:walnut+legs", reasons)
    if (" rattan " in materials or " ротанг " in materials or " wicker " in materials or "плет" in desc) and (
        " sea " in desc or "морск" in desc or "coastal" in desc
    ):
        add_score(scores, "coastal", 2.0, "materials:rattan+coastal", reasons)
    if " velvet " in materials or "бархат" in materials or "velvet" in desc:
        if " gold " in materials or "золото" in desc or "золот" in desc or "mirror" in desc:
            add_score(scores, "art_deco", 2.0, "materials:velvet+gold", reasons)
        else:
            add_score(scores, "hollywood_regency", 1.5, "materials:velvet", reasons)
    if " chesterfield " in title or "chesterfield" in desc or "честерфилд" in desc:
        add_score(scores, "traditional", 3.0, "shape:chesterfield", reasons)
        tags.update({"chesterfield", "english_country"})
    if "датск" in desc or "danish" in desc or "hygge" in desc:
        add_score(scores, "scandinavian", 3.0, "description:danish", reasons)
        tags.add("danish")
    if "натуральн" in desc and ("дерев" in desc or "wood" in desc) and ("лаконич" in desc or "clean lines" in desc):
        add_score(scores, "modern_organic", 2.0, "description:natural+laconic", reasons)
    if "криволин" in desc or "curved" in desc or "organic shape" in desc:
        add_score(scores, "modern_organic", 1.5, "description:curved", reasons)
    if (" ванн " in category or "смесители" in category or "дозатор" in title or "душ" in title) and site == "sancos":
        add_score(scores, "contemporary", 2.5, "site_category:sancos_modern_bath", reasons)
        add_score(scores, "minimalist", 1.0, "site_category:sancos_modern_bath", reasons)
    if site == "sancos":
        add_score(scores, "contemporary", 1.5, "site_default:sancos", reasons)
    if site == "loftdesigne":
        add_score(scores, "contemporary", 1.5, "site_default:loftdesigne", reasons)
        if ("металл" in materials or " metal " in materials) and ("кожа" in materials or " leather " in materials):
            add_score(scores, "industrial", 2.0, "site_materials:loftdesigne_industrial", reasons)
    if site == "imodern":
        add_score(scores, "contemporary", 1.5, "site_default:imodern", reasons)
        if "лаконич" in desc:
            add_score(scores, "minimalist", 2.0, "description:lakonic", reasons)
        if "орех" in desc or " walnut " in desc:
            add_score(scores, "midcentury_modern", 1.5, "description:walnut", reasons)
        if "сканди" in desc:
            add_score(scores, "scandinavian", 2.0, "description:scandi", reasons)
    if site == "homeconcept":
        add_score(scores, "contemporary", 1.0, "site_default:homeconcept", reasons)
        if "датск" in desc or "сканди" in desc or "umage" in normalize(row["brand"]):
            add_score(scores, "scandinavian", 2.5, "brand_or_desc:homeconcept_scandi", reasons)
        if "растен" in desc or "treez" in normalize(row["brand"]):
            add_score(scores, "biophilic", 2.0, "brand_or_desc:treez", reasons)
    return scores, tags, reasons


def merge_scores(*parts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = defaultdict(float)
    for part in parts:
        for style, weight in part.items():
            merged[style] += weight
    return merged


def choose_primary(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    ordered = sorted(
        scores.items(),
        key=lambda item: (-item[1], STYLE_PRIORITY.index(item[0]) if item[0] in STYLE_PRIORITY else 999, item[0]),
    )
    return ordered[0][0]


def secondary_styles(scores: dict[str, float], primary: str | None) -> list[str]:
    if not primary:
        return []
    top_score = scores.get(primary, 0.0)
    out: list[str] = []
    for style, score in sorted(
        scores.items(),
        key=lambda item: (-item[1], STYLE_PRIORITY.index(item[0]) if item[0] in STYLE_PRIORITY else 999, item[0]),
    ):
        if style == primary:
            continue
        if score < 2.0:
            continue
        if score < top_score - 2.0:
            continue
        out.append(style)
    return out[:3]


def confidence_from_reasons(primary: str | None, scores: dict[str, float], reasons: list[dict[str, Any]]) -> float | None:
    if not primary:
        return None
    max_score = scores.get(primary, 0.0)
    reason_texts = {entry["reason"] for entry in reasons if entry["style"] == primary}
    if any(text.startswith("raw_style:") for text in reason_texts):
        return 0.95
    if any(text.startswith("explicit_text:") for text in reason_texts):
        return 0.9 if max_score >= 5 else 0.82
    if any(text.startswith("brand_prior:") for text in reason_texts):
        return 0.74
    if any(text.startswith("site_default:") for text in reason_texts):
        return 0.46 if max_score < 3.5 else 0.58
    if max_score >= 5:
        return 0.85
    if max_score >= 3:
        return 0.7
    return 0.55


def method_from_reasons(primary: str | None, reasons: list[dict[str, Any]]) -> str | None:
    if not primary:
        return None
    methods = [entry["reason"] for entry in reasons if entry["style"] == primary]
    if not methods:
        return None
    if any(text.startswith("raw_style:") for text in methods):
        return "raw_style"
    if any(text.startswith("explicit_text:") for text in methods):
        return "text_signal"
    if any(text.startswith("brand_prior:") for text in methods):
        return "brand_prior"
    if any(text.startswith("site_default:") for text in methods):
        return "site_default"
    return "heuristic"


def infer_raw_primary_for_priors(style: str | None) -> str | None:
    scores, _, _ = map_raw_style(style)
    return choose_primary(scores)


def build_brand_priors(rows: list[sqlite3.Row]) -> dict[tuple[str, str], tuple[str, float]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        brand = (row["brand"] or "").strip()
        site = (row["source_site"] or "").strip()
        if not brand or not site:
            continue
        primary = infer_raw_primary_for_priors(row["style"])
        if not primary:
            continue
        counts[(site, brand)][primary] += 1

    priors: dict[tuple[str, str], tuple[str, float]] = {}
    for key, counter in counts.items():
        total = sum(counter.values())
        if total < 3:
            continue
        primary, count = counter.most_common(1)[0]
        share = count / total
        if share >= 0.55:
            priors[key] = (primary, share)
    return priors


def classify_row(row: sqlite3.Row, brand_priors: dict[tuple[str, str], tuple[str, float]]) -> dict[str, Any]:
    raw_scores, raw_tags, raw_reasons = map_raw_style(row["style"])
    text = normalize(" ".join(
        str(value)
        for value in (
            row["style"],
            row["title"],
            row["description"],
            row["category_raw"],
            row["brand"],
            row["collection"],
            row["materials"],
            row["color"],
        )
        if value not in (None, "")
    ))
    text_scores, text_tags, text_reasons = infer_from_text(text)
    heuristic_scores, heuristic_tags, heuristic_reasons = infer_from_materials_and_category(row)

    brand_scores: dict[str, float] = defaultdict(float)
    brand_reasons: list[dict[str, Any]] = []
    brand = (row["brand"] or "").strip()
    site = (row["source_site"] or "").strip()
    if brand and site and (site, brand) in brand_priors:
        prior_style, share = brand_priors[(site, brand)]
        add_score(brand_scores, prior_style, 2.0 + share, f"brand_prior:{brand}", brand_reasons)

    scores = merge_scores(raw_scores, text_scores, heuristic_scores, brand_scores)
    primary = choose_primary(scores)
    tags = set(raw_tags) | set(text_tags) | set(heuristic_tags)
    tags.update(secondary_styles(scores, primary))

    # Keep modifier tags and avoid duplicating the primary style.
    tags.discard(primary or "")

    reasons = raw_reasons + text_reasons + heuristic_reasons + brand_reasons
    confidence = confidence_from_reasons(primary, scores, reasons)
    method = method_from_reasons(primary, reasons)

    return {
        "style_family_web": STYLE_FAMILIES.get(primary),
        "style_primary_web": primary,
        "style_tags_web_json": json_dumps(sorted(tags)) if tags else None,
        "style_confidence_web": confidence,
        "style_method_web": method,
        "style_signals_web_json": json_dumps(reasons[:24]) if reasons else None,
    }


def write_outputs(
    db_path: Path,
    taxonomy_json: Path,
    taxonomy_md: Path,
    report_json: Path,
    report_md: Path,
) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    taxonomy_payload = {
        "taxonomy_version": "web_style_taxonomy/v1",
        "sources": SOURCE_REFS,
        "styles": STYLE_TAXONOMY,
        "modifier_tags": [
            "provence",
            "chesterfield",
            "english_country",
            "italian",
            "scandi",
            "danish",
            "eco",
            "biophilic",
            "global",
            "loft",
            "vintage",
        ],
    }
    taxonomy_json.write_text(json_dumps(taxonomy_payload) + "\n", encoding="utf-8")

    md_lines = [
        "# Web Style Taxonomy",
        "",
        "Merged from internet style guides and then used to classify supplier items.",
        "",
        "## Sources",
    ]
    for ref in SOURCE_REFS.values():
        md_lines.append(f"- [{ref['title']}]({ref['url']})")
        md_lines.append(f"  {ref['note']}")
    md_lines += ["", "## Core Styles"]
    for style in STYLE_TAXONOMY:
        src_titles = ", ".join(SOURCE_REFS[source]["title"] for source in style["sources"])
        md_lines.append(f"- `{style['id']}` — family `{style['family']}`; supported by: {src_titles}")
    md_lines += ["", "## Modifier Tags", "- `provence`", "- `chesterfield`", "- `english_country`", "- `italian`", "- `scandi`", "- `danish`", "- `eco`", "- `biophilic`", "- `global`", "- `loft`", "- `vintage`"]
    taxonomy_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    total_all = con.execute("SELECT COUNT(*) FROM supplier_catalog_one_table").fetchone()[0]
    filled_all = con.execute(
        "SELECT COUNT(*) FROM supplier_catalog_one_table WHERE style_primary_web IS NOT NULL"
    ).fetchone()[0]
    total_accessible = con.execute(
        "SELECT COUNT(*) FROM supplier_catalog_one_table WHERE has_model_url=1 OR has_asset_local=1"
    ).fetchone()[0]
    filled_accessible = con.execute(
        "SELECT COUNT(*) FROM supplier_catalog_one_table WHERE (has_model_url=1 OR has_asset_local=1) AND style_primary_web IS NOT NULL"
    ).fetchone()[0]

    by_style_all = [
        dict(row)
        for row in con.execute(
            """
            SELECT style_primary_web, COUNT(*) AS row_count
            FROM supplier_catalog_one_table
            GROUP BY style_primary_web
            ORDER BY row_count DESC, style_primary_web
            """
        )
    ]
    by_style_accessible = [
        dict(row)
        for row in con.execute(
            """
            SELECT style_primary_web, COUNT(*) AS row_count
            FROM supplier_catalog_one_table
            WHERE has_model_url=1 OR has_asset_local=1
            GROUP BY style_primary_web
            ORDER BY row_count DESC, style_primary_web
            """
        )
    ]
    by_site_style_all = [
        dict(row)
        for row in con.execute(
            """
            SELECT source_site, style_primary_web, COUNT(*) AS row_count
            FROM supplier_catalog_one_table
            GROUP BY source_site, style_primary_web
            ORDER BY source_site, row_count DESC, style_primary_web
            """
        )
    ]
    by_site_style_accessible = [
        dict(row)
        for row in con.execute(
            """
            SELECT source_site, style_primary_web, COUNT(*) AS row_count
            FROM supplier_catalog_one_table
            WHERE has_model_url=1 OR has_asset_local=1
            GROUP BY source_site, style_primary_web
            ORDER BY source_site, row_count DESC, style_primary_web
            """
        )
    ]
    by_method_all = [
        dict(row)
        for row in con.execute(
            """
            SELECT style_method_web, COUNT(*) AS row_count
            FROM supplier_catalog_one_table
            GROUP BY style_method_web
            ORDER BY row_count DESC, style_method_web
            """
        )
    ]
    by_method_accessible = [
        dict(row)
        for row in con.execute(
            """
            SELECT style_method_web, COUNT(*) AS row_count
            FROM supplier_catalog_one_table
            WHERE has_model_url=1 OR has_asset_local=1
            GROUP BY style_method_web
            ORDER BY row_count DESC, style_method_web
            """
        )
    ]

    report_payload = {
        "db_path": str(db_path),
        "taxonomy_version": "web_style_taxonomy/v1",
        "all_items_total": total_all,
        "all_items_classified": filled_all,
        "all_items_coverage": round(filled_all / total_all, 4) if total_all else None,
        "accessible_items_total": total_accessible,
        "accessible_items_classified": filled_accessible,
        "accessible_items_coverage": round(filled_accessible / total_accessible, 4) if total_accessible else None,
        "by_style_all": by_style_all,
        "by_style_accessible": by_style_accessible,
        "by_site_style_all": by_site_style_all,
        "by_site_style_accessible": by_site_style_accessible,
        "by_method_all": by_method_all,
        "by_method_accessible": by_method_accessible,
    }
    report_json.write_text(json_dumps(report_payload) + "\n", encoding="utf-8")

    md = [
        "# Supplier Style Backfill Report",
        "",
        f"- All items: `{total_all}`",
        f"- Classified all items: `{filled_all}`",
        f"- All-items coverage: `{report_payload['all_items_coverage']}`",
        "",
        f"- Accessible items: `{total_accessible}`",
        f"- Classified accessible items: `{filled_accessible}`",
        f"- Coverage: `{report_payload['accessible_items_coverage']}`",
        "",
        "## By Style (All Items)",
    ]
    for row in by_style_all:
        md.append(f"- `{row['style_primary_web']}` — `{row['row_count']}`")
    md += ["", "## By Method (All Items)"]
    for row in by_method_all:
        md.append(f"- `{row['style_method_web']}` — `{row['row_count']}`")
    report_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill a web-derived style taxonomy into supplier_catalog_one_table.db")
    parser.add_argument(
        "--db",
        default="data/sourse/suppliers/supplier_catalog_one_table.db",
        help="Path to supplier_catalog_one_table.db",
    )
    parser.add_argument(
        "--only-accessible",
        action="store_true",
        help="Only classify rows that already have a model URL or a local asset",
    )
    parser.add_argument(
        "--taxonomy-json",
        default="data/sourse/suppliers/style_taxonomy_web_20260423.json",
        help="Path to write the taxonomy JSON",
    )
    parser.add_argument(
        "--taxonomy-md",
        default="data/sourse/suppliers/style_taxonomy_web_20260423.md",
        help="Path to write the taxonomy Markdown",
    )
    parser.add_argument(
        "--report-json",
        default="data/sourse/suppliers/style_backfill_report_20260423.json",
        help="Path to write the backfill report JSON",
    )
    parser.add_argument(
        "--report-md",
        default="data/sourse/suppliers/style_backfill_report_20260423.md",
        help="Path to write the backfill report Markdown",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    taxonomy_json = Path(args.taxonomy_json)
    taxonomy_md = Path(args.taxonomy_md)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    ensure_columns(con, "supplier_catalog_one_table")

    base_rows = con.execute(
        """
        SELECT rowid AS _rowid, *
        FROM supplier_catalog_one_table
        """
    ).fetchall()

    brand_priors = build_brand_priors(base_rows)

    where = "WHERE has_model_url=1 OR has_asset_local=1" if args.only_accessible else ""
    target_rows = con.execute(
        f"""
        SELECT rowid AS _rowid, *
        FROM supplier_catalog_one_table
        {where}
        """
    ).fetchall()

    updates: list[tuple[Any, ...]] = []
    for row in target_rows:
        inferred = classify_row(row, brand_priors)
        updates.append(
            (
                inferred["style_family_web"],
                inferred["style_primary_web"],
                inferred["style_tags_web_json"],
                inferred["style_confidence_web"],
                inferred["style_method_web"],
                inferred["style_signals_web_json"],
                row["_rowid"],
            )
        )

    con.executemany(
        """
        UPDATE supplier_catalog_one_table
        SET
            style_family_web=?,
            style_primary_web=?,
            style_tags_web_json=?,
            style_confidence_web=?,
            style_method_web=?,
            style_signals_web_json=?
        WHERE rowid=?
        """,
        updates,
    )
    con.commit()
    con.close()

    write_outputs(db_path, taxonomy_json, taxonomy_md, report_json, report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
