#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/ChooseObject/choose_obj_from_prepared.py

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


PREPARED_INFO_DEFAULT = "data/sourse/3D-FRONT/prepared_model_info.json"
FUTURE_ROOT_DEFAULT = "data/sourse/3D-FRONT/3D-FUTURE-model"
DEFAULT_OUT = "data/input/objects.json"


# ------------------------------------------------------------
# Нормализация текста
# ------------------------------------------------------------
_word_re = re.compile(r"[А-Яа-яA-Za-z0-9/\-]+")


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


# ------------------------------------------------------------
# Категории и ограничения
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Алиасы категорий: русский + английский + смешанный prompt
# ------------------------------------------------------------
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
        "письмен", "рабоч", "desk", "work desk"
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
        "офисн кресл", "рабоч кресл", "office chair", "lounge chair"
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
        "светиль", "люстр", "ceiling lamp", "ceiling light", "lamp", "light"
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


# ------------------------------------------------------------
# Room metrics
# ------------------------------------------------------------
def _as_xy_point(pt: Any) -> tuple[float, float]:
    if isinstance(pt, dict):
        return float(pt["x"]), float(pt["y"])
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    raise RuntimeError(f"Неверная точка floor_polygon: {pt!r}")


def load_room_metrics(room_json_path: str) -> dict[str, Any]:
    p = Path(room_json_path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))

    room = data.get("room")
    if not isinstance(room, dict):
        raise RuntimeError("room.json: нет поля room")

    fp = room.get("floor_polygon")
    if not isinstance(fp, list) or len(fp) < 3:
        raise RuntimeError("room.json: floor_polygon должен содержать >= 3 точек")

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


# ------------------------------------------------------------
# Prepared assets
# ------------------------------------------------------------
def load_prepared_info(path: str) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("prepared_model_info.json должен быть массивом")
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


# ------------------------------------------------------------
# Будущий серверный LLM-хук
# ------------------------------------------------------------
def draft_layout_request_via_llm_stub(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
) -> Optional[dict[str, Any]]:
    _ = prompt_text, room_metrics, available_categories
    return None


# ------------------------------------------------------------
# Prompt parsing
# ------------------------------------------------------------
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


def _find_number_before_alias(prompt_n: str, alias_n: str) -> Optional[int]:
    idx = prompt_n.find(alias_n)
    if idx == -1:
        return None

    left = prompt_n[:idx].strip()
    if not left:
        return None

    tokens = left.split()
    tail = tokens[-4:] if len(tokens) >= 4 else tokens

    for t in reversed(tail):
        if t.isdigit():
            return max(1, int(t))
        if t in NUMBER_WORDS:
            return NUMBER_WORDS[t]

    return None


def _contains_alias(prompt_n: str, alias_n: str) -> bool:
    """
    Для коротких английских фраз ищем обычное вхождение.
    Для русских стемов тоже достаточно substring matching.
    """
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
    """
    Эвристика по умолчанию, если количество явно не найдено.
    Для спальни обычно 2 тумбочки.
    """
    if category == "Nightstand":
        if "двуспаль" in prompt_n or "king" in prompt_n or "double bed" in prompt_n:
            return 2
        return 1
    return 1


def _map_requested_category_to_available(requested_category: str, available_categories: list[str]) -> Optional[str]:
    """
    Если точной категории нет, ищем ближайшую из available.
    """
    if requested_category in available_categories:
        return requested_category

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
    """
    Извлекает категории и количества из русского/английского/смешанного prompt.
    Работает по стемам и не требует точных словоформ.
    """
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

    # Отдельная логика для света:
    # если есть общее упоминание света, но конкретная лампа не вытащилась,
    # выбираем первую доступную категорию освещения.
    if not any(k in result for k in LIGHT_PRIORITY):
        if any(x in prompt_n for x in ["светиль", "люстр", "lamp", "light", "lighting"]):
            for k in LIGHT_PRIORITY:
                mapped = _map_requested_category_to_available(k, available_categories)
                if mapped is not None:
                    result[mapped] = max(result[mapped], 1)
                    break

    # Fuzzy fallback только если вообще ничего не нашли
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

        # не больше 5 категорий из fallback
        for _, cat in scored[:5]:
            result[cat] += 1

    return dict(result)

def heuristic_layout_request(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
) -> dict[str, Any]:
    """
    1) Сначала пытаемся явно извлечь предметы из prompt.
    2) Если не получилось — fallback по типу комнаты.
    3) Категории из priors тоже прогоняем через fuzzy map к available.
    """
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

def build_layout_request(
    prompt_text: str,
    room_metrics: dict[str, Any],
    available_categories: list[str],
) -> dict[str, Any]:
    structured = maybe_parse_prompt_as_structured_json(prompt_text)
    if structured is not None:
        return structured

    llm_result = draft_layout_request_via_llm_stub(prompt_text, room_metrics, available_categories)
    if llm_result is not None:
        return llm_result

    return heuristic_layout_request(prompt_text, room_metrics, available_categories)


# ------------------------------------------------------------
# Выбор моделей
# ------------------------------------------------------------
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

        # 1) Сразу отфильтровываем только те модели, для которых реально есть OBJ
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

        # 2) Считаем score только для реально существующих моделей
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in existing_pool:
            s = category_score(prompt_text, rec, category)
            scored.append((s, rec))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 3) Берём top-K, но не случайно из мусора, а из реально хороших
        top_k = min(10, len(scored))
        ranked_pool = [x[1] for x in scored[:top_k]]

        added_for_category = 0
        used_model_ids: set[str] = set()

        # сначала пытаемся брать разные модели
        for rec in ranked_pool:
            if added_for_category >= count:
                break

            model_id = str(rec["model_id"])
            if model_id in used_model_ids:
                continue

            obj_path = model_obj_path(future_root, model_id)
            if obj_path is None:
                continue

            # ВАЖНО: sizes в 3D-FRONT boxes.npz — это half-size,
            # поэтому в objects.json нужно писать полный размер
            sx_mm = max(1, int(round(float(rec["size_x"]) * 2.0 * 1000.0)))  # width
            sy_mm = max(1, int(round(float(rec["size_z"]) * 2.0 * 1000.0)))  # depth
            sz_mm = max(1, int(round(float(rec["size_y"]) * 2.0 * 1000.0)))  # height

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

        # если разных моделей не хватило, разрешаем повторы из ranked_pool
        while added_for_category < count and ranked_pool:
            rec = rng.choice(ranked_pool)
            model_id = str(rec["model_id"])
            obj_path = model_obj_path(future_root, model_id)
            if obj_path is None:
                break

            sx_mm = max(1, int(round(float(rec["size_x"]) * 2.0 * 1000.0)))
            sy_mm = max(1, int(round(float(rec["size_y"]) * 2.0 * 1000.0)))
            sz_mm = max(1, int(round(float(rec["size_z"]) * 2.0 * 1000.0)))

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

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Выбор предметов из prepared_model_info.json")
    p.add_argument("--room-json", required=True, help="Путь к room.json")
    p.add_argument("--prompt", default=None, help="Текст prompt")
    p.add_argument("--prompt-file", default=None, help="Файл с prompt")
    p.add_argument("--prepared-info", default=PREPARED_INFO_DEFAULT)
    p.add_argument("--future-root", default=FUTURE_ROOT_DEFAULT)
    p.add_argument("--out", default=DEFAULT_OUT, help="Куда сохранить objects.json")
    p.add_argument("--run-dir", default=None, help="Папка run для логов выбора")
    p.add_argument("--seed", type=int, default=0)
    return p


def read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt is not None:
        return str(prompt)
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    raise RuntimeError("Нужно передать --prompt или --prompt-file")


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

    layout_request = build_layout_request(prompt_text, room_metrics, available_categories)
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

        (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        (run_dir / "chooser_request.json").write_text(
            json.dumps(
                {
                    "room_metrics": room_metrics,
                    "layout_request": layout_request,
                    "seed": int(args.seed),
                },
                ensure_ascii=False,
                indent=2,
            ),
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