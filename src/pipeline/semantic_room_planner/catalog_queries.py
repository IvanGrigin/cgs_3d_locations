from __future__ import annotations

from typing import Any

from .llm_client import call_json_llm


QUERY_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "desk": {
        "ru": ["письменный стол светлый дуб", "рабочий стол светлое дерево", "компьютерный стол черные ножки"],
        "en": ["light oak work desk", "modern student desk", "computer desk black metal legs"],
        "de": ["Schreibtisch Eiche hell", "Computertisch schwarze Beine"],
    },
    "office_chair": {
        "ru": ["кресло компьютерное черное", "офисное кресло черное", "рабочее кресло ткань черное"],
        "en": ["black office chair", "black computer chair", "fabric office chair"],
        "de": ["schwarzer Bürostuhl", "schwarzer Schreibtischstuhl"],
    },
    "bed": {
        "ru": ["кровать бежевая тканевая", "кровать односпальная бежевая", "кровать с мягким изголовьем"],
        "en": ["beige fabric bed", "single bed upholstered beige", "upholstered bed beige"],
        "de": ["beiges Polsterbett", "Einzelbett beige"],
    },
    "wardrobe": {
        "ru": ["шкаф белый", "шкаф для одежды белый", "шкаф светлый современный"],
        "en": ["white wardrobe", "modern white clothes wardrobe", "light wardrobe"],
        "de": ["weißer Kleiderschrank", "moderner heller Kleiderschrank"],
    },
    "dresser": {"ru": ["комод белый современный", "комод светлый дуб"], "en": ["modern white dresser", "light oak dresser"], "de": ["weiße Kommode", "Kommode Eiche hell"]},
    "shelf": {"ru": ["стеллаж светлый дуб", "книжный стеллаж современный"], "en": ["light oak shelf", "modern book shelf"], "de": ["Regal Eiche hell", "modernes Bücherregal"]},
    "bookcase": {"ru": ["стеллаж светлый дуб", "книжный стеллаж современный"], "en": ["light oak bookcase", "modern bookshelf"], "de": ["Bücherregal Eiche hell"]},
    "nightstand": {"ru": ["прикроватная тумба светлый дуб", "тумба прикроватная современная"], "en": ["light oak nightstand", "modern bedside table"], "de": ["Nachttisch Eiche hell"]},
    "table_lamp": {"ru": ["настольная лампа черная", "лампа настольная с белым абажуром"], "en": ["black table lamp", "desk lamp warm white shade"], "de": ["schwarze Tischlampe", "Schreibtischlampe"]},
    "laptop": {"ru": ["ноутбук серебристый", "ноутбук для учебы"], "en": ["silver laptop", "student laptop"], "de": ["silberner Laptop"]},
    "monitor": {"ru": ["монитор черный", "монитор для рабочего стола"], "en": ["black monitor", "desk computer monitor"], "de": ["schwarzer Monitor"]},
    "keyboard": {"ru": ["клавиатура черная", "компьютерная клавиатура"], "en": ["black keyboard", "computer keyboard"], "de": ["schwarze Tastatur"]},
    "mouse": {"ru": ["компьютерная мышь черная", "мышь для ноутбука"], "en": ["black computer mouse"], "de": ["schwarze Computermaus"]},
    "mug": {"ru": ["кружка белая керамическая", "кружка кофе белая"], "en": ["white ceramic mug", "coffee mug white"], "de": ["weiße Keramiktasse"]},
    "water_bottle": {"ru": ["бутылка для воды прозрачная", "бутылка воды синяя"], "en": ["transparent blue water bottle"], "de": ["transparente Trinkflasche blau"]},
    "notebook": {"ru": ["блокнот белый", "блокнот для учебы"], "en": ["student notebook", "white notebook"], "de": ["Notizbuch"]},
    "desk_organizer": {"ru": ["органайзер для стола черный", "настольный органайзер металл"], "en": ["black desk organizer", "metal desk organizer"], "de": ["schwarzer Schreibtisch Organizer"]},
    "pillow": {"ru": ["подушка белая", "подушка декоративная светлая"], "en": ["white pillow", "decorative pillow"], "de": ["weißes Kissen"]},
    "blanket": {"ru": ["одеяло бежевое", "плед бежевый"], "en": ["beige blanket", "beige throw blanket"], "de": ["beige Decke"]},
    "rug": {"ru": ["ковер бежево зеленый", "ковер современный для спальни"], "en": ["beige green rug", "modern bedroom rug"], "de": ["beiger grüner Teppich"]},
    "wall_art": {"ru": ["настенный постер зеленый бежевый", "постер в тонкой раме"], "en": ["muted green beige wall art"], "de": ["Wandbild grün beige"]},
    "mirror": {"ru": ["зеркало в черной раме", "настенное зеркало"], "en": ["black framed mirror", "wall mirror"], "de": ["Wandspiegel schwarzer Rahmen"]},
    "plant": {"ru": ["декоративное растение в белом горшке", "комнатное растение"], "en": ["decorative plant white pot", "indoor plant"], "de": ["Zimmerpflanze weißer Topf"]},
    "storage_box": {"ru": ["коробка для хранения бежевая", "тканевая коробка для хранения"], "en": ["beige storage box", "fabric storage box"], "de": ["beige Aufbewahrungsbox"]},
    "book": {"ru": ["книга декоративная", "книга для полки"], "en": ["decorative book", "book for shelf"], "de": ["dekoratives Buch"]},
    "phone": {"ru": ["телефон черный", "смартфон черный"], "en": ["black smartphone"], "de": ["schwarzes Smartphone"]},
    "toy_car": {
        "ru": ["игрушечная машинка", "модель автомобиля игрушка", "машинка для мальчика"],
        "en": ["toy car", "diecast car model", "kids toy car"],
        "de": ["Spielzeugauto", "Modellauto Spielzeug"],
    },
    "car_model": {
        "ru": ["модель автомобиля игрушка", "коллекционная машинка", "игрушечная модель машины"],
        "en": ["diecast car model", "toy car model", "collectible toy car"],
        "de": ["Modellauto Spielzeug", "Spielzeugauto Modell"],
    },
    "car_poster": {
        "ru": ["постер с машиной", "настенный постер автомобиль", "постер гоночная машина"],
        "en": ["car poster", "racing car wall poster", "automotive wall art"],
        "de": ["Auto Poster", "Rennwagen Wandposter"],
    },
    "racing_wall_art": {
        "ru": ["постер гоночная машина", "настенный постер автомобиль", "постер с машиной"],
        "en": ["racing car poster", "car wall art", "race car print"],
        "de": ["Rennwagen Poster", "Auto Wandbild"],
    },
    "racing_rug": {
        "ru": ["детский ковёр дорога", "ковёр с дорогами для машинок", "игровой коврик дорога"],
        "en": ["kids road rug", "racing road play rug", "car play mat rug"],
        "de": ["Kinderteppich Straße", "Spielteppich Straße"],
    },
    "road_play_mat": {
        "ru": ["игровой коврик дорога", "коврик с дорогами для машинок", "детский коврик дорога"],
        "en": ["road play mat", "kids car road play mat", "toy car play mat"],
        "de": ["Spielteppich Straße", "Straßen Spielmatte"],
    },
    "toy_storage_box": {
        "ru": ["коробка для машинок", "ящик для игрушечных машинок", "коробка для хранения игрушек"],
        "en": ["toy car storage box", "kids toy storage box", "car toy organizer"],
        "de": ["Spielzeugauto Aufbewahrungsbox", "Spielzeugkiste"],
    },
    "car_decor": {
        "ru": ["автомобильный декор детский", "декор машинка для полки", "фигурка машина декор"],
        "en": ["car decor for kids room", "car shelf decor", "car figurine decor"],
        "de": ["Auto Deko Kinderzimmer", "Auto Figur Dekoration"],
    },
}


def fallback_catalog_queries(obj: dict[str, Any]) -> dict[str, Any]:
    sc = str(obj.get("subclass") or "")
    if sc in QUERY_TEMPLATES:
        queries = QUERY_TEMPLATES[sc]
        return {"object_id": obj["id"], "catalog_queries": queries, "must_match": [x for x in sc.split("_") if x], "should_match": [x for x in [obj.get("color"), obj.get("material")] if x], "negative_keywords": ["children", "bar"] if "chair" in sc else []}
    color = str(obj.get("color") or "").strip()
    label = str(obj.get("label_en") or sc.replace("_", " "))
    ru_label = str(obj.get("label_ru") or label)
    material = str(obj.get("material") or "").strip()
    prefix = f"{color} " if color else ""
    ru_query = " ".join(x for x in [ru_label, color, material] if x).strip()
    return {"object_id": obj["id"], "catalog_queries": {"ru": [ru_query, f"{ru_label} {color}".strip()], "en": [f"{prefix}{label}".strip(), label], "de": [f"{prefix}{label}".strip()]}, "must_match": [x for x in sc.split("_") if x], "should_match": [x for x in [color, material] if x], "negative_keywords": ["children", "bar"] if "chair" in sc else []}


def generate_catalog_queries(objects: list[dict[str, Any]], llm_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    provider = str((llm_settings or {}).get("provider") or "none").strip().lower()
    use_llm_catalog = bool((llm_settings or {}).get("use_llm_catalog_queries", False))
    if provider == "none" or not use_llm_catalog:
        return {"schema": "catalog_queries/v1", "items": [fallback_catalog_queries(obj) for obj in objects], "source": "fallback_templates"}

    max_objects = max(1, min(int((llm_settings or {}).get("llm_catalog_max_objects", 8)), len(objects)))
    limited = objects[:max_objects]
    fallback_by_id = {obj["id"]: fallback_catalog_queries(obj) for obj in objects}
    prompt_objects = [
        {
            "object_id": obj["id"],
            "subclass": obj.get("subclass"),
            "label_ru": obj.get("label_ru"),
            "label_en": obj.get("label_en"),
            "color": obj.get("color"),
            "material": obj.get("material"),
        }
        for obj in limited
    ]
    messages = [
        {
            "role": "system",
            "content": "Return strict JSON only. Generate catalog text queries only. Do not output coordinates, placement, yaw, bbox, asset ids, or mesh paths.",
        },
        {
            "role": "user",
            "content": (
                "Return schema catalog_queries/v1 with key items. Each item must contain object_id, catalog_queries.ru/en/de arrays, "
                "must_match, should_match, negative_keywords. Generate queries for these objects as one batch:\n"
                f"{prompt_objects}"
            ),
        },
    ]
    try:
        call_settings = dict(llm_settings or {})
        call_settings.pop("use_llm_catalog_queries", None)
        call_settings.pop("llm_catalog_max_objects", None)
        data = call_json_llm(messages, **{**call_settings, "step_name": "12_catalog_queries_batch"})
        llm_items = data.get("items") if isinstance(data.get("items"), list) else []
        out_by_id = dict(fallback_by_id)
        for item in llm_items:
            if isinstance(item, dict) and item.get("object_id") in out_by_id:
                out_by_id[item["object_id"]] = item
        return {"schema": "catalog_queries/v1", "items": [out_by_id[obj["id"]] for obj in objects], "source": "llm_batch_with_fallback", "llm_catalog_max_objects": max_objects}
    except Exception as exc:
        return {"schema": "catalog_queries/v1", "items": [fallback_by_id[obj["id"]] for obj in objects], "source": "fallback_after_llm_catalog_error", "llm_error": str(exc)}
