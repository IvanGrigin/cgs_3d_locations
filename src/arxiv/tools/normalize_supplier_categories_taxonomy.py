#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TAXONOMY_VERSION = "supplier_category_100/v1"

TAXONOMY = [
    "chair",
    "dining_chair",
    "office_chair",
    "armchair",
    "sofa",
    "modular_sofa",
    "sofa_bed",
    "bench",
    "stool",
    "ottoman_pouf",
    "bed",
    "bunk_child_bed",
    "baby_crib",
    "desk",
    "dining_table",
    "kitchen_table",
    "coffee_table",
    "side_table",
    "console_table",
    "bedside_table",
    "dressing_table",
    "wardrobe",
    "sliding_builtin_wardrobe",
    "children_wardrobe",
    "low_cabinet",
    "wall_cabinet",
    "tall_cabinet",
    "chest_of_drawers",
    "sideboard",
    "tv_stand",
    "bookcase",
    "shelving_unit",
    "wall_shelf",
    "shoe_cabinet",
    "clothes_rack",
    "coat_rack",
    "storage_box_basket",
    "kitchen_base_cabinet",
    "kitchen_wall_cabinet",
    "kitchen_tall_cabinet",
    "kitchen_drawer_unit",
    "countertop",
    "kitchen_island",
    "backsplash",
    "kitchen_sink",
    "kitchen_faucet",
    "kitchenware",
    "food_drink",
    "refrigerator_freezer",
    "oven",
    "microwave",
    "cooktop_hob",
    "extractor_hood",
    "dishwasher",
    "washing_machine",
    "dryer_washer_dryer",
    "small_kitchen_appliance",
    "vacuum_cleaner",
    "climate_appliance",
    "iron_household_appliance",
    "toilet_bidet",
    "bathroom_sink",
    "bathtub",
    "shower_cabin",
    "shower_system",
    "shower_parts",
    "bathroom_faucet_mixer",
    "hygiene_shower",
    "towel_radiator",
    "bathroom_accessory",
    "bathroom_shelf_caddy",
    "bathroom_furniture",
    "ceiling_lamp",
    "chandelier",
    "pendant_lamp",
    "wall_light",
    "floor_lamp",
    "table_lamp",
    "desk_lamp",
    "recessed_spot_track_light",
    "led_strip",
    "tv_projector_screen",
    "computer_monitor",
    "laptop_computer_keyboard_mouse",
    "phone_tablet",
    "speaker_soundbar",
    "router_camera_smart_home",
    "hardware_handle_knob",
    "hook_rail",
    "mirror",
    "wall_art_frame_panel",
    "sculpture_decor_set",
    "plant_planter_vase",
    "rug",
    "curtain_blinds",
    "toy_hobby_game",
    "luggage_bag",
    "door",
    "material_texture_surface",
    "exterior_fence",
]

TAXONOMY_SET = set(TAXONOMY)


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


def _has(text: str, *parts: str) -> bool:
    return any(part in text for part in parts)


def _json_load(value: Any, default: Any) -> Any:
    if value in (None, "", b""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _row_value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except Exception:
        return getattr(row, key, None)


def _extract_h1(raw_html: str) -> str | None:
    if not raw_html:
        return None
    m = re.search(r"<h1[^>]*>(.*?)</h1>", raw_html, re.I | re.S)
    if not m:
        return None
    text = re.sub(r"<[^>]+>", " ", m.group(1))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _looks_like_placeholder_title(title: str) -> bool:
    if not title:
        return True
    t = _norm(title)
    return bool(
        re.fullmatch(r"\d+\s*model", t)
        or re.fullmatch(r"\d+[a-z\-_/ ]*", t)
        or t in {"model", "3d model"}
    )


def _effective_title(row: Any) -> str:
    title = str(_row_value(row, "title") or "").strip()
    h1 = _extract_h1(str(_row_value(row, "raw_html") or ""))
    if _looks_like_placeholder_title(title) and h1:
        return h1
    return title or (h1 or "")


def _source_text(row: Any) -> dict[str, str]:
    title = _effective_title(row)
    category_raw = str(_row_value(row, "category_raw") or "")
    legacy_norm = str(_row_value(row, "category_norm") or "")
    product_url = str(_row_value(row, "product_url") or "")
    description = str(_row_value(row, "description") or "")
    materials = str(_row_value(row, "materials") or "")
    images = " ".join(str(x) for x in _json_load(_row_value(row, "images_json"), []) if x)
    raw_html = str(_row_value(row, "raw_html") or "")
    text = _norm(" ".join([title, category_raw, legacy_norm, product_url, description, materials, images]))
    return {
        "title": _norm(title),
        "category_raw": _norm(category_raw),
        "legacy_norm": _norm(legacy_norm),
        "product_url": _norm(product_url),
        "description": _norm(description),
        "materials": _norm(materials),
        "images": _norm(images),
        "raw_html": _norm(raw_html),
        "text": text,
        "source_site": _norm(row["source_site"]),
    }


def _classify_appliance(text: str) -> tuple[str | None, str | None]:
    if _has(text, "стирал", "washing machine"):
        return "washing_machine", "title.appliance.washing_machine"
    if _has(text, "сушиль", "washer dryer", "washer-dryer", "dryer"):
        return "dryer_washer_dryer", "title.appliance.dryer"
    if _has(text, "посудом", "dishwasher"):
        return "dishwasher", "title.appliance.dishwasher"
    if _has(text, "микровол", "свч", "microwave"):
        return "microwave", "title.appliance.microwave"
    if _has(text, "духов", "oven"):
        return "oven", "title.appliance.oven"
    if _has(text, "вароч", "cooktop", "hob"):
        return "cooktop_hob", "title.appliance.cooktop"
    if _has(text, "вытяж", "hood"):
        return "extractor_hood", "title.appliance.hood"
    if _has(text, "холодиль", "refrigerator", "fridge", "freezer"):
        return "refrigerator_freezer", "title.appliance.fridge"
    if _has(text, "пылесос", "vacuum"):
        return "vacuum_cleaner", "title.appliance.vacuum"
    if _has(text, "кондиционер", "humidifier", "air purifier", "heater", "обогревател", "очиститель воздуха", "увлажнител"):
        return "climate_appliance", "title.appliance.climate"
    if _has(text, "утюг", "iron"):
        return "iron_household_appliance", "title.appliance.iron"
    if _has(text, "кофемаш", "чайник", "тостер", "blender", "mixer", "mixeur", "coffee machine", "juicer", "соковыжим", "мультиварк", "гриль"):
        return "small_kitchen_appliance", "title.appliance.small"
    return None, None


def _classify_electronics(text: str) -> tuple[str | None, str | None]:
    if _has(text, "iphone", "ipad", "promax", "pro max", "phone", "телефон", "смартфон", "tablet", "планшет"):
        return "phone_tablet", "title.electronics.phone_tablet"
    if _has(text, "soundbar", "speaker", "колонк", "акустик"):
        return "speaker_soundbar", "title.electronics.speaker"
    if _has(text, "router", "wi-fi", "wifi", "камера", "camera", "smart home", "умный дом"):
        return "router_camera_smart_home", "title.electronics.router_camera"
    if _has(text, "laptop", "notebook", "клавиатур", "keyboard", "mouse", "мышь", "компьютер"):
        return "laptop_computer_keyboard_mouse", "title.electronics.computer"
    if _has(text, "monitor", "display", "монитор"):
        return "computer_monitor", "title.electronics.monitor"
    if _has(text, "телевиз", "projector", "экран", "projection screen", "projector screen") or re.search(
        r"(^|[^a-z])tv([^a-z]|$)", text
    ):
        return "tv_projector_screen", "title.electronics.tv"
    return None, None


def _classify_lighting(text: str, category_raw: str) -> tuple[str | None, str | None]:
    if _has(text, "бра", "sconce", "wall light") or _has(category_raw, "бра"):
        return "wall_light", "lighting.wall"
    if _has(text, "торшер", "floor lamp") or _has(category_raw, "торшер"):
        return "floor_lamp", "lighting.floor"
    if _has(text, "настольная лампа", "table lamp") or _has(category_raw, "настольный"):
        if _has(text, "desk lamp", "рабочая лампа", "лампа для стола"):
            return "desk_lamp", "lighting.desk_lamp"
        return "table_lamp", "lighting.table_lamp"
    if _has(text, "люстра", "chandelier"):
        return "chandelier", "lighting.chandelier"
    if _has(text, "подвес", "pendant lamp", "pendant light", "hanging lamp", "suspension") or _has(category_raw, "подвесной"):
        return "pendant_lamp", "lighting.pendant"
    if _has(category_raw, "потолочный", "ceiling lamp") or _has(text, "ceiling lamp", "потолочный светильник"):
        return "ceiling_lamp", "lighting.ceiling"
    if _has(category_raw, "встроенный", "технический") or _has(text, "spot", "track light", "recessed", "встроенный светильник"):
        return "recessed_spot_track_light", "lighting.recessed"
    if _has(text, "led strip", "светодиодная лента"):
        return "led_strip", "lighting.led_strip"
    return None, None


def _classify_bathroom(text: str, category_raw: str) -> tuple[str | None, str | None]:
    if _has(category_raw, "гигиенические души") or _has(text, "гигиеническ"):
        return "hygiene_shower", "bathroom.hygiene_shower"
    if _has(category_raw, "комплектующие для душевых систем") or _has(text, "комплектующ", "душевое плечо", "верхняя лейка", "ручная лейка", "душевой шланг"):
        return "shower_parts", "bathroom.shower_parts"
    if _has(category_raw, "душевые системы") or _has(text, "душевая система", "shower system", "душевая стойка", "душевой комплект"):
        return "shower_system", "bathroom.shower_system"
    if _has(category_raw, "душевая кабина") or _has(text, "душевая кабина", "shower cabin"):
        return "shower_cabin", "bathroom.shower_cabin"
    if _has(category_raw, "мебель для ванной", "зеркальные шкафы", "санузел > мебель"):
        return "bathroom_furniture", "bathroom.furniture"
    if _has(category_raw, "унитаз", "биде") or _has(text, "унитаз", "bidet", "toilet"):
        return "toilet_bidet", "bathroom.toilet_bidet"
    if _has(category_raw, "для ванной") and _has(category_raw, "смесители"):
        return "bathroom_faucet_mixer", "bathroom.bath_faucet"
    if _has(category_raw, "для раковины", "смеситель") and not _has(category_raw, "кухни", "кухня"):
        if _has(category_raw, "для биде") or _has(text, "для биде"):
            return "bathroom_faucet_mixer", "bathroom.bidet_faucet"
        return "bathroom_faucet_mixer", "bathroom.faucet_mixer"
    if _has(category_raw, "раковины", "умывальники") or (_has(text, "раковин", "умывальник", "washbasin") and not _has(category_raw, "кухня", "мойка")):
        return "bathroom_sink", "bathroom.sink"
    if _has(category_raw, "полотенцесушитель") or _has(text, "полотенцесуш"):
        return "towel_radiator", "bathroom.towel_radiator"
    if _has(category_raw, "ванны") or _has(text, "ванна", "bathtub") or _has(category_raw, "рамы для ванн"):
        return "bathtub", "bathroom.bathtub"
    if _has(category_raw, "полки и крючки") and _has(text, "полк"):
        return "bathroom_shelf_caddy", "bathroom.shelf_caddy"
    if _has(category_raw, "аксессуары", "декор для санузла", "каталог продукции > аксессуары"):
        if _has(text, "полк"):
            return "bathroom_shelf_caddy", "bathroom.accessory_shelf"
        return "bathroom_accessory", "bathroom.accessory"
    return None, None


def _classify_kitchen(text: str, category_raw: str) -> tuple[str | None, str | None]:
    if _has(category_raw, "смесители для кухни", "кухня > смеситель") or _has(text, "kitchen faucet", "смеситель для кухни"):
        return "kitchen_faucet", "kitchen.faucet"
    if _has(category_raw, "кухня > мойка") or (_has(text, "мойка") and _has(category_raw, "кухня")):
        return "kitchen_sink", "kitchen.sink"
    if _has(category_raw, "кухня > мелочь для кухни"):
        return "kitchenware", "kitchen.kitchenware"
    if _has(category_raw, "кухня > еда и напитки"):
        return "food_drink", "kitchen.food_drink"
    if _has(category_raw, "кухня > техника", "техника > бытовая техника"):
        cat, rule = _classify_appliance(text)
        if cat:
            return cat, rule
        return "small_kitchen_appliance", "kitchen.tech_fallback"
    if _has(category_raw, "столешниц"):
        return "countertop", "kitchen.countertop"
    return None, None


def _classify_decor_misc(text: str, category_raw: str) -> tuple[str | None, str | None]:
    if _has(category_raw, "текстуры >", "материалы >"):
        return "material_texture_surface", "misc.material"
    if _has(category_raw, "двери") or _has(text, "door", "дверь"):
        return "door", "misc.door"
    if _has(category_raw, "ограждение") or _has(text, "fence", "огражден"):
        return "exterior_fence", "misc.fence"
    if _has(category_raw, "ковры") or _has(text, "rug", "ковер"):
        return "rug", "misc.rug"
    if _has(category_raw, "шторы") or _has(text, "curtain", "blind", "штор", "жалюз"):
        return "curtain_blinds", "misc.curtain"
    if _has(category_raw, "растения", "кашпо", "искусственные растения") or _has(text, "plant", "tree", "букет", "кашпо", "ваза"):
        return "plant_planter_vase", "misc.plant"
    if _has(category_raw, "багеты", "картины", "принты", "постеры", "лепнина") or _has(text, "картина", "print", "poster", "frame", "панно", "baguette"):
        return "wall_art_frame_panel", "misc.wall_art"
    if _has(category_raw, "скульптуры", "декоративный набор", "статуэтки", "сувениры", "другие предметы интерьера", "декор > другие предметы интерьера") or _has(text, "статуэт", "sculpt", "figur", "skull", "decor set"):
        return "sculpture_decor_set", "misc.sculpture_decor"
    if _has(category_raw, "чемодан") or _has(text, "luggage", "bag", "чемодан", "сумк"):
        return "luggage_bag", "misc.luggage"
    if _has(category_raw, "телефоны"):
        return "phone_tablet", "misc.phone"
    if _has(category_raw, "транспорт >", "скрипты >", "бильярд", "другие модели > разное", "другие предметы для детской", "игрушки", "магазин"):
        return "toy_hobby_game", "misc.toy_hobby_fallback"
    return None, None


def _classify_furniture(text: str, category_raw: str) -> tuple[str | None, str | None]:
    if _has(category_raw, "мебель > кресла", "кресла"):
        if _has(text, "office chair", "офисн", "компьютерн"):
            return "office_chair", "furniture.office_chair_from_raw"
        return "armchair", "furniture.armchair_from_raw"
    if _has(category_raw, "мебель > офисная мебель"):
        if _has(text, "chair", "стул", "кресло"):
            if _has(text, "office chair", "офисн", "компьютерн"):
                return "office_chair", "furniture.office_chair_office_raw"
            return "office_chair", "furniture.office_chair_generic"
        return "desk", "furniture.desk_office_raw"
    if _has(text, "office chair", "офисн") and _has(text, "chair", "стул", "кресло"):
        return "office_chair", "furniture.office_chair"
    if _has(text, "барный стул", "bar stool", "табурет", "stool"):
        return "stool", "furniture.stool"
    if _has(text, "банкет", "пуф", "пуфик", "footstool", "ottoman", "pouf"):
        return "ottoman_pouf", "furniture.ottoman"
    if _has(text, "лавка", "скам", "bench"):
        return "bench", "furniture.bench"
    if _has(text, "dining chair", "обеденный стул"):
        return "dining_chair", "furniture.dining_chair"
    if _has(text, "кресло", "armchair", "lounge chair", "easy chair"):
        return "armchair", "furniture.armchair"
    if _has(text, "стул", "chair"):
        return "chair", "furniture.chair"

    if _has(text, "sofa bed", "диван-кровать", "sleeper"):
        return "sofa_bed", "furniture.sofa_bed"
    if _has(text, "sectional", "модульн", "chaise", "corner", "угловой диван"):
        return "modular_sofa", "furniture.modular_sofa"
    if _has(text, "диван", "sofa", "seater", "loveseat"):
        return "sofa", "furniture.sofa"

    if _has(text, "bunk bed", "двухъярус") or _has(category_raw, "детская") and _has(text, "кровать"):
        return "bunk_child_bed", "furniture.bunk_bed"
    if _has(text, "crib", "кроватка", "baby bed"):
        return "baby_crib", "furniture.crib"
    if _has(text, "кровать", "bed"):
        return "bed", "furniture.bed"

    if _has(text, "письменный стол", "desk", "рабочий стол"):
        return "desk", "furniture.desk"
    if _has(text, "туалетный столик", "dressing table"):
        return "dressing_table", "furniture.dressing_table"
    if _has(text, "прикроват", "nightstand", "bedside"):
        return "bedside_table", "furniture.bedside_table"
    if _has(text, "консоль", "console table"):
        return "console_table", "furniture.console_table"
    if _has(text, "кофейный стол", "журналь", "coffee table"):
        return "coffee_table", "furniture.coffee_table"
    if _has(text, "side table", "приставной стол", "end table"):
        return "side_table", "furniture.side_table"
    if _has(text, "обеденный стол", "dining table"):
        return "dining_table", "furniture.dining_table"
    if _has(text, "кухонный стол", "bar table", "kitchen table"):
        return "kitchen_table", "furniture.kitchen_table"
    if _has(text, "стол", "table"):
        return "dining_table", "furniture.table_fallback"

    if _has(text, "шкаф-купе", "sliding wardrobe"):
        return "sliding_builtin_wardrobe", "furniture.sliding_wardrobe"
    if _has(category_raw, "детская > шкафы") or (_has(text, "детск") and _has(text, "шкаф", "wardrobe")):
        return "children_wardrobe", "furniture.children_wardrobe"
    if _has(text, "обувниц", "shoe cabinet"):
        return "shoe_cabinet", "furniture.shoe_cabinet"
    if _has(text, "tv stand", "тумба под телевизор", "тв тумба"):
        return "tv_stand", "furniture.tv_stand"
    if _has(text, "сервант", "буфет", "sideboard"):
        return "sideboard", "furniture.sideboard"
    if _has(text, "сундук", "trunk", "storage chest"):
        return "storage_box_basket", "furniture.storage_trunk"
    if _has(text, "комод", "dresser", "chest"):
        return "chest_of_drawers", "furniture.chest_of_drawers"
    if _has(text, "пенал") or _has(category_raw, "высокие шкафы"):
        return "tall_cabinet", "furniture.tall_cabinet"
    if _has(text, "навесной шкаф", "wall cabinet"):
        return "wall_cabinet", "furniture.wall_cabinet"
    if _has(text, "шкаф", "wardrobe", "closet", "cabinet"):
        return "wardrobe", "furniture.wardrobe"
    if _has(text, "bookcase", "держател для книг"):
        return "bookcase", "furniture.bookcase"
    if _has(text, "стеллаж", "shelving", "shelf unit"):
        return "shelving_unit", "furniture.shelving_unit"
    if _has(text, "полка", "wall shelf"):
        return "wall_shelf", "furniture.wall_shelf"
    if _has(text, "тумба", "low cabinet"):
        return "low_cabinet", "furniture.low_cabinet"

    if _has(text, "стойка для одежды", "clothes rack"):
        return "clothes_rack", "furniture.clothes_rack"
    if _has(text, "напольная вешалка", "coat rack", "вешало", "гардеробная стойка"):
        return "coat_rack", "furniture.coat_rack"
    if _has(text, "крюч", "hook", "rail"):
        return "hook_rail", "furniture.hook_rail"

    return None, None


def _classify_from_legacy_norm(legacy_norm: str, text: str) -> tuple[str | None, str | None]:
    if not legacy_norm or legacy_norm == "unknown":
        return None, None
    mapping = {
        "chair": "chair",
        "armchair": "armchair",
        "sofa": "sofa",
        "bed": "bed",
        "table": "dining_table",
        "coffee_table": "coffee_table",
        "console_table": "console_table",
        "desk": "desk",
        "cabinet": "low_cabinet",
        "sideboard": "sideboard",
        "bookcase": "shelving_unit",
        "mirror": "mirror",
        "lamp": "table_lamp",
        "wall_light": "wall_light",
        "floor_lamp": "floor_lamp",
        "chandelier": "chandelier",
        "hook": "hook_rail",
        "shelf": "wall_shelf",
        "clothes_rack": "clothes_rack",
        "ottoman": "ottoman_pouf",
        "plant": "plant_planter_vase",
        "artificial_plant": "plant_planter_vase",
        "artificial_tree": "plant_planter_vase",
        "planter": "plant_planter_vase",
        "bath_fixture": "shower_system",
        "bath_accessory": "bathroom_accessory",
    }
    category = mapping.get(legacy_norm)
    if category:
        if category == "dining_table":
            refined, rule = _classify_furniture(text, "")
            if refined:
                return refined, f"legacy.refined.{rule}"
        if category == "table_lamp":
            refined, rule = _classify_lighting(text, "")
            if refined:
                return refined, f"legacy.refined.{rule}"
        return category, f"legacy.{legacy_norm}"
    return None, None


def infer_category_from_mapping(row: Any) -> tuple[str, str, str]:
    src = _source_text(row)
    text = src["text"]
    category_raw = src["category_raw"]
    effective_title = _effective_title(row)

    def done(category: str, rule: str) -> tuple[str, str, str]:
        return category, rule, effective_title

    category, rule = _classify_appliance(text)
    if category:
        return done(category, rule or "appliance")

    category, rule = _classify_electronics(text)
    if category:
        return done(category, rule or "electronics")

    category, rule = _classify_bathroom(text, category_raw)
    if category:
        return done(category, rule or "bathroom")

    category, rule = _classify_kitchen(text, category_raw)
    if category:
        return done(category, rule or "kitchen")

    category, rule = _classify_lighting(text, category_raw)
    if category:
        return done(category, rule or "lighting")

    if _has(text, "зеркал", "mirror"):
        return done("mirror", "title.mirror")

    category, rule = _classify_furniture(text, category_raw)
    if category:
        return done(category, rule or "furniture")

    category, rule = _classify_decor_misc(text, category_raw)
    if category:
        return done(category, rule or "misc")

    category, rule = _classify_from_legacy_norm(src["legacy_norm"], text)
    if category:
        return done(category, rule or "legacy")

    if _has(category_raw, "все"):
        category, rule = _classify_furniture(text, category_raw)
        if category:
            return done(category, f"raw.vse.{rule}")
        category, rule = _classify_lighting(text, category_raw)
        if category:
            return done(category, f"raw.vse.{rule}")
        category, rule = _classify_decor_misc(text, category_raw)
        if category:
            return done(category, f"raw.vse.{rule}")

    if _has(src["source_site"], "loftdesigne"):
        if _has(src["raw_html"], "bed-depth.png"):
            return done("bed", "loft.image.bed")
        if _has(src["raw_html"], "chair.png"):
            return done("chair", "loft.image.chair")
        if _has(src["raw_html"], "office-chair.png"):
            return done("office_chair", "loft.image.office_chair")
        if _has(src["raw_html"], "rect-bar-chair.png", "round-bar-chair.png"):
            return done("stool", "loft.image.stool")
        if _has(src["raw_html"], "armchair.png"):
            return done("armchair", "loft.image.armchair")
        if _has(src["raw_html"], "couch.png"):
            return done("sofa", "loft.image.sofa")
        if _has(src["raw_html"], "console.png"):
            return done("console_table", "loft.image.console_table")
        if _has(src["raw_html"], "rect-coffee-table.png", "round-coffee-table.png"):
            return done("coffee_table", "loft.image.coffee_table")
        if _has(src["raw_html"], "rect-table_2.png", "table-round_2.png", "square-table.png"):
            return done("dining_table", "loft.image.table")
        if _has(src["raw_html"], "bar-table-scheme.png"):
            return done("kitchen_table", "loft.image.bar_table")
        if _has(src["raw_html"], "shkaf-scheme.png"):
            return done("wardrobe", "loft.image.wardrobe")
        if _has(src["raw_html"], "komod-scheme.png"):
            return done("chest_of_drawers", "loft.image.chest_of_drawers")
        if _has(src["raw_html"], "stellaz-scheme.png"):
            return done("shelving_unit", "loft.image.shelving_unit")
        if _has(src["raw_html"], "polka-scheme.png"):
            return done("wall_shelf", "loft.image.wall_shelf")
        if _has(src["raw_html"], "tumba-scheme.png"):
            return done("low_cabinet", "loft.image.low_cabinet")
        if _has(src["raw_html"], "mirror-rect_2.png", "mirror-round-hang-2.png", "mirror-1.png", "zerkalo-krugloe.png"):
            return done("mirror", "loft.image.mirror")
        if _has(src["raw_html"], "light-floor-round.png"):
            return done("floor_lamp", "loft.image.floor_lamp")
        if _has(src["raw_html"], "light-table-round.png", "light-table-rect.png"):
            return done("table_lamp", "loft.image.table_lamp")
        if _has(src["raw_html"], "light-ceiling-round.png"):
            return done("pendant_lamp", "loft.image.pendant_lamp")
        if _has(src["raw_html"], "icon-no-lamps.svg"):
            return done("table_lamp", "loft.image.lamp_fallback")

    return done("sculpture_decor_set", "fallback.decor")


def _infer_category(row: Any) -> tuple[str, str]:
    category, rule, _title = infer_category_from_mapping(row)
    return category, rule


def _build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = Counter()
    by_rule = Counter()
    samples = defaultdict(list)
    for item in results:
        by_category[item["category_norm"]] += 1
        by_rule[item["category_rule"]] += 1
        bucket = item["category_norm"]
        if len(samples[bucket]) < 5:
            samples[bucket].append(
                {
                    "title": item["title"],
                    "source_site": item["source_site"],
                    "category_raw": item["category_raw"],
                    "product_url": item["product_url"],
                    "category_rule": item["category_rule"],
                }
            )
    return {
        "schema": "supplier_category_taxonomy_report/v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "item_count": len(results),
        "categories": [
            {
                "category_norm": key,
                "count": by_category[key],
                "sample_items": samples[key],
            }
            for key in sorted(by_category, key=lambda x: (-by_category[x], x))
        ],
        "rules": [
            {"category_rule": key, "count": by_rule[key]}
            for key in sorted(by_rule, key=lambda x: (-by_rule[x], x))
        ],
    }


def _update_row(con: sqlite3.Connection, row: sqlite3.Row, category: str, rule: str) -> None:
    extra = _json_load(row["extra_json"], {})
    if not isinstance(extra, dict):
        extra = {}
    if "legacy_category_norm" not in extra:
        extra["legacy_category_norm"] = row["category_norm"]
    extra["category_taxonomy_version"] = TAXONOMY_VERSION
    extra["category_rule"] = rule
    extra["category_effective_title"] = _effective_title(row)
    con.execute(
        "UPDATE supplier_product SET category_norm=?, extra_json=? WHERE unique_key=?",
        (category, json.dumps(extra, ensure_ascii=False, sort_keys=True), row["unique_key"]),
    )


def normalize_db(db_path: Path, apply: bool, report_path: Path | None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT unique_key, source_site, category_raw, category_norm, title, product_url,
                   description, materials, images_json, extra_json, raw_html,
                   width_cm, depth_cm, height_cm
            FROM supplier_product
            ORDER BY source_site, title, unique_key
            """
        ).fetchall()

        for row in rows:
            category, rule = _infer_category(row)
            if category not in TAXONOMY_SET:
                raise RuntimeError(f"Category '{category}' is outside taxonomy for {row['unique_key']}")
            effective_title = _effective_title(row)
            results.append(
                {
                    "unique_key": row["unique_key"],
                    "source_site": row["source_site"],
                    "title": effective_title,
                    "product_url": row["product_url"],
                    "category_raw": row["category_raw"],
                    "old_category_norm": row["category_norm"],
                    "category_norm": category,
                    "category_rule": rule,
                }
            )
            if apply:
                _update_row(con, row, category, rule)

        if apply:
            con.commit()

    report = _build_report(results)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize supplier_product.category_norm to the 100-category taxonomy.")
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--apply", action="store_true", help="Write updates back to DB")
    ap.add_argument("--report", help="Optional JSON report path")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else None
    report = normalize_db(db_path=db_path, apply=bool(args.apply), report_path=report_path)

    print(f"taxonomy_version = {TAXONOMY_VERSION}")
    print(f"items = {report['item_count']}")
    print(f"categories = {len(report['categories'])}")
    print("top_categories =")
    for item in report["categories"][:20]:
        print(f"  {item['category_norm']}: {item['count']}")
    if report_path:
        print(f"report = {report_path}")
    print(f"applied = {bool(args.apply)}")


if __name__ == "__main__":
    main()
