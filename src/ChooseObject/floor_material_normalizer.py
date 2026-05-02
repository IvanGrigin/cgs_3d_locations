from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FloorMaterial:
    version: str = "floor_material.v1"
    source: str = "domlenta"
    sku: str = ""
    name: str = ""
    brand: str = ""
    product_url: str = ""
    price: float | None = None
    price_currency: str = ""
    availability: str = "unknown"
    material_type: str = "unknown_floor_material"
    surface_group: str = "floor_covering"
    decor: str | None = None
    decor_name: str | None = None
    design: str | None = None
    tone: str | None = None
    tone_family: str | None = None
    gloss: str | None = None
    class_: int | None = None
    thickness_mm: float | None = None
    plank_width_mm: float | None = None
    plank_length_mm: float | None = None
    package_area_m2: float | None = None
    chamfer: str | None = None
    water_resistant: bool = False
    warm_floor_compatible: bool | None = None
    country: str | None = None
    description: str = ""
    raw_properties: dict[str, Any] = field(default_factory=dict)
    image_urls: list[str] = field(default_factory=list)
    local_image_paths: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    room_suitability: list[str] = field(default_factory=list)
    bad_for: list[str] = field(default_factory=list)
    search_text: str = ""
    parse_status: str = "ok"

    @property
    def class_value(self) -> int | None:
        return self.class_

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["class"] = data.pop("class_")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FloorMaterial":
        data = dict(data)
        if "class" in data:
            data["class_"] = data.pop("class")
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\xa0", " ")).strip()


def _lower(text: Any) -> str:
    return _norm(text).lower().replace("ё", "е")


def _safe_json(value: str, default: Any) -> Any:
    value = _norm(value)
    if not value:
        return default
    try:
        parsed = json.loads(value)
        return parsed if parsed is not None else default
    except Exception:
        return default


def _parse_float(value: Any) -> float | None:
    text = _norm(value)
    if not text:
        return None
    text = text.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else None


def _prop(props: dict[str, Any], names: list[str]) -> str:
    lowered = {_lower(k): v for k, v in props.items()}
    for name in names:
        key = _lower(name)
        if key in lowered:
            return _norm(lowered[key])
    return ""


def load_domlenta_products(products_csv: Path) -> list[dict[str, str]]:
    with Path(products_csv).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_availability(value: str) -> str:
    text = _lower(value)
    if "instock" in text or "in_stock" in text or "в наличии" in text:
        return "in_stock"
    if "outofstock" in text or "out_of_stock" in text or "нет в наличии" in text:
        return "out_of_stock"
    return "unknown"


def normalize_material_type(text: str) -> str:
    t = _lower(text)
    if "керамогранит" in t:
        return "porcelain_tile"
    if "инженерная доска" in t:
        return "engineered_wood"
    if "паркет" in t or "паркетная доска" in t:
        return "parquet_board"
    if "кварцвинил" in t or "spc" in t or "lvt" in t or "пвх" in t:
        return "vinyl_or_spc"
    if "ламинат" in t:
        return "laminate"
    if "линолеум" in t:
        return "linoleum"
    if "плитка" in t:
        return "ceramic_tile"
    return "unknown_floor_material"


def normalize_design_and_decor(text: str) -> tuple[str | None, str | None]:
    t = _lower(text)
    decor = None
    design = None
    for needle, value in [("дуб", "oak"), ("ясень", "ash"), ("орех", "walnut"), ("сосна", "pine")]:
        if needle in t:
            decor = value
            design = "wood"
            break
    if "дерево" in t or "деревян" in t or "под дерево" in t:
        design = "wood"
    if "бетон" in t:
        decor = decor or "concrete"
        design = "concrete"
    if "камень" in t:
        decor = decor or "stone"
        design = "stone"
    if "мрамор" in t:
        decor = decor or "marble"
        design = "marble"
    if "плитка" in t:
        design = design or "tile"
    if "однотон" in t:
        decor = decor or "plain"
        design = "plain"
    return decor, design


def normalize_tone(text: str) -> tuple[str | None, str | None]:
    t = _lower(text)
    explicit = [
        ("светл", "light", None), ("темн", "dark", "brown"), ("натурал", "natural", "brown"),
        ("сер", "gray", "gray"), ("беж", "beige", "beige"), ("коричнев", "brown", "brown"),
        ("бел", "white", "white"), ("черн", "black", "black"),
    ]
    for needle, tone, family in explicit:
        if needle in t:
            return tone, family or tone
    if "white" in t or "бел" in t:
        return "white", "white"
    if "gray" in t or "grey" in t or "сер" in t:
        return "gray", "gray"
    if "натурал" in t:
        return "natural", "brown"
    if any(x in t for x in ["табак", "венге", "шоколад", "dark"]):
        return "dark", "brown"
    if "дуб" in t:
        return "natural", "brown"
    return None, None


def normalize_chamfer(text: str) -> str | None:
    t = _lower(text)
    if "без фаски" in t:
        return "none"
    if "четырехсторон" in t or "4v" in t:
        return "four_sided"
    if "двухсторон" in t or "2v" in t:
        return "two_sided"
    return None


def _package_area(name: str, props: dict[str, Any]) -> float | None:
    value = _prop(props, [
        "Площадь в упаковке",
        "Количество м² в упаковке",
        "Площадь упаковки",
        "м² в упаковке",
        "Кол-во м2 (м.п.) в коробке",
        "Кол-во м² (м.п.) в коробке",
    ])
    parsed = _parse_float(value)
    if parsed:
        return parsed
    matches = re.findall(r"(\d+(?:[,.]\d+)?)\s*м²", name.lower())
    return _parse_float(matches[-1]) if matches else None


def _style_tags(material_type: str, decor: str | None, design: str | None, tone: str | None) -> list[str]:
    tags = {material_type}
    if design:
        tags.add(design)
    if decor:
        tags.add(decor)
    if tone:
        tags.add(tone)
    if design == "wood":
        tags.update(["natural", "warm"])
    if decor == "oak":
        tags.update(["oak", "wood"])
    if decor == "ash":
        tags.update(["ash", "light", "natural"])
    if decor == "walnut":
        tags.update(["walnut", "classic", "premium"])
    if design == "concrete":
        tags.update(["loft", "minimalism", "industrial"])
    if design == "marble":
        tags.update(["classic", "baroque", "premium"])
    if tone == "light":
        tags.update(["scandinavian", "japandi", "minimalism"])
    if tone == "dark":
        tags.update(["loft", "classic", "baroque"])
    if tone == "gray":
        tags.update(["minimalism", "loft", "contemporary"])
    if tone == "beige":
        tags.update(["japandi", "scandinavian", "contemporary"])
    return sorted(tags)


def _room_suitability(material_type: str, water_resistant: bool) -> tuple[list[str], list[str]]:
    mapping = {
        "laminate": (["bedroom", "living_room", "office", "hallway"], ["bathroom"] if not water_resistant else []),
        "parquet_board": (["bedroom", "living_room", "office"], ["bathroom", "wet_zone"]),
        "engineered_wood": (["bedroom", "living_room", "office"], ["bathroom", "wet_zone"]),
        "vinyl_or_spc": (["kitchen", "hallway", "bathroom", "living_room"], []),
        "ceramic_tile": (["bathroom", "kitchen", "hallway"], ["bedroom"]),
        "porcelain_tile": (["bathroom", "kitchen", "hallway", "living_room"], ["bedroom"]),
        "linoleum": (["kitchen", "hallway", "utility"], []),
    }
    return mapping.get(material_type, ([], []))


def normalize_product(row: dict[str, Any]) -> FloorMaterial:
    props = _safe_json(row.get("properties_json", ""), {})
    images = _safe_json(row.get("images_json", ""), [])
    local_images = _safe_json(row.get("local_image_paths_json", ""), [])
    room_recs = _safe_json(row.get("room_recommendations_json", ""), [])
    style_recs = _safe_json(row.get("style_recommendations_json", ""), [])
    if not isinstance(props, dict):
        props = {}
    if not isinstance(images, list):
        images = []
    if not isinstance(local_images, list):
        local_images = []
    if not isinstance(room_recs, list):
        room_recs = []
    if not isinstance(style_recs, list):
        style_recs = []

    name = _norm(row.get("name"))
    description = _norm(row.get("description"))
    all_text = " ".join([name, description, " ".join(f"{k} {v}" for k, v in props.items())])
    type_text = " ".join([_prop(props, ["Тип", "Тип товара", "Категория"]), name, row.get("categories", ""), row.get("breadcrumbs", "")])
    material_type = normalize_material_type(type_text)
    decor, design = normalize_design_and_decor(all_text)
    prop_design = _prop(props, ["Дизайн"])
    if prop_design:
        p_decor, p_design = normalize_design_and_decor(prop_design)
        decor = decor or p_decor
        design = design or p_design
    tone, tone_family = normalize_tone(" ".join([_prop(props, ["Оттенок", "Цвет", "Основной цвет"]), name]))
    water = any(x in _lower(all_text) for x in ["влагостойкий", "влагозащит", "aqua", "water resistant", "aquaout", "влагостойкость"])
    warm_floor = None
    if any(x in _lower(all_text) for x in ["теплый пол", "warm floor", "underfloor heating"]):
        warm_floor = True

    suitability, bad_for = _room_suitability(material_type, water)
    explicit_rooms = [
        _norm(item.get("room"))
        for item in room_recs
        if isinstance(item, dict) and _norm(item.get("room"))
    ]
    if explicit_rooms:
        suitability = sorted(set(suitability).union(explicit_rooms))
        bad_for = [room for room in bad_for if room not in suitability]
    decor_name = _prop(props, ["Название декора", "Декор"]) or None
    class_value = _parse_int(_prop(props, ["Класс", "Класс износостойкости", "Класс применения"]) or name)
    product_url = _norm(row.get("url") or row.get("final_url"))
    source = "mosplitka" if "mosplitka.ru" in product_url else "domlenta"

    material = FloorMaterial(
        source=source,
        sku=_norm(row.get("sku")) or (product_url.rstrip("/").split("-")[-1] if product_url else ""),
        name=name,
        brand=_norm(row.get("brand")) or _prop(props, ["Бренд"]),
        product_url=product_url,
        price=_parse_float(row.get("price")),
        price_currency=_norm(row.get("price_currency")) or "RUB",
        availability=normalize_availability(_norm(row.get("availability"))),
        material_type=material_type,
        decor=decor,
        decor_name=decor_name,
        design=design,
        tone=tone,
        tone_family=tone_family,
        class_=class_value,
        thickness_mm=_parse_float(_prop(props, ["Толщина планки", "Толщина", "Толщина покрытия", "Толщина, мм"])),
        plank_width_mm=_parse_float(_prop(props, ["Ширина планки", "Ширина", "Ширина доски"])),
        plank_length_mm=_parse_float(_prop(props, ["Длина планки", "Длина", "Длина доски"])),
        package_area_m2=_package_area(name, props),
        chamfer=normalize_chamfer(" ".join([_prop(props, ["Фаска"]), name])),
        water_resistant=water,
        warm_floor_compatible=warm_floor,
        country=_prop(props, ["Страна", "Страна производства"]) or None,
        description=description or _prop(props, ["Описание"]),
        raw_properties=props,
        image_urls=[_norm(x) for x in images if _norm(x)],
        local_image_paths=[_norm(x) for x in local_images if _norm(x)],
        room_suitability=suitability,
        bad_for=bad_for,
        parse_status=_norm(row.get("parse_status")) or "ok",
    )
    material.style_tags = _style_tags(material.material_type, material.decor, material.design, material.tone)
    explicit_styles = [
        _norm(item.get("style"))
        for item in style_recs
        if isinstance(item, dict) and _norm(item.get("style"))
    ]
    if explicit_styles:
        material.style_tags = sorted(set(material.style_tags).union(explicit_styles))
    material.search_text = _lower(" ".join([
        material.name, material.brand, material.description, material.material_type,
        material.decor or "", material.decor_name or "", material.design or "", material.tone or "",
        row.get("recommendations_text", ""),
        " ".join(f"{k} {v}" for k, v in props.items()),
    ]))
    return material


def write_jsonl(materials: list[FloorMaterial], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for material in materials:
            f.write(json.dumps(material.to_dict(), ensure_ascii=False) + "\n")


def normalize_domlenta_catalog(products_csv: Path, out_jsonl: Path) -> list[FloorMaterial]:
    materials: list[FloorMaterial] = []
    for row in load_domlenta_products(products_csv):
        try:
            materials.append(normalize_product(row))
        except Exception:
            materials.append(FloorMaterial(name=_norm(row.get("name")), product_url=_norm(row.get("url"))))
    write_jsonl(materials, out_jsonl)
    return materials


def load_normalized_materials(path: Path) -> list[FloorMaterial]:
    materials: list[FloorMaterial] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                materials.append(FloorMaterial.from_dict(json.loads(line)))
    return materials
