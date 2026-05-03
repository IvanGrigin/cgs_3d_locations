from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WallMaterial:
    version: str = "wall_material.v1"
    source: str = "domlenta"
    sku: str = ""
    name: str = ""
    brand: str = ""
    product_url: str = ""
    price: float | None = None
    price_currency: str = "RUB"
    availability: str = "unknown"
    material_type: str = "unknown_wall_material"
    surface_group: str = "wall_covering"
    color: str | None = None
    tone: str | None = None
    pattern: str | None = None
    finish: str | None = None
    base_material: str | None = None
    width_cm: float | None = None
    length_m: float | None = None
    country: str | None = None
    description: str = ""
    raw_properties: dict[str, Any] = field(default_factory=dict)
    image_urls: list[str] = field(default_factory=list)
    local_image_paths: list[str] = field(default_factory=list)
    average_rgb: list[int] | None = None
    average_hex: str | None = None
    dominant_colors_rgb: list[list[int]] = field(default_factory=list)
    dominant_colors_hex: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    room_suitability: list[str] = field(default_factory=list)
    search_text: str = ""
    parse_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WallMaterial":
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in dict(data).items() if k in allowed})


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def _lower(value: Any) -> str:
    return _norm(value).lower().replace("ё", "е")


def _safe_json(value: str, default: Any) -> Any:
    text = _norm(value)
    if not text:
        return default
    try:
        parsed = json.loads(text)
        return parsed if parsed is not None else default
    except Exception:
        return default


def _parse_float(value: Any) -> float | None:
    text = _norm(value).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _prop(props: dict[str, Any], names: list[str]) -> str:
    lowered = {_lower(k): v for k, v in props.items()}
    for name in names:
        key = _lower(name)
        if key in lowered:
            return _norm(lowered[key])
    return ""


def _rgb_to_hex(rgb: list[int] | tuple[int, int, int] | None) -> str | None:
    if not rgb or len(rgb) != 3:
        return None
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(x))) for x in rgb))


def normalize_availability(value: str) -> str:
    text = _lower(value)
    if "instock" in text or "in_stock" in text or "в наличии" in text:
        return "in_stock"
    if "outofstock" in text or "out_of_stock" in text or "нет в наличии" in text:
        return "out_of_stock"
    return "unknown"


def normalize_wall_material_type(text: str) -> str:
    t = _lower(text)
    if "фотооб" in t or "фото об" in t:
        return "photo_wallpaper"
    if "жидкие обои" in t:
        return "liquid_wallpaper"
    if "обои" in t:
        return "wallpaper"
    if "панел" in t:
        return "wall_panel"
    if "краск" in t:
        return "paint"
    return "unknown_wall_material"


def normalize_base_material(text: str) -> str | None:
    t = _lower(text)
    if "флизелин" in t or "fliz" in t:
        return "nonwoven"
    if "винил" in t:
        return "vinyl"
    if "бумаг" in t:
        return "paper"
    if "стеклооб" in t:
        return "fiberglass"
    return None


def normalize_color_and_tone(text: str) -> tuple[str | None, str | None]:
    t = _lower(text)
    checks = [
        ("бел", "white", "light"),
        ("молоч", "white", "light"),
        ("сер", "gray", "neutral"),
        ("графит", "gray", "dark"),
        ("беж", "beige", "warm_light"),
        ("крем", "beige", "warm_light"),
        ("песоч", "beige", "warm_light"),
        ("корич", "brown", "warm_dark"),
        ("зелен", "green", "accent"),
        ("олив", "green", "accent"),
        ("син", "blue", "accent"),
        ("голуб", "blue", "accent"),
        ("роз", "pink", "accent"),
        ("красн", "red", "accent"),
        ("желт", "yellow", "warm_light"),
        ("золот", "gold", "warm_light"),
        ("черн", "black", "dark"),
    ]
    for needle, color, tone in checks:
        if needle in t:
            return color, tone
    if "темн" in t:
        return None, "dark"
    if "светл" in t:
        return None, "light"
    return None, None


def normalize_pattern(text: str) -> str | None:
    t = _lower(text)
    checks = [
        ("однотон", "plain"),
        ("фон", "plain"),
        ("бетон", "concrete"),
        ("кирпич", "brick"),
        ("камень", "stone"),
        ("мрамор", "marble"),
        ("дерев", "wood"),
        ("полоск", "stripe"),
        ("геометр", "geometric"),
        ("листь", "botanical"),
        ("лист", "botanical"),
        ("цвет", "floral"),
        ("растен", "botanical"),
        ("орнамент", "ornament"),
        ("венз", "ornament"),
        ("дамаск", "damask"),
        ("детск", "kids"),
        ("динозав", "kids"),
        ("текстил", "textile"),
        ("лен", "textile"),
        ("штукатур", "plaster"),
    ]
    for needle, pattern in checks:
        if needle in t:
            return pattern
    return None


def _style_tags(material_type: str, color: str | None, tone: str | None, pattern: str | None) -> list[str]:
    tags = {material_type}
    if color:
        tags.add(color)
    if tone:
        tags.add(tone)
    if pattern:
        tags.add(pattern)
    if color in {"white", "beige", "gray"} or tone in {"light", "neutral", "warm_light"}:
        tags.update(["scandinavian", "japandi", "minimalism", "contemporary"])
    if color in {"black", "brown"} or tone in {"dark", "warm_dark"}:
        tags.update(["loft", "classic"])
    if pattern in {"concrete", "brick", "plaster"}:
        tags.update(["loft", "industrial", "minimalism"])
    if pattern in {"damask", "ornament", "marble"}:
        tags.update(["classic", "baroque"])
    if pattern in {"botanical", "floral"}:
        tags.update(["classic", "contemporary"])
    if pattern == "plain":
        tags.update(["minimalism", "scandinavian"])
    return sorted(tags)


def _sample_image_pixels(image_path: Path, max_pixels: int = 5000) -> list[tuple[int, int, int]]:
    try:
        from PIL import Image
    except ImportError:
        return []
    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((220, 220))
        pixels = list(image.getdata())
    except Exception:
        return []
    usable: list[tuple[int, int, int]] = []
    for r, g, b in pixels:
        if r >= 246 and g >= 246 and b >= 246:
            continue
        if r <= 6 and g <= 6 and b <= 6:
            continue
        usable.append((r, g, b))
    if len(usable) <= max_pixels:
        return usable
    step = max(1, len(usable) // max_pixels)
    return usable[::step][:max_pixels]


def _kmeans_rgb(pixels: list[tuple[int, int, int]], k: int = 5, iterations: int = 12) -> list[list[int]]:
    if not pixels:
        return []
    k = max(1, min(k, len(pixels)))
    ordered = sorted(pixels, key=lambda p: 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
    centers = [ordered[int((i + 0.5) * len(ordered) / k)] for i in range(k)]
    labels = [0] * len(pixels)
    for _ in range(iterations):
        buckets: list[list[tuple[int, int, int]]] = [[] for _ in range(k)]
        for idx, pixel in enumerate(pixels):
            best = min(range(k), key=lambda ci: sum((pixel[c] - centers[ci][c]) ** 2 for c in range(3)))
            labels[idx] = best
            buckets[best].append(pixel)
        new_centers = []
        for ci, bucket in enumerate(buckets):
            if not bucket:
                new_centers.append(centers[ci])
            else:
                new_centers.append(tuple(int(round(sum(p[c] for p in bucket) / len(bucket))) for c in range(3)))
        if new_centers == centers:
            break
        centers = new_centers
    counts = [labels.count(i) for i in range(k)]
    ranked = sorted(range(k), key=lambda i: counts[i], reverse=True)
    return [[int(x) for x in centers[i]] for i in ranked if counts[i] > 0]


def analyze_wallpaper_colors(base_dir: Path, local_image_paths: list[str], k: int = 5) -> dict[str, Any]:
    pixels: list[tuple[int, int, int]] = []
    for raw in local_image_paths[:3]:
        path = Path(raw)
        if not path.is_absolute():
            path = Path(base_dir) / path
        if path.exists():
            pixels.extend(_sample_image_pixels(path))
    if not pixels:
        return {"average_rgb": None, "average_hex": None, "dominant_colors_rgb": [], "dominant_colors_hex": []}
    avg = [int(round(sum(p[i] for p in pixels) / len(pixels))) for i in range(3)]
    palette = _kmeans_rgb(pixels, k=k)
    return {
        "average_rgb": avg,
        "average_hex": _rgb_to_hex(avg),
        "dominant_colors_rgb": palette,
        "dominant_colors_hex": [_rgb_to_hex(rgb) for rgb in palette if _rgb_to_hex(rgb)],
    }


def normalize_product(row: dict[str, Any], base_dir: Path | None = None, analyze_images: bool = True) -> WallMaterial:
    props = _safe_json(row.get("properties_json", ""), {})
    images = _safe_json(row.get("images_json", ""), [])
    local_images = _safe_json(row.get("local_image_paths_json", ""), [])
    if not isinstance(props, dict):
        props = {}
    if not isinstance(images, list):
        images = []
    if not isinstance(local_images, list):
        local_images = []

    name = _norm(row.get("name"))
    description = _norm(row.get("description"))
    all_text = " ".join([name, description, row.get("categories", ""), row.get("breadcrumbs", ""), " ".join(f"{k} {v}" for k, v in props.items())])
    color, tone = normalize_color_and_tone(" ".join([_prop(props, ["Цвет", "Основной цвет", "Оттенок"]), name, description]))
    pattern = normalize_pattern(" ".join([_prop(props, ["Рисунок", "Дизайн", "Узор", "Коллекция"]), name, description]))
    local_paths = [_norm(x) for x in local_images if _norm(x)]
    color_info = analyze_wallpaper_colors(base_dir or Path("."), local_paths) if analyze_images and local_paths else {}
    material_type = normalize_wall_material_type(all_text)
    base_material = normalize_base_material(all_text)
    material = WallMaterial(
        sku=_norm(row.get("sku")) or (_norm(row.get("url")).rstrip("/").split("-")[-1] if _norm(row.get("url")) else ""),
        name=name,
        brand=_norm(row.get("brand")) or _prop(props, ["Бренд"]),
        product_url=_norm(row.get("url") or row.get("final_url")),
        price=_parse_float(row.get("price")),
        price_currency=_norm(row.get("price_currency")) or "RUB",
        availability=normalize_availability(_norm(row.get("availability"))),
        material_type=material_type,
        color=color,
        tone=tone,
        pattern=pattern,
        finish=_prop(props, ["Фактура", "Поверхность"]) or None,
        base_material=base_material,
        width_cm=_parse_float(_prop(props, ["Ширина", "Ширина рулона"])),
        length_m=_parse_float(_prop(props, ["Длина", "Длина рулона"])),
        country=_prop(props, ["Страна", "Страна производства"]) or None,
        description=description or _prop(props, ["Описание"]),
        raw_properties=props,
        image_urls=[_norm(x) for x in images if _norm(x)],
        local_image_paths=local_paths,
        average_rgb=color_info.get("average_rgb"),
        average_hex=color_info.get("average_hex"),
        dominant_colors_rgb=color_info.get("dominant_colors_rgb") or [],
        dominant_colors_hex=color_info.get("dominant_colors_hex") or [],
        room_suitability=["bedroom", "living_room", "office", "children", "hallway"],
        parse_status=_norm(row.get("parse_status")) or "ok",
    )
    material.style_tags = _style_tags(material.material_type, material.color, material.tone, material.pattern)
    material.search_text = _lower(" ".join([
        material.name, material.brand, material.description, material.material_type,
        material.color or "", material.tone or "", material.pattern or "", material.base_material or "",
        " ".join(f"{k} {v}" for k, v in props.items()),
    ]))
    return material


def load_domlenta_products(products_csv: Path) -> list[dict[str, str]]:
    with Path(products_csv).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(materials: list[WallMaterial], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for material in materials:
            f.write(json.dumps(material.to_dict(), ensure_ascii=False) + "\n")


def normalize_domlenta_wallpapers_catalog(products_csv: Path, out_jsonl: Path, analyze_images: bool = True) -> list[WallMaterial]:
    base_dir = Path(products_csv).resolve().parent
    materials: list[WallMaterial] = []
    for row in load_domlenta_products(products_csv):
        try:
            materials.append(normalize_product(row, base_dir=base_dir, analyze_images=analyze_images))
        except Exception:
            materials.append(WallMaterial(name=_norm(row.get("name")), product_url=_norm(row.get("url"))))
    write_jsonl(materials, out_jsonl)
    return materials


def load_normalized_wall_materials(path: Path) -> list[WallMaterial]:
    path = Path(path).expanduser()
    if path.is_dir():
        return _load_wall_materials_from_dir(path)
    materials: list[WallMaterial] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            material = _wall_material_from_catalog_item(item, path.parent)
            if material is not None:
                materials.append(material)
    return _dedupe_wall_materials(materials)


def _surface_catalog_paths(root: Path) -> list[Path]:
    preferred_names = {"normalized_wall_materials.jsonl"}
    out: list[Path] = []
    for file_path in sorted(root.rglob("*.jsonl")):
        name = file_path.name
        if name in preferred_names:
            out.append(file_path)
            continue
        if "surface_materials" in name and "test" not in name:
            out.append(file_path)
    return out


def _load_wall_materials_from_dir(root: Path) -> list[WallMaterial]:
    materials: list[WallMaterial] = []
    for file_path in _surface_catalog_paths(root):
        try:
            materials.extend(load_normalized_wall_materials(file_path))
        except Exception:
            continue
    return _dedupe_wall_materials(materials)


def _dedupe_wall_materials(materials: list[WallMaterial]) -> list[WallMaterial]:
    out: list[WallMaterial] = []
    seen: set[tuple[str, str, str]] = set()
    for material in materials:
        key = (
            str(material.source or "").strip(),
            str(material.sku or "").strip(),
            str(material.product_url or material.name or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(material)
    return out


def _abs_local_paths(paths: list[Any], base_dir: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in paths:
        text = _norm(value)
        if not text:
            continue
        p = Path(text).expanduser()
        candidates = [p] if p.is_absolute() else [base_dir / p, Path.cwd() / p]
        resolved = None
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
            except Exception:
                pass
            if candidate.is_file():
                resolved = candidate
                break
        if resolved is None:
            resolved = candidates[0].resolve()
        resolved_text = str(resolved)
        if resolved_text not in seen:
            seen.add(resolved_text)
            out.append(resolved_text)
    return out


def _wall_material_from_catalog_item(item: dict[str, Any], base_dir: Path) -> WallMaterial | None:
    if not isinstance(item, dict):
        return None
    version = str(item.get("version") or "")
    if version.startswith("wall_material"):
        material = WallMaterial.from_dict(item)
        material.local_image_paths = _abs_local_paths(material.local_image_paths, base_dir)
        return material
    if not version.startswith("surface_material"):
        return None

    normalized = item.get("normalized") if isinstance(item.get("normalized"), dict) else {}
    if normalized.get("is_selectable_wall") is not True:
        return None
    image = item.get("material_image") if isinstance(item.get("material_image"), dict) else {}
    image_urls = [str(image.get("image_url") or "").strip()] if image.get("image_url") else []
    local_candidates = [image.get("source_path"), image.get("path")]
    raw_props = item.get("raw_properties") if isinstance(item.get("raw_properties"), dict) else {}
    text_facts = item.get("text_facts") if isinstance(item.get("text_facts"), dict) else {}
    desc = str(item.get("text_description_ru") or "")
    material_type = str(normalized.get("material_type") or "unknown_wall_material")
    style_tags = [str(x) for x in (normalized.get("style_tags") or []) if str(x).strip()]
    rooms = [str(x) for x in (normalized.get("rooms") or []) if str(x).strip()]
    color = str(normalized.get("base_color") or "") or None
    pattern = str(normalized.get("visual_pattern") or "") or None
    search_text = _lower(" ".join([
        str(item.get("name") or ""),
        str(item.get("brand") or ""),
        desc,
        material_type,
        color or "",
        str(normalized.get("precise_color_ru") or ""),
        str(normalized.get("tone") or ""),
        pattern or "",
        " ".join(f"{k} {v}" for k, v in raw_props.items()),
        " ".join(f"{k} {v}" for k, v in text_facts.items()),
    ]))
    return WallMaterial(
        version="wall_material.v1",
        source=str(item.get("source") or ""),
        sku=str(item.get("sku") or ""),
        name=str(item.get("name") or ""),
        brand=str(item.get("brand") or ""),
        product_url=str(item.get("url") or ""),
        price=item.get("price"),
        price_currency=str(item.get("price_currency") or "RUB"),
        availability=normalize_availability(str(item.get("availability") or "")),
        material_type=material_type,
        color=color,
        tone=str(normalized.get("tone") or "") or None,
        pattern=pattern,
        finish=str(normalized.get("surface_finish") or "") or None,
        base_material=str(text_facts.get("base_material") or raw_props.get("Материал основы") or "") or None,
        width_cm=_parse_float(normalized.get("width_cm")),
        length_m=_parse_float(normalized.get("length_m")),
        country=str(raw_props.get("Страна") or text_facts.get("country") or "") or None,
        description=desc,
        raw_properties=raw_props,
        image_urls=[x for x in image_urls if x],
        local_image_paths=_abs_local_paths(local_candidates, base_dir),
        style_tags=sorted(set(style_tags + [material_type])),
        room_suitability=rooms,
        search_text=search_text,
        parse_status="ok",
    )
