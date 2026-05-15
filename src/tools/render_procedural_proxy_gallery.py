#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/tools/render_procedural_proxy_gallery.py

Blender-only preview tool for procedural supplier proxy fallback.

Purpose
-------
Build a 10x10 gallery of 100 category-aware procedural proxy objects before
integrating proxy fallback into the main pipeline.

Input sources
-------------
1. data/sourse/suppliers/supplier_catalog_canonical.json
   Used as design references: category, title, dimensions, colors, materials,
   VLM description, image_color_features, preview paths.

2. data/floor_materials
   Used as optional local texture source. The script recursively samples local
   images and assigns them to some proxy materials by broad material family.

Output
------
- .blend file with a grid of 100 proxy objects.
- Optional render image.
- JSON report with selected supplier references and proxy/material decisions.

Usage from repository root
--------------------------
/Applications/Blender.app/Contents/MacOS/Blender -b --python src/tools/render_procedural_proxy_gallery.py -- \
  --supplier-catalog data/sourse/suppliers/supplier_catalog_canonical.json \
  --materials-root data/floor_materials \
  --out-blend out/proxy_gallery/procedural_proxy_gallery_10x10.blend \
  --out-render out/proxy_gallery/procedural_proxy_gallery_10x10.png \
  --out-report out/proxy_gallery/procedural_proxy_gallery_10x10.report.json \
  --grid-cols 10 \
  --grid-rows 10 \
  --seed 123
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import bpy


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "src" / "Plasement") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src" / "Plasement"))

try:
    from Plasement.procedural_proxy_blender import build_procedural_proxy
except Exception:
    from procedural_proxy_blender import build_procedural_proxy


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CATEGORY_TO_PROXY_CATEGORY = {
    "bed": "bed",
    "single_bed": "bed",
    "double_bed": "bed",
    "sofa": "sofa",
    "armchair": "armchair",
    "chair": "chair",
    "dining_chair": "chair",
    "wardrobe": "wardrobe",
    "cabinet": "cabinet",
    "dresser": "dresser",
    "sideboard": "dresser",
    "nightstand": "nightstand",
    "tv_stand": "tv_stand",
    "bookshelf": "bookshelf",
    "bookcase": "bookshelf",
    "shelf": "bookshelf",
    "desk": "desk",
    "dining_table": "dining_table",
    "coffee_table": "coffee_table",
    "side_table": "side_table",
    "console_table": "console_table",
    "floor_lamp": "floor_lamp",
    "table_lamp": "table_lamp",
    "pendant_lamp": "ceiling_light",
    "chandelier": "ceiling_light",
    "ceiling_light": "ceiling_light",
    "wall_light": "wall_light",
    "mirror": "mirror",
    "wall_art": "wall_art",
    "rug": "rug",
    "decor_vase": "decor_vase",
    "vase": "decor_vase",
    "decor_books": "decor_books",
    "decor_box": "decor_box",
    "bathroom_sink": "bathroom_sink",
    "toilet": "toilet",
    "shoe_cabinet": "shoe_cabinet",
    "entry_bench": "bench",
    "bench": "bench",
    "plant": "decor_vase",
}

DEFAULT_SIZE_BY_PROXY_CATEGORY = {
    "bed": [1.80, 2.10, 0.55],
    "sofa": [2.40, 1.00, 0.85],
    "armchair": [0.85, 0.85, 0.90],
    "chair": [0.50, 0.55, 0.90],
    "wardrobe": [0.90, 0.60, 2.20],
    "cabinet": [0.95, 0.45, 1.20],
    "dresser": [1.20, 0.45, 0.85],
    "nightstand": [0.45, 0.42, 0.55],
    "tv_stand": [1.60, 0.42, 0.52],
    "bookshelf": [0.85, 0.36, 2.05],
    "desk": [1.25, 0.65, 0.75],
    "dining_table": [1.45, 0.85, 0.75],
    "coffee_table": [1.10, 0.62, 0.42],
    "side_table": [0.45, 0.45, 0.55],
    "console_table": [1.10, 0.35, 0.82],
    "floor_lamp": [0.38, 0.38, 1.60],
    "table_lamp": [0.30, 0.30, 0.55],
    "ceiling_light": [0.55, 0.55, 0.45],
    "wall_light": [0.22, 0.08, 0.32],
    "mirror": [0.70, 0.04, 1.60],
    "wall_art": [0.90, 0.05, 0.65],
    "rug": [1.80, 2.40, 0.03],
    "decor_vase": [0.22, 0.22, 0.42],
    "decor_books": [0.34, 0.24, 0.10],
    "decor_box": [0.32, 0.24, 0.16],
    "bathroom_sink": [0.65, 0.45, 0.25],
    "toilet": [0.38, 0.65, 0.75],
    "shoe_cabinet": [0.90, 0.32, 0.90],
    "bench": [1.20, 0.42, 0.45],
}

MATERIAL_FAMILY_BY_PROXY_CATEGORY = {
    "bed": "fabric_wood",
    "sofa": "fabric",
    "armchair": "fabric",
    "chair": "fabric",
    "wardrobe": "wood",
    "cabinet": "wood",
    "dresser": "wood",
    "nightstand": "wood",
    "tv_stand": "wood",
    "bookshelf": "wood",
    "desk": "wood",
    "dining_table": "wood",
    "coffee_table": "wood",
    "side_table": "wood",
    "console_table": "wood",
    "floor_lamp": "metal",
    "table_lamp": "metal_glass",
    "ceiling_light": "metal_glass",
    "wall_light": "metal_glass",
    "mirror": "mirror",
    "wall_art": "painted",
    "rug": "fabric",
    "decor_vase": "ceramic",
    "decor_books": "paper",
    "decor_box": "painted",
    "bathroom_sink": "ceramic",
    "toilet": "ceramic",
    "shoe_cabinet": "wood",
    "bench": "fabric_wood",
}

FAMILY_KEYWORDS = {
    "wood": ("wood", "oak", "walnut", "mdf", "дерев", "дуб", "орех", "шпон", "мдф"),
    "fabric": ("fabric", "textile", "linen", "velvet", "cloth", "ткан", "текстил", "велюр", "лен"),
    "leather": ("leather", "кожа", "экокожа"),
    "metal": ("metal", "steel", "iron", "chrome", "brass", "металл", "сталь", "желез", "хром", "латун"),
    "glass": ("glass", "crystal", "стекло", "хрусталь"),
    "ceramic": ("ceramic", "porcelain", "керами", "фарфор"),
    "stone": ("stone", "marble", "granite", "камень", "мрамор", "гранит"),
    "painted": ("paint", "color", "wallpaper", "обои", "краска", "декор"),
}

COLOR_TEXT_TO_HEX = {
    "black": "#111111",
    "white": "#eeeeee",
    "gray": "#808080",
    "grey": "#808080",
    "beige": "#c9b79c",
    "brown": "#8a5f3d",
    "red": "#9f2d2d",
    "green": "#4f7d4a",
    "blue": "#3b5f91",
    "yellow": "#d6b84a",
    "orange": "#c97a32",
    "purple": "#705090",
    "черный": "#111111",
    "чёрный": "#111111",
    "белый": "#eeeeee",
    "серый": "#808080",
    "бежевый": "#c9b79c",
    "коричневый": "#8a5f3d",
    "красный": "#9f2d2d",
    "зеленый": "#4f7d4a",
    "зелёный": "#4f7d4a",
    "синий": "#3b5f91",
    "желтый": "#d6b84a",
    "жёлтый": "#d6b84a",
}


def _repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def read_json(path: str | Path) -> Any:
    p = _repo_path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    p = _repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text


def _category_norm(item: dict[str, Any]) -> str:
    return _norm_text(item.get("category_norm") or item.get("semantic_group") or item.get("category_raw") or item.get("title"))


def proxy_category_for_item(item: dict[str, Any]) -> str | None:
    category = _category_norm(item)
    if category in CATEGORY_TO_PROXY_CATEGORY:
        return CATEGORY_TO_PROXY_CATEGORY[category]

    title_blob = _norm_text(" ".join(str(item.get(k) or "") for k in ("title", "category_raw", "description", "vlm_description_text")))
    keyword_map = (
        (("bed", "кровать"), "bed"),
        (("sofa", "диван"), "sofa"),
        (("armchair", "кресл"), "armchair"),
        (("chair", "стул"), "chair"),
        (("wardrobe", "шкаф"), "wardrobe"),
        (("dresser", "sideboard", "комод", "сундук"), "dresser"),
        (("nightstand", "bedside", "тумб"), "nightstand"),
        (("bookshelf", "bookcase", "стеллаж"), "bookshelf"),
        (("desk", "письменный стол"), "desk"),
        (("dining table", "обеденный стол"), "dining_table"),
        (("coffee table", "журнальный стол"), "coffee_table"),
        (("side table", "приставной стол"), "side_table"),
        (("floor lamp", "торшер"), "floor_lamp"),
        (("table lamp", "настольная лампа"), "table_lamp"),
        (("pendant", "chandelier", "люстр", "подвесной"), "ceiling_light"),
        (("mirror", "зеркало"), "mirror"),
        (("wall art", "painting", "картина", "панно"), "wall_art"),
        (("rug", "carpet", "ковер", "ковёр"), "rug"),
        (("vase", "ваза"), "decor_vase"),
        (("sink", "basin", "раковин", "умывальник"), "bathroom_sink"),
        (("toilet", "унитаз"), "toilet"),
    )
    for tokens, mapped in keyword_map:
        if any(token in title_blob for token in tokens):
            return mapped
    return None


def candidate_dimensions_m(item: dict[str, Any], proxy_category: str) -> list[float]:
    dims = item.get("dimensions_cm") if isinstance(item.get("dimensions_cm"), dict) else {}
    width = dims.get("width") or item.get("width_cm")
    depth = dims.get("depth") or item.get("depth_cm")
    height = dims.get("height") or item.get("height_cm")
    fallback = DEFAULT_SIZE_BY_PROXY_CATEGORY.get(proxy_category, [0.8, 0.6, 0.8])

    try:
        values = [float(width) / 100.0, float(depth) / 100.0, float(height) / 100.0]
        if all(v > 0.01 for v in values):
            return values
    except Exception:
        pass
    return list(fallback)


def normalize_preview_size(size_m: list[float], proxy_category: str) -> list[float]:
    """Normalize huge real catalog sizes for a readable gallery cell.

    The main pipeline should keep layout bbox sizes. This gallery can scale large
    references down to fit a grid cell.
    """
    sx, sy, sz = [max(float(v), 0.02) for v in size_m]

    max_xy = max(sx, sy)
    max_z = sz
    scale = 1.0
    if max_xy > 1.8:
        scale = min(scale, 1.8 / max_xy)
    if max_z > 2.2:
        scale = min(scale, 2.2 / max_z)
    if max_xy < 0.22 and proxy_category not in {"table_lamp", "decor_vase", "decor_books", "decor_box"}:
        scale = max(scale, 0.22 / max_xy)

    sx, sy, sz = sx * scale, sy * scale, sz * scale

    # Keep gallery objects visible but category-proportional.
    if proxy_category == "ceiling_light":
        sz = max(sz, 0.35)
    elif proxy_category in {"wall_art", "mirror"}:
        sz = max(sz, 0.55)
    elif proxy_category in {"rug"}:
        sz = min(max(sz, 0.03), 0.06)
    elif proxy_category in {"decor_books", "decor_box"}:
        sz = max(sz, 0.10)
    elif proxy_category in {"decor_vase"}:
        sz = max(sz, 0.28)

    return [round(sx, 4), round(sy, 4), round(sz, 4)]


def _extract_color_hex_from_image_features(item: dict[str, Any], rank: int = 0) -> str | None:
    features = item.get("image_color_features") if isinstance(item.get("image_color_features"), dict) else {}
    colors = features.get("colors") if isinstance(features.get("colors"), dict) else {}
    top5 = colors.get("top5") if isinstance(colors.get("top5"), list) else []
    if top5 and len(top5) > rank and isinstance(top5[rank], dict) and top5[rank].get("hex"):
        return str(top5[rank]["hex"])
    one = colors.get("one") if isinstance(colors.get("one"), list) else []
    if one and isinstance(one[0], dict) and one[0].get("hex"):
        return str(one[0]["hex"])
    return None


def _extract_color_hex_from_text(item: dict[str, Any]) -> str | None:
    blob = _norm_text(" ".join(str(item.get(k) or "") for k in ("color", "title", "materials", "description", "vlm_description_text")))
    for token, hex_value in COLOR_TEXT_TO_HEX.items():
        if token in blob:
            return hex_value
    return None


def infer_material_family(item: dict[str, Any], proxy_category: str) -> str:
    blob = _norm_text(" ".join(str(item.get(k) or "") for k in ("materials", "description", "vlm_description_text", "title", "category_raw", "category_norm")))

    for family, tokens in FAMILY_KEYWORDS.items():
        if any(token in blob for token in tokens):
            if family == "glass" and proxy_category in {"ceiling_light", "table_lamp", "wall_light"}:
                return "metal_glass"
            return family

    return MATERIAL_FAMILY_BY_PROXY_CATEGORY.get(proxy_category, "painted")


def family_material_params(family: str) -> dict[str, float]:
    if family in {"metal", "metal_glass", "wood_metal"}:
        return {"roughness": 0.32, "metallic": 0.65, "alpha": 1.0}
    if family == "mirror":
        return {"roughness": 0.02, "metallic": 1.0, "alpha": 1.0}
    if family == "glass":
        return {"roughness": 0.05, "metallic": 0.0, "alpha": 0.35}
    if family in {"ceramic", "stone"}:
        return {"roughness": 0.22, "metallic": 0.0, "alpha": 1.0}
    if family in {"fabric", "leather", "fabric_wood"}:
        return {"roughness": 0.72, "metallic": 0.0, "alpha": 1.0}
    if family in {"wood", "wood_metal"}:
        return {"roughness": 0.55, "metallic": 0.0, "alpha": 1.0}
    return {"roughness": 0.55, "metallic": 0.0, "alpha": 1.0}


def resolve_material(item: dict[str, Any], proxy_category: str, texture_index: dict[str, list[Path]], rng: random.Random) -> dict[str, Any]:
    family = infer_material_family(item, proxy_category)
    base_hex = _extract_color_hex_from_image_features(item, 0) or _extract_color_hex_from_text(item) or "#b8b8b8"
    secondary_hex = _extract_color_hex_from_image_features(item, 1) or "#eeeeee"
    params = family_material_params(family)

    texture_path = None
    candidates = texture_index.get(family) or []
    if candidates and proxy_category not in {"mirror", "ceiling_light", "floor_lamp", "table_lamp", "wall_light"}:
        texture_path = str(rng.choice(candidates))

    return {
        "schema": "proxy_material/v1",
        "base_color_hex": base_hex,
        "secondary_color_hex": secondary_hex,
        "material_family": family,
        "roughness": params["roughness"],
        "metallic": params["metallic"],
        "alpha": params["alpha"],
        "texture_path": texture_path,
        "source": "supplier_catalog_canonical+floor_materials",
    }


def collect_texture_index(materials_root: Path, max_per_family: int, rng: random.Random) -> dict[str, list[Path]]:
    if not materials_root.exists():
        return {}

    all_images: list[Path] = []
    for p in materials_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            all_images.append(p.resolve())

    rng.shuffle(all_images)
    index: dict[str, list[Path]] = {family: [] for family in FAMILY_KEYWORDS}

    for p in all_images:
        text = _norm_text(str(p))
        assigned: list[str] = []
        if "oboykin" in text or "wallpaper" in text or "обои" in text:
            assigned.append("painted")
        if "mosplitka" in text or "tile" in text or "plitka" in text or "керам" in text:
            assigned.extend(["ceramic", "stone"])
        if "basisrf" in text or "mdf" in text or "wood" in text or "дуб" in text or "шпон" in text:
            assigned.append("wood")
        if "shtorystore" in text or "curtain" in text or "fabric" in text or "ткан" in text:
            assigned.append("fabric")
        if "texture_crops" in text:
            assigned.extend(["fabric", "wood", "stone", "painted"])

        if not assigned:
            continue
        for family in assigned:
            if family not in index:
                index[family] = []
            if len(index[family]) < max_per_family:
                index[family].append(p)

    return {k: v for k, v in index.items() if v}


def stratified_supplier_items(catalog: dict[str, Any], total: int, rng: random.Random) -> list[dict[str, Any]]:
    items = [item for item in catalog.get("items", []) if isinstance(item, dict)]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        proxy_category = proxy_category_for_item(item)
        if not proxy_category:
            continue
        buckets.setdefault(proxy_category, []).append(item)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    preferred_order = [
        "bed",
        "sofa",
        "armchair",
        "chair",
        "wardrobe",
        "dresser",
        "nightstand",
        "tv_stand",
        "bookshelf",
        "desk",
        "dining_table",
        "coffee_table",
        "side_table",
        "console_table",
        "floor_lamp",
        "table_lamp",
        "ceiling_light",
        "wall_light",
        "mirror",
        "wall_art",
        "rug",
        "decor_vase",
        "decor_books",
        "decor_box",
        "bathroom_sink",
        "toilet",
        "shoe_cabinet",
        "bench",
    ]

    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < total and cursor < total * 10:
        category = preferred_order[cursor % len(preferred_order)]
        bucket = buckets.get(category) or []
        if bucket:
            selected.append(bucket.pop())
        cursor += 1

    # If canonical has fewer mapped categories than expected, fill from any bucket.
    remaining = [item for bucket in buckets.values() for item in bucket]
    rng.shuffle(remaining)
    while len(selected) < total and remaining:
        selected.append(remaining.pop())

    return selected[:total]


def make_proxy_item(
    *,
    supplier_item: dict[str, Any],
    proxy_category: str,
    index: int,
    cell_x: float,
    cell_y: float,
    material: dict[str, Any],
) -> dict[str, Any]:
    raw_size = candidate_dimensions_m(supplier_item, proxy_category)
    size = normalize_preview_size(raw_size, proxy_category)
    sx, sy, sz = size

    mount_type = "floor"
    z = sz / 2.0
    if proxy_category in {"ceiling_light"}:
        mount_type = "ceiling"
        z = 2.4
    elif proxy_category in {"wall_light", "wall_art", "mirror"}:
        mount_type = "wall"
        z = 1.35
    elif proxy_category in {"rug"}:
        z = 0.015

    item_id = f"proxy_gallery_{index:03d}_{proxy_category}"
    title = str(supplier_item.get("title") or proxy_category)

    return {
        "id": item_id,
        "name": title,
        "category": proxy_category,
        "position_m": [cell_x, cell_y, z],
        "size_m": size,
        "rotation_deg": 0.0,
        "yaw_deg": 0.0,
        "yaw_rad": 0.0,
        "mount_type": mount_type,
        "asset": {
            "kind": "procedural_proxy",
            "mesh_fit_mode": "fit",
            "proxy_schema": "procedural_proxy/v1",
            "proxy_category": proxy_category,
            "proxy_source": "supplier_catalog_gallery_preview",
            "supplier_unique_key": supplier_item.get("unique_key"),
            "supplier_title": title,
            "material": material,
        },
        "source": {
            "placement_source": "procedural_proxy_gallery",
            "asset_source": "supplier_catalog_procedural_proxy",
            "supplier_proxy": True,
            "supplier_unique_key": supplier_item.get("unique_key"),
            "supplier_source_site": supplier_item.get("source_site"),
            "supplier_product_url": supplier_item.get("product_url") or supplier_item.get("model_page_url"),
        },
        "meta": {
            "gallery_index": index,
            "raw_supplier_category_norm": supplier_item.get("category_norm"),
            "raw_supplier_semantic_group": supplier_item.get("semantic_group"),
            "raw_supplier_dimensions_m": raw_size,
            "preview_normalized_size_m": size,
            "supplier_candidate": compact_supplier_item(supplier_item),
        },
    }


def compact_supplier_item(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "unique_key",
        "source_site",
        "title",
        "brand",
        "category_raw",
        "category_norm",
        "semantic_group",
        "product_url",
        "model_page_url",
        "model_download_url",
        "model_download_landing_url",
        "asset_status",
        "asset_format",
        "asset_local_path",
        "preview_local_path",
        "price_value",
        "price_currency",
        "style",
        "color",
        "materials",
        "dimensions_cm",
        "vlm_description_summary",
    ]
    return {k: item.get(k) for k in keys if k in item}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_floor_grid(cols: int, rows: int, spacing: float) -> None:
    mat = bpy.data.materials.new("gallery_grid_floor_mat")
    mat.diffuse_color = (0.78, 0.78, 0.78, 1.0)

    total_w = cols * spacing
    total_d = rows * spacing
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=((cols - 1) * spacing / 2.0, (rows - 1) * spacing / 2.0, -0.025))
    floor = bpy.context.object
    floor.name = "proxy_gallery_floor"
    floor.dimensions = (total_w + spacing, total_d + spacing, 0.03)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    floor.data.materials.append(mat)

    line_mat = bpy.data.materials.new("gallery_grid_line_mat")
    line_mat.diffuse_color = (0.25, 0.25, 0.25, 1.0)
    for c in range(cols + 1):
        x = (c - 0.5) * spacing
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, (rows - 1) * spacing / 2.0, 0.005))
        line = bpy.context.object
        line.name = f"grid_x_{c:02d}"
        line.dimensions = (0.02, total_d + spacing, 0.01)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        line.data.materials.append(line_mat)
    for r in range(rows + 1):
        y = (r - 0.5) * spacing
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=((cols - 1) * spacing / 2.0, y, 0.006))
        line = bpy.context.object
        line.name = f"grid_y_{r:02d}"
        line.dimensions = (total_w + spacing, 0.02, 0.01)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        line.data.materials.append(line_mat)


def make_label(text: str, x: float, y: float, z: float, max_len: int = 34) -> None:
    label = text[: max_len - 1] + "…" if len(text) > max_len else text
    bpy.ops.object.text_add(location=(x, y, z), rotation=(math.radians(75), 0.0, 0.0))
    obj = bpy.context.object
    obj.name = "proxy_gallery_label"
    obj.data.body = label
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = 0.12


def add_lights(cols: int, rows: int, spacing: float) -> None:
    cx = (cols - 1) * spacing / 2.0
    cy = (rows - 1) * spacing / 2.0
    bpy.ops.object.light_add(type="SUN", location=(cx, cy, 8.0))
    sun = bpy.context.object
    sun.name = "proxy_gallery_sun"
    sun.data.energy = 2.2
    sun.rotation_euler = (math.radians(45), 0.0, math.radians(35))

    bpy.ops.object.light_add(type="AREA", location=(cx, cy - rows * spacing * 0.35, 6.0))
    area = bpy.context.object
    area.name = "proxy_gallery_area_light"
    area.data.energy = 700.0
    area.data.size = max(cols, rows) * spacing * 0.8


def add_camera(cols: int, rows: int, spacing: float, resolution_x: int, resolution_y: int) -> None:
    cx = (cols - 1) * spacing / 2.0
    cy = (rows - 1) * spacing / 2.0
    max_dim = max(cols, rows) * spacing
    cam_z = max_dim * 1.15
    cam_y = cy - max_dim * 0.95
    bpy.ops.object.camera_add(location=(cx, cam_y, cam_z), rotation=(math.radians(60), 0.0, 0.0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.name = "proxy_gallery_camera"
    cam.data.lens = 26
    cam.data.type = "PERSP"

    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.context.scene.eevee.taa_render_samples = 64


def build_gallery(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(int(args.seed))
    catalog_path = _repo_path(args.supplier_catalog)
    materials_root = _repo_path(args.materials_root)

    catalog = read_json(catalog_path)
    total = int(args.grid_cols) * int(args.grid_rows)
    selected = stratified_supplier_items(catalog, total, rng)
    texture_index = collect_texture_index(materials_root, max_per_family=int(args.max_textures_per_family), rng=rng)

    clear_scene()
    make_floor_grid(int(args.grid_cols), int(args.grid_rows), float(args.spacing))

    report_items: list[dict[str, Any]] = []
    for idx, supplier_item in enumerate(selected):
        row = idx // int(args.grid_cols)
        col = idx % int(args.grid_cols)
        x = col * float(args.spacing)
        y = row * float(args.spacing)
        proxy_category = proxy_category_for_item(supplier_item) or "cabinet"
        material = resolve_material(supplier_item, proxy_category, texture_index, rng)
        proxy_item = make_proxy_item(
            supplier_item=supplier_item,
            proxy_category=proxy_category,
            index=idx + 1,
            cell_x=x,
            cell_y=y,
            material=material,
        )
        try:
            obj = build_procedural_proxy(proxy_item)
            obj.name = proxy_item["id"]
        except Exception as exc:
            print(f"[WARN] failed to build proxy {idx + 1}: {proxy_category}: {type(exc).__name__}: {exc}")
            continue

        make_label(f"{idx + 1:02d} {proxy_category}", x, y - float(args.spacing) * 0.43, 0.04)
        report_items.append(
            {
                "index": idx + 1,
                "proxy_category": proxy_category,
                "supplier_unique_key": supplier_item.get("unique_key"),
                "supplier_title": supplier_item.get("title"),
                "supplier_category_norm": supplier_item.get("category_norm"),
                "supplier_source_site": supplier_item.get("source_site"),
                "size_m": proxy_item.get("size_m"),
                "material": material,
            }
        )

    add_lights(int(args.grid_cols), int(args.grid_rows), float(args.spacing))
    add_camera(int(args.grid_cols), int(args.grid_rows), float(args.spacing), int(args.resolution_x), int(args.resolution_y))

    out_blend = _repo_path(args.out_blend)
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    render_path = None
    if args.out_render:
        out_render = _repo_path(args.out_render)
        out_render.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(out_render)
        bpy.ops.render.render(write_still=True)
        render_path = str(out_render)

    report = {
        "schema": "procedural_proxy_gallery_report/v1",
        "supplier_catalog": str(catalog_path),
        "materials_root": str(materials_root),
        "out_blend": str(out_blend),
        "out_render": render_path,
        "grid_cols": int(args.grid_cols),
        "grid_rows": int(args.grid_rows),
        "requested_count": total,
        "built_count": len(report_items),
        "texture_index_counts": {k: len(v) for k, v in texture_index.items()},
        "items": report_items,
    }

    if args.out_report:
        write_json(args.out_report, report)

    return report


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Render 10x10 procedural supplier proxy gallery in Blender.")
    ap.add_argument("--supplier-catalog", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    ap.add_argument("--materials-root", default="data/floor_materials")
    ap.add_argument("--out-blend", default="out/proxy_gallery/procedural_proxy_gallery_10x10.blend")
    ap.add_argument("--out-render", default="out/proxy_gallery/procedural_proxy_gallery_10x10.png")
    ap.add_argument("--out-report", default="out/proxy_gallery/procedural_proxy_gallery_10x10.report.json")
    ap.add_argument("--grid-cols", type=int, default=10)
    ap.add_argument("--grid-rows", type=int, default=10)
    ap.add_argument("--spacing", type=float, default=2.65)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-textures-per-family", type=int, default=80)
    ap.add_argument("--resolution-x", type=int, default=2400)
    ap.add_argument("--resolution-y", type=int, default=1800)
    return ap


def _args_after_blender_separator(argv: list[str]) -> list[str]:
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def main() -> None:
    parser = build_cli()
    args = parser.parse_args(_args_after_blender_separator(sys.argv))
    report = build_gallery(args)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
