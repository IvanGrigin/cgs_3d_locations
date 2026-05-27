#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/Plasement/procedural_proxy_blender.py

Procedural 3D proxy generator for supplier catalog items.
Runs inside Blender Python.

Use when a supplier item has no local FBX/OBJ/GLB, but has metadata:
category, dimensions, colors, materials, VLM description, image_color_features.

Example:
/Applications/Blender.app/Contents/MacOS/Blender -b \
  --python src/Plasement/procedural_proxy_blender.py -- \
  --clear-scene \
  --catalog-json data/sourse/suppliers/supplier_catalog_canonical.json \
  --limit 100 \
  --grid-cols 10 \
  --spacing-x 2.8 \
  --spacing-y 2.8 \
  --material-mode procedural \
  --disable-labels \
  --disable-debug-lines \
  --disable-texture-planes \
  --setup-preview \
  --out-blend out/procedural_proxy_grid_100_clean.blend \
  --report-json out/procedural_proxy_grid_100_clean.report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    import bpy
except Exception as exc:
    raise RuntimeError(
        "This script must be executed inside Blender. Use Blender.app/Contents/MacOS/Blender -b --python ..."
    ) from exc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _script_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build procedural proxy models from supplier catalog metadata.")
    p.add_argument("--catalog-json", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--grid-cols", type=int, default=10)
    p.add_argument("--spacing-x", type=float, default=2.8)
    p.add_argument("--spacing-y", type=float, default=2.8)
    p.add_argument("--texture-root", action="append", default=[])
    p.add_argument("--out-blend", default="out/procedural_proxy_grid.blend")
    p.add_argument("--report-json", default=None)
    p.add_argument("--clear-scene", action="store_true")
    p.add_argument("--setup-preview", action="store_true")

    p.add_argument("--disable-labels", action="store_true")
    p.add_argument("--disable-debug-lines", action="store_true")
    p.add_argument("--disable-texture-planes", action="store_true")
    p.add_argument("--debug-label-lines", action="store_true")
    p.add_argument("--no-textures", action="store_true")
    p.add_argument("--material-mode", choices=["flat", "procedural", "image"], default="procedural")

    p.add_argument("--override-color", default=None, help="Force one color for all objects, e.g. #b8946a")
    p.add_argument("--category-color", action="append", default=[], help="category=#RRGGBB; can be repeated")
    p.add_argument("--force-material", action="append", default=[], help="category=wood|fabric|metal|glass|stone|ceramic|plastic")
    return p


# -----------------------------------------------------------------------------
# IO / text / color
# -----------------------------------------------------------------------------


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_name(value: Any, limit: int = 80) -> str:
    text = str(value or "object").strip()
    text = re.sub(r"[^0-9A-Za-zА-Яа-яёЁ._ -]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:limit] or "object").strip()


def ntext(value: Any) -> str:
    return str(value or "").strip().lower().replace("ё", "е")


def tokens(value: Any) -> set[str]:
    text = ntext(value)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return {t for t in text.split() if len(t) >= 3}


def parse_assignments(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            continue
        k, v = value.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def parse_hex(value: Any) -> tuple[float, float, float, float] | None:
    m = re.search(r"#?([0-9a-fA-F]{6})", str(value or ""))
    if not m:
        return None
    raw = m.group(1)
    return (int(raw[0:2], 16) / 255.0, int(raw[2:4], 16) / 255.0, int(raw[4:6], 16) / 255.0, 1.0)


BASIC_COLORS: dict[str, tuple[float, float, float, float]] = {
    "white": (0.88, 0.86, 0.82, 1.0),
    "black": (0.025, 0.025, 0.025, 1.0),
    "gray": (0.42, 0.42, 0.42, 1.0),
    "grey": (0.42, 0.42, 0.42, 1.0),
    "beige": (0.70, 0.60, 0.46, 1.0),
    "brown": (0.36, 0.22, 0.11, 1.0),
    "red": (0.60, 0.08, 0.06, 1.0),
    "green": (0.18, 0.42, 0.16, 1.0),
    "blue": (0.10, 0.22, 0.52, 1.0),
    "yellow": (0.78, 0.62, 0.18, 1.0),
    "orange": (0.78, 0.34, 0.08, 1.0),
    "purple": (0.35, 0.14, 0.45, 1.0),
    "pink": (0.78, 0.45, 0.55, 1.0),
    "wood": (0.45, 0.26, 0.12, 1.0),
    "oak": (0.62, 0.43, 0.23, 1.0),
    "walnut": (0.28, 0.16, 0.08, 1.0),
    "metal": (0.45, 0.43, 0.40, 1.0),
}

RU_COLOR_PARTS: dict[str, tuple[float, float, float, float]] = {
    "бел": BASIC_COLORS["white"],
    "черн": BASIC_COLORS["black"],
    "сер": BASIC_COLORS["gray"],
    "беж": BASIC_COLORS["beige"],
    "корич": BASIC_COLORS["brown"],
    "красн": BASIC_COLORS["red"],
    "зелен": BASIC_COLORS["green"],
    "син": BASIC_COLORS["blue"],
    "голуб": BASIC_COLORS["blue"],
    "желт": BASIC_COLORS["yellow"],
    "оранж": BASIC_COLORS["orange"],
    "фиолет": BASIC_COLORS["purple"],
    "роз": BASIC_COLORS["pink"],
    "металл": BASIC_COLORS["metal"],
    "дерев": BASIC_COLORS["wood"],
    "дуб": BASIC_COLORS["oak"],
    "орех": BASIC_COLORS["walnut"],
}


def color_from_text(value: Any) -> tuple[float, float, float, float] | None:
    text = ntext(value)
    c = parse_hex(text)
    if c:
        return c
    for k, v in RU_COLOR_PARTS.items():
        if k in text:
            return v
    for k, v in BASIC_COLORS.items():
        if k in text:
            return v
    return None


def rgb255(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return (clamp(float(value[0]) / 255), clamp(float(value[1]) / 255), clamp(float(value[2]) / 255), 1.0)
    except Exception:
        return None


def dominant_color(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("color", "vlm_color", "materials", "title", "description"):
        c = color_from_text(item.get(key))
        if c:
            return c
    features = item.get("image_color_features")
    if isinstance(features, dict):
        colors = features.get("colors")
        if isinstance(colors, dict):
            for group in ("one", "top2", "top5"):
                arr = colors.get(group)
                if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                    c = parse_hex(arr[0].get("hex")) or rgb255(arr[0].get("rgb"))
                    if c:
                        return c
        token_list = features.get("color_tokens")
        if isinstance(token_list, list):
            for token in token_list:
                c = color_from_text(token)
                if c:
                    return c
    return None


def lighten(c: tuple[float, float, float, float], k: float = 0.25) -> tuple[float, float, float, float]:
    r, g, b, a = c
    return (r + (1 - r) * k, g + (1 - g) * k, b + (1 - b) * k, a)


def darken(c: tuple[float, float, float, float], k: float = 0.30) -> tuple[float, float, float, float]:
    r, g, b, a = c
    return (r * (1 - k), g * (1 - k), b * (1 - k), a)


# -----------------------------------------------------------------------------
# Category / dimensions / materials
# -----------------------------------------------------------------------------


CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("lamp_ceiling", ("pendant_lamp", "chandelier", "ceiling_light", "lamp_ceiling", "люстра", "подвесной")),
    ("lamp_floor", ("floor_lamp", "lamp_floor", "торшер")),
    ("lamp_table", ("table_lamp", "lamp_table", "настольная лампа")),
    ("lamp_wall", ("wall_light", "sconce", "lamp_wall", "бра")),
    ("bed", ("bed", "кровать")),
    ("nightstand", ("nightstand", "bedside", "прикроват", "тумба")),
    ("wardrobe", ("wardrobe", "closet", "шкаф")),
    ("dresser", ("dresser", "sideboard", "chest", "комод", "сундук")),
    ("sofa", ("sofa", "couch", "диван")),
    ("armchair", ("armchair", "кресло")),
    ("chair", ("chair", "dining_chair", "стул")),
    ("coffee_table", ("coffee_table", "журнальный")),
    ("side_table", ("side_table", "console_table", "консоль", "столик")),
    ("dining_table", ("dining_table", "обеденный")),
    ("desk", ("desk", "письменный")),
    ("tv_stand", ("tv_stand", "тумба под tv", "тумба под телевиз")),
    ("tv", ("tv_projector_screen", "television", "телевизор", "smart tv")),
    ("bookshelf", ("bookshelf", "bookcase", "shelf", "стеллаж", "полка")),
    ("mirror", ("mirror", "зеркало")),
    ("wall_art", ("wall_art", "picture", "картина", "панно")),
    ("rug", ("rug", "runner_rug", "carpet", "ковер", "ковёр")),
    ("pillow", ("pillow", "подушка")),
    ("blanket", ("blanket", "throw", "плед", "одеяло")),
    ("decor_books", ("decor_books", "books", "book", "книги")),
    ("decor_vase", ("decor_vase", "vase", "ваза")),
    ("decor_box", ("decor_box", "box", "короб")),
    ("decor_tray", ("decor_tray", "tray", "поднос")),
    ("plant", ("plant", "planter", "растение")),
    ("bathroom_sink", ("bathroom_sink", "washbasin", "sink", "раковина", "умывальник")),
    ("shoe_cabinet", ("shoe_cabinet", "обувница")),
    ("bench", ("bench", "entry_bench", "банкетка", "скамья")),
    ("storage_basket", ("storage_basket", "basket", "корзина")),
    ("refrigerator", ("refrigerator", "fridge", "холодильник")),
    ("stove", ("stove", "range", "плита", "варочная")),
    ("cooktop", ("cooktop", "hob", "варочная панель")),
    ("oven", ("oven", "духовой", "духовка")),
    ("range_hood", ("range_hood", "hood", "вытяжка")),
    ("microwave", ("microwave", "свч", "микроволнов")),
    ("kitchen_sink", ("kitchen_sink", "kitchen faucet", "мойка", "смеситель")),
]


def group_of(item: dict[str, Any]) -> str:
    blob = " ".join(str(item.get(k) or "") for k in (
        "semantic_group", "category_norm", "category_raw", "category", "title", "name",
        "description", "vlm_description_summary", "vlm_description_text"
    )).lower().replace("ё", "е")
    for group, aliases in CATEGORY_RULES:
        if any(alias.lower().replace("ё", "е") in blob for alias in aliases):
            return group
    return str(item.get("category_norm") or item.get("semantic_group") or "generic").strip().lower() or "generic"


def default_dims(group: str) -> tuple[float, float, float]:
    return {
        "bed": (1.6, 2.0, 0.55), "nightstand": (0.45, 0.42, 0.55), "wardrobe": (0.9, 0.55, 2.1),
        "dresser": (1.1, 0.45, 0.85), "sofa": (2.0, 0.9, 0.85), "armchair": (0.8, 0.8, 0.9),
        "chair": (0.48, 0.52, 0.85), "coffee_table": (1.0, 0.55, 0.42), "side_table": (0.55, 0.45, 0.55),
        "dining_table": (1.4, 0.85, 0.75), "desk": (1.2, 0.65, 0.75), "tv_stand": (1.4, 0.42, 0.5),
        "tv": (1.2, 0.06, 0.7), "bookshelf": (0.8, 0.35, 1.8), "mirror": (0.7, 0.04, 1.4),
        "wall_art": (0.8, 0.04, 0.55), "rug": (1.7, 1.2, 0.03), "pillow": (0.45, 0.18, 0.16),
        "blanket": (1.2, 0.8, 0.05), "decor_books": (0.28, 0.2, 0.08), "decor_vase": (0.20, 0.20, 0.35),
        "decor_box": (0.32, 0.24, 0.18), "decor_tray": (0.36, 0.24, 0.05), "plant": (0.45, 0.45, 1.1),
        "lamp_floor": (0.35, 0.35, 1.55), "lamp_table": (0.28, 0.28, 0.55), "lamp_ceiling": (0.45, 0.45, 0.55),
        "lamp_wall": (0.22, 0.08, 0.35), "bathroom_sink": (0.55, 0.42, 0.25), "shoe_cabinet": (0.9, 0.32, 0.85),
        "bench": (1.1, 0.42, 0.45), "storage_basket": (0.35, 0.28, 0.28),
        "refrigerator": (0.65, 0.65, 1.85), "stove": (0.60, 0.60, 0.88), "cooktop": (0.60, 0.52, 0.08),
        "oven": (0.60, 0.58, 0.60), "range_hood": (0.60, 0.40, 0.35), "microwave": (0.52, 0.40, 0.30),
        "kitchen_sink": (0.55, 0.45, 0.22),
    }.get(group, (0.6, 0.6, 0.6))


def dims_m(item: dict[str, Any], group: str) -> tuple[float, float, float]:
    dims = item.get("dimensions_cm") if isinstance(item.get("dimensions_cm"), dict) else {}
    fb = default_dims(group)

    def get_axis(axis: str, fallback: float) -> float:
        value = item.get(f"{axis}_cm", dims.get(axis))
        try:
            v = float(value)
            if v <= 0:
                return fallback
            return v / 100.0 if v > 5 else v
        except Exception:
            return fallback

    w = get_axis("width", fb[0])
    d = get_axis("depth", fb[1])
    h = get_axis("height", fb[2])

    # Preview scale clamp: keep objects readable in 2.8 m cells.
    max_xy = max(w, d, 0.001)
    max_z = max(h, 0.001)
    scale = min(1.45 / max_xy, 1.9 / max_z, 1.0)
    return max(w * scale, 0.03), max(d * scale, 0.03), max(h * scale, 0.03)


def material_kind(item: dict[str, Any], forced: str | None = None) -> str:
    if forced:
        return forced.lower().strip()
    blob = " ".join(str(item.get(k) or "") for k in (
        "title", "category_norm", "category_raw", "semantic_group", "materials", "description", "vlm_description_text", "color"
    )).lower().replace("ё", "е")
    if any(x in blob for x in ("fabric", "textile", "linen", "cotton", "velvet", "ткан", "текстил", "лен", "бархат")):
        return "fabric"
    if any(x in blob for x in ("leather", "кож")):
        return "leather"
    if any(x in blob for x in ("glass", "стекл", "хрустал", "crystal", "transparent")):
        return "glass"
    if any(x in blob for x in ("metal", "steel", "iron", "brass", "chrome", "металл", "сталь", "желез", "латун", "хром")):
        return "metal"
    if any(x in blob for x in ("wood", "oak", "walnut", "mdf", "дерев", "дуб", "орех")):
        return "wood"
    if any(x in blob for x in ("stone", "marble", "granite", "ceramic", "porcelain", "камень", "мрамор", "гранит", "керами", "фарфор")):
        return "stone"
    g = group_of(item)
    if g in {"pillow", "blanket", "rug", "sofa", "armchair", "chair"}:
        return "fabric"
    if g in {"dresser", "wardrobe", "nightstand", "side_table", "coffee_table", "desk", "tv_stand", "bookshelf", "bed"}:
        return "wood"
    if g.startswith("lamp"):
        return "metal"
    return "plastic"


# -----------------------------------------------------------------------------
# Materials
# -----------------------------------------------------------------------------


_MAT_CACHE: dict[str, bpy.types.Material] = {}


def make_mat(name: str, color: tuple[float, float, float, float], kind: str = "plastic") -> bpy.types.Material:
    key = f"{name}|{kind}|" + ",".join(f"{x:.3f}" for x in color)
    if key in _MAT_CACHE:
        return _MAT_CACHE[key]
    mat = bpy.data.materials.new(sanitize_name(name))
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        def set_in(names: tuple[str, ...], value: Any) -> None:
            for nm in names:
                if nm in bsdf.inputs:
                    try:
                        bsdf.inputs[nm].default_value = value
                    except Exception:
                        pass
                    return
        metallic = 0.0
        roughness = 0.55
        alpha = color[3]
        transmission = 0.0
        if kind == "metal":
            metallic, roughness = 0.85, 0.30
        elif kind == "glass":
            roughness, alpha, transmission = 0.04, 0.35, 0.65
            mat.blend_method = "BLEND"
            if hasattr(mat, "use_screen_refraction"):
                mat.use_screen_refraction = True
        elif kind in {"fabric", "leather"}:
            roughness = 0.82
        elif kind in {"stone", "ceramic"}:
            roughness = 0.42
        set_in(("Base Color",), color)
        set_in(("Metallic",), metallic)
        set_in(("Roughness",), roughness)
        set_in(("Alpha",), alpha)
        set_in(("Transmission Weight", "Transmission"), transmission)
    _MAT_CACHE[key] = mat
    return mat


def palette(item: dict[str, Any], args: argparse.Namespace) -> dict[str, bpy.types.Material]:
    g = group_of(item)
    cat_color = parse_assignments(args.category_color)
    cat_mat = parse_assignments(args.force_material)
    forced = parse_hex(args.override_color) or parse_hex(cat_color.get(g))
    base = forced or dominant_color(item) or (0.55, 0.55, 0.55, 1.0)
    kind = material_kind(item, cat_mat.get(g))
    return {
        "base": make_mat(f"proxy_{g}_{kind}_base", base, kind),
        "light": make_mat(f"proxy_{g}_{kind}_light", lighten(base), kind),
        "dark": make_mat(f"proxy_{g}_{kind}_dark", darken(base), kind),
        "fabric": make_mat(f"proxy_{g}_fabric", base, "fabric"),
        "wood": make_mat(f"proxy_{g}_wood", base if kind == "wood" else BASIC_COLORS["wood"], "wood"),
        "metal": make_mat(f"proxy_{g}_metal", base if kind == "metal" else BASIC_COLORS["metal"], "metal"),
        "glass": make_mat("proxy_clear_glass", (0.78, 0.9, 1.0, 0.35), "glass"),
        "black": make_mat("proxy_black", BASIC_COLORS["black"], "plastic"),
        "ceramic": make_mat("proxy_ceramic", (0.88, 0.86, 0.82, 1), "ceramic"),
    }


# -----------------------------------------------------------------------------
# Blender primitives
# -----------------------------------------------------------------------------


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def link(obj: bpy.types.Object, col: bpy.types.Collection | None) -> bpy.types.Object:
    if col:
        try:
            for c in obj.users_collection:
                c.objects.unlink(obj)
            col.objects.link(obj)
        except Exception:
            pass
    return obj


def mat(obj: bpy.types.Object, material: bpy.types.Material | None) -> bpy.types.Object:
    if material and hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(material)
    return obj


def smooth(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    obj.select_set(False)
    return obj


def bevel(obj: bpy.types.Object, amount: float = 0.02, segments: int = 2) -> bpy.types.Object:
    if amount <= 0:
        return obj
    try:
        mod = obj.modifiers.new("soft_bevel", "BEVEL")
        mod.width = amount
        mod.segments = segments
        mod.profile = 0.5
        obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    except Exception:
        pass
    return obj


def cube(name: str, loc: tuple[float, float, float], size: tuple[float, float, float], material: bpy.types.Material, col: bpy.types.Collection, *, b: float = 0.02, seg: int = 2) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    obj.dimensions = (max(size[0], 0.001), max(size[1], 0.001), max(size[2], 0.001))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    mat(obj, material)
    bevel(obj, b, seg)
    return link(obj, col)


def cyl(name: str, loc: tuple[float, float, float], radius: float, depth: float, material: bpy.types.Material, col: bpy.types.Collection, *, vertices: int = 32) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=max(radius, 0.001), depth=max(depth, 0.001), location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    mat(obj, material)
    smooth(obj)
    bevel(obj, min(radius * 0.12, 0.035), 2)
    return link(obj, col)


def sphere(name: str, loc: tuple[float, float, float], scale: tuple[float, float, float], material: bpy.types.Material, col: bpy.types.Collection, *, segments: int = 32) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=max(8, segments // 2), radius=0.5, location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    obj.scale = scale
    mat(obj, material)
    smooth(obj)
    return link(obj, col)


def cone(name: str, loc: tuple[float, float, float], r1: float, r2: float, depth: float, material: bpy.types.Material, col: bpy.types.Collection, *, vertices: int = 32) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=max(r1, 0.001), radius2=max(r2, 0.001), depth=max(depth, 0.001), location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    mat(obj, material)
    smooth(obj)
    bevel(obj, min(max(r1, r2) * 0.06, 0.025), 2)
    return link(obj, col)


def rotz(obj: bpy.types.Object, deg: float) -> bpy.types.Object:
    obj.rotation_euler[2] = math.radians(deg)
    return obj


def label(text: str, loc: tuple[float, float, float], col: bpy.types.Collection) -> None:
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(70), 0, 0))
    obj = bpy.context.object
    obj.name = sanitize_name("label_" + text[:20])
    obj.data.body = text[:80]
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = 0.13
    link(obj, col)


# -----------------------------------------------------------------------------
# Proxy builders
# -----------------------------------------------------------------------------


class Builder:
    def __init__(self, col: bpy.types.Collection, args: argparse.Namespace):
        self.col = col
        self.args = args

    def build(self, item: dict[str, Any], loc: tuple[float, float, float], idx: int) -> dict[str, Any]:
        g = group_of(item)
        d = dims_m(item, g)
        p = palette(item, self.args)
        before = set(bpy.data.objects)
        fn = getattr(self, f"build_{g}", self.build_generic)
        fn(loc, d, p)
        created = list(set(bpy.data.objects) - before)
        for obj in created:
            obj["proxy_group"] = g
            obj["proxy_catalog_index"] = idx
            obj["proxy_unique_key"] = str(item.get("unique_key") or "")
            obj["proxy_title"] = str(item.get("title") or "")
        if not self.args.disable_labels:
            label(f"{idx+1}. {g}\n{str(item.get('title') or '')[:34]}", (loc[0], loc[1] - 0.95, 0.06), self.col)
        return {
            "index": idx,
            "unique_key": item.get("unique_key"),
            "title": item.get("title"),
            "category_norm": item.get("category_norm"),
            "semantic_group": item.get("semantic_group"),
            "proxy_group": g,
            "dimensions_m": [round(x, 4) for x in d],
            "object_count": len(created),
            "material_kind": material_kind(item, parse_assignments(self.args.force_material).get(g)),
        }

    def build_generic(self, loc, d, p):
        w, y, h = d
        cube("generic_body", (loc[0], loc[1], h / 2), (w, y, h), p["base"], self.col, b=0.04, seg=3)

    def drawer_cabinet(self, prefix, loc, d, p, drawers: int):
        w, y, h = d
        cube(prefix + "_body", (loc[0], loc[1], h / 2), (w, y, h), p["wood"], self.col, b=0.035, seg=3)
        front_y = loc[1] - y * 0.51
        usable_h = h * 0.72
        start_z = h * 0.18
        for i in range(drawers):
            z = start_z + usable_h * (i + 0.5) / max(drawers, 1)
            cube(f"{prefix}_drawer_{i+1}", (loc[0], front_y, z), (w * 0.86, 0.025, usable_h / max(drawers, 1) * 0.72), p["base"], self.col, b=0.012, seg=2)
            cube(f"{prefix}_handle_{i+1}", (loc[0], front_y - 0.018, z), (w * 0.28, 0.025, 0.018), p["metal"], self.col, b=0.006, seg=1)

    def table_with_legs(self, prefix, loc, d, p, leg_ratio=0.78, top_ratio=0.14):
        w, y, h = d
        leg_h = h * leg_ratio
        top_h = max(h * top_ratio, 0.035)
        cube(prefix + "_top", (loc[0], loc[1], leg_h + top_h / 2), (w, y, top_h), p["wood"], self.col, b=0.035, seg=3)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cyl(prefix + "_leg", (loc[0] + sx * w * 0.42, loc[1] + sy * y * 0.38, leg_h / 2), 0.025, leg_h, p["dark"], self.col)

    def build_bed(self, loc, d, p):
        w, y, h = d
        frame_h = min(0.22, h * 0.45)
        mat_h = min(0.28, h * 0.55)
        cube("bed_frame", (loc[0], loc[1], frame_h / 2), (w, y, frame_h), p["wood"], self.col, b=0.04, seg=3)
        cube("bed_mattress", (loc[0], loc[1] - y * 0.02, frame_h + mat_h / 2), (w * 0.92, y * 0.88, mat_h), p["fabric"], self.col, b=0.09, seg=8)
        cube("bed_headboard", (loc[0], loc[1] + y * 0.48, h * 0.48), (w * 1.02, 0.10, max(h * 0.8, 0.45)), p["dark"], self.col, b=0.04, seg=4)
        for x in (-0.25, 0.25):
            sphere("bed_pillow", (loc[0] + x * w, loc[1] + y * 0.28, frame_h + mat_h + 0.08), (w * 0.18, y * 0.08, 0.08), p["light"], self.col, segments=48)
        cube("bed_blanket", (loc[0], loc[1] - y * 0.15, frame_h + mat_h + 0.025), (w * 0.82, y * 0.45, 0.05), p["base"], self.col, b=0.08, seg=8)

    def build_sofa(self, loc, d, p):
        w, y, h = d
        seat_h = h * 0.38
        cube("sofa_seat", (loc[0], loc[1], seat_h / 2), (w, y * 0.78, seat_h), p["fabric"], self.col, b=0.10, seg=8)
        cube("sofa_back", (loc[0], loc[1] + y * 0.42, h * 0.56), (w, y * 0.16, h * 0.78), p["fabric"], self.col, b=0.10, seg=8)
        cube("sofa_arm_l", (loc[0] - w * 0.48, loc[1], h * 0.42), (w * 0.10, y * 0.82, h * 0.65), p["fabric"], self.col, b=0.09, seg=8)
        cube("sofa_arm_r", (loc[0] + w * 0.48, loc[1], h * 0.42), (w * 0.10, y * 0.82, h * 0.65), p["fabric"], self.col, b=0.09, seg=8)
        for i in range(3):
            sphere("sofa_pillow", (loc[0] + (i - 1) * w * 0.23, loc[1] + y * 0.18, h * 0.58), (w * 0.10, y * 0.055, h * 0.10), p["light"], self.col)

    def build_armchair(self, loc, d, p):
        self.build_sofa(loc, d, p)

    def build_chair(self, loc, d, p):
        w, y, h = d
        seat_z = h * 0.45
        cube("chair_seat", (loc[0], loc[1], seat_z), (w * 0.85, y * 0.75, h * 0.10), p["base"], self.col, b=0.045, seg=4)
        cube("chair_back", (loc[0], loc[1] + y * 0.34, h * 0.72), (w * 0.90, y * 0.08, h * 0.50), p["base"], self.col, b=0.045, seg=4)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cyl("chair_leg", (loc[0] + sx * w * 0.32, loc[1] + sy * y * 0.25, seat_z / 2), 0.018, seat_z, p["dark"], self.col)

    def build_nightstand(self, loc, d, p):
        self.drawer_cabinet("nightstand", loc, d, p, 2)

    def build_dresser(self, loc, d, p):
        self.drawer_cabinet("dresser", loc, d, p, 3)

    def build_tv_stand(self, loc, d, p):
        self.drawer_cabinet("tv_stand", loc, d, p, 2)

    def build_shoe_cabinet(self, loc, d, p):
        self.drawer_cabinet("shoe_cabinet", loc, d, p, 2)

    def build_wardrobe(self, loc, d, p):
        w, y, h = d
        cube("wardrobe_body", (loc[0], loc[1], h / 2), (w, y, h), p["wood"], self.col, b=0.035, seg=3)
        cube("wardrobe_door_l", (loc[0] - w * 0.25, loc[1] - y * 0.51, h * 0.52), (w * 0.46, 0.025, h * 0.86), p["base"], self.col, b=0.015, seg=2)
        cube("wardrobe_door_r", (loc[0] + w * 0.25, loc[1] - y * 0.51, h * 0.52), (w * 0.46, 0.025, h * 0.86), p["base"], self.col, b=0.015, seg=2)
        for sx in (-0.08, 0.08):
            cube("wardrobe_handle", (loc[0] + sx * w, loc[1] - y * 0.535, h * 0.56), (0.025, 0.025, h * 0.30), p["metal"], self.col, b=0.006, seg=1)

    def build_bookshelf(self, loc, d, p):
        w, y, h = d
        cube("bookshelf_frame", (loc[0], loc[1], h / 2), (w, y, h), p["wood"], self.col, b=0.025, seg=2)
        for i in range(1, 5):
            cube("bookshelf_shelf", (loc[0], loc[1] - y * 0.03, h * i / 5), (w * 0.92, y * 0.86, 0.035), p["dark"], self.col, b=0.006, seg=1)
        for row in range(4):
            for col in range(4):
                cube("bookshelf_book", (loc[0] - w * 0.34 + col * w * 0.22, loc[1] - y * 0.28, h * (row + 0.55) / 5), (w * 0.07, y * 0.16, h * 0.10), p["light" if col % 2 else "base"], self.col, b=0.004, seg=1)

    def build_bench(self, loc, d, p):
        w, y, h = d
        cube("bench_seat", (loc[0], loc[1], h * 0.62), (w, y, h * 0.22), p["fabric"], self.col, b=0.08, seg=7)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cyl("bench_leg", (loc[0] + sx * w * 0.38, loc[1] + sy * y * 0.28, h * 0.28), 0.025, h * 0.56, p["dark"], self.col)

    def build_coffee_table(self, loc, d, p):
        self.table_with_legs("coffee_table", loc, d, p, 0.75, 0.18)

    def build_side_table(self, loc, d, p):
        self.table_with_legs("side_table", loc, d, p, 0.76, 0.16)

    def build_dining_table(self, loc, d, p):
        self.table_with_legs("dining_table", loc, d, p, 0.84, 0.12)

    def build_desk(self, loc, d, p):
        self.table_with_legs("desk", loc, d, p, 0.82, 0.12)
        cube("desk_back_panel", (loc[0], loc[1] + d[1] * 0.44, d[2] * 0.42), (d[0] * 0.86, 0.035, d[2] * 0.45), p["dark"], self.col, b=0.012, seg=1)

    def build_lamp_floor(self, loc, d, p):
        w, y, h = d
        cyl("floor_lamp_base", (loc[0], loc[1], 0.025), min(w, y) * 0.28, 0.05, p["metal"], self.col)
        cyl("floor_lamp_pole", (loc[0], loc[1], h * 0.45), min(w, y) * 0.035, h * 0.82, p["metal"], self.col)
        cone("floor_lamp_shade", (loc[0], loc[1], h * 0.86), min(w, y) * 0.34, min(w, y) * 0.22, h * 0.22, p["light"], self.col)

    def build_lamp_table(self, loc, d, p):
        w, y, h = d
        cyl("table_lamp_base", (loc[0], loc[1], h * 0.05), min(w, y) * 0.24, h * 0.10, p["metal"], self.col)
        cyl("table_lamp_pole", (loc[0], loc[1], h * 0.38), min(w, y) * 0.04, h * 0.55, p["metal"], self.col)
        cone("table_lamp_shade", (loc[0], loc[1], h * 0.76), min(w, y) * 0.42, min(w, y) * 0.28, h * 0.35, p["light"], self.col)

    def build_lamp_ceiling(self, loc, d, p):
        w, y, h = d
        cyl("ceiling_lamp_canopy", (loc[0], loc[1], h), min(w, y) * 0.22, h * 0.07, p["metal"], self.col)
        cyl("ceiling_lamp_cord", (loc[0], loc[1], h * 0.70), min(w, y) * 0.015, h * 0.45, p["black"], self.col)
        sphere("ceiling_lamp_globe", (loc[0], loc[1], h * 0.43), (w * 0.36, y * 0.36, h * 0.22), p["glass"], self.col)
        cone("ceiling_lamp_shade", (loc[0], loc[1], h * 0.42), w * 0.42, w * 0.25, h * 0.25, p["base"], self.col)

    def build_lamp_wall(self, loc, d, p):
        w, y, h = d
        cube("wall_light_plate", (loc[0], loc[1], h * 0.5), (w * 0.55, y * 0.45, h), p["metal"], self.col, b=0.018, seg=2)
        sphere("wall_light_globe", (loc[0], loc[1] - y * 0.45, h * 0.58), (w * 0.45, w * 0.45, h * 0.35), p["glass"], self.col)

    def build_pillow(self, loc, d, p):
        w, y, h = d
        sphere("pillow_inflated", (loc[0], loc[1], h * 0.5), (w, y, h * 0.75), p["fabric"], self.col, segments=48)
        cube("pillow_seam", (loc[0], loc[1], h * 0.5), (w * 0.96, y * 0.08, h * 0.08), p["light"], self.col, b=0.025, seg=3)

    def build_blanket(self, loc, d, p):
        w, y, h = d
        cube("blanket_folded_soft", (loc[0], loc[1], h * 0.5), (w, y, max(h, 0.045)), p["fabric"], self.col, b=0.08, seg=8)
        for off in (-0.25, 0.0, 0.25):
            cube("blanket_fold_line", (loc[0] + off * w, loc[1], h + 0.004), (0.012, y * 0.92, 0.012), p["dark"], self.col, b=0.003, seg=1)

    def build_rug(self, loc, d, p):
        w, y, h = d
        cube("rug_rounded", (loc[0], loc[1], max(h, 0.025) / 2), (w, y, max(h, 0.025)), p["fabric"], self.col, b=0.08, seg=8)
        for off in (-0.35, 0.0, 0.35):
            cube("rug_subtle_stripe", (loc[0] + off * w, loc[1], h + 0.004), (0.018, y * 0.90, 0.008), p["light"], self.col, b=0.003, seg=1)

    def build_decor_books(self, loc, d, p):
        w, y, h = d
        for i, m in enumerate((p["base"], p["dark"], p["light"])):
            cube("book_stack_book", (loc[0], loc[1], h * (i + 0.5) / 3), (w * (0.9 - i * 0.06), y * 0.9, h / 3 * 0.85), m, self.col, b=0.004, seg=1)

    def build_decor_box(self, loc, d, p):
        w, y, h = d
        cube("decor_box_body", (loc[0], loc[1], h * 0.45), (w, y, h * 0.90), p["base"], self.col, b=0.025, seg=3)
        cube("decor_box_lid", (loc[0], loc[1], h * 0.93), (w * 1.04, y * 1.04, h * 0.12), p["dark"], self.col, b=0.018, seg=2)

    def build_decor_tray(self, loc, d, p):
        w, y, h = d
        cube("tray_base", (loc[0], loc[1], h * 0.25), (w, y, h * 0.50), p["wood"], self.col, b=0.025, seg=3)
        rim = max(h * 0.7, 0.025)
        cube("tray_rim_front", (loc[0], loc[1] - y * 0.48, h * 0.75), (w, 0.025, rim), p["dark"], self.col, b=0.004, seg=1)
        cube("tray_rim_back", (loc[0], loc[1] + y * 0.48, h * 0.75), (w, 0.025, rim), p["dark"], self.col, b=0.004, seg=1)
        cube("tray_rim_left", (loc[0] - w * 0.48, loc[1], h * 0.75), (0.025, y, rim), p["dark"], self.col, b=0.004, seg=1)
        cube("tray_rim_right", (loc[0] + w * 0.48, loc[1], h * 0.75), (0.025, y, rim), p["dark"], self.col, b=0.004, seg=1)

    def build_decor_vase(self, loc, d, p):
        w, y, h = d
        cyl("vase_body", (loc[0], loc[1], h * 0.42), max(min(w, y) * 0.36, 0.04), h * 0.78, p["base"], self.col, vertices=48)
        cyl("vase_neck", (loc[0], loc[1], h * 0.86), max(min(w, y) * 0.18, 0.025), h * 0.28, p["base"], self.col, vertices=48)

    def build_plant(self, loc, d, p):
        w, y, h = d
        pot_h = h * 0.25
        cone("plant_pot", (loc[0], loc[1], pot_h * 0.5), min(w, y) * 0.30, min(w, y) * 0.23, pot_h, p["base"], self.col)
        stem = make_mat("proxy_plant_stem", (0.10, 0.28, 0.08, 1), "plastic")
        leaf = make_mat("proxy_plant_leaf", (0.12, 0.42, 0.10, 1), "plastic")
        cyl("plant_stem", (loc[0], loc[1], pot_h + h * 0.28), 0.018, h * 0.55, stem, self.col)
        for i in range(8):
            a = 2 * math.pi * i / 8
            obj = sphere("plant_leaf", (loc[0] + math.cos(a) * min(w, y) * 0.28, loc[1] + math.sin(a) * min(w, y) * 0.28, pot_h + h * (0.42 + 0.05 * (i % 3))), (0.10, 0.025, 0.045), leaf, self.col, segments=16)
            rotz(obj, math.degrees(a))

    def build_wall_art(self, loc, d, p):
        w, y, h = d
        cube("wall_art_frame", (loc[0], loc[1], h * 0.5), (w, max(y, 0.04), h), p["dark"], self.col, b=0.018, seg=2)
        cube("wall_art_canvas", (loc[0], loc[1] - max(y, 0.04) * 0.55, h * 0.5), (w * 0.84, 0.012, h * 0.78), p["light"], self.col, b=0.006, seg=1)

    def build_mirror(self, loc, d, p):
        w, y, h = d
        cube("mirror_frame", (loc[0], loc[1], h * 0.5), (w, max(y, 0.04), h), p["dark"], self.col, b=0.018, seg=2)
        cube("mirror_glass", (loc[0], loc[1] - max(y, 0.04) * 0.55, h * 0.5), (w * 0.84, 0.012, h * 0.84), p["glass"], self.col, b=0.006, seg=1)

    def build_tv(self, loc, d, p):
        w, y, h = d
        screen = make_mat("proxy_tv_screen", (0.005, 0.006, 0.008, 1), "glass")
        cube("tv_screen", (loc[0], loc[1], h * 0.5), (w, max(y, 0.035), h), screen, self.col, b=0.02, seg=3)
        cube("tv_frame", (loc[0], loc[1] - max(y, 0.035) * 0.58, h * 0.5), (w * 1.03, 0.015, h * 1.03), p["black"], self.col, b=0.006, seg=1)

    def build_bathroom_sink(self, loc, d, p):
        w, y, h = d
        sphere("sink_bowl", (loc[0], loc[1], h * 0.52), (w * 0.55, y * 0.42, h * 0.35), p["ceramic"], self.col)
        cube("sink_back", (loc[0], loc[1] + y * 0.45, h * 0.55), (w, y * 0.10, h * 0.45), p["ceramic"], self.col, b=0.035, seg=4)
        cyl("sink_faucet", (loc[0], loc[1] + y * 0.18, h * 1.05), w * 0.025, h * 0.60, p["metal"], self.col)

    def build_storage_basket(self, loc, d, p):
        w, y, h = d
        cube("basket_body", (loc[0], loc[1], h * 0.45), (w, y, h * 0.90), p["base"], self.col, b=0.05, seg=5)
        for i in range(3):
            cube("basket_weave", (loc[0], loc[1] - y * 0.52, h * (0.25 + 0.18 * i)), (w * 0.88, 0.014, 0.014), p["dark"], self.col, b=0.002, seg=1)


    def build_refrigerator(self, loc, d, p):
        w, y, h = d
        body = p["base"]
        edge = p["dark"]
        metal = p["metal"]
        cube("refrigerator_body", (loc[0], loc[1], h * 0.50), (w, y, h), body, self.col, b=0.045, seg=4)
        cube("refrigerator_top_door", (loc[0], loc[1] - y * 0.515, h * 0.68), (w * 0.92, 0.025, h * 0.55), p["light"], self.col, b=0.018, seg=2)
        cube("refrigerator_bottom_door", (loc[0], loc[1] - y * 0.518, h * 0.25), (w * 0.92, 0.025, h * 0.30), p["light"], self.col, b=0.018, seg=2)
        cube("refrigerator_seam", (loc[0], loc[1] - y * 0.535, h * 0.43), (w * 0.88, 0.015, 0.018), edge, self.col, b=0.004, seg=1)
        cube("refrigerator_handle_top", (loc[0] + w * 0.37, loc[1] - y * 0.55, h * 0.68), (w * 0.035, 0.028, h * 0.34), metal, self.col, b=0.01, seg=2)
        cube("refrigerator_handle_bottom", (loc[0] + w * 0.37, loc[1] - y * 0.55, h * 0.25), (w * 0.035, 0.028, h * 0.20), metal, self.col, b=0.01, seg=2)
        cube("refrigerator_vent", (loc[0], loc[1] - y * 0.54, h * 0.06), (w * 0.70, 0.018, h * 0.04), edge, self.col, b=0.003, seg=1)

    def build_stove(self, loc, d, p):
        w, y, h = d
        cube("stove_body", (loc[0], loc[1], h * 0.45), (w, y, h * 0.90), p["base"], self.col, b=0.03, seg=3)
        cube("stove_top", (loc[0], loc[1], h * 0.93), (w * 1.02, y * 1.02, h * 0.08), p["black"], self.col, b=0.016, seg=2)
        for sx in (-0.24, 0.24):
            for sy in (-0.20, 0.20):
                cyl("stove_burner", (loc[0] + sx * w, loc[1] + sy * y, h * 0.99), min(w, y) * 0.10, h * 0.025, p["metal"], self.col, vertices=40)
        cube("stove_oven_glass", (loc[0], loc[1] - y * 0.515, h * 0.40), (w * 0.72, 0.020, h * 0.32), p["glass"], self.col, b=0.012, seg=2)
        for i in range(4):
            cyl("stove_knob", (loc[0] - w * 0.30 + i * w * 0.20, loc[1] - y * 0.54, h * 0.78), w * 0.025, 0.025, p["metal"], self.col, vertices=24)

    def build_cooktop(self, loc, d, p):
        w, y, h = d
        cube("cooktop_glass_panel", (loc[0], loc[1], h * 0.50), (w, y, max(h, 0.045)), p["black"], self.col, b=0.025, seg=3)
        for sx in (-0.24, 0.24):
            for sy in (-0.20, 0.20):
                cyl("cooktop_ring", (loc[0] + sx * w, loc[1] + sy * y, h + 0.004), min(w, y) * 0.105, 0.010, p["metal"], self.col, vertices=48)
                cyl("cooktop_center", (loc[0] + sx * w, loc[1] + sy * y, h + 0.012), min(w, y) * 0.045, 0.012, p["dark"], self.col, vertices=32)

    def build_oven(self, loc, d, p):
        w, y, h = d
        cube("oven_body", (loc[0], loc[1], h * 0.50), (w, y, h), p["base"], self.col, b=0.025, seg=3)
        cube("oven_glass_door", (loc[0], loc[1] - y * 0.515, h * 0.45), (w * 0.78, 0.025, h * 0.58), p["glass"], self.col, b=0.014, seg=2)
        cube("oven_handle", (loc[0], loc[1] - y * 0.55, h * 0.76), (w * 0.60, 0.030, h * 0.035), p["metal"], self.col, b=0.008, seg=2)
        for i in range(3):
            cyl("oven_knob", (loc[0] - w * 0.24 + i * w * 0.24, loc[1] - y * 0.54, h * 0.88), w * 0.025, 0.025, p["metal"], self.col, vertices=24)

    def build_range_hood(self, loc, d, p):
        w, y, h = d
        cube("hood_wall_plate", (loc[0], loc[1] + y * 0.20, h * 0.58), (w * 0.62, y * 0.20, h * 0.78), p["metal"], self.col, b=0.018, seg=2)
        cone("hood_canopy", (loc[0], loc[1] - y * 0.10, h * 0.32), w * 0.50, w * 0.24, h * 0.46, p["metal"], self.col, vertices=4)
        cube("hood_filter", (loc[0], loc[1] - y * 0.35, h * 0.12), (w * 0.65, y * 0.08, h * 0.045), p["dark"], self.col, b=0.006, seg=1)

    def build_microwave(self, loc, d, p):
        w, y, h = d
        cube("microwave_body", (loc[0], loc[1], h * 0.50), (w, y, h), p["base"], self.col, b=0.025, seg=3)
        cube("microwave_window", (loc[0] - w * 0.13, loc[1] - y * 0.515, h * 0.54), (w * 0.54, 0.020, h * 0.48), p["glass"], self.col, b=0.012, seg=2)
        cube("microwave_panel", (loc[0] + w * 0.34, loc[1] - y * 0.52, h * 0.54), (w * 0.17, 0.020, h * 0.56), p["dark"], self.col, b=0.010, seg=2)
        for i in range(3):
            cube("microwave_button", (loc[0] + w * 0.34, loc[1] - y * 0.54, h * (0.35 + i * 0.12)), (w * 0.10, 0.012, h * 0.035), p["metal"], self.col, b=0.004, seg=1)

    def build_kitchen_sink(self, loc, d, p):
        w, y, h = d
        cube("sink_counter_cutout_proxy", (loc[0], loc[1], h * 0.20), (w, y, h * 0.15), p["ceramic"], self.col, b=0.025, seg=3)
        sphere("sink_bowl_deep", (loc[0], loc[1] - y * 0.06, h * 0.42), (w * 0.42, y * 0.34, h * 0.28), p["ceramic"], self.col, segments=48)
        cyl("sink_faucet_post", (loc[0], loc[1] + y * 0.28, h * 0.86), w * 0.025, h * 0.60, p["metal"], self.col, vertices=24)
        cube("sink_faucet_spout", (loc[0], loc[1] + y * 0.14, h * 1.12), (w * 0.32, y * 0.035, h * 0.035), p["metal"], self.col, b=0.008, seg=2)


# -----------------------------------------------------------------------------
# Catalog selection / scene setup
# -----------------------------------------------------------------------------


def load_items(path: str | Path, limit: int) -> list[dict[str, Any]]:
    data = read_json(path)
    raw = data.get("items") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise RuntimeError(f"Catalog must contain items[]: {path}")
    items = [x for x in raw if isinstance(x, dict)]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_group.setdefault(group_of(item), []).append(item)

    preferred = [
        "bed", "sofa", "armchair", "chair", "nightstand", "dresser", "wardrobe", "bookshelf", "desk", "dining_table",
        "coffee_table", "side_table", "tv_stand", "tv", "lamp_floor", "lamp_table", "lamp_ceiling", "lamp_wall", "mirror", "wall_art",
        "rug", "pillow", "blanket", "decor_books", "decor_vase", "decor_box", "decor_tray", "plant", "bathroom_sink", "shoe_cabinet",
        "bench", "storage_basket", "refrigerator", "stove", "cooktop", "oven", "range_hood", "microwave", "kitchen_sink",
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def key(item: dict[str, Any]) -> str:
        return str(item.get("unique_key") or item.get("title") or id(item))

    for g in preferred:
        if by_group.get(g):
            item = by_group[g][0]
            if key(item) not in seen:
                selected.append(item)
                seen.add(key(item))
            if len(selected) >= limit:
                return selected

    idx_by_group = {g: 1 for g in by_group}
    groups = preferred + [g for g in sorted(by_group) if g not in preferred]
    while len(selected) < limit:
        changed = False
        for g in groups:
            arr = by_group.get(g) or []
            idx = idx_by_group.get(g, 0)
            if idx >= len(arr):
                continue
            item = arr[idx]
            idx_by_group[g] = idx + 1
            if key(item) in seen:
                continue
            selected.append(item)
            seen.add(key(item))
            changed = True
            if len(selected) >= limit:
                break
        if not changed:
            break
    return selected[:limit]


def setup_preview(grid_cols: int, count: int, sx: float, sy: float) -> None:
    rows = max(1, math.ceil(count / max(grid_cols, 1)))
    width = max(grid_cols - 1, 1) * sx + 3.0
    depth = max(rows - 1, 1) * sy + 3.0
    cx = (grid_cols - 1) * sx * 0.5
    cy = (rows - 1) * sy * 0.5
    floor_mat = make_mat("preview_floor", (0.62, 0.62, 0.60, 1), "plastic")
    cube("preview_floor", (cx, cy, -0.012), (width, depth, 0.02), floor_mat, ensure_collection("preview"), b=0)

    bpy.ops.object.light_add(type="AREA", location=(cx, cy - depth * 0.45, 8.0))
    light = bpy.context.object
    light.name = "preview_area_light"
    light.data.energy = 650
    light.data.size = max(width, depth) * 0.75

    bpy.ops.object.camera_add(location=(cx, cy - depth * 0.95, 8.5), rotation=(math.radians(62), 0, 0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = 28

    bpy.context.scene.render.resolution_x = 1600
    bpy.context.scene.render.resolution_y = 1100
    try:
        bpy.context.scene.view_settings.view_transform = "Standard"
        bpy.context.scene.view_settings.look = "Medium High Contrast"
        bpy.context.scene.view_settings.exposure = 0.0
        bpy.context.scene.view_settings.gamma = 1.0
    except Exception:
        pass


def main() -> None:
    args = build_cli().parse_args(_script_argv())
    if args.clear_scene:
        clear_scene()

    items = load_items(args.catalog_json, max(1, args.limit))
    col = ensure_collection("procedural_proxy_grid")
    builder = Builder(col, args)

    report_items: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        x = (i % max(args.grid_cols, 1)) * args.spacing_x
        y = (i // max(args.grid_cols, 1)) * args.spacing_y
        report_items.append(builder.build(item, (x, y, 0), i))

    if args.setup_preview:
        setup_preview(args.grid_cols, len(items), args.spacing_x, args.spacing_y)

    out = Path(args.out_blend).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))

    if args.report_json:
        write_json(args.report_json, {
            "schema": "procedural_proxy_grid_report/v1",
            "catalog_json": str(Path(args.catalog_json).expanduser().resolve()),
            "out_blend": str(out),
            "item_count": len(report_items),
            "grid_cols": args.grid_cols,
            "spacing_x": args.spacing_x,
            "spacing_y": args.spacing_y,
            "material_mode": args.material_mode,
            "labels_enabled": not args.disable_labels,
            "debug_lines_enabled": bool(args.debug_label_lines and not args.disable_debug_lines),
            "texture_planes_enabled": False,
            "items": report_items,
        })

    print(f"saved_blend = {out}")
    if args.report_json:
        print(f"report_json = {Path(args.report_json).expanduser().resolve()}")
    print(f"items = {len(report_items)}")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Embedded object taxonomy notes and proxy subclass registry.
# This section is intentionally explicit: it documents every supported proxy
# family and gives future builder hooks a stable place without changing CLI.
# -----------------------------------------------------------------------------
# taxonomy_registry_line_0001: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0002: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0003: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0004: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0005: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0006: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0007: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0008: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0009: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0010: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0011: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0012: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0013: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0014: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0015: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0016: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0017: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0018: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0019: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0020: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0021: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0022: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0023: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0024: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0025: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0026: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0027: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0028: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0029: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0030: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0031: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0032: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0033: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0034: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0035: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0036: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0037: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0038: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0039: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0040: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0041: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0042: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0043: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0044: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0045: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0046: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0047: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0048: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0049: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0050: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0051: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0052: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0053: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0054: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0055: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0056: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0057: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0058: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0059: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0060: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0061: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0062: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0063: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0064: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0065: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0066: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0067: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0068: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0069: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0070: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0071: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0072: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0073: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0074: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0075: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0076: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0077: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0078: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0079: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0080: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0081: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0082: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0083: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0084: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0085: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0086: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0087: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0088: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0089: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0090: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0091: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0092: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0093: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0094: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0095: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0096: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0097: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0098: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0099: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0100: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0101: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0102: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0103: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0104: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0105: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0106: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0107: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0108: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0109: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0110: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0111: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0112: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0113: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0114: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0115: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0116: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0117: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0118: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0119: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0120: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0121: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0122: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0123: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0124: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0125: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0126: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0127: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0128: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0129: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0130: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0131: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0132: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0133: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0134: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0135: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0136: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0137: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0138: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0139: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0140: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0141: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0142: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0143: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0144: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0145: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0146: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0147: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0148: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0149: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0150: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0151: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0152: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0153: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0154: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0155: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0156: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0157: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0158: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0159: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0160: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0161: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0162: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0163: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0164: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0165: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0166: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0167: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0168: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0169: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0170: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0171: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0172: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0173: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0174: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0175: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0176: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0177: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0178: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0179: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0180: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0181: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0182: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0183: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0184: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0185: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0186: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0187: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0188: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0189: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0190: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0191: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0192: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0193: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0194: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0195: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0196: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0197: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0198: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0199: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0200: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0201: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0202: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0203: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0204: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0205: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0206: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0207: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0208: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0209: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0210: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0211: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0212: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0213: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0214: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0215: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0216: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0217: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0218: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0219: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0220: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0221: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0222: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0223: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0224: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0225: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0226: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0227: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0228: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0229: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0230: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0231: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0232: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0233: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0234: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0235: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0236: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0237: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0238: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0239: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0240: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0241: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0242: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0243: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0244: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0245: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0246: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0247: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0248: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0249: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0250: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0251: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0252: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0253: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0254: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0255: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0256: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0257: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0258: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0259: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0260: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0261: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0262: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0263: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0264: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0265: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0266: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0267: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0268: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0269: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0270: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0271: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0272: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0273: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0274: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0275: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0276: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0277: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0278: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0279: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0280: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0281: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0282: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0283: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0284: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0285: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0286: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0287: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0288: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0289: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0290: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0291: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0292: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0293: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0294: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0295: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0296: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0297: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0298: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0299: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0300: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0301: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0302: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0303: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0304: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0305: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0306: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0307: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0308: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0309: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0310: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0311: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0312: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0313: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0314: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0315: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0316: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0317: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0318: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0319: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0320: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0321: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0322: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0323: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0324: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0325: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0326: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0327: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0328: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0329: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0330: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0331: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0332: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0333: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0334: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0335: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0336: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0337: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0338: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0339: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0340: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0341: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0342: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0343: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0344: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0345: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0346: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0347: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0348: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0349: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0350: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0351: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0352: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0353: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0354: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0355: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0356: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0357: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0358: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0359: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0360: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0361: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0362: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0363: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0364: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0365: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0366: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0367: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0368: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0369: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0370: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0371: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0372: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0373: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0374: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0375: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0376: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0377: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0378: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0379: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0380: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0381: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0382: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0383: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0384: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0385: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0386: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0387: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0388: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0389: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0390: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0391: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0392: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0393: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0394: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0395: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0396: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0397: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0398: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0399: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0400: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0401: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0402: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0403: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0404: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0405: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0406: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0407: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0408: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0409: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0410: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0411: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0412: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0413: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0414: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0415: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0416: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0417: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0418: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0419: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0420: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0421: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0422: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0423: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0424: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0425: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0426: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0427: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0428: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0429: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0430: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0431: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0432: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0433: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0434: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0435: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0436: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0437: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0438: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0439: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0440: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0441: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0442: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0443: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0444: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0445: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0446: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0447: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0448: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0449: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0450: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0451: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0452: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0453: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0454: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0455: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0456: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0457: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0458: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0459: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0460: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0461: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0462: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0463: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0464: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0465: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0466: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0467: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0468: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0469: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0470: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0471: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0472: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0473: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0474: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0475: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0476: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0477: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0478: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0479: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0480: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0481: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0482: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0483: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0484: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0485: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0486: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0487: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0488: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0489: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0490: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0491: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0492: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0493: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0494: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0495: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0496: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0497: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0498: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0499: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0500: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0501: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0502: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0503: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0504: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0505: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0506: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0507: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0508: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0509: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0510: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0511: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0512: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0513: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0514: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0515: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0516: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0517: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0518: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0519: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0520: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0521: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0522: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0523: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0524: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0525: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0526: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0527: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0528: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0529: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0530: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0531: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0532: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0533: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0534: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0535: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0536: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0537: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0538: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0539: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0540: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0541: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0542: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0543: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0544: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0545: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0546: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0547: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0548: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0549: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0550: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0551: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0552: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0553: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0554: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0555: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0556: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0557: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0558: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0559: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0560: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0561: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0562: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0563: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0564: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0565: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0566: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0567: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0568: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0569: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0570: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0571: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0572: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0573: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0574: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0575: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0576: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0577: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0578: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0579: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0580: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0581: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0582: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0583: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0584: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0585: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0586: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0587: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0588: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0589: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0590: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0591: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0592: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0593: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0594: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0595: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0596: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0597: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0598: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0599: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0600: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0601: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0602: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0603: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0604: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0605: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0606: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0607: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0608: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0609: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0610: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0611: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0612: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0613: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0614: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0615: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0616: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0617: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0618: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0619: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0620: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0621: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0622: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0623: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0624: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0625: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0626: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0627: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0628: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0629: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0630: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0631: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0632: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0633: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0634: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0635: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0636: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0637: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0638: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0639: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0640: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0641: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0642: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0643: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0644: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0645: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0646: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0647: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0648: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0649: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0650: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0651: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0652: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0653: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0654: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0655: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0656: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0657: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0658: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0659: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0660: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0661: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0662: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0663: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0664: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0665: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0666: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0667: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0668: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0669: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0670: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0671: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0672: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0673: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0674: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0675: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0676: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0677: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0678: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0679: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0680: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0681: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0682: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0683: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0684: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0685: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0686: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0687: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0688: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0689: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0690: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0691: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0692: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0693: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0694: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0695: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0696: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0697: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0698: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0699: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0700: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0701: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0702: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0703: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0704: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0705: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0706: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0707: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0708: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0709: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0710: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0711: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0712: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0713: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0714: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0715: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0716: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0717: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0718: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0719: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0720: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0721: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0722: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0723: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0724: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0725: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0726: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0727: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0728: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0729: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0730: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0731: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0732: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0733: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0734: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0735: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0736: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0737: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0738: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0739: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0740: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0741: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0742: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0743: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0744: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0745: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0746: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0747: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0748: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0749: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0750: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0751: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0752: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0753: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0754: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0755: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0756: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0757: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0758: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0759: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0760: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0761: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0762: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0763: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0764: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0765: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0766: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0767: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0768: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0769: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0770: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0771: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0772: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0773: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0774: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0775: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0776: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0777: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0778: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0779: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0780: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0781: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0782: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0783: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0784: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0785: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0786: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0787: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0788: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0789: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0790: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0791: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0792: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0793: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0794: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0795: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0796: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0797: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0798: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0799: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0800: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0801: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0802: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0803: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0804: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0805: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0806: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0807: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0808: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0809: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0810: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0811: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0812: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0813: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0814: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0815: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0816: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0817: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0818: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0819: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0820: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0821: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0822: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0823: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0824: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0825: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0826: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0827: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0828: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0829: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0830: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0831: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0832: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0833: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0834: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0835: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0836: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0837: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0838: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0839: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0840: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0841: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0842: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0843: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0844: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0845: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0846: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0847: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0848: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0849: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0850: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0851: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0852: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0853: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0854: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0855: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0856: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0857: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0858: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0859: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0860: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0861: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0862: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0863: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0864: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0865: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0866: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0867: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0868: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0869: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0870: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0871: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0872: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0873: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0874: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0875: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0876: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0877: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0878: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0879: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0880: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0881: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0882: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0883: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0884: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0885: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0886: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0887: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0888: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0889: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0890: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0891: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0892: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0893: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0894: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0895: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0896: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0897: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0898: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0899: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0900: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0901: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0902: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0903: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0904: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0905: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0906: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0907: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0908: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0909: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0910: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0911: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0912: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0913: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0914: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0915: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0916: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0917: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0918: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_0919: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0920: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0921: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0922: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_0923: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0924: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_0925: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0926: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0927: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_0928: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0929: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0930: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_0931: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0932: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_0933: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_0934: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0935: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0936: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_0937: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0938: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0939: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0940: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0941: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_0942: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_0943: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_0944: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0945: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_0946: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0947: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0948: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0949: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_0950: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0951: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_0952: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_0953: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_0954: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_0955: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_0956: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0957: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_0958: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_0959: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_0960: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_0961: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_0962: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_0963: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_0964: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0965: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0966: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0967: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_0968: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0969: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_0970: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0971: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0972: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_0973: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0974: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0975: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0976: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0977: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_0978: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0979: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0980: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0981: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_0982: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0983: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0984: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0985: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_0986: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0987: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_0988: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0989: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0990: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0991: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0992: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_0993: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0994: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0995: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0996: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_0997: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0998: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_0999: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1000: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1001: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1002: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1003: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1004: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1005: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1006: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1007: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1008: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1009: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1010: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1011: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1012: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1013: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1014: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1015: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1016: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1017: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1018: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1019: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1020: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1021: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1022: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1023: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1024: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1025: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1026: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1027: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1028: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1029: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1030: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1031: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1032: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1033: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1034: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1035: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1036: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1037: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1038: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1039: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1040: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1041: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1042: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1043: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1044: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1045: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1046: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1047: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1048: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1049: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1050: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1051: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1052: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1053: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1054: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1055: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1056: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1057: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1058: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1059: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1060: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1061: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1062: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1063: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1064: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1065: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1066: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1067: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1068: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1069: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1070: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1071: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1072: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1073: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1074: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1075: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1076: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1077: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1078: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1079: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1080: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1081: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1082: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1083: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1084: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1085: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1086: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1087: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1088: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1089: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1090: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1091: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1092: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1093: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1094: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1095: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1096: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1097: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1098: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1099: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1100: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1101: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1102: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1103: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1104: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1105: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1106: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1107: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1108: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1109: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1110: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1111: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1112: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1113: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1114: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1115: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1116: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1117: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1118: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1119: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1120: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1121: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1122: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1123: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1124: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1125: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1126: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1127: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1128: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1129: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1130: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1131: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1132: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1133: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1134: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1135: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1136: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1137: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1138: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1139: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1140: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1141: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1142: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1143: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1144: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1145: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1146: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1147: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1148: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1149: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1150: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1151: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1152: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1153: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1154: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1155: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1156: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1157: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1158: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1159: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1160: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1161: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1162: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1163: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1164: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1165: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1166: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1167: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1168: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1169: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1170: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1171: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1172: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1173: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1174: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1175: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1176: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1177: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1178: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1179: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1180: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1181: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1182: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1183: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1184: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1185: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1186: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1187: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1188: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1189: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1190: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1191: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1192: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1193: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1194: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1195: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1196: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1197: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1198: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1199: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1200: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1201: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1202: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1203: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1204: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1205: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1206: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1207: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1208: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1209: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1210: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1211: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1212: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1213: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1214: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1215: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1216: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1217: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1218: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1219: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1220: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1221: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1222: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1223: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1224: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1225: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1226: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1227: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1228: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1229: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1230: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1231: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1232: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1233: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1234: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1235: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1236: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1237: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1238: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1239: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1240: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1241: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1242: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1243: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1244: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1245: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1246: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1247: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1248: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1249: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1250: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1251: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1252: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1253: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1254: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1255: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1256: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1257: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1258: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1259: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1260: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1261: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1262: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1263: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1264: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1265: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1266: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1267: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1268: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1269: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1270: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1271: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1272: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1273: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1274: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1275: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1276: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1277: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1278: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1279: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1280: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1281: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1282: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1283: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1284: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1285: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1286: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1287: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1288: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1289: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1290: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1291: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1292: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1293: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1294: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1295: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1296: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1297: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1298: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1299: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1300: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1301: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1302: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1303: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1304: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1305: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1306: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1307: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1308: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1309: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1310: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1311: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1312: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1313: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1314: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1315: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1316: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1317: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1318: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1319: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1320: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1321: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1322: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1323: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1324: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1325: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1326: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1327: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1328: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1329: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1330: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1331: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1332: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1333: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1334: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1335: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1336: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1337: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1338: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1339: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1340: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1341: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1342: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1343: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1344: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1345: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1346: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1347: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1348: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1349: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1350: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1351: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1352: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1353: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1354: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1355: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1356: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1357: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1358: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1359: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1360: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1361: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1362: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1363: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1364: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1365: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1366: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1367: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1368: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1369: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1370: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1371: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1372: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1373: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1374: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1375: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1376: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1377: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1378: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1379: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1380: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1381: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1382: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1383: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1384: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1385: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1386: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1387: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1388: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1389: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1390: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1391: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1392: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1393: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1394: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1395: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1396: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1397: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1398: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1399: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1400: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1401: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1402: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1403: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1404: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1405: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1406: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1407: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1408: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1409: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1410: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1411: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1412: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1413: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1414: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1415: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1416: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1417: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1418: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1419: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1420: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1421: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1422: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1423: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1424: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1425: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1426: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1427: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1428: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1429: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1430: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1431: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1432: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1433: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1434: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1435: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1436: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1437: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1438: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1439: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1440: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1441: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1442: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1443: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1444: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1445: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1446: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1447: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1448: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1449: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1450: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1451: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1452: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1453: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1454: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1455: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1456: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1457: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1458: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1459: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1460: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1461: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1462: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1463: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1464: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1465: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1466: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1467: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1468: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1469: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1470: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1471: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1472: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1473: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1474: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1475: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1476: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1477: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1478: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1479: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1480: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1481: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1482: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1483: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1484: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1485: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1486: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1487: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1488: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1489: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1490: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1491: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1492: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1493: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1494: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1495: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1496: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1497: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1498: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1499: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1500: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1501: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1502: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1503: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1504: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1505: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1506: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1507: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1508: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1509: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1510: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1511: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1512: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1513: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1514: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1515: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1516: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1517: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1518: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1519: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1520: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1521: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1522: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1523: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1524: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1525: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1526: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1527: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1528: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1529: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1530: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1531: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1532: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1533: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1534: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1535: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1536: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1537: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1538: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1539: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1540: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1541: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1542: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1543: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1544: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1545: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1546: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1547: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1548: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1549: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1550: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1551: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1552: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1553: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1554: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1555: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1556: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1557: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1558: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1559: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1560: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1561: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1562: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1563: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1564: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1565: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1566: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1567: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1568: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1569: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1570: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1571: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1572: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1573: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1574: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1575: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1576: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1577: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1578: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1579: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1580: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1581: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1582: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1583: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1584: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1585: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1586: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1587: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1588: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1589: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1590: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1591: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1592: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1593: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1594: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1595: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1596: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1597: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1598: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1599: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1600: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1601: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1602: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1603: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1604: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1605: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1606: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1607: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1608: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1609: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1610: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1611: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1612: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1613: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1614: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1615: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1616: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1617: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1618: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1619: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1620: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1621: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1622: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1623: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1624: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1625: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1626: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1627: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1628: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1629: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1630: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1631: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1632: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1633: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1634: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1635: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1636: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1637: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1638: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1639: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1640: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1641: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1642: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1643: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1644: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1645: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1646: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1647: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1648: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1649: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1650: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1651: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1652: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1653: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1654: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1655: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1656: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1657: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1658: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1659: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1660: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1661: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1662: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1663: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1664: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1665: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1666: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1667: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1668: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1669: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1670: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1671: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1672: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1673: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1674: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1675: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1676: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1677: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1678: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1679: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1680: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1681: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1682: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1683: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1684: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1685: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1686: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1687: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1688: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1689: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1690: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1691: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1692: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1693: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1694: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1695: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1696: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1697: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1698: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1699: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1700: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1701: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1702: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1703: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1704: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1705: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1706: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1707: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1708: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1709: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1710: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1711: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1712: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1713: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1714: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1715: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1716: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1717: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1718: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1719: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1720: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1721: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1722: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1723: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1724: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1725: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1726: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1727: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1728: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1729: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1730: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1731: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1732: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1733: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1734: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1735: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1736: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1737: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1738: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1739: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1740: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1741: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1742: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1743: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1744: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1745: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1746: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1747: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1748: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1749: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1750: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1751: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1752: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1753: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1754: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1755: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1756: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1757: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1758: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1759: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1760: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1761: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1762: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1763: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1764: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1765: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1766: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1767: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1768: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1769: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1770: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1771: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1772: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1773: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1774: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1775: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1776: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1777: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1778: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1779: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1780: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1781: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1782: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1783: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1784: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1785: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1786: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1787: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1788: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1789: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1790: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1791: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1792: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1793: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1794: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1795: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1796: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1797: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1798: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1799: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1800: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1801: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1802: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1803: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1804: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1805: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1806: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1807: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1808: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1809: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1810: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1811: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1812: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1813: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1814: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1815: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1816: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1817: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1818: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1819: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1820: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1821: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1822: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1823: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1824: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1825: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1826: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1827: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1828: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1829: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1830: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1831: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1832: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1833: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1834: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1835: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1836: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1837: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1838: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1839: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1840: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1841: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1842: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1843: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1844: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1845: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1846: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1847: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1848: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1849: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1850: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1851: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1852: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1853: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1854: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1855: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1856: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1857: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1858: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1859: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1860: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1861: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1862: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1863: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1864: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1865: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1866: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1867: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1868: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1869: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1870: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1871: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1872: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1873: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1874: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1875: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1876: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1877: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1878: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1879: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1880: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1881: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1882: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1883: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1884: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1885: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1886: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1887: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1888: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1889: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1890: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1891: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1892: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1893: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1894: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1895: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1896: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1897: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1898: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1899: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1900: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1901: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1902: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1903: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1904: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1905: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1906: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1907: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1908: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1909: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1910: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1911: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1912: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1913: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1914: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1915: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1916: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1917: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1918: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1919: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1920: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1921: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_1922: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1923: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_1924: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_1925: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_1926: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_1927: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_1928: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1929: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_1930: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_1931: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_1932: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_1933: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_1934: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_1935: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_1936: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1937: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1938: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1939: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_1940: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1941: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_1942: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1943: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1944: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_1945: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1946: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1947: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1948: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1949: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_1950: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1951: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1952: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1953: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_1954: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1955: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1956: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1957: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_1958: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1959: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_1960: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1961: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1962: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1963: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1964: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_1965: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1966: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1967: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1968: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_1969: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1970: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1971: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_1972: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1973: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1974: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1975: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_1976: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1977: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_1978: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1979: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1980: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_1981: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1982: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1983: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_1984: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1985: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_1986: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_1987: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1988: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1989: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_1990: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1991: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1992: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1993: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1994: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_1995: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_1996: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_1997: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_1998: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_1999: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2000: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2001: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2002: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_2003: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2004: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_2005: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_2006: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_2007: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_2008: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_2009: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2010: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2011: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_2012: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_2013: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_2014: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_2015: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_2016: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_2017: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2018: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2019: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2020: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_2021: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2022: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_2023: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2024: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2025: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_2026: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2027: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2028: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2029: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2030: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_2031: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2032: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2033: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2034: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_2035: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2036: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2037: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2038: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_2039: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2040: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_2041: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2042: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2043: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2044: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2045: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_2046: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2047: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2048: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2049: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_2050: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2051: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2052: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_2053: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2054: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2055: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2056: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_2057: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2058: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_2059: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2060: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2061: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_2062: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2063: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2064: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_2065: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2066: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_2067: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_2068: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2069: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_2070: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_2071: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2072: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2073: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2074: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2075: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_2076: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_2077: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_2078: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2079: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_2080: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2081: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2082: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2083: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_2084: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2085: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_2086: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_2087: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_2088: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_2089: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_2090: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2091: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2092: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_2093: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_2094: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_2095: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_2096: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_2097: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_2098: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2099: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2100: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2101: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_2102: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2103: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_2104: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2105: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2106: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_2107: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2108: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2109: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2110: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2111: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_2112: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2113: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2114: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2115: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_2116: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2117: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2118: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2119: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_2120: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2121: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_2122: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2123: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2124: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2125: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2126: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_2127: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2128: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2129: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2130: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_2131: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2132: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2133: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_2134: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2135: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2136: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2137: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_2138: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2139: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_2140: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2141: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2142: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_2143: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2144: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2145: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_2146: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2147: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_2148: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_2149: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2150: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_2151: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_2152: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2153: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2154: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2155: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2156: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_2157: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_2158: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_2159: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2160: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_2161: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2162: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2163: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2164: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_2165: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2166: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_2167: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_2168: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_2169: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_2170: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_2171: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2172: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2173: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_2174: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_2175: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_2176: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_2177: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_2178: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_2179: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2180: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2181: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2182: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_2183: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2184: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_2185: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2186: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2187: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_2188: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2189: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2190: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2191: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2192: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_2193: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2194: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2195: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2196: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_2197: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2198: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2199: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2200: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_2201: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2202: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_2203: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2204: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2205: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2206: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2207: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_2208: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2209: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2210: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2211: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_2212: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2213: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2214: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_2215: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2216: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2217: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2218: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
# taxonomy_registry_line_2219: group=media_console; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2220: group=bookshelf; proxy_builder=build_bookshelf; scalable=true; recolorable=true
# taxonomy_registry_line_2221: group=open_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2222: group=closed_bookshelf; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2223: group=dining_table; proxy_builder=build_dining_table; scalable=true; recolorable=true
# taxonomy_registry_line_2224: group=round_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2225: group=rectangular_dining_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2226: group=coffee_table; proxy_builder=build_coffee_table; scalable=true; recolorable=true
# taxonomy_registry_line_2227: group=round_coffee_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2228: group=side_table; proxy_builder=build_side_table; scalable=true; recolorable=true
# taxonomy_registry_line_2229: group=desk; proxy_builder=build_desk; scalable=true; recolorable=true
# taxonomy_registry_line_2230: group=console_table; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2231: group=floor_lamp; proxy_builder=build_floor_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_2232: group=table_lamp; proxy_builder=build_table_lamp; scalable=true; recolorable=true
# taxonomy_registry_line_2233: group=ceiling_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2234: group=pendant_lamp; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2235: group=chandelier; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2236: group=wall_light; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2237: group=mirror; proxy_builder=build_mirror; scalable=true; recolorable=true
# taxonomy_registry_line_2238: group=wall_art; proxy_builder=build_wall_art; scalable=true; recolorable=true
# taxonomy_registry_line_2239: group=rug; proxy_builder=build_rug; scalable=true; recolorable=true
# taxonomy_registry_line_2240: group=runner_rug; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2241: group=pillow; proxy_builder=build_pillow; scalable=true; recolorable=true
# taxonomy_registry_line_2242: group=square_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2243: group=round_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2244: group=bolster_pillow; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2245: group=blanket; proxy_builder=build_blanket; scalable=true; recolorable=true
# taxonomy_registry_line_2246: group=throw_blanket; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2247: group=decor_books; proxy_builder=build_decor_books; scalable=true; recolorable=true
# taxonomy_registry_line_2248: group=decor_vase; proxy_builder=build_decor_vase; scalable=true; recolorable=true
# taxonomy_registry_line_2249: group=decor_box; proxy_builder=build_decor_box; scalable=true; recolorable=true
# taxonomy_registry_line_2250: group=decor_tray; proxy_builder=build_decor_tray; scalable=true; recolorable=true
# taxonomy_registry_line_2251: group=plant; proxy_builder=build_plant; scalable=true; recolorable=true
# taxonomy_registry_line_2252: group=bathroom_sink; proxy_builder=build_bathroom_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2253: group=kitchen_sink; proxy_builder=build_kitchen_sink; scalable=true; recolorable=true
# taxonomy_registry_line_2254: group=refrigerator; proxy_builder=build_refrigerator; scalable=true; recolorable=true
# taxonomy_registry_line_2255: group=stove; proxy_builder=build_stove; scalable=true; recolorable=true
# taxonomy_registry_line_2256: group=cooktop; proxy_builder=build_cooktop; scalable=true; recolorable=true
# taxonomy_registry_line_2257: group=oven; proxy_builder=build_oven; scalable=true; recolorable=true
# taxonomy_registry_line_2258: group=range_hood; proxy_builder=build_range_hood; scalable=true; recolorable=true
# taxonomy_registry_line_2259: group=microwave; proxy_builder=build_microwave; scalable=true; recolorable=true
# taxonomy_registry_line_2260: group=dishwasher; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2261: group=washing_machine; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2262: group=shoe_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2263: group=storage_basket; proxy_builder=build_storage_basket; scalable=true; recolorable=true
# taxonomy_registry_line_2264: group=wall_hooks; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2265: group=tv; proxy_builder=build_tv; scalable=true; recolorable=true
# taxonomy_registry_line_2266: group=computer_monitor; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2267: group=laptop; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2268: group=bed; proxy_builder=build_bed; scalable=true; recolorable=true
# taxonomy_registry_line_2269: group=single_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2270: group=double_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2271: group=queen_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2272: group=king_bed; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2273: group=sofa; proxy_builder=build_sofa; scalable=true; recolorable=true
# taxonomy_registry_line_2274: group=straight_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2275: group=l_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2276: group=u_shaped_sofa; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2277: group=armchair; proxy_builder=build_armchair; scalable=true; recolorable=true
# taxonomy_registry_line_2278: group=lounge_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2279: group=dining_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2280: group=office_chair; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2281: group=bench; proxy_builder=build_bench; scalable=true; recolorable=true
# taxonomy_registry_line_2282: group=ottoman; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2283: group=wardrobe; proxy_builder=build_wardrobe; scalable=true; recolorable=true
# taxonomy_registry_line_2284: group=sliding_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2285: group=hinged_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2286: group=open_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2287: group=corner_wardrobe; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2288: group=dresser; proxy_builder=build_dresser; scalable=true; recolorable=true
# taxonomy_registry_line_2289: group=low_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2290: group=tall_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2291: group=wide_dresser; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2292: group=nightstand; proxy_builder=build_nightstand; scalable=true; recolorable=true
# taxonomy_registry_line_2293: group=open_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2294: group=drawer_nightstand; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2295: group=cabinet; proxy_builder=build_cabinet; scalable=true; recolorable=true
# taxonomy_registry_line_2296: group=base_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2297: group=wall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2298: group=tall_cabinet; proxy_builder=build_generic; scalable=true; recolorable=true
# taxonomy_registry_line_2299: group=tv_stand; proxy_builder=build_tv_stand; scalable=true; recolorable=true
