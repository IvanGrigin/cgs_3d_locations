#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/Plasement/procedural_object_factory_blender.py

Procedural object factory for interior_object_taxonomy/v2.
Runs inside Blender Python.

Goal:
- one reusable file that can create procedural fallback 3D objects for all taxonomy subclasses;
- every subclass listed in interior_object_taxonomy/v2 is addressable by name;
- geometry is parametric: any color, material, width/depth/height can be supplied;
- generated proxies are usable when supplier FBX/OBJ/GLB is absent;
- objects are composed from reusable parts: cushions, legs, panels, handles, doors,
  shelves, screens, appliance doors, bulbs, bowls, etc.

Usage examples:

/Applications/Blender.app/Contents/MacOS/Blender -b \
  --python src/Plasement/procedural_object_factory_blender.py -- \
  --build-all \
  --grid-cols 12 \
  --out-blend out/procedural_taxonomy_v2_all.blend \
  --report-json out/procedural_taxonomy_v2_all.report.json

/Applications/Blender.app/Contents/MacOS/Blender -b \
  --python src/Plasement/procedural_object_factory_blender.py -- \
  --subclass l_shaped_sofa \
  --width 2.8 --depth 1.8 --height 0.85 \
  --color '#8a7662' --material fabric \
  --out-blend out/l_shaped_sofa.blend

This file intentionally avoids external Python dependencies. Only Blender bpy/mathutils
and the Python standard library are used.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import bpy
    from mathutils import Vector
except Exception as exc:  # pragma: no cover - this must run in Blender
    raise RuntimeError(
        "This module must be executed inside Blender. Use: "
        "/Applications/Blender.app/Contents/MacOS/Blender -b --python ..."
    ) from exc


# =============================================================================
# Taxonomy registry
# =============================================================================

# base_type -> subclasses. The list is copied from interior_object_taxonomy/v2.
TAXONOMY: dict[str, list[str]] = {
    # casegood
    "wardrobe": [
        "hinged_wardrobe", "sliding_wardrobe", "open_wardrobe", "mirrored_wardrobe", "corner_wardrobe",
        "built_in_wardrobe", "single_wardrobe", "double_wardrobe", "triple_wardrobe", "full_wall_wardrobe",
        "floor_to_ceiling_wardrobe",
    ],
    "dresser": [
        "low_dresser", "tall_dresser", "wide_dresser", "narrow_dresser", "chest_of_drawers",
        "lingerie_chest", "bedroom_dresser", "hallway_dresser", "changing_dresser", "vanity_dresser",
    ],
    "nightstand": [
        "open_nightstand", "drawer_nightstand", "cabinet_nightstand", "floating_nightstand", "round_nightstand",
        "square_nightstand", "narrow_nightstand", "paired_nightstands",
    ],
    "cabinet": [
        "base_cabinet", "wall_cabinet", "tall_cabinet", "corner_cabinet", "glass_cabinet", "open_cabinet",
        "closed_cabinet", "drawer_cabinet", "kitchen_cabinet", "bathroom_cabinet", "living_room_cabinet",
        "hallway_cabinet", "office_cabinet", "freestanding_cabinet", "wall_mounted_cabinet", "built_in_cabinet",
    ],
    "tv_stand": [
        "low_tv_stand", "long_tv_stand", "floating_tv_stand", "corner_tv_stand", "media_console",
        "tv_wall_unit", "tv_stand_with_storage", "open_shelf_tv_stand", "closed_tv_stand",
    ],
    "bookshelf": [
        "open_bookshelf", "closed_bookshelf", "glass_door_bookshelf", "ladder_bookshelf", "cube_bookshelf",
        "modular_bookshelf", "corner_bookshelf", "wall_mounted_bookshelf", "built_in_bookshelf", "low_bookshelf",
        "narrow_bookshelf", "tall_bookshelf", "full_wall_bookshelf",
    ],

    # table
    "dining_table": [
        "round_dining_table", "oval_dining_table", "square_dining_table", "rectangular_dining_table",
        "extendable_dining_table", "two_seat_dining_table", "four_seat_dining_table", "six_seat_dining_table",
        "eight_seat_dining_table", "pedestal_dining_table", "four_leg_dining_table", "trestle_dining_table",
        "crossed_leg_dining_table",
    ],
    "coffee_table": [
        "round_coffee_table", "oval_coffee_table", "square_coffee_table", "rectangular_coffee_table",
        "irregular_coffee_table", "storage_coffee_table", "lift_top_coffee_table", "nesting_coffee_table",
        "ottoman_coffee_table", "tray_coffee_table",
    ],
    "side_table": [
        "round_side_table", "square_side_table", "pedestal_side_table", "c_table", "nesting_side_table",
        "small_side_table", "sofa_side_table", "bedside_side_table",
    ],
    "desk": [
        "writing_desk", "computer_desk", "office_desk", "corner_desk", "l_shaped_desk", "standing_desk",
        "secretary_desk", "gaming_desk", "compact_desk", "desk_with_drawers", "desk_with_shelves", "simple_desk",
    ],
    "console_table": [
        "narrow_console_table", "hallway_console_table", "sofa_console_table", "wall_console_table",
        "console_table_with_drawers", "open_console_table", "decorative_console_table",
    ],

    # seating
    "chair": [
        "dining_chair", "office_chair", "desk_chair", "accent_chair", "kitchen_chair", "bar_chair",
        "folding_chair", "lounge_chair", "low_back_chair", "high_back_chair", "open_back_chair",
        "solid_back_chair", "curved_back_chair", "ladder_back_chair", "armless_chair", "chair_with_armrests",
        "four_leg_chair", "sled_base_chair", "swivel_chair", "cantilever_chair", "wheeled_chair",
    ],
    "armchair": [
        "lounge_armchair", "club_armchair", "wingback_armchair", "recliner_armchair", "rocking_armchair",
        "swivel_armchair", "accent_armchair", "reading_armchair", "compact_armchair", "wide_armchair",
        "oversized_armchair",
    ],
    "sofa": [
        "straight_sofa", "l_shaped_sofa", "u_shaped_sofa", "curved_sofa", "modular_sofa", "sectional_sofa",
        "loveseat", "three_seat_sofa", "four_seat_sofa", "large_sofa", "sofa_bed", "chaise_sofa",
        "recliner_sofa", "corner_sofa", "armless_sofa", "one_arm_sofa", "two_arm_sofa",
    ],
    "bench": [
        "dining_bench", "entryway_bench", "bedroom_bench", "storage_bench", "window_bench", "outdoor_bench",
        "backless_bench", "bench_with_back", "bench_with_armrests", "straight_bench", "corner_bench", "curved_bench",
    ],
    "ottoman": [
        "round_ottoman", "square_ottoman", "rectangular_ottoman", "storage_ottoman", "pouf_ottoman",
        "footstool", "cocktail_ottoman", "modular_ottoman",
    ],

    # lighting
    "pendant_lamp": [
        "single_pendant_lamp", "multi_pendant_lamp", "linear_pendant_lamp", "cluster_pendant_lamp",
        "globe_pendant_lamp", "dome_pendant_lamp", "cone_pendant_lamp", "drum_pendant_lamp",
        "dining_pendant_lamp", "kitchen_island_pendant_lamp", "bedside_pendant_lamp",
    ],
    "ceiling_light": [
        "flush_mount_ceiling_light", "semi_flush_ceiling_light", "chandelier", "track_light", "recessed_light",
        "ceiling_spotlight", "led_panel", "ceiling_fan_light", "round_ceiling_light", "square_ceiling_light",
        "linear_ceiling_light",
    ],
    "floor_lamp": [
        "arc_floor_lamp", "tripod_floor_lamp", "torchiere_floor_lamp", "reading_floor_lamp",
        "multi_head_floor_lamp", "slim_floor_lamp", "globe_floor_lamp",
    ],
    "table_lamp": [
        "bedside_table_lamp", "desk_lamp", "decorative_table_lamp", "task_table_lamp", "mushroom_table_lamp",
        "globe_table_lamp", "banker_lamp",
    ],
    "wall_light": [
        "wall_sconce", "up_down_wall_light", "picture_light", "swing_arm_wall_light", "bedside_wall_light",
        "bathroom_wall_light", "wall_spotlight", "linear_wall_light",
    ],

    # soft
    "bed": [
        "single_bed", "twin_bed", "double_bed", "queen_bed", "king_bed", "bunk_bed", "loft_bed", "crib",
        "platform_bed", "storage_bed", "canopy_bed", "upholstered_bed", "metal_frame_bed", "wooden_frame_bed",
        "daybed", "sofa_bed", "bed_without_headboard", "low_headboard_bed", "high_headboard_bed",
        "upholstered_headboard_bed", "wooden_headboard_bed",
    ],
    "pillow": [
        "sleeping_pillow", "decorative_pillow", "throw_pillow", "bolster_pillow", "body_pillow", "lumbar_pillow",
        "square_pillow", "rectangular_pillow", "round_pillow", "cylindrical_pillow",
    ],
    "blanket": [
        "bed_blanket", "throw_blanket", "duvet", "quilt", "comforter", "bedspread", "plaid",
        "folded_blanket", "draped_blanket", "flat_blanket",
    ],
    "rug": [
        "rectangular_rug", "round_rug", "oval_rug", "runner_rug", "irregular_rug", "living_room_rug",
        "bedroom_rug", "dining_rug", "hallway_runner", "bathroom_rug", "small_rug", "medium_rug",
        "large_rug", "area_rug",
    ],

    # decor
    "mirror": [
        "round_mirror", "oval_mirror", "rectangular_mirror", "square_mirror", "arched_mirror", "irregular_mirror",
        "wall_mirror", "floor_mirror", "vanity_mirror", "bathroom_mirror", "decorative_mirror", "full_length_mirror",
    ],
    "wall_art": [
        "framed_picture", "canvas_art", "poster", "wall_panel", "tapestry", "wall_sculpture", "photo_frame",
        "gallery_wall_set", "portrait_wall_art", "landscape_wall_art", "square_wall_art", "vertical_triptych",
        "horizontal_triptych",
    ],
    "decor_vase": [
        "tall_vase", "short_vase", "round_vase", "cylindrical_vase", "narrow_neck_vase", "floor_vase",
        "table_vase", "decorative_bottle_vase", "empty_vase", "vase_with_flowers", "vase_with_branches",
    ],
    "decor_box": [
        "small_decor_box", "storage_box", "jewelry_box", "book_box", "lidded_box", "woven_box",
        "decorative_container",
    ],
    "decor_tray": [
        "round_tray", "rectangular_tray", "oval_tray", "serving_tray", "coffee_table_tray", "vanity_tray",
        "mirrored_tray",
    ],
    "decor_books": [
        "stacked_books", "standing_books", "open_book", "book_set", "coffee_table_books", "bookshelf_books",
        "single_book", "small_book_stack", "large_book_stack",
    ],
    "plant": [
        "floor_plant", "table_plant", "hanging_plant", "wall_plant", "potted_plant", "planter_box",
        "small_succulent", "large_leaf_plant", "artificial_plant", "small_plant", "medium_plant", "large_plant",
        "tall_plant",
    ],

    # appliance
    "refrigerator": [
        "compact_refrigerator", "under_counter_refrigerator", "single_door_refrigerator", "top_freezer_refrigerator",
        "bottom_freezer_refrigerator", "two_door_refrigerator", "side_by_side_refrigerator",
        "french_door_refrigerator", "built_in_refrigerator", "wine_cooler",
    ],
    "stove": [
        "freestanding_stove", "gas_stove", "electric_stove", "induction_stove", "compact_stove", "range_cooker",
    ],
    "cooktop": [
        "gas_cooktop", "electric_cooktop", "induction_cooktop", "two_burner_cooktop", "four_burner_cooktop",
        "five_burner_cooktop",
    ],
    "oven": [
        "built_in_oven", "freestanding_oven", "double_oven", "compact_oven", "wall_oven",
    ],
    "microwave": [
        "countertop_microwave", "built_in_microwave", "over_range_microwave", "microwave_oven_combo",
        "compact_microwave",
    ],
    "dishwasher": [
        "built_in_dishwasher", "freestanding_dishwasher", "slim_dishwasher", "countertop_dishwasher",
    ],
    "washing_machine": [
        "front_load_washing_machine", "top_load_washing_machine", "built_in_washing_machine", "washer_dryer_combo",
    ],
    "range_hood": [
        "wall_mount_range_hood", "built_in_range_hood", "island_range_hood", "telescopic_range_hood",
    ],
    "small_kitchen_appliance": [
        "kettle", "coffee_machine", "toaster", "blender", "food_processor", "air_fryer", "multicooker",
    ],

    # electronics
    "tv": ["wall_mounted_tv", "tv_on_stand", "large_tv", "compact_tv"],
    "monitor": ["single_monitor", "dual_monitor_setup", "ultrawide_monitor", "curved_monitor", "monitor_on_arm"],
    "laptop": ["closed_laptop", "open_laptop", "gaming_laptop", "ultrabook"],
    "desktop_computer": ["pc_tower", "mini_pc", "all_in_one_computer", "gaming_pc_tower"],
    "phone": ["smartphone", "desk_phone", "cordless_phone", "phone_charging_stand"],
    "tablet": ["tablet_flat", "tablet_on_stand"],
    "keyboard": ["computer_keyboard", "compact_keyboard", "gaming_keyboard"],
    "mouse": ["computer_mouse", "mouse_pad"],
    "speaker": ["bookshelf_speaker", "floorstanding_speaker", "soundbar", "smart_speaker"],

    # architectural
    "door": [
        "hinged_door", "single_hinged_door", "double_hinged_door", "sliding_door", "pocket_door",
        "folding_door", "glass_door", "balcony_door", "entrance_door",
    ],
    "window": ["single_window", "double_window", "panoramic_window", "corner_window", "skylight"],
    "partition": ["solid_partition", "glass_partition", "folding_partition", "screen_partition"],

    # storage accessories
    "storage_box": [
        "cardboard_box", "plastic_storage_box", "fabric_storage_box", "lidded_storage_box", "transparent_storage_box",
        "moving_box", "decorative_storage_box",
    ],
    "basket": ["laundry_basket", "woven_basket", "storage_basket", "toy_basket"],
    "shelf_item": ["small_storage_bin", "document_tray", "file_box"],

    # books/media
    "book": ["single_book", "open_book", "closed_book", "standing_book", "book_stack", "large_book_stack", "coffee_table_book", "cookbook"],
    "magazine": ["single_magazine", "magazine_stack", "magazine_holder"],
    "folder": ["document_folder", "ring_binder", "paper_stack"],

    # bathroom fixtures
    "toilet": ["floor_mounted_toilet", "wall_hung_toilet", "compact_toilet"],
    "sink": ["bathroom_sink", "pedestal_sink", "wall_mounted_sink", "vanity_sink", "vessel_sink", "kitchen_sink", "double_kitchen_sink"],
    "shower": ["shower_cabin", "walk_in_shower", "corner_shower"],
    "bathtub": ["rectangular_bathtub", "freestanding_bathtub", "corner_bathtub"],
}

SUBCLASS_TO_BASE: dict[str, str] = {}
for _base, _subs in TAXONOMY.items():
    for _sub in _subs:
        # Some taxonomy names intentionally repeat across different base types.
        # Keep the first explicit meaning for direct function generation and allow
        # base_type override through BuildSpec when needed.
        SUBCLASS_TO_BASE.setdefault(_sub, _base)


@dataclass(frozen=True)
class BuildSpec:
    subclass: str
    base_type: str | None = None
    width: float | None = None
    depth: float | None = None
    height: float | None = None
    color: str | tuple[float, float, float, float] | None = None
    material: str | None = None
    collection_name: str = "procedural_taxonomy_objects"
    name: str | None = None


# =============================================================================
# Utility and material helpers
# =============================================================================

_MAT_CACHE: dict[str, bpy.types.Material] = {}

COLOR_TABLE: dict[str, tuple[float, float, float, float]] = {
    "white": (0.88, 0.86, 0.82, 1.0),
    "black": (0.02, 0.02, 0.02, 1.0),
    "gray": (0.45, 0.45, 0.45, 1.0),
    "grey": (0.45, 0.45, 0.45, 1.0),
    "beige": (0.70, 0.62, 0.50, 1.0),
    "brown": (0.38, 0.23, 0.12, 1.0),
    "wood": (0.50, 0.31, 0.14, 1.0),
    "oak": (0.65, 0.46, 0.25, 1.0),
    "walnut": (0.28, 0.16, 0.08, 1.0),
    "red": (0.62, 0.08, 0.06, 1.0),
    "green": (0.12, 0.40, 0.12, 1.0),
    "blue": (0.08, 0.18, 0.50, 1.0),
    "yellow": (0.80, 0.64, 0.18, 1.0),
    "orange": (0.84, 0.38, 0.08, 1.0),
    "pink": (0.78, 0.42, 0.55, 1.0),
    "purple": (0.34, 0.12, 0.45, 1.0),
    "metal": (0.54, 0.52, 0.50, 1.0),
    "glass": (0.72, 0.90, 1.00, 0.34),
    "ceramic": (0.92, 0.90, 0.86, 1.0),
    "fabric": (0.58, 0.55, 0.52, 1.0),
}

RU_COLOR_HINTS: dict[str, str] = {
    "бел": "white", "черн": "black", "сер": "gray", "беж": "beige", "корич": "brown",
    "дерев": "wood", "дуб": "oak", "орех": "walnut", "красн": "red", "зелен": "green",
    "син": "blue", "голуб": "blue", "желт": "yellow", "оранж": "orange", "роз": "pink",
    "фиолет": "purple", "металл": "metal", "стекл": "glass", "керам": "ceramic",
}

DEFAULT_DIMS: dict[str, tuple[float, float, float]] = {
    "wardrobe": (0.9, 0.55, 2.1), "dresser": (1.05, 0.45, 0.85), "nightstand": (0.45, 0.42, 0.55),
    "cabinet": (0.8, 0.45, 0.9), "tv_stand": (1.4, 0.42, 0.48), "bookshelf": (0.85, 0.35, 1.75),
    "dining_table": (1.45, 0.85, 0.75), "coffee_table": (1.0, 0.55, 0.42), "side_table": (0.48, 0.42, 0.55),
    "desk": (1.2, 0.65, 0.75), "console_table": (1.05, 0.32, 0.78),
    "chair": (0.48, 0.52, 0.85), "armchair": (0.82, 0.82, 0.9), "sofa": (2.1, 0.9, 0.85),
    "bench": (1.15, 0.42, 0.45), "ottoman": (0.62, 0.50, 0.42),
    "pendant_lamp": (0.38, 0.38, 0.85), "ceiling_light": (0.55, 0.55, 0.22), "floor_lamp": (0.38, 0.38, 1.55),
    "table_lamp": (0.28, 0.28, 0.55), "wall_light": (0.24, 0.10, 0.35),
    "bed": (1.6, 2.05, 0.65), "pillow": (0.48, 0.18, 0.16), "blanket": (1.2, 0.8, 0.06), "rug": (1.7, 1.2, 0.03),
    "mirror": (0.72, 0.04, 1.25), "wall_art": (0.8, 0.04, 0.55), "decor_vase": (0.22, 0.22, 0.36),
    "decor_box": (0.32, 0.24, 0.18), "decor_tray": (0.36, 0.24, 0.05), "decor_books": (0.30, 0.22, 0.10),
    "plant": (0.45, 0.45, 1.0),
    "refrigerator": (0.72, 0.70, 1.85), "stove": (0.60, 0.62, 0.86), "cooktop": (0.60, 0.52, 0.06),
    "oven": (0.60, 0.58, 0.60), "microwave": (0.52, 0.40, 0.32), "dishwasher": (0.60, 0.60, 0.82),
    "washing_machine": (0.60, 0.60, 0.85), "range_hood": (0.70, 0.42, 0.55), "small_kitchen_appliance": (0.30, 0.25, 0.32),
    "tv": (1.20, 0.06, 0.70), "monitor": (0.62, 0.06, 0.42), "laptop": (0.34, 0.24, 0.18),
    "desktop_computer": (0.26, 0.46, 0.48), "phone": (0.08, 0.015, 0.16), "tablet": (0.20, 0.015, 0.28),
    "keyboard": (0.44, 0.14, 0.025), "mouse": (0.08, 0.12, 0.035), "speaker": (0.24, 0.22, 0.42),
    "door": (0.85, 0.05, 2.05), "window": (1.20, 0.05, 1.20), "partition": (1.4, 0.06, 2.0),
    "storage_box": (0.42, 0.34, 0.28), "basket": (0.40, 0.34, 0.38), "shelf_item": (0.32, 0.24, 0.16),
    "book": (0.22, 0.16, 0.035), "magazine": (0.30, 0.22, 0.015), "folder": (0.32, 0.24, 0.035),
    "toilet": (0.42, 0.68, 0.75), "sink": (0.60, 0.46, 0.32), "shower": (0.9, 0.9, 2.0), "bathtub": (1.65, 0.72, 0.58),
}


def script_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def sanitize_name(value: Any, limit: int = 96) -> str:
    text = str(value or "object").strip()
    text = re.sub(r"[^0-9A-Za-zА-Яа-яёЁ._ -]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return (text[:limit] or "object")


def parse_color(value: str | tuple[float, float, float, float] | None, fallback: str = "fabric") -> tuple[float, float, float, float]:
    if isinstance(value, tuple):
        return value
    text = str(value or "").strip().lower().replace("ё", "е")
    m = re.search(r"#?([0-9a-fA-F]{6})", text)
    if m:
        raw = m.group(1)
        return (int(raw[0:2], 16) / 255.0, int(raw[2:4], 16) / 255.0, int(raw[4:6], 16) / 255.0, 1.0)
    if text in COLOR_TABLE:
        return COLOR_TABLE[text]
    for hint, name in RU_COLOR_HINTS.items():
        if hint in text:
            return COLOR_TABLE[name]
    return COLOR_TABLE.get(fallback, COLOR_TABLE["fabric"])


def lighten(color: tuple[float, float, float, float], k: float = 0.24) -> tuple[float, float, float, float]:
    r, g, b, a = color
    return (r + (1 - r) * k, g + (1 - g) * k, b + (1 - b) * k, a)


def darken(color: tuple[float, float, float, float], k: float = 0.30) -> tuple[float, float, float, float]:
    r, g, b, a = color
    return (r * (1 - k), g * (1 - k), b * (1 - k), a)


def get_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def link_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for old in list(obj.users_collection):
        try:
            old.objects.unlink(obj)
        except Exception:
            pass
    col.objects.link(obj)
    return obj


def make_material(name: str, color: tuple[float, float, float, float], kind: str = "plastic") -> bpy.types.Material:
    key = f"{name}|{kind}|" + ",".join(f"{x:.3f}" for x in color)
    if key in _MAT_CACHE:
        return _MAT_CACHE[key]
    mat = bpy.data.materials.new(sanitize_name(name))
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        def set_input(names: tuple[str, ...], val: Any) -> None:
            for nm in names:
                if nm in bsdf.inputs:
                    try:
                        bsdf.inputs[nm].default_value = val
                    except Exception:
                        pass
                    return
        metallic = 0.0
        roughness = 0.55
        alpha = color[3]
        transmission = 0.0
        if kind == "metal":
            metallic, roughness = 0.8, 0.28
        elif kind == "glass":
            roughness, alpha, transmission = 0.03, min(alpha, 0.38), 0.65
            mat.blend_method = "BLEND"
            mat.use_screen_refraction = True if hasattr(mat, "use_screen_refraction") else False
        elif kind in {"fabric", "leather"}:
            roughness = 0.82
        elif kind in {"wood", "ceramic", "stone"}:
            roughness = 0.45
        set_input(("Base Color",), color)
        set_input(("Metallic",), metallic)
        set_input(("Roughness",), roughness)
        set_input(("Alpha",), alpha)
        set_input(("Transmission Weight", "Transmission"), transmission)
    _MAT_CACHE[key] = mat
    return mat


@dataclass
class Palette:
    base: bpy.types.Material
    light: bpy.types.Material
    dark: bpy.types.Material
    fabric: bpy.types.Material
    wood: bpy.types.Material
    metal: bpy.types.Material
    glass: bpy.types.Material
    black: bpy.types.Material
    ceramic: bpy.types.Material
    rubber: bpy.types.Material
    screen: bpy.types.Material


def make_palette(color: str | tuple[float, float, float, float] | None, material_kind: str | None = None) -> Palette:
    kind = material_kind or "fabric"
    base_color = parse_color(color, fallback=kind if kind in COLOR_TABLE else "fabric")
    return Palette(
        base=make_material("m_base", base_color, kind),
        light=make_material("m_light", lighten(base_color), kind),
        dark=make_material("m_dark", darken(base_color), kind),
        fabric=make_material("m_fabric", base_color, "fabric"),
        wood=make_material("m_wood", base_color if kind == "wood" else COLOR_TABLE["wood"], "wood"),
        metal=make_material("m_metal", base_color if kind == "metal" else COLOR_TABLE["metal"], "metal"),
        glass=make_material("m_glass", COLOR_TABLE["glass"], "glass"),
        black=make_material("m_black", COLOR_TABLE["black"], "plastic"),
        ceramic=make_material("m_ceramic", COLOR_TABLE["ceramic"], "ceramic"),
        rubber=make_material("m_rubber", (0.015, 0.015, 0.014, 1.0), "plastic"),
        screen=make_material("m_screen", (0.005, 0.006, 0.008, 1.0), "glass"),
    )


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def apply_material(obj: bpy.types.Object, material: bpy.types.Material) -> bpy.types.Object:
    if hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(material)
    return obj


def shade_smooth(obj: bpy.types.Object) -> bpy.types.Object:
    try:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        obj.select_set(False)
    except Exception:
        pass
    return obj


def add_bevel(obj: bpy.types.Object, amount: float = 0.02, segments: int = 2) -> bpy.types.Object:
    if amount <= 0:
        return obj
    try:
        mod = obj.modifiers.new("proxy_bevel", "BEVEL")
        mod.width = amount
        mod.segments = segments
        mod.profile = 0.5
        obj.modifiers.new("proxy_weighted_normals", "WEIGHTED_NORMAL")
    except Exception:
        pass
    return obj


def cube_obj(name: str, loc: tuple[float, float, float], size: tuple[float, float, float], material: bpy.types.Material, col: bpy.types.Collection, bevel: float = 0.02, segments: int = 2) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    obj.dimensions = (max(size[0], 0.001), max(size[1], 0.001), max(size[2], 0.001))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    apply_material(obj, material)
    add_bevel(obj, bevel, segments)
    return link_to_collection(obj, col)


def cylinder_obj(name: str, loc: tuple[float, float, float], radius: float, depth: float, material: bpy.types.Material, col: bpy.types.Collection, vertices: int = 32, bevel: float = 0.0) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=max(radius, 0.001), depth=max(depth, 0.001), location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    apply_material(obj, material)
    shade_smooth(obj)
    add_bevel(obj, bevel, 2)
    return link_to_collection(obj, col)


def sphere_obj(name: str, loc: tuple[float, float, float], scale: tuple[float, float, float], material: bpy.types.Material, col: bpy.types.Collection, segments: int = 32) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=max(8, segments // 2), radius=0.5, location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    obj.scale = scale
    apply_material(obj, material)
    shade_smooth(obj)
    return link_to_collection(obj, col)


def cone_obj(name: str, loc: tuple[float, float, float], r1: float, r2: float, depth: float, material: bpy.types.Material, col: bpy.types.Collection, vertices: int = 32) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=max(r1, 0.001), radius2=max(r2, 0.001), depth=max(depth, 0.001), location=loc)
    obj = bpy.context.object
    obj.name = sanitize_name(name)
    apply_material(obj, material)
    shade_smooth(obj)
    add_bevel(obj, min(max(r1, r2) * 0.04, 0.02), 2)
    return link_to_collection(obj, col)


def rotate_z(obj: bpy.types.Object, degrees: float) -> bpy.types.Object:
    obj.rotation_euler[2] = math.radians(degrees)
    return obj


def rotate_x(obj: bpy.types.Object, degrees: float) -> bpy.types.Object:
    obj.rotation_euler[0] = math.radians(degrees)
    return obj


def tag_object(obj: bpy.types.Object, spec: BuildSpec, base_type: str, part: str) -> None:
    obj["taxonomy_schema"] = "interior_object_taxonomy/v2"
    obj["taxonomy_subclass"] = spec.subclass
    obj["taxonomy_base_type"] = base_type
    obj["procedural_part"] = part


def tag_created(before: set[bpy.types.Object], spec: BuildSpec, base_type: str) -> list[bpy.types.Object]:
    created = [obj for obj in bpy.data.objects if obj not in before]
    for obj in created:
        tag_object(obj, spec, base_type, obj.name)
    return created


# =============================================================================
# Reusable geometry parts
# =============================================================================

class Parts:
    def __init__(self, col: bpy.types.Collection, palette: Palette):
        self.col = col
        self.p = palette

    def panel(self, name: str, loc, size, mat: bpy.types.Material | None = None, bevel: float = 0.015) -> bpy.types.Object:
        return cube_obj(name, loc, size, mat or self.p.base, self.col, bevel=bevel, segments=2)

    def soft_box(self, name: str, loc, size, mat: bpy.types.Material | None = None, bevel: float = 0.08, segments: int = 8) -> bpy.types.Object:
        return cube_obj(name, loc, size, mat or self.p.fabric, self.col, bevel=bevel, segments=segments)

    def cushion(self, name: str, loc, size, mat: bpy.types.Material | None = None) -> bpy.types.Object:
        return sphere_obj(name, loc, (size[0] * 0.5, size[1] * 0.5, size[2] * 0.5), mat or self.p.fabric, self.col, segments=48)

    def leg(self, name: str, loc, height: float, radius: float = 0.02, mat: bpy.types.Material | None = None) -> bpy.types.Object:
        return cylinder_obj(name, loc, radius, height, mat or self.p.dark, self.col, vertices=18)

    def four_legs(self, prefix: str, loc, w: float, d: float, leg_h: float, radius: float = 0.02, mat: bpy.types.Material | None = None) -> None:
        for sx in (-1, 1):
            for sy in (-1, 1):
                self.leg(f"{prefix}_leg", (loc[0] + sx * w * 0.42, loc[1] + sy * d * 0.36, leg_h / 2), leg_h, radius, mat)

    def drawer_faces(self, prefix: str, loc, w: float, d: float, h: float, drawers: int = 3) -> None:
        front_y = loc[1] - d * 0.51
        usable_h = h * 0.72
        start_z = h * 0.18
        for i in range(drawers):
            z = start_z + usable_h * (i + 0.5) / max(drawers, 1)
            face_h = usable_h / max(drawers, 1) * 0.70
            self.panel(f"{prefix}_drawer_{i+1}", (loc[0], front_y, z), (w * 0.86, 0.025, face_h), self.p.light, bevel=0.01)
            self.panel(f"{prefix}_handle_{i+1}", (loc[0], front_y - 0.018, z), (w * 0.25, 0.018, 0.018), self.p.metal, bevel=0.004)

    def door_pair(self, prefix: str, loc, w: float, d: float, h: float, mat: bpy.types.Material | None = None) -> None:
        front_y = loc[1] - d * 0.51
        self.panel(f"{prefix}_door_l", (loc[0] - w * 0.25, front_y, h * 0.52), (w * 0.47, 0.025, h * 0.86), mat or self.p.light, bevel=0.012)
        self.panel(f"{prefix}_door_r", (loc[0] + w * 0.25, front_y, h * 0.52), (w * 0.47, 0.025, h * 0.86), mat or self.p.light, bevel=0.012)
        self.panel(f"{prefix}_handle_l", (loc[0] - w * 0.08, front_y - 0.018, h * 0.55), (0.018, 0.018, h * 0.24), self.p.metal, bevel=0.003)
        self.panel(f"{prefix}_handle_r", (loc[0] + w * 0.08, front_y - 0.018, h * 0.55), (0.018, 0.018, h * 0.24), self.p.metal, bevel=0.003)

    def burners(self, prefix: str, loc, w: float, d: float, z: float, count: int = 4) -> None:
        positions = [(-0.24, -0.18), (0.24, -0.18), (-0.24, 0.18), (0.24, 0.18)]
        if count == 2:
            positions = [(-0.22, 0.0), (0.22, 0.0)]
        if count == 5:
            positions.append((0.0, 0.0))
        for i, (sx, sy) in enumerate(positions[:count], 1):
            cylinder_obj(f"{prefix}_burner_{i}", (loc[0] + sx * w, loc[1] + sy * d, z), min(w, d) * 0.085, 0.014, self.p.black, self.col, vertices=32)

    def screen(self, prefix: str, loc, w: float, d: float, h: float) -> None:
        self.panel(f"{prefix}_screen", (loc[0], loc[1], h * 0.5), (w, max(d, 0.025), h), self.p.screen, bevel=0.018)
        self.panel(f"{prefix}_frame", (loc[0], loc[1] - max(d, 0.025) * 0.60, h * 0.5), (w * 1.03, 0.012, h * 1.03), self.p.black, bevel=0.004)


# =============================================================================
# Main factory
# =============================================================================

class ProceduralObjectFactory:
    def __init__(self, collection_name: str = "procedural_taxonomy_objects"):
        self.collection = get_collection(collection_name)

    def build(self, spec: BuildSpec, loc: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[bpy.types.Object]:
        base_type = spec.base_type or SUBCLASS_TO_BASE.get(spec.subclass)
        if base_type is None:
            base_type = self.infer_base_type(spec.subclass)
        dims = self.dimensions(base_type, spec)
        mat_kind = spec.material or self.default_material(base_type, spec.subclass)
        palette = make_palette(spec.color, mat_kind)
        parts = Parts(self.collection, palette)
        before = set(bpy.data.objects)
        method = getattr(self, f"build_base_{base_type}", self.build_base_generic)
        method(parts, spec.subclass, loc, dims)
        return tag_created(before, spec, base_type)

    def dimensions(self, base_type: str, spec: BuildSpec) -> tuple[float, float, float]:
        w, d, h = DEFAULT_DIMS.get(base_type, (0.6, 0.6, 0.6))
        w = float(spec.width) if spec.width is not None else w
        d = float(spec.depth) if spec.depth is not None else d
        h = float(spec.height) if spec.height is not None else h
        # Basic subclass size tuning.
        s = spec.subclass
        if any(k in s for k in ("compact", "small", "single", "two_seat", "narrow")):
            w *= 0.75
        if any(k in s for k in ("large", "king", "eight_seat", "full_wall", "side_by_side", "french_door", "oversized")):
            w *= 1.35
        if any(k in s for k in ("wide", "long", "six_seat", "four_seat")):
            w *= 1.25
        if any(k in s for k in ("tall", "floor_to_ceiling", "tower", "high_back", "double_oven")):
            h *= 1.25
        if any(k in s for k in ("low", "flush")):
            h *= 0.70
        return max(w, 0.02), max(d, 0.02), max(h, 0.02)

    def default_material(self, base_type: str, subclass: str) -> str:
        if base_type in {"sofa", "armchair", "chair", "bench", "ottoman", "bed", "pillow", "blanket", "rug"}:
            return "fabric"
        if base_type in {"wardrobe", "dresser", "nightstand", "cabinet", "tv_stand", "bookshelf", "dining_table", "coffee_table", "side_table", "desk", "console_table"}:
            return "wood"
        if base_type in {"refrigerator", "stove", "cooktop", "oven", "microwave", "dishwasher", "washing_machine", "range_hood", "pendant_lamp", "ceiling_light", "floor_lamp", "table_lamp", "wall_light"}:
            return "metal"
        if base_type in {"sink", "toilet", "bathtub"}:
            return "ceramic"
        if base_type in {"mirror", "window", "shower"}:
            return "glass"
        return "plastic"

    def infer_base_type(self, subclass: str) -> str:
        text = subclass.lower()
        for base, subs in TAXONOMY.items():
            if base in text or subclass in subs:
                return base
        return "generic"

    # ------------------------------------------------------------------
    # Casegoods
    # ------------------------------------------------------------------

    def build_base_wardrobe(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "corner" in subclass:
            p.panel("corner_wardrobe_left", (loc[0] - w * 0.16, loc[1], h / 2), (w * 0.68, depth * 0.55, h), p.p.wood)
            p.panel("corner_wardrobe_right", (loc[0] + w * 0.18, loc[1] - depth * 0.18, h / 2), (w * 0.55, depth * 0.68, h), p.p.wood)
            return
        p.panel("wardrobe_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.wood, bevel=0.025)
        if "open" in subclass:
            for i in range(1, 4):
                p.panel(f"wardrobe_shelf_{i}", (loc[0], loc[1] - depth * 0.18, h * i / 4), (w * 0.9, depth * 0.68, 0.025), p.p.dark, bevel=0.004)
            p.panel("wardrobe_hanging_rail", (loc[0], loc[1] - depth * 0.20, h * 0.72), (w * 0.70, 0.035, 0.035), p.p.metal, bevel=0.004)
        elif "sliding" in subclass:
            front = loc[1] - depth * 0.51
            p.panel("wardrobe_slide_back", (loc[0] - w * 0.18, front + 0.03, h * 0.52), (w * 0.55, 0.018, h * 0.86), p.p.light)
            p.panel("wardrobe_slide_front", (loc[0] + w * 0.18, front, h * 0.52), (w * 0.55, 0.018, h * 0.86), p.p.base)
        else:
            door_mat = p.p.glass if "mirror" in subclass else p.p.light
            p.door_pair("wardrobe", loc, w, depth, h, door_mat)

    def build_base_dresser(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "vanity" in subclass:
            self.build_base_desk(p, "desk_with_drawers", loc, (w, depth, max(h * 0.88, 0.72)))
            self.build_base_mirror(p, "vanity_mirror", (loc[0], loc[1] + depth * 0.40, 0), (w * 0.35, 0.04, 0.55))
            return
        p.panel("dresser_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.wood, bevel=0.025)
        drawers = 5 if "tall" in subclass or "lingerie" in subclass else 2 if "low" in subclass else 4 if "wide" in subclass else 3
        p.drawer_faces("dresser", loc, w, depth, h, drawers)
        if "changing" in subclass:
            p.soft_box("changing_pad", (loc[0], loc[1], h + 0.04), (w * 0.72, depth * 0.70, 0.08), p.p.fabric, bevel=0.05)

    def build_base_nightstand(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "round" in subclass:
            cylinder_obj("round_nightstand_body", (loc[0], loc[1], h * 0.48), min(w, depth) * 0.48, h * 0.96, p.p.wood, p.col, vertices=40, bevel=0.01)
            cylinder_obj("round_nightstand_top", (loc[0], loc[1], h + 0.02), min(w, depth) * 0.50, 0.04, p.p.light, p.col, vertices=40)
            return
        if "paired" in subclass:
            self.build_base_nightstand(p, "drawer_nightstand", (loc[0] - w * 0.58, loc[1], loc[2]), d)
            self.build_base_nightstand(p, "drawer_nightstand", (loc[0] + w * 0.58, loc[1], loc[2]), d)
            return
        p.panel("nightstand_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.wood, bevel=0.025)
        if "open" in subclass:
            p.panel("nightstand_shelf", (loc[0], loc[1] - depth * 0.05, h * 0.55), (w * 0.86, depth * 0.82, 0.025), p.p.dark)
        elif "cabinet" in subclass:
            p.door_pair("nightstand", loc, w, depth, h, p.p.light)
        else:
            p.drawer_faces("nightstand", loc, w, depth, h, 2)

    def build_base_cabinet(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "corner" in subclass:
            p.panel("corner_cabinet_a", (loc[0] - w * 0.18, loc[1], h / 2), (w * 0.64, depth * 0.52, h), p.p.wood)
            p.panel("corner_cabinet_b", (loc[0] + w * 0.16, loc[1] - depth * 0.16, h / 2), (w * 0.52, depth * 0.64, h), p.p.wood)
            return
        if "wall" in subclass or "built_in" in subclass:
            h = max(h * 0.70, 0.45)
            loc = (loc[0], loc[1], loc[2] + 0.8)
        p.panel("cabinet_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.wood)
        if "open" in subclass:
            for i in range(1, 3):
                p.panel(f"cabinet_shelf_{i}", (loc[0], loc[1], h * i / 3), (w * 0.9, depth * 0.86, 0.025), p.p.dark)
        elif "glass" in subclass:
            p.door_pair("cabinet", loc, w, depth, h, p.p.glass)
        elif "drawer" in subclass or "base" in subclass or "kitchen" in subclass:
            p.drawer_faces("cabinet", loc, w, depth, h, 3)
        else:
            p.door_pair("cabinet", loc, w, depth, h, p.p.light)

    def build_base_tv_stand(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "wall_unit" in subclass:
            p.panel("tv_wall_unit_base", (loc[0], loc[1], h * 0.30), (w, depth, h * 0.6), p.p.wood)
            p.panel("tv_wall_unit_panel", (loc[0], loc[1] + depth * 0.45, h * 1.35), (w, 0.06, h * 1.7), p.p.dark)
            p.screen("tv_wall_unit_screen", (loc[0], loc[1] + depth * 0.40, h * 1.45), w * 0.50, 0.04, h * 0.78)
            return
        if "corner" in subclass:
            p.panel("corner_tv_stand_a", (loc[0] - w * 0.16, loc[1], h / 2), (w * 0.68, depth * 0.52, h), p.p.wood)
            p.panel("corner_tv_stand_b", (loc[0] + w * 0.16, loc[1] - depth * 0.16, h / 2), (w * 0.52, depth * 0.68, h), p.p.wood)
            return
        p.panel("tv_stand_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.wood)
        if "open" in subclass:
            for x in (-0.25, 0.25):
                p.panel("tv_stand_shelf", (loc[0] + x * w, loc[1], h * 0.56), (w * 0.40, depth * 0.86, 0.025), p.p.dark)
        else:
            p.drawer_faces("tv_stand", loc, w, depth, h, 2)

    def build_base_bookshelf(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "ladder" in subclass:
            p.panel("ladder_left", (loc[0] - w * 0.38, loc[1], h / 2), (0.04, depth * 0.20, h), p.p.wood)
            p.panel("ladder_right", (loc[0] + w * 0.38, loc[1], h / 2), (0.04, depth * 0.20, h), p.p.wood)
            for i in range(4):
                p.panel(f"ladder_shelf_{i+1}", (loc[0], loc[1], h * (0.18 + i * 0.20)), (w * (0.95 - i * 0.13), depth * (0.80 - i * 0.08), 0.03), p.p.dark)
            return
        p.panel("bookshelf_frame", (loc[0], loc[1], h / 2), (w, depth, h), p.p.wood)
        if "cube" in subclass or "modular" in subclass:
            p.panel("bookshelf_v", (loc[0], loc[1], h / 2), (0.03, depth * 0.94, h * 0.94), p.p.dark)
            p.panel("bookshelf_h", (loc[0], loc[1], h / 2), (w * 0.94, depth * 0.94, 0.03), p.p.dark)
        else:
            for i in range(1, 5):
                p.panel(f"bookshelf_shelf_{i}", (loc[0], loc[1], h * i / 5), (w * 0.94, depth * 0.92, 0.03), p.p.dark)
        if "glass" in subclass or "closed" in subclass:
            p.door_pair("bookshelf", loc, w, depth, h, p.p.glass if "glass" in subclass else p.p.light)
        for row in range(4):
            for col in range(4):
                p.panel("bookshelf_book", (loc[0] - w * 0.34 + col * w * 0.22, loc[1] - depth * 0.33, h * (row + 0.55) / 5), (w * 0.06, depth * 0.14, h * 0.10), p.p.base if col % 2 else p.p.light, bevel=0.003)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def table_rect(self, p: Parts, prefix: str, loc, d, with_drawers: bool = False, trestle: bool = False, crossed: bool = False) -> None:
        w, depth, h = d
        top_h = max(0.04, h * 0.10)
        leg_h = h - top_h
        p.panel(f"{prefix}_top", (loc[0], loc[1], leg_h + top_h / 2), (w, depth, top_h), p.p.wood, bevel=0.025)
        if trestle:
            p.panel(f"{prefix}_trestle_l", (loc[0] - w * 0.30, loc[1], leg_h * 0.45), (0.06, depth * 0.75, leg_h * 0.90), p.p.dark)
            p.panel(f"{prefix}_trestle_r", (loc[0] + w * 0.30, loc[1], leg_h * 0.45), (0.06, depth * 0.75, leg_h * 0.90), p.p.dark)
        elif crossed:
            a = p.panel(f"{prefix}_cross_a", (loc[0], loc[1], leg_h * 0.44), (w * 0.78, 0.04, leg_h * 0.90), p.p.dark)
            rotate_z(a, 24)
            b = p.panel(f"{prefix}_cross_b", (loc[0], loc[1], leg_h * 0.44), (w * 0.78, 0.04, leg_h * 0.90), p.p.dark)
            rotate_z(b, -24)
        else:
            p.four_legs(prefix, loc, w, depth, leg_h, 0.022)
        if with_drawers:
            p.drawer_faces(f"{prefix}_drawers", (loc[0], loc[1], leg_h * 0.55), w * 0.38, depth * 0.25, leg_h * 0.32, 2)

    def table_round(self, p: Parts, prefix: str, loc, d) -> None:
        w, depth, h = d
        r = min(w, depth) * 0.45
        top_h = max(0.04, h * 0.09)
        cylinder_obj(f"{prefix}_round_top", (loc[0], loc[1], h - top_h / 2), r, top_h, p.p.wood, p.col, vertices=56)
        cylinder_obj(f"{prefix}_pedestal", (loc[0], loc[1], h * 0.45), r * 0.12, h * 0.78, p.p.metal, p.col, vertices=32)
        cylinder_obj(f"{prefix}_base", (loc[0], loc[1], 0.035), r * 0.32, 0.07, p.p.metal, p.col, vertices=48)

    def build_base_dining_table(self, p: Parts, subclass: str, loc, d) -> None:
        if "round" in subclass or "pedestal" in subclass:
            self.table_round(p, "dining_table", loc, d)
        elif "oval" in subclass:
            p.soft_box("dining_oval_top", (loc[0], loc[1], d[2] * 0.93), (d[0], d[1], max(0.04, d[2] * 0.09)), p.p.wood, bevel=0.12, segments=10)
            cylinder_obj("dining_oval_ped_l", (loc[0] - d[0] * 0.18, loc[1], d[2] * 0.44), min(d[0], d[1]) * 0.08, d[2] * 0.78, p.p.metal, p.col)
            cylinder_obj("dining_oval_ped_r", (loc[0] + d[0] * 0.18, loc[1], d[2] * 0.44), min(d[0], d[1]) * 0.08, d[2] * 0.78, p.p.metal, p.col)
        elif "trestle" in subclass:
            self.table_rect(p, "dining_trestle", loc, d, trestle=True)
        elif "crossed" in subclass:
            self.table_rect(p, "dining_crossed", loc, d, crossed=True)
        else:
            self.table_rect(p, "dining_table", loc, d)

    def build_base_coffee_table(self, p: Parts, subclass: str, loc, d) -> None:
        if "round" in subclass or "tray" in subclass:
            self.table_round(p, "coffee_table", loc, d)
        elif "oval" in subclass or "irregular" in subclass:
            p.soft_box("coffee_oval_top", (loc[0], loc[1], d[2] * 0.86), (d[0], d[1], max(0.03, d[2] * 0.14)), p.p.wood, bevel=0.10, segments=8)
            p.four_legs("coffee_oval", loc, d[0], d[1], d[2] * 0.72, 0.018)
        elif "nesting" in subclass:
            self.table_rect(p, "coffee_nest_big", (loc[0] - d[0] * 0.12, loc[1], loc[2]), (d[0] * 0.65, d[1] * 0.65, d[2]))
            self.table_rect(p, "coffee_nest_small", (loc[0] + d[0] * 0.18, loc[1] - d[1] * 0.12, loc[2]), (d[0] * 0.46, d[1] * 0.46, d[2] * 0.78))
        elif "storage" in subclass or "lift_top" in subclass:
            p.panel("coffee_storage_body", (loc[0], loc[1], d[2] * 0.40), (d[0], d[1], d[2] * 0.80), p.p.wood)
            p.drawer_faces("coffee_storage", loc, d[0], d[1], d[2], 1)
            p.panel("coffee_storage_top", (loc[0], loc[1], d[2] * 0.92), (d[0] * 1.02, d[1] * 1.02, 0.04), p.p.light)
        elif "ottoman" in subclass:
            self.build_base_ottoman(p, "cocktail_ottoman", loc, d)
        else:
            self.table_rect(p, "coffee_table", loc, d)

    def build_base_side_table(self, p: Parts, subclass: str, loc, d) -> None:
        if "round" in subclass or "pedestal" in subclass:
            self.table_round(p, "side_table", loc, d)
        elif "c_table" in subclass:
            p.panel("c_table_top", (loc[0], loc[1], d[2]), (d[0], d[1], 0.035), p.p.wood)
            p.panel("c_table_side", (loc[0] - d[0] * 0.42, loc[1], d[2] * 0.5), (0.04, 0.04, d[2]), p.p.metal)
            p.panel("c_table_base", (loc[0], loc[1], 0.03), (d[0], d[1], 0.035), p.p.metal)
        elif "nesting" in subclass:
            self.build_base_coffee_table(p, "nesting_coffee_table", loc, d)
        else:
            self.table_rect(p, "side_table", loc, d)

    def build_base_desk(self, p: Parts, subclass: str, loc, d) -> None:
        if "corner" in subclass or "l_shaped" in subclass:
            p.panel("desk_l_top_a", (loc[0] - d[0] * 0.12, loc[1], d[2]), (d[0] * 0.78, d[1] * 0.55, 0.045), p.p.wood)
            p.panel("desk_l_top_b", (loc[0] + d[0] * 0.22, loc[1] - d[1] * 0.22, d[2]), (d[0] * 0.45, d[1] * 0.90, 0.045), p.p.wood)
            p.four_legs("desk_l", loc, d[0] * 0.8, d[1], d[2], 0.02)
        else:
            self.table_rect(p, "desk", loc, d, with_drawers=("drawer" in subclass or "secretary" in subclass or "gaming" in subclass))
            if "shelves" in subclass or "secretary" in subclass:
                p.panel("desk_upper_shelf", (loc[0], loc[1] + d[1] * 0.38, d[2] + 0.35), (d[0] * 0.80, 0.16, 0.04), p.p.wood)
                p.panel("desk_upper_back", (loc[0], loc[1] + d[1] * 0.45, d[2] + 0.20), (d[0] * 0.85, 0.04, 0.35), p.p.dark)

    def build_base_console_table(self, p: Parts, subclass: str, loc, d) -> None:
        self.table_rect(p, "console_table", loc, d, with_drawers=("drawer" in subclass))

    # ------------------------------------------------------------------
    # Seating
    # ------------------------------------------------------------------

    def build_base_chair(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "bar" in subclass:
            h *= 1.25
        seat_h = max(h * 0.45, 0.42)
        p.soft_box("chair_seat", (loc[0], loc[1], seat_h), (w * 0.84, depth * 0.72, h * 0.10), p.p.fabric if "lounge" in subclass or "accent" in subclass else p.p.base, bevel=0.045)
        if "open_back" in subclass or "ladder_back" in subclass:
            for i in range(3):
                p.panel(f"chair_back_slat_{i}", (loc[0], loc[1] + depth * 0.34, h * (0.58 + i * 0.10)), (w * 0.80, 0.035, 0.035), p.p.base)
        elif "curved" in subclass:
            back = p.soft_box("chair_curved_back", (loc[0], loc[1] + depth * 0.32, h * 0.72), (w * 0.88, depth * 0.08, h * 0.46), p.p.fabric, bevel=0.08)
            rotate_z(back, 0)
        else:
            p.panel("chair_back", (loc[0], loc[1] + depth * 0.34, h * 0.72), (w * 0.86, depth * 0.07, h * (0.55 if "high_back" in subclass else 0.42)), p.p.base, bevel=0.035)
        if "armrests" in subclass:
            p.panel("chair_arm_l", (loc[0] - w * 0.43, loc[1], h * 0.48), (w * 0.08, depth * 0.62, h * 0.12), p.p.base)
            p.panel("chair_arm_r", (loc[0] + w * 0.43, loc[1], h * 0.48), (w * 0.08, depth * 0.62, h * 0.12), p.p.base)
        if "sled" in subclass or "cantilever" in subclass:
            p.panel("chair_sled_l", (loc[0] - w * 0.30, loc[1], seat_h * 0.45), (0.035, depth * 0.85, seat_h * 0.90), p.p.metal)
            p.panel("chair_sled_r", (loc[0] + w * 0.30, loc[1], seat_h * 0.45), (0.035, depth * 0.85, seat_h * 0.90), p.p.metal)
        elif "swivel" in subclass or "wheeled" in subclass or "office" in subclass:
            cylinder_obj("chair_pedestal", (loc[0], loc[1], seat_h * 0.45), 0.035, seat_h * 0.80, p.p.metal, p.col)
            cylinder_obj("chair_star_base", (loc[0], loc[1], 0.04), w * 0.28, 0.035, p.p.metal, p.col, vertices=5)
            if "wheeled" in subclass or "office" in subclass:
                for i in range(5):
                    a = 2 * math.pi * i / 5
                    cylinder_obj("chair_wheel", (loc[0] + math.cos(a) * w * 0.32, loc[1] + math.sin(a) * w * 0.32, 0.04), 0.025, 0.025, p.p.rubber, p.col, vertices=16)
        else:
            p.four_legs("chair", loc, w, depth, seat_h, 0.018)

    def build_base_armchair(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        seat_h = max(h * 0.34, 0.32)
        p.soft_box("armchair_seat", (loc[0], loc[1], seat_h * 0.58), (w * 0.88, depth * 0.78, seat_h * 1.16), p.p.fabric)
        p.soft_box("armchair_back", (loc[0], loc[1] + depth * 0.30, seat_h + h * 0.20), (w * 0.92, depth * 0.16, h * (0.50 if "wingback" in subclass else 0.42)), p.p.fabric)
        p.soft_box("armchair_arm_l", (loc[0] - w * 0.40, loc[1], h * 0.40), (w * 0.16, depth * 0.78, h * 0.54), p.p.fabric)
        p.soft_box("armchair_arm_r", (loc[0] + w * 0.40, loc[1], h * 0.40), (w * 0.16, depth * 0.78, h * 0.54), p.p.fabric)
        if "wingback" in subclass:
            p.soft_box("wing_l", (loc[0] - w * 0.44, loc[1] + depth * 0.24, h * 0.70), (w * 0.10, depth * 0.10, h * 0.50), p.p.fabric)
            p.soft_box("wing_r", (loc[0] + w * 0.44, loc[1] + depth * 0.24, h * 0.70), (w * 0.10, depth * 0.10, h * 0.50), p.p.fabric)
        if "rocking" in subclass:
            p.panel("rocker_l", (loc[0] - w * 0.27, loc[1], 0.05), (0.035, depth * 0.95, 0.035), p.p.wood)
            p.panel("rocker_r", (loc[0] + w * 0.27, loc[1], 0.05), (0.035, depth * 0.95, 0.035), p.p.wood)
        elif "swivel" in subclass:
            cylinder_obj("armchair_pedestal", (loc[0], loc[1], 0.16), 0.035, 0.32, p.p.metal, p.col)
            cylinder_obj("armchair_round_base", (loc[0], loc[1], 0.035), w * 0.28, 0.07, p.p.metal, p.col)
        else:
            p.four_legs("armchair", loc, w * 0.72, depth * 0.60, 0.20, 0.018)

    def build_base_sofa(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "l_shaped" in subclass or "corner" in subclass or "chaise" in subclass:
            self.sofa_l(p, loc, d)
        elif "u_shaped" in subclass:
            self.sofa_u(p, loc, d)
        elif "modular" in subclass or "sectional" in subclass:
            self.sofa_modular(p, loc, d)
        elif "curved" in subclass:
            self.sofa_curved(p, loc, d)
        else:
            seats = 2 if "loveseat" in subclass else 4 if "four" in subclass or "large" in subclass else 3
            self.sofa_straight(p, loc, d, seats=seats, arms=("armless" not in subclass), one_arm=("one_arm" in subclass), sleeper=("sofa_bed" in subclass))

    def sofa_straight(self, p: Parts, loc, d, seats: int = 3, arms: bool = True, one_arm: bool = False, sleeper: bool = False) -> None:
        w, depth, h = d
        seat_h = h * 0.34
        p.soft_box("sofa_base", (loc[0], loc[1], seat_h * 0.55), (w, depth * 0.84, seat_h * 1.1), p.p.fabric)
        p.soft_box("sofa_back", (loc[0], loc[1] + depth * 0.32, seat_h + h * 0.23), (w, depth * 0.16, h * 0.46), p.p.fabric)
        if arms:
            p.soft_box("sofa_arm_l", (loc[0] - w * 0.46, loc[1], h * 0.40), (w * 0.09, depth * 0.84, h * 0.58), p.p.fabric)
            if not one_arm:
                p.soft_box("sofa_arm_r", (loc[0] + w * 0.46, loc[1], h * 0.40), (w * 0.09, depth * 0.84, h * 0.58), p.p.fabric)
        for i in range(seats):
            x = loc[0] + (i - (seats - 1) / 2) * w * 0.24
            p.soft_box(f"sofa_seat_cushion_{i+1}", (x, loc[1], seat_h + 0.055), (w * 0.20, depth * 0.52, 0.10), p.p.light)
            p.cushion(f"sofa_back_pillow_{i+1}", (x, loc[1] + depth * 0.13, seat_h + h * 0.35), (w * 0.13, depth * 0.12, h * 0.15), p.p.base)
        if sleeper:
            p.panel("sofa_bed_pullout_line", (loc[0], loc[1] - depth * 0.43, seat_h * 0.85), (w * 0.76, 0.018, 0.018), p.p.dark)

    def sofa_l(self, p: Parts, loc, d) -> None:
        w, depth, h = max(d[0], 2.2), max(d[1], 1.45), d[2]
        seat_h = h * 0.34
        p.soft_box("sofa_l_main", (loc[0] - w * 0.12, loc[1], seat_h * 0.55), (w * 0.62, depth * 0.52, seat_h * 1.1), p.p.fabric)
        p.soft_box("sofa_l_chaise", (loc[0] + w * 0.22, loc[1] - depth * 0.18, seat_h * 0.55), (w * 0.42, depth * 0.68, seat_h * 1.1), p.p.fabric)
        p.soft_box("sofa_l_back", (loc[0] - w * 0.12, loc[1] + depth * 0.22, seat_h + h * 0.22), (w * 0.62, depth * 0.14, h * 0.44), p.p.fabric)
        p.soft_box("sofa_l_side_back", (loc[0] + w * 0.42, loc[1] - depth * 0.12, seat_h + h * 0.22), (depth * 0.14, depth * 0.62, h * 0.44), p.p.fabric)
        p.cushion("sofa_l_pillow_1", (loc[0] - w * 0.20, loc[1] + depth * 0.10, seat_h + h * 0.33), (w * 0.16, depth * 0.12, h * 0.14), p.p.light)
        p.cushion("sofa_l_pillow_2", (loc[0] + w * 0.12, loc[1] - depth * 0.26, seat_h + h * 0.20), (w * 0.14, depth * 0.12, h * 0.13), p.p.light)

    def sofa_u(self, p: Parts, loc, d) -> None:
        w, depth, h = max(d[0], 2.8), max(d[1], 1.7), d[2]
        seat_h = h * 0.34
        p.soft_box("sofa_u_center", (loc[0], loc[1], seat_h * 0.55), (w * 0.54, depth * 0.42, seat_h * 1.1), p.p.fabric)
        p.soft_box("sofa_u_left", (loc[0] - w * 0.28, loc[1] - depth * 0.10, seat_h * 0.55), (w * 0.28, depth * 0.68, seat_h * 1.1), p.p.fabric)
        p.soft_box("sofa_u_right", (loc[0] + w * 0.28, loc[1] - depth * 0.10, seat_h * 0.55), (w * 0.28, depth * 0.68, seat_h * 1.1), p.p.fabric)
        p.soft_box("sofa_u_back", (loc[0], loc[1] + depth * 0.18, seat_h + h * 0.22), (w * 0.54, depth * 0.13, h * 0.44), p.p.fabric)

    def sofa_modular(self, p: Parts, loc, d) -> None:
        w, depth, h = d
        module_w = max(w * 0.28, 0.55)
        module_d = max(depth * 0.45, 0.55)
        for i, (ox, oy) in enumerate([(-module_w, 0), (0, 0), (module_w, 0), (module_w, -module_d * 0.95)], 1):
            p.soft_box(f"sofa_module_{i}", (loc[0] + ox, loc[1] + oy, h * 0.18), (module_w, module_d, h * 0.36), p.p.fabric)
            p.soft_box(f"sofa_module_back_{i}", (loc[0] + ox, loc[1] + oy + module_d * 0.34, h * 0.48), (module_w, module_d * 0.12, h * 0.42), p.p.fabric)

    def sofa_curved(self, p: Parts, loc, d) -> None:
        w, depth, h = d
        for i in range(5):
            angle = -40 + i * 20
            x = loc[0] + math.sin(math.radians(angle)) * w * 0.35
            y = loc[1] + math.cos(math.radians(angle)) * depth * 0.15
            obj = p.soft_box(f"curved_sofa_segment_{i}", (x, y, h * 0.20), (w * 0.22, depth * 0.55, h * 0.40), p.p.fabric)
            rotate_z(obj, angle)

    def build_base_bench(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "corner" in subclass:
            p.soft_box("corner_bench_a", (loc[0] - w * 0.15, loc[1], h * 0.45), (w * 0.70, depth, h * 0.35), p.p.fabric)
            p.soft_box("corner_bench_b", (loc[0] + w * 0.22, loc[1] - depth * 0.20, h * 0.45), (w * 0.42, depth * 0.85, h * 0.35), p.p.fabric)
        else:
            p.soft_box("bench_seat", (loc[0], loc[1], h * 0.62), (w, depth, h * 0.22), p.p.fabric)
            if "back" in subclass:
                p.soft_box("bench_back", (loc[0], loc[1] + depth * 0.42, h * 0.80), (w, depth * 0.14, h * 0.48), p.p.fabric)
            if "armrests" in subclass:
                p.soft_box("bench_arm_l", (loc[0] - w * 0.48, loc[1], h * 0.60), (w * 0.08, depth, h * 0.40), p.p.fabric)
                p.soft_box("bench_arm_r", (loc[0] + w * 0.48, loc[1], h * 0.60), (w * 0.08, depth, h * 0.40), p.p.fabric)
            if "storage" in subclass:
                p.panel("bench_storage_front", (loc[0], loc[1] - depth * 0.50, h * 0.34), (w * 0.8, 0.02, h * 0.22), p.p.light)
            p.four_legs("bench", loc, w, depth, h * 0.45, 0.022)

    def build_base_ottoman(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "round" in subclass or "pouf" in subclass:
            cylinder_obj("ottoman_round", (loc[0], loc[1], h * 0.5), min(w, depth) * 0.46, h, p.p.fabric, p.col, vertices=48, bevel=0.02)
        elif "modular" in subclass:
            for i, ox in enumerate((-0.25, 0.25), 1):
                p.soft_box(f"ottoman_mod_{i}", (loc[0] + ox * w, loc[1], h * 0.5), (w * 0.48, depth, h), p.p.fabric)
        else:
            p.soft_box("ottoman_rect", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.fabric)
        if "storage" in subclass:
            p.panel("ottoman_lid_line", (loc[0], loc[1] - depth * 0.52, h * 0.78), (w * 0.8, 0.012, 0.012), p.p.dark)

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def build_base_pendant_lamp(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "linear" in subclass or "island" in subclass:
            p.panel("linear_pendant_bar", (loc[0], loc[1], h * 0.78), (w * 1.8, 0.05, 0.05), p.p.metal)
            for i in range(3):
                x = loc[0] + (i - 1) * w * 0.55
                cylinder_obj("linear_pendant_cord", (x, loc[1], h * 0.58), 0.01, h * 0.34, p.p.black, p.col)
                cone_obj("linear_pendant_shade", (x, loc[1], h * 0.38), w * 0.16, w * 0.10, h * 0.18, p.p.base, p.col)
            return
        count = 5 if "cluster" in subclass else 3 if "multi" in subclass else 1
        for i in range(count):
            angle = 2 * math.pi * i / max(count, 1)
            x = loc[0] + (math.cos(angle) * w * 0.25 if count > 1 else 0)
            y = loc[1] + (math.sin(angle) * depth * 0.25 if count > 1 else 0)
            cylinder_obj("pendant_cord", (x, y, h * 0.65), 0.01, h * 0.48, p.p.black, p.col)
            if "globe" in subclass:
                sphere_obj("pendant_globe", (x, y, h * 0.34), (w * 0.32, depth * 0.32, w * 0.32), p.p.glass, p.col)
            elif "dome" in subclass or "drum" in subclass:
                cylinder_obj("pendant_drum", (x, y, h * 0.34), w * 0.28, h * 0.18, p.p.base, p.col, vertices=48)
            else:
                cone_obj("pendant_cone", (x, y, h * 0.34), w * 0.30, w * 0.16, h * 0.22, p.p.base, p.col)

    def build_base_ceiling_light(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "track" in subclass:
            p.panel("track_bar", (loc[0], loc[1], h * 0.95), (w * 1.8, 0.04, 0.04), p.p.metal)
            for i in range(4):
                x = loc[0] + (i - 1.5) * w * 0.42
                cone_obj("track_spot", (x, loc[1], h * 0.75), 0.08, 0.05, 0.14, p.p.base, p.col)
            return
        if "led_panel" in subclass or "square" in subclass or "linear" in subclass:
            p.panel("ceiling_panel", (loc[0], loc[1], h * 0.95), (w * (1.8 if "linear" in subclass else 1.0), depth * 0.75, max(h * 0.14, 0.035)), p.p.light)
            return
        if "fan" in subclass:
            cylinder_obj("fan_motor", (loc[0], loc[1], h * 0.78), w * 0.12, h * 0.18, p.p.metal, p.col)
            for i in range(4):
                blade = p.panel("fan_blade", (loc[0] + math.cos(i * math.pi / 2) * w * 0.32, loc[1] + math.sin(i * math.pi / 2) * w * 0.32, h * 0.78), (w * 0.55, 0.06, 0.018), p.p.wood)
                rotate_z(blade, math.degrees(i * math.pi / 2))
            sphere_obj("fan_light_globe", (loc[0], loc[1], h * 0.58), (w * 0.18, depth * 0.18, h * 0.14), p.p.glass, p.col)
            return
        cylinder_obj("ceiling_light_base", (loc[0], loc[1], h * 0.88), min(w, depth) * 0.38, max(h * 0.18, 0.04), p.p.metal, p.col, vertices=48)
        sphere_obj("ceiling_light_diffuser", (loc[0], loc[1], h * 0.72), (w * 0.38, depth * 0.38, h * 0.24), p.p.glass, p.col)
        if "chandelier" in subclass:
            for i in range(6):
                a = 2 * math.pi * i / 6
                sphere_obj("chandelier_crystal", (loc[0] + math.cos(a) * w * 0.30, loc[1] + math.sin(a) * depth * 0.30, h * 0.46), (0.035, 0.035, 0.08), p.p.glass, p.col)

    def build_base_floor_lamp(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        cylinder_obj("floor_lamp_base", (loc[0], loc[1], 0.03), min(w, depth) * 0.28, 0.06, p.p.metal, p.col)
        if "tripod" in subclass:
            for i in range(3):
                a = 2 * math.pi * i / 3
                leg = p.panel("tripod_leg", (loc[0] + math.cos(a) * w * 0.12, loc[1] + math.sin(a) * depth * 0.12, h * 0.35), (0.025, 0.025, h * 0.72), p.p.metal)
                rotate_z(leg, math.degrees(a))
        elif "arc" in subclass:
            pole = p.panel("arc_floor_lamp_pole", (loc[0] + w * 0.20, loc[1], h * 0.55), (0.025, 0.025, h * 0.92), p.p.metal)
            rotate_z(pole, -12)
        else:
            cylinder_obj("floor_lamp_pole", (loc[0], loc[1], h * 0.45), min(w, depth) * 0.035, h * 0.82, p.p.metal, p.col)
        heads = 3 if "multi_head" in subclass else 1
        for i in range(heads):
            a = 2 * math.pi * i / max(heads, 1)
            x = loc[0] + (math.cos(a) * w * 0.20 if heads > 1 else 0)
            y = loc[1] + (math.sin(a) * depth * 0.20 if heads > 1 else 0)
            if "globe" in subclass:
                sphere_obj("floor_lamp_globe", (x, y, h * 0.86), (w * 0.18, depth * 0.18, h * 0.07), p.p.glass, p.col)
            else:
                cone_obj("floor_lamp_shade", (x, y, h * 0.86), min(w, depth) * 0.32, min(w, depth) * 0.20, h * 0.20, p.p.light, p.col)

    def build_base_table_lamp(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "banker" in subclass:
            cylinder_obj("banker_base", (loc[0], loc[1], h * 0.05), w * 0.22, h * 0.10, p.p.metal, p.col)
            p.panel("banker_arm", (loc[0], loc[1], h * 0.42), (0.025, 0.025, h * 0.60), p.p.metal)
            p.panel("banker_shade", (loc[0], loc[1], h * 0.78), (w * 0.80, depth * 0.30, h * 0.16), p.p.glass)
            return
        cylinder_obj("table_lamp_base", (loc[0], loc[1], h * 0.05), min(w, depth) * 0.24, h * 0.10, p.p.metal, p.col)
        cylinder_obj("table_lamp_pole", (loc[0], loc[1], h * 0.38), min(w, depth) * 0.035, h * 0.55, p.p.metal, p.col)
        if "globe" in subclass or "mushroom" in subclass:
            sphere_obj("table_lamp_globe", (loc[0], loc[1], h * 0.75), (w * 0.34, depth * 0.34, h * 0.18), p.p.glass, p.col)
        else:
            cone_obj("table_lamp_shade", (loc[0], loc[1], h * 0.76), min(w, depth) * 0.42, min(w, depth) * 0.28, h * 0.34, p.p.light, p.col)

    def build_base_wall_light(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "picture" in subclass or "linear" in subclass or "bathroom" in subclass:
            p.panel("wall_linear_light", (loc[0], loc[1], h * 0.5), (w * 1.4, depth, h * 0.18), p.p.light)
            p.panel("wall_linear_backplate", (loc[0], loc[1] + depth * 0.3, h * 0.5), (w * 1.5, 0.025, h * 0.24), p.p.metal)
            return
        if "swing_arm" in subclass:
            p.panel("wall_light_plate", (loc[0], loc[1], h * 0.5), (w * 0.32, depth, h * 0.60), p.p.metal)
            arm = p.panel("swing_arm", (loc[0], loc[1] - depth * 1.8, h * 0.60), (w * 0.80, 0.025, 0.025), p.p.metal)
            rotate_z(arm, -20)
            sphere_obj("swing_arm_globe", (loc[0] + w * 0.36, loc[1] - depth * 2.2, h * 0.55), (w * 0.20, w * 0.20, h * 0.20), p.p.glass, p.col)
            return
        p.panel("wall_light_plate", (loc[0], loc[1], h * 0.5), (w * 0.55, depth, h), p.p.metal)
        sphere_obj("wall_light_globe", (loc[0], loc[1] - depth * 0.60, h * 0.58), (w * 0.38, w * 0.38, h * 0.25), p.p.glass, p.col)

    # ------------------------------------------------------------------
    # Soft objects
    # ------------------------------------------------------------------

    def build_base_bed(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "bunk" in subclass or "loft" in subclass:
            self.build_bunk_bed(p, subclass, loc, d)
            return
        frame_h = min(0.24, h * 0.36)
        mat_h = min(0.28, h * 0.40)
        p.panel("bed_frame", (loc[0], loc[1], frame_h / 2), (w, depth, frame_h), p.p.wood)
        p.soft_box("bed_mattress", (loc[0], loc[1], frame_h + mat_h / 2), (w * 0.94, depth * 0.92, mat_h), p.p.light, bevel=0.08)
        if "without_headboard" not in subclass:
            hb_h = 0.45 if "low_headboard" in subclass else 0.95 if "high_headboard" in subclass else 0.72
            hb_mat = p.p.fabric if "upholstered" in subclass else p.p.wood
            p.soft_box("bed_headboard", (loc[0], loc[1] + depth * 0.48, hb_h / 2 + frame_h), (w * 1.02, 0.10, hb_h), hb_mat, bevel=0.04)
        pillows = 1 if any(k in subclass for k in ("single", "twin", "crib", "daybed")) else 2
        for i in range(pillows):
            x = loc[0] + (i - (pillows - 1) / 2) * w * 0.34
            p.soft_box(f"bed_pillow_{i+1}", (x, loc[1] + depth * 0.28, frame_h + mat_h + 0.06), (w * 0.28, depth * 0.15, 0.10), p.p.fabric, bevel=0.07)
        if "storage" in subclass:
            p.drawer_faces("bed_storage", loc, w, depth, frame_h + 0.12, 3)
        if "canopy" in subclass:
            post_h = 1.8
            for sx in (-1, 1):
                for sy in (-1, 1):
                    cylinder_obj("canopy_post", (loc[0] + sx * w * 0.47, loc[1] + sy * depth * 0.47, post_h / 2), 0.025, post_h, p.p.metal, p.col)
            p.panel("canopy_top_front", (loc[0], loc[1] - depth * 0.47, post_h), (w, 0.025, 0.025), p.p.metal)
            p.panel("canopy_top_back", (loc[0], loc[1] + depth * 0.47, post_h), (w, 0.025, 0.025), p.p.metal)

    def build_bunk_bed(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        level_h = h * 0.42
        for level in (0.38, 0.78):
            z = h * level
            p.panel("bunk_frame", (loc[0], loc[1], z), (w, depth, 0.16), p.p.wood)
            p.soft_box("bunk_mattress", (loc[0], loc[1], z + 0.12), (w * 0.92, depth * 0.90, 0.16), p.p.light, bevel=0.06)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cylinder_obj("bunk_post", (loc[0] + sx * w * 0.48, loc[1] + sy * depth * 0.48, h * 0.5), 0.03, h, p.p.wood, p.col)
        for i in range(5):
            p.panel("bunk_ladder_step", (loc[0] + w * 0.52, loc[1] - depth * 0.25, h * (0.18 + i * 0.12)), (0.18, 0.025, 0.025), p.p.metal)

    def build_base_pillow(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "bolster" in subclass or "cylindrical" in subclass:
            cyl = cylinder_obj("pillow_bolster", (loc[0], loc[1], h * 0.5), min(depth, h) * 0.48, w, p.p.fabric, p.col, vertices=40)
            cyl.rotation_euler[1] = math.radians(90)
            return
        if "round" in subclass:
            sphere_obj("pillow_round", (loc[0], loc[1], h * 0.5), (w * 0.5, w * 0.5, h * 0.5), p.p.fabric, p.col, segments=48)
            return
        if "body" in subclass:
            w *= 1.8
        if "lumbar" in subclass:
            w *= 1.25
            h *= 0.75
        p.soft_box("pillow_soft_rect", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.fabric, bevel=0.08)
        p.panel("pillow_seam", (loc[0], loc[1], h * 0.5), (w * 0.95, depth * 0.06, h * 0.06), p.p.light, bevel=0.01)

    def build_base_blanket(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "folded" in subclass:
            w *= 0.70
            depth *= 0.55
            h = max(h * 2.0, 0.07)
        p.soft_box("blanket_body", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.fabric, bevel=0.06)
        folds = 4 if "quilt" in subclass or "plaid" in subclass else 2
        for i in range(folds):
            x = loc[0] + (i - (folds - 1) / 2) * w / max(folds, 1)
            p.panel("blanket_fold", (x, loc[1], h + 0.004), (0.010, depth * 0.90, 0.006), p.p.dark, bevel=0.002)
        if "draped" in subclass:
            flap = p.soft_box("blanket_drape_flap", (loc[0], loc[1] - depth * 0.44, h * 0.34), (w * 0.88, 0.06, h * 1.6), p.p.fabric, bevel=0.04)
            rotate_x(flap, 15)

    def build_base_rug(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        h = max(h, 0.025)
        if "round" in subclass:
            cylinder_obj("round_rug", (loc[0], loc[1], h / 2), min(w, depth) * 0.48, h, p.p.fabric, p.col, vertices=72)
        elif "oval" in subclass or "irregular" in subclass:
            p.soft_box("oval_rug", (loc[0], loc[1], h / 2), (w, depth, h), p.p.fabric, bevel=0.16, segments=12)
        elif "runner" in subclass or "hallway" in subclass:
            p.soft_box("runner_rug", (loc[0], loc[1], h / 2), (w * 0.65, depth * 1.8, h), p.p.fabric, bevel=0.06)
        else:
            p.soft_box("rect_rug", (loc[0], loc[1], h / 2), (w, depth, h), p.p.fabric, bevel=0.06)
        for off in (-0.33, 0, 0.33):
            p.panel("rug_stripe", (loc[0] + off * w, loc[1], h + 0.003), (0.014, depth * 0.88, 0.006), p.p.light, bevel=0.001)

    # ------------------------------------------------------------------
    # Decor
    # ------------------------------------------------------------------

    def build_base_mirror(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "round" in subclass:
            cylinder_obj("round_mirror_frame", (loc[0], loc[1], h * 0.5), w * 0.50, max(depth, 0.03), p.p.metal, p.col, vertices=72)
            cylinder_obj("round_mirror_glass", (loc[0], loc[1] - depth * 0.6, h * 0.5), w * 0.42, 0.012, p.p.glass, p.col, vertices=72)
        elif "oval" in subclass or "arched" in subclass:
            p.soft_box("oval_mirror_frame", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.metal, bevel=0.12)
            p.soft_box("oval_mirror_glass", (loc[0], loc[1] - depth * 0.60, h * 0.5), (w * 0.82, 0.012, h * 0.84), p.p.glass, bevel=0.09)
        else:
            p.panel("mirror_frame", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.metal, bevel=0.025)
            p.panel("mirror_glass", (loc[0], loc[1] - depth * 0.55, h * 0.5), (w * 0.84, 0.012, h * 0.86), p.p.glass, bevel=0.008)

    def build_base_wall_art(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "gallery" in subclass or "triptych" in subclass:
            count = 3
            for i in range(count):
                x = loc[0] + (i - 1) * w * 0.35
                p.panel("gallery_frame", (x, loc[1], h * 0.5), (w * 0.28, depth, h * 0.70), p.p.dark)
                p.panel("gallery_art", (x, loc[1] - depth * 0.58, h * 0.5), (w * 0.23, 0.012, h * 0.58), p.p.light)
            return
        if "sculpture" in subclass:
            for i in range(5):
                a = 2 * math.pi * i / 5
                sphere_obj("wall_sculpture_piece", (loc[0] + math.cos(a) * w * 0.22, loc[1], h * (0.5 + math.sin(a) * 0.18)), (w * 0.10, depth * 0.5, h * 0.08), p.p.base, p.col)
            return
        p.panel("wall_art_frame", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.dark)
        p.panel("wall_art_canvas", (loc[0], loc[1] - depth * 0.58, h * 0.5), (w * 0.84, 0.012, h * 0.78), p.p.light)

    def build_base_decor_vase(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "floor" in subclass:
            h *= 1.8
            w *= 1.4
            depth *= 1.4
        if "short" in subclass:
            h *= 0.65
        body_r = min(w, depth) * (0.40 if "round" in subclass else 0.32)
        neck_r = body_r * (0.38 if "narrow" in subclass or "bottle" in subclass else 0.55)
        cylinder_obj("vase_body", (loc[0], loc[1], h * 0.38), body_r, h * 0.72, p.p.base, p.col, vertices=48)
        cylinder_obj("vase_neck", (loc[0], loc[1], h * 0.78), neck_r, h * 0.34, p.p.base, p.col, vertices=48)
        if "flowers" in subclass or "branches" in subclass:
            stem_mat = make_material("plant_stem", (0.10, 0.30, 0.08, 1), "plastic")
            flower_mat = make_material("flower", (0.8, 0.25, 0.35, 1), "plastic")
            for i in range(5):
                a = 2 * math.pi * i / 5
                cylinder_obj("vase_stem", (loc[0] + math.cos(a) * w * 0.04, loc[1] + math.sin(a) * depth * 0.04, h * 1.05), 0.006, h * 0.50, stem_mat, p.col)
                if "flowers" in subclass:
                    sphere_obj("vase_flower", (loc[0] + math.cos(a) * w * 0.10, loc[1] + math.sin(a) * depth * 0.10, h * 1.32), (0.035, 0.035, 0.035), flower_mat, p.col, segments=16)

    def build_base_decor_box(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "book" in subclass:
            self.build_base_book(p, "closed_book", loc, (w, depth, h))
            return
        p.panel("decor_box_body", (loc[0], loc[1], h * 0.45), (w, depth, h * 0.90), p.p.base)
        if "lid" in subclass or "jewelry" in subclass or "storage" in subclass:
            p.panel("decor_box_lid", (loc[0], loc[1], h * 0.93), (w * 1.04, depth * 1.04, h * 0.12), p.p.dark)
        if "woven" in subclass:
            for i in range(3):
                p.panel("woven_line", (loc[0], loc[1] - depth * 0.52, h * (0.25 + i * 0.18)), (w * 0.9, 0.012, 0.012), p.p.dark)

    def build_base_decor_tray(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "round" in subclass:
            cylinder_obj("round_tray_base", (loc[0], loc[1], h * 0.25), min(w, depth) * 0.48, h * 0.50, p.p.base, p.col, vertices=56)
            cylinder_obj("round_tray_rim", (loc[0], loc[1], h * 0.65), min(w, depth) * 0.50, h * 0.18, p.p.dark, p.col, vertices=56)
            return
        p.panel("tray_base", (loc[0], loc[1], h * 0.25), (w, depth, h * 0.50), p.p.base)
        rim_h = max(h * 0.7, 0.025)
        p.panel("tray_rim_front", (loc[0], loc[1] - depth * 0.48, h * 0.75), (w, 0.025, rim_h), p.p.dark)
        p.panel("tray_rim_back", (loc[0], loc[1] + depth * 0.48, h * 0.75), (w, 0.025, rim_h), p.p.dark)
        p.panel("tray_rim_left", (loc[0] - w * 0.48, loc[1], h * 0.75), (0.025, depth, rim_h), p.p.dark)
        p.panel("tray_rim_right", (loc[0] + w * 0.48, loc[1], h * 0.75), (0.025, depth, rim_h), p.p.dark)

    def build_base_decor_books(self, p: Parts, subclass: str, loc, d) -> None:
        self.build_base_book(p, "book_stack" if "stack" in subclass or "coffee" in subclass else "standing_book" if "standing" in subclass or "bookshelf" in subclass else "open_book" if "open" in subclass else "closed_book", loc, d)

    def build_base_plant(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "hanging" in subclass:
            loc = (loc[0], loc[1], loc[2] + 0.8)
        pot_h = h * 0.25
        if "planter_box" in subclass:
            p.panel("plant_rect_planter", (loc[0], loc[1], pot_h * 0.5), (w, depth * 0.55, pot_h), p.p.base)
        else:
            cone_obj("plant_pot", (loc[0], loc[1], pot_h * 0.5), min(w, depth) * 0.30, min(w, depth) * 0.22, pot_h, p.p.base, p.col)
        stem_mat = make_material("plant_stem", (0.10, 0.28, 0.08, 1), "plastic")
        leaf_mat = make_material("plant_leaf", (0.12, 0.42, 0.10, 1), "plastic")
        if "succulent" in subclass:
            for i in range(10):
                a = 2 * math.pi * i / 10
                leaf = sphere_obj("succulent_leaf", (loc[0] + math.cos(a) * w * 0.12, loc[1] + math.sin(a) * depth * 0.12, pot_h + h * 0.06), (0.08, 0.025, 0.035), leaf_mat, p.col, segments=16)
                rotate_z(leaf, math.degrees(a))
            return
        cylinder_obj("plant_stem", (loc[0], loc[1], pot_h + h * 0.25), 0.014, h * 0.50, stem_mat, p.col)
        leaf_count = 12 if "large_leaf" in subclass or "large" in subclass else 7
        for i in range(leaf_count):
            a = 2 * math.pi * i / leaf_count
            leaf = sphere_obj("plant_leaf", (loc[0] + math.cos(a) * w * 0.25, loc[1] + math.sin(a) * depth * 0.25, pot_h + h * (0.32 + 0.04 * (i % 4))), (0.11, 0.026, 0.050), leaf_mat, p.col, segments=16)
            rotate_z(leaf, math.degrees(a))

    # ------------------------------------------------------------------
    # Appliances
    # ------------------------------------------------------------------

    def build_base_refrigerator(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("fridge_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.metal)
        front = loc[1] - depth * 0.51
        if "side_by_side" in subclass or "french" in subclass or "two_door" in subclass:
            p.panel("fridge_left_door", (loc[0] - w * 0.24, front, h * 0.53), (w * 0.47, 0.02, h * 0.88), p.p.light)
            p.panel("fridge_right_door", (loc[0] + w * 0.24, front, h * 0.53), (w * 0.47, 0.02, h * 0.88), p.p.light)
            p.panel("fridge_handle_l", (loc[0] - w * 0.03, front - 0.012, h * 0.54), (0.018, 0.012, h * 0.58), p.p.metal)
            p.panel("fridge_handle_r", (loc[0] + w * 0.03, front - 0.012, h * 0.54), (0.018, 0.012, h * 0.58), p.p.metal)
            if "french" in subclass:
                p.panel("fridge_bottom_freezer", (loc[0], front - 0.01, h * 0.16), (w * 0.88, 0.02, h * 0.22), p.p.base)
        elif "wine" in subclass:
            p.panel("wine_glass_door", (loc[0], front, h * 0.52), (w * 0.86, 0.02, h * 0.80), p.p.glass)
            for i in range(5):
                p.panel("wine_shelf", (loc[0], loc[1] - depth * 0.10, h * (0.18 + i * 0.14)), (w * 0.75, depth * 0.55, 0.018), p.p.dark)
        else:
            freezer_top = "top_freezer" in subclass
            split_z = h * (0.72 if freezer_top else 0.35)
            p.panel("fridge_door_main", (loc[0], front, split_z * 0.50), (w * 0.92, 0.02, split_z * 0.90), p.p.light)
            p.panel("fridge_door_secondary", (loc[0], front, split_z + (h - split_z) * 0.50), (w * 0.92, 0.02, (h - split_z) * 0.86), p.p.base)
            p.panel("fridge_handle", (loc[0] + w * 0.36, front - 0.012, h * 0.52), (0.018, 0.012, h * 0.44), p.p.metal)

    def build_base_stove(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("stove_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.metal)
        p.panel("stove_oven_window", (loc[0], loc[1] - depth * 0.515, h * 0.42), (w * 0.65, 0.018, h * 0.32), p.p.screen)
        count = 2 if "compact" in subclass else 4
        p.burners("stove", loc, w, depth, h + 0.012, count)
        for i in range(count):
            cylinder_obj("stove_knob", (loc[0] - w * 0.25 + i * w * 0.16, loc[1] - depth * 0.53, h * 0.80), 0.018, 0.018, p.p.metal, p.col, vertices=24)

    def build_base_cooktop(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("cooktop_glass", (loc[0], loc[1], max(h, 0.045) / 2), (w, depth, max(h, 0.045)), p.p.screen)
        count = 2 if "two_burner" in subclass else 5 if "five_burner" in subclass else 4
        p.burners("cooktop", loc, w, depth, max(h, 0.045) + 0.012, count)

    def build_base_oven(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "double" in subclass:
            h *= 1.55
        p.panel("oven_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.metal)
        sections = 2 if "double" in subclass else 1
        for i in range(sections):
            z = h * (0.30 + i * 0.38) if sections == 2 else h * 0.46
            p.panel("oven_window", (loc[0], loc[1] - depth * 0.515, z), (w * 0.68, 0.018, h * 0.24), p.p.screen)
            p.panel("oven_handle", (loc[0], loc[1] - depth * 0.535, z + h * 0.18), (w * 0.55, 0.018, 0.018), p.p.metal)

    def build_base_microwave(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("microwave_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.metal)
        p.panel("microwave_window", (loc[0] - w * 0.10, loc[1] - depth * 0.515, h * 0.52), (w * 0.62, 0.018, h * 0.62), p.p.screen)
        p.panel("microwave_control_panel", (loc[0] + w * 0.34, loc[1] - depth * 0.516, h * 0.52), (w * 0.16, 0.018, h * 0.65), p.p.black)
        for i in range(2):
            cylinder_obj("microwave_knob", (loc[0] + w * 0.34, loc[1] - depth * 0.535, h * (0.38 + i * 0.22)), 0.018, 0.016, p.p.metal, p.col, vertices=24)

    def build_base_dishwasher(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("dishwasher_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.metal)
        p.panel("dishwasher_front", (loc[0], loc[1] - depth * 0.515, h * 0.52), (w * 0.92, 0.018, h * 0.84), p.p.light)
        p.panel("dishwasher_handle", (loc[0], loc[1] - depth * 0.535, h * 0.82), (w * 0.45, 0.018, 0.018), p.p.metal)
        p.panel("dishwasher_control", (loc[0], loc[1] - depth * 0.536, h * 0.92), (w * 0.55, 0.012, 0.035), p.p.black)

    def build_base_washing_machine(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("washer_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.light)
        front = loc[1] - depth * 0.515
        if "top_load" in subclass:
            p.panel("washer_top_lid", (loc[0], loc[1], h + 0.014), (w * 0.86, depth * 0.78, 0.025), p.p.base)
            p.panel("washer_control", (loc[0], front, h * 0.88), (w * 0.78, 0.012, h * 0.12), p.p.black)
        else:
            cylinder_obj("washer_round_door", (loc[0], front, h * 0.45), min(w, h) * 0.24, 0.025, p.p.glass, p.col, vertices=56)
            p.panel("washer_control", (loc[0], front - 0.01, h * 0.84), (w * 0.74, 0.014, h * 0.10), p.p.black)

    def build_base_range_hood(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "island" in subclass:
            cylinder_obj("island_hood_chimney", (loc[0], loc[1], h * 0.75), min(w, depth) * 0.16, h * 0.50, p.p.metal, p.col, vertices=4)
            p.panel("island_hood_body", (loc[0], loc[1], h * 0.36), (w, depth, h * 0.32), p.p.metal)
        elif "telescopic" in subclass or "built_in" in subclass:
            p.panel("telescopic_hood_body", (loc[0], loc[1], h * 0.45), (w, depth * 0.55, h * 0.42), p.p.metal)
            p.panel("telescopic_hood_slide", (loc[0], loc[1] - depth * 0.28, h * 0.32), (w * 0.92, depth * 0.32, h * 0.10), p.p.light)
        else:
            cone_obj("wall_hood_pyramid", (loc[0], loc[1], h * 0.42), max(w, depth) * 0.45, max(w, depth) * 0.22, h * 0.58, p.p.metal, p.col, vertices=4)
            cylinder_obj("wall_hood_chimney", (loc[0], loc[1], h * 0.82), min(w, depth) * 0.15, h * 0.36, p.p.metal, p.col, vertices=4)

    def build_base_small_kitchen_appliance(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "kettle" in subclass:
            cylinder_obj("kettle_body", (loc[0], loc[1], h * 0.45), min(w, depth) * 0.32, h * 0.70, p.p.metal, p.col, vertices=40)
            p.panel("kettle_handle", (loc[0] + w * 0.32, loc[1], h * 0.52), (0.035, depth * 0.12, h * 0.42), p.p.black)
            cone_obj("kettle_spout", (loc[0] - w * 0.32, loc[1], h * 0.58), 0.04, 0.015, 0.12, p.p.metal, p.col)
        elif "coffee" in subclass:
            p.panel("coffee_machine_body", (loc[0], loc[1], h * 0.52), (w, depth, h), p.p.black)
            cylinder_obj("coffee_machine_portafilter", (loc[0], loc[1] - depth * 0.52, h * 0.45), w * 0.12, 0.06, p.p.metal, p.col, vertices=24)
            p.panel("coffee_machine_cup", (loc[0], loc[1] - depth * 0.25, h * 0.16), (w * 0.28, depth * 0.20, h * 0.22), p.p.ceramic)
        elif "toaster" in subclass:
            p.soft_box("toaster_body", (loc[0], loc[1], h * 0.45), (w, depth, h * 0.78), p.p.metal, bevel=0.05)
            p.panel("toaster_slot_1", (loc[0] - w * 0.16, loc[1], h * 0.86), (w * 0.22, depth * 0.70, 0.012), p.p.black)
            p.panel("toaster_slot_2", (loc[0] + w * 0.16, loc[1], h * 0.86), (w * 0.22, depth * 0.70, 0.012), p.p.black)
        elif "blender" in subclass:
            p.panel("blender_base", (loc[0], loc[1], h * 0.18), (w * 0.65, depth * 0.65, h * 0.30), p.p.black)
            cone_obj("blender_jar", (loc[0], loc[1], h * 0.62), w * 0.26, w * 0.18, h * 0.60, p.p.glass, p.col)
        else:
            p.soft_box("small_appliance_body", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.metal, bevel=0.05)
            p.panel("small_appliance_panel", (loc[0], loc[1] - depth * 0.515, h * 0.55), (w * 0.62, 0.012, h * 0.38), p.p.black)

    # ------------------------------------------------------------------
    # Electronics
    # ------------------------------------------------------------------

    def build_base_tv(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.screen("tv", loc, w, depth, h)
        if "on_stand" in subclass:
            cylinder_obj("tv_neck", (loc[0], loc[1], -0.05), 0.025, 0.20, p.p.metal, p.col)
            p.panel("tv_base", (loc[0], loc[1], -0.16), (w * 0.36, depth * 4.0, 0.035), p.p.metal)

    def build_base_monitor(self, p: Parts, subclass: str, loc, d) -> None:
        count = 2 if "dual" in subclass else 1
        w, depth, h = d
        for i in range(count):
            x = loc[0] + (i - (count - 1) / 2) * w * 0.90
            p.screen("monitor", (x, loc[1], loc[2]), w * (1.5 if "ultrawide" in subclass else 1.0), depth, h)
            cylinder_obj("monitor_stand", (x, loc[1], -0.10), 0.018, 0.20, p.p.metal, p.col)
            p.panel("monitor_base", (x, loc[1], -0.21), (w * 0.34, depth * 3.0, 0.026), p.p.metal)
        if "arm" in subclass:
            p.panel("monitor_arm", (loc[0], loc[1] + 0.06, h * 0.32), (w * 0.55, 0.025, 0.025), p.p.metal)

    def build_base_laptop(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("laptop_keyboard_base", (loc[0], loc[1], 0.015), (w, depth, 0.030), p.p.metal)
        p.panel("laptop_keyboard", (loc[0], loc[1] - depth * 0.08, 0.035), (w * 0.82, depth * 0.38, 0.006), p.p.black)
        if "closed" not in subclass:
            screen = p.panel("laptop_screen", (loc[0], loc[1] + depth * 0.42, h * 0.50), (w, 0.025, h), p.p.screen)
            rotate_x(screen, -68)

    def build_base_desktop_computer(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "all_in_one" in subclass:
            self.build_base_monitor(p, "single_monitor", loc, (w * 2.0, 0.05, h * 1.1))
            return
        if "mini" in subclass:
            p.panel("mini_pc", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.black)
        else:
            p.panel("pc_tower", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.black)
            if "gaming" in subclass:
                p.panel("pc_glass_side", (loc[0] - w * 0.51, loc[1], h * 0.52), (0.014, depth * 0.82, h * 0.76), p.p.glass)
                sphere_obj("pc_rgb_fan", (loc[0] - w * 0.52, loc[1], h * 0.55), (w * 0.20, w * 0.20, w * 0.04), make_material("rgb_fan", (0.1, 0.4, 0.9, 1), "plastic"), p.col, segments=24)

    def build_base_phone(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "desk_phone" in subclass or "cordless" in subclass:
            p.panel("desk_phone_base", (loc[0], loc[1], h * 0.24), (w * 2.0, depth * 2.0, h * 0.40), p.p.black)
            p.panel("desk_phone_receiver", (loc[0], loc[1] + depth * 0.45, h * 0.62), (w * 2.0, depth * 0.35, h * 0.20), p.p.dark)
        else:
            p.panel("smartphone_body", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.screen, bevel=0.02)
            if "stand" in subclass:
                stand = p.panel("phone_stand", (loc[0], loc[1] + depth * 1.5, h * 0.4), (w * 0.85, 0.025, h * 0.60), p.p.metal)
                rotate_x(stand, -18)

    def build_base_tablet(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "stand" in subclass:
            screen = p.panel("tablet_screen_stand", (loc[0], loc[1], h * 0.5), (w, 0.025, h), p.p.screen)
            rotate_x(screen, -15)
            p.panel("tablet_stand_base", (loc[0], loc[1] + 0.06, 0.025), (w * 0.45, depth * 3.0, 0.04), p.p.metal)
        else:
            p.panel("tablet_flat", (loc[0], loc[1], depth * 0.5), (w, h, depth), p.p.screen, bevel=0.015)

    def build_base_keyboard(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("keyboard_base", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.black, bevel=0.012)
        key_rows = 4
        key_cols = 10 if "compact" in subclass else 14
        for r in range(key_rows):
            for c in range(key_cols):
                p.panel("keyboard_key", (loc[0] - w * 0.42 + c * w * 0.84 / key_cols, loc[1] - depth * 0.32 + r * depth * 0.20, h + 0.004), (w * 0.035, depth * 0.09, 0.004), p.p.dark, bevel=0.001)

    def build_base_mouse(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "pad" in subclass:
            p.soft_box("mouse_pad", (loc[0], loc[1], h * 0.5), (w * 2.2, depth * 1.7, h), p.p.fabric, bevel=0.02)
        else:
            sphere_obj("mouse_body", (loc[0], loc[1], h * 0.5), (w * 0.5, depth * 0.5, h * 0.5), p.p.black, p.col, segments=32)
            p.panel("mouse_split", (loc[0], loc[1], h * 0.92), (0.006, depth * 0.50, 0.003), p.p.dark, bevel=0.001)

    def build_base_speaker(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "soundbar" in subclass:
            p.panel("soundbar", (loc[0], loc[1], h * 0.5), (w * 2.2, depth * 0.45, h * 0.42), p.p.black, bevel=0.025)
            return
        if "floorstanding" in subclass:
            h *= 1.6
        if "smart" in subclass:
            cylinder_obj("smart_speaker", (loc[0], loc[1], h * 0.5), min(w, depth) * 0.38, h, p.p.black, p.col, vertices=48)
            return
        p.panel("speaker_box", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.black)
        for z in (h * 0.32, h * 0.68):
            cylinder_obj("speaker_driver", (loc[0], loc[1] - depth * 0.51, z), w * 0.22, 0.016, p.p.dark, p.col, vertices=40)

    # ------------------------------------------------------------------
    # Architectural
    # ------------------------------------------------------------------

    def build_base_door(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "double" in subclass:
            p.panel("door_leaf_l", (loc[0] - w * 0.25, loc[1], h / 2), (w * 0.48, depth, h), p.p.wood)
            p.panel("door_leaf_r", (loc[0] + w * 0.25, loc[1], h / 2), (w * 0.48, depth, h), p.p.wood)
        elif "sliding" in subclass or "pocket" in subclass:
            p.panel("sliding_door_track", (loc[0], loc[1], h + 0.03), (w * 1.4, depth * 0.65, 0.035), p.p.metal)
            p.panel("sliding_door_leaf", (loc[0] + w * 0.18, loc[1], h / 2), (w, depth, h), p.p.wood)
        elif "folding" in subclass:
            for i in range(4):
                leaf = p.panel("folding_door_leaf", (loc[0] - w * 0.36 + i * w * 0.24, loc[1], h / 2), (w * 0.22, depth, h), p.p.wood)
                rotate_z(leaf, -8 if i % 2 else 8)
        else:
            p.panel("door_leaf", (loc[0], loc[1], h / 2), (w, depth, h), p.p.glass if "glass" in subclass else p.p.wood)
        p.panel("door_handle", (loc[0] + w * 0.36, loc[1] - depth * 0.70, h * 0.52), (0.08, 0.018, 0.025), p.p.metal)

    def build_base_window(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "panoramic" in subclass:
            w *= 1.8
            h *= 1.3
        p.panel("window_frame_outer", (loc[0], loc[1], h / 2), (w, depth, h), p.p.metal)
        p.panel("window_glass", (loc[0], loc[1] - depth * 0.55, h / 2), (w * 0.88, 0.012, h * 0.86), p.p.glass)
        if "double" in subclass:
            p.panel("window_mullion", (loc[0], loc[1] - depth * 0.60, h / 2), (0.025, 0.014, h * 0.86), p.p.metal)
        if "corner" in subclass:
            side = p.panel("corner_window_side", (loc[0] + w * 0.52, loc[1] - w * 0.22, h / 2), (w * 0.45, depth, h), p.p.glass)
            rotate_z(side, 90)

    def build_base_partition(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "folding" in subclass or "screen" in subclass:
            for i in range(3):
                panel = p.panel("partition_screen_panel", (loc[0] + (i - 1) * w * 0.32, loc[1], h / 2), (w * 0.30, depth, h), p.p.glass if "glass" in subclass else p.p.wood)
                rotate_z(panel, -12 if i % 2 else 12)
        else:
            p.panel("partition_panel", (loc[0], loc[1], h / 2), (w, depth, h), p.p.glass if "glass" in subclass else p.p.wood)

    # ------------------------------------------------------------------
    # Storage, books, bathroom
    # ------------------------------------------------------------------

    def build_base_storage_box(self, p: Parts, subclass: str, loc, d) -> None:
        self.build_base_decor_box(p, subclass, loc, d)

    def build_base_basket(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "woven" in subclass or "toy" in subclass:
            cylinder_obj("round_basket", (loc[0], loc[1], h * 0.45), min(w, depth) * 0.42, h * 0.90, p.p.base, p.col, vertices=40)
        else:
            p.soft_box("basket_body", (loc[0], loc[1], h * 0.45), (w, depth, h * 0.90), p.p.base, bevel=0.04)
        for i in range(4):
            p.panel("basket_weave", (loc[0], loc[1] - depth * 0.52, h * (0.20 + i * 0.16)), (w * 0.86, 0.012, 0.012), p.p.dark, bevel=0.001)

    def build_base_shelf_item(self, p: Parts, subclass: str, loc, d) -> None:
        if "tray" in subclass:
            self.build_base_decor_tray(p, "rectangular_tray", loc, d)
        else:
            self.build_base_storage_box(p, subclass, loc, d)

    def build_base_book(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "open" in subclass:
            left = p.panel("open_book_left", (loc[0] - w * 0.24, loc[1], h * 0.5), (w * 0.48, depth, h), p.p.light, bevel=0.006)
            right = p.panel("open_book_right", (loc[0] + w * 0.24, loc[1], h * 0.5), (w * 0.48, depth, h), p.p.light, bevel=0.006)
            rotate_z(left, 5)
            rotate_z(right, -5)
        elif "standing" in subclass:
            for i in range(5):
                p.panel("standing_book", (loc[0] - w * 0.35 + i * w * 0.18, loc[1], depth * 0.5), (w * 0.12, h, depth), p.p.base if i % 2 else p.p.light, bevel=0.003)
        elif "stack" in subclass:
            count = 5 if "large" in subclass else 3
            for i in range(count):
                p.panel("book_stack_item", (loc[0], loc[1], h * (i + 0.5)), (w * (1 - 0.05 * i), depth, h * 0.82), p.p.base if i % 2 else p.p.light, bevel=0.003)
        else:
            p.panel("closed_book", (loc[0], loc[1], h * 0.5), (w, depth, h), p.p.base, bevel=0.004)
            p.panel("book_pages", (loc[0], loc[1] - depth * 0.48, h * 0.5), (w * 0.90, 0.012, h * 0.72), p.p.light, bevel=0.002)

    def build_base_magazine(self, p: Parts, subclass: str, loc, d) -> None:
        if "holder" in subclass:
            p.panel("mag_holder_left", (loc[0] - d[0] * 0.38, loc[1], d[2] * 1.5), (0.025, d[1], d[2] * 3.0), p.p.metal)
            p.panel("mag_holder_right", (loc[0] + d[0] * 0.38, loc[1], d[2] * 1.5), (0.025, d[1], d[2] * 3.0), p.p.metal)
            self.build_base_magazine(p, "magazine_stack", loc, d)
        else:
            self.build_base_book(p, "book_stack" if "stack" in subclass else "closed_book", loc, d)

    def build_base_folder(self, p: Parts, subclass: str, loc, d) -> None:
        self.build_base_book(p, "standing_book" if "binder" in subclass else "book_stack" if "paper_stack" in subclass else "closed_book", loc, d)

    def build_base_toilet(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.soft_box("toilet_bowl", (loc[0], loc[1] - depth * 0.10, h * 0.32), (w * 0.82, depth * 0.62, h * 0.35), p.p.ceramic, bevel=0.08)
        cylinder_obj("toilet_seat", (loc[0], loc[1] - depth * 0.16, h * 0.54), w * 0.30, 0.035, p.p.ceramic, p.col, vertices=48)
        p.panel("toilet_tank", (loc[0], loc[1] + depth * 0.34, h * 0.66), (w * 0.86, depth * 0.22, h * 0.36), p.p.ceramic)
        if "wall_hung" in subclass:
            p.panel("toilet_wall_plate", (loc[0], loc[1] + depth * 0.48, h * 0.60), (w * 0.72, 0.03, h * 0.32), p.p.metal)

    def build_base_sink(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "double_kitchen" in subclass:
            p.panel("double_sink_counter_cut", (loc[0], loc[1], h * 0.5), (w, depth, h * 0.35), p.p.metal)
            for sx in (-0.25, 0.25):
                sphere_obj("sink_bowl", (loc[0] + sx * w, loc[1], h * 0.52), (w * 0.20, depth * 0.33, h * 0.25), p.p.ceramic, p.col)
        elif "vanity" in subclass:
            self.build_base_cabinet(p, "bathroom_cabinet", loc, (w, depth, h * 1.6))
            sphere_obj("vanity_sink_bowl", (loc[0], loc[1] - depth * 0.05, h * 1.72), (w * 0.34, depth * 0.28, h * 0.22), p.p.ceramic, p.col)
        elif "pedestal" in subclass:
            cylinder_obj("sink_pedestal", (loc[0], loc[1], h * 0.55), w * 0.12, h * 1.1, p.p.ceramic, p.col, vertices=32)
            sphere_obj("pedestal_sink_bowl", (loc[0], loc[1], h * 1.13), (w * 0.50, depth * 0.40, h * 0.28), p.p.ceramic, p.col)
        elif "wall_mounted" in subclass:
            p.panel("wall_mounted_sink_backplate", (loc[0], loc[1] + depth * 0.43, h * 0.62), (w * 0.82, 0.035, h * 0.58), p.p.ceramic)
            sphere_obj("wall_mounted_sink_bowl", (loc[0], loc[1] - depth * 0.02, h * 0.62), (w * 0.50, depth * 0.38, h * 0.30), p.p.ceramic, p.col)
            cylinder_obj("wall_mounted_sink_trap", (loc[0], loc[1] + depth * 0.18, h * 0.34), w * 0.025, h * 0.45, p.p.metal, p.col, vertices=24)
        else:
            sphere_obj("sink_bowl", (loc[0], loc[1], h * 0.55), (w * 0.50, depth * 0.38, h * 0.30), p.p.ceramic, p.col)
        cylinder_obj("sink_faucet", (loc[0], loc[1] + depth * 0.22, h * 1.10), w * 0.020, h * 0.55, p.p.metal, p.col)

    def build_base_shower(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "wet_room" in subclass or "floor_drain" in subclass:
            drain_radius = max(min(w, depth) * 0.42, 0.045)
            p.panel("wet_room_drain_plate", (loc[0], loc[1], 0.012), (drain_radius * 2.2, drain_radius * 2.2, 0.018), p.p.metal)
            p.panel("wet_room_drain_slot_a", (loc[0], loc[1], 0.026), (drain_radius * 1.55, drain_radius * 0.12, 0.012), p.p.base)
            p.panel("wet_room_drain_slot_b", (loc[0], loc[1], 0.028), (drain_radius * 0.12, drain_radius * 1.55, 0.012), p.p.base)
            return
        p.panel("shower_base", (loc[0], loc[1], 0.04), (w, depth, 0.08), p.p.ceramic)
        if "corner" in subclass:
            p.panel("shower_glass_a", (loc[0] - w * 0.35, loc[1], h * 0.5), (0.025, depth, h), p.p.glass)
            p.panel("shower_glass_b", (loc[0], loc[1] - depth * 0.35, h * 0.5), (w, 0.025, h), p.p.glass)
        else:
            p.panel("shower_glass_front", (loc[0], loc[1] - depth * 0.48, h * 0.5), (w, 0.025, h), p.p.glass)
            p.panel("shower_glass_side", (loc[0] - w * 0.48, loc[1], h * 0.5), (0.025, depth, h), p.p.glass)
        cylinder_obj("shower_head", (loc[0], loc[1] + depth * 0.35, h * 0.82), w * 0.055, 0.045, p.p.metal, p.col)

    def build_base_bathtub(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        if "corner" in subclass:
            p.soft_box("corner_bathtub", (loc[0], loc[1], h * 0.42), (w, depth, h * 0.84), p.p.ceramic, bevel=0.10)
        elif "freestanding" in subclass:
            p.soft_box("freestanding_bathtub", (loc[0], loc[1], h * 0.45), (w, depth, h * 0.82), p.p.ceramic, bevel=0.18)
        else:
            p.soft_box("rect_bathtub", (loc[0], loc[1], h * 0.45), (w, depth, h * 0.82), p.p.ceramic, bevel=0.08)
        p.soft_box("bathtub_inner", (loc[0], loc[1], h * 0.62), (w * 0.78, depth * 0.62, h * 0.22), p.p.glass, bevel=0.08)

    def build_base_generic(self, p: Parts, subclass: str, loc, d) -> None:
        w, depth, h = d
        p.panel("generic_proxy_body", (loc[0], loc[1], h / 2), (w, depth, h), p.p.base)


# =============================================================================
# Public helpers and dynamic one-function-per-subclass API
# =============================================================================

_FACTORY_CACHE: ProceduralObjectFactory | None = None


def get_factory(collection_name: str = "procedural_taxonomy_objects") -> ProceduralObjectFactory:
    global _FACTORY_CACHE
    if _FACTORY_CACHE is None or _FACTORY_CACHE.collection.name != collection_name:
        _FACTORY_CACHE = ProceduralObjectFactory(collection_name)
    return _FACTORY_CACHE


def build_taxonomy_object(
    subclass: str,
    *,
    base_type: str | None = None,
    width: float | None = None,
    depth: float | None = None,
    height: float | None = None,
    color: str | tuple[float, float, float, float] | None = None,
    material: str | None = None,
    loc: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collection_name: str = "procedural_taxonomy_objects",
) -> list[bpy.types.Object]:
    spec = BuildSpec(
        subclass=subclass,
        base_type=base_type,
        width=width,
        depth=depth,
        height=height,
        color=color,
        material=material,
        collection_name=collection_name,
    )
    return get_factory(collection_name).build(spec, loc=loc)


def build_from_catalog_item(
    item: dict[str, Any],
    *,
    loc: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fallback_subclass: str = "closed_cabinet",
    collection_name: str = "procedural_taxonomy_objects",
) -> list[bpy.types.Object]:
    subclass = str(item.get("subclass") or item.get("taxonomy_subclass") or "").strip()
    base_type = str(item.get("base_type") or item.get("semantic_group") or item.get("category_norm") or "").strip() or None
    if not subclass:
        subclass = infer_subclass_from_catalog_item(item, fallback=fallback_subclass)
    dims = item.get("dimensions_cm") if isinstance(item.get("dimensions_cm"), dict) else {}
    def dim(axis: str) -> float | None:
        value = item.get(f"{axis}_cm", dims.get(axis))
        try:
            return float(value) / 100.0 if value is not None else None
        except Exception:
            return None
    color = item.get("color")
    if isinstance(item.get("image_color_features"), dict):
        colors = item["image_color_features"].get("colors") if isinstance(item["image_color_features"].get("colors"), dict) else {}
        top5 = colors.get("top5") if isinstance(colors, dict) else None
        if isinstance(top5, list) and top5 and isinstance(top5[0], dict) and top5[0].get("hex"):
            color = top5[0]["hex"]
    material = infer_material_from_catalog_item(item)
    return build_taxonomy_object(
        subclass,
        base_type=base_type if base_type in TAXONOMY else None,
        width=dim("width"),
        depth=dim("depth"),
        height=dim("height"),
        color=color,
        material=material,
        loc=loc,
        collection_name=collection_name,
    )


def infer_material_from_catalog_item(item: dict[str, Any]) -> str:
    text = " ".join(str(item.get(k) or "") for k in ("materials", "description", "title", "category_norm", "category_raw", "vlm_description_text")).lower().replace("ё", "е")
    if any(x in text for x in ("fabric", "textile", "velvet", "ткан", "текстил", "бархат")):
        return "fabric"
    if any(x in text for x in ("leather", "кож")):
        return "leather"
    if any(x in text for x in ("glass", "стекл", "хрустал", "crystal", "transparent")):
        return "glass"
    if any(x in text for x in ("metal", "steel", "iron", "chrome", "металл", "сталь", "желез", "хром")):
        return "metal"
    if any(x in text for x in ("wood", "oak", "walnut", "дерев", "дуб", "орех", "mdf")):
        return "wood"
    if any(x in text for x in ("ceramic", "porcelain", "керам", "фарфор")):
        return "ceramic"
    return "fabric"


def infer_subclass_from_catalog_item(item: dict[str, Any], fallback: str = "closed_cabinet") -> str:
    text = " ".join(str(item.get(k) or "") for k in ("title", "category_norm", "category_raw", "semantic_group", "description", "vlm_description_text")).lower().replace("ё", "е")
    rules = [
        ("side_by_side_refrigerator", ("side by side", "side-by-side")),
        ("single_door_refrigerator", ("fridge", "refrigerator", "холодильник")),
        ("wall_mount_range_hood", ("hood", "вытяж")),
        ("countertop_microwave", ("microwave", "свч", "микроволн")),
        ("freestanding_stove", ("stove", "плита")),
        ("four_burner_cooktop", ("cooktop", "варочная")),
        ("straight_sofa", ("sofa", "диван")),
        ("l_shaped_sofa", ("corner sofa", "угловой диван", "l shaped sofa", "l-shaped sofa")),
        ("dining_chair", ("chair", "стул")),
        ("lounge_armchair", ("armchair", "кресл")),
        ("double_bed", ("bed", "кровать")),
        ("drawer_nightstand", ("nightstand", "прикроват")),
        ("chest_of_drawers", ("dresser", "комод", "chest")),
        ("hinged_wardrobe", ("wardrobe", "шкаф")),
        ("open_bookshelf", ("bookshelf", "bookcase", "стеллаж")),
        ("rectangular_dining_table", ("dining table", "обеденный стол")),
        ("rectangular_coffee_table", ("coffee table", "журналь")),
        ("round_side_table", ("side table", "приставной")),
        ("writing_desk", ("desk", "рабочий стол", "письменный")),
        ("wall_mounted_tv", ("tv", "television", "телевизор")),
        ("single_monitor", ("monitor", "монитор")),
        ("open_laptop", ("laptop", "ноутбук")),
        ("floor_mounted_toilet", ("toilet", "унитаз")),
        ("bathroom_sink", ("sink", "раковина", "умывальник")),
        ("rectangular_bathtub", ("bathtub", "ванна")),
        ("shower_cabin", ("shower", "душ")),
        ("single_pendant_lamp", ("pendant", "подвес")),
        ("flush_mount_ceiling_light", ("ceiling light", "потолочный")),
        ("slim_floor_lamp", ("floor lamp", "торшер")),
        ("desk_lamp", ("desk lamp", "настольная лампа")),
        ("wall_sconce", ("sconce", "бра")),
        ("round_mirror", ("mirror", "зеркало")),
        ("framed_picture", ("wall art", "картина", "панно")),
        ("tall_vase", ("vase", "ваза")),
        ("stacked_books", ("book", "книга")),
        ("potted_plant", ("plant", "растение")),
        ("rectangular_rug", ("rug", "ковер", "ковёр")),
    ]
    for subclass, aliases in rules:
        if any(a in text for a in aliases):
            return subclass
    return fallback


# Dynamic public functions: build_<subclass>(...). This creates explicit callable
# names for every taxonomy subclass without duplicating hundreds of wrappers.
def _make_subclass_builder(subclass: str, base_type: str) -> Callable[..., list[bpy.types.Object]]:
    def _builder(
        *,
        width: float | None = None,
        depth: float | None = None,
        height: float | None = None,
        color: str | tuple[float, float, float, float] | None = None,
        material: str | None = None,
        loc: tuple[float, float, float] = (0.0, 0.0, 0.0),
        collection_name: str = "procedural_taxonomy_objects",
    ) -> list[bpy.types.Object]:
        return build_taxonomy_object(
            subclass,
            base_type=base_type,
            width=width,
            depth=depth,
            height=height,
            color=color,
            material=material,
            loc=loc,
            collection_name=collection_name,
        )
    _builder.__name__ = f"build_{subclass}"
    _builder.__doc__ = f"Build procedural proxy for taxonomy subclass `{subclass}` / base_type `{base_type}`."
    return _builder


for _base_type, _subclasses in TAXONOMY.items():
    for _subclass in _subclasses:
        globals()[f"build_{_subclass}"] = _make_subclass_builder(_subclass, _base_type)


# =============================================================================
# CLI / preview grid
# =============================================================================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build procedural interior objects for interior_object_taxonomy/v2.")
    p.add_argument("--subclass", default=None, help="Single taxonomy subclass to build, e.g. l_shaped_sofa")
    p.add_argument("--base-type", default=None)
    p.add_argument("--width", type=float, default=None)
    p.add_argument("--depth", type=float, default=None)
    p.add_argument("--height", type=float, default=None)
    p.add_argument("--color", default="#b8aa98")
    p.add_argument("--material", default=None)
    p.add_argument("--build-all", action="store_true", help="Build one object for every taxonomy subclass.")
    p.add_argument("--limit", type=int, default=0, help="Limit for --build-all; 0 means all.")
    p.add_argument("--grid-cols", type=int, default=12)
    p.add_argument("--spacing-x", type=float, default=2.6)
    p.add_argument("--spacing-y", type=float, default=2.6)
    p.add_argument("--catalog-json", default=None, help="Optional supplier catalog JSON to build from catalog items.")
    p.add_argument("--catalog-limit", type=int, default=100)
    p.add_argument("--out-blend", default="out/procedural_taxonomy_objects.blend")
    p.add_argument("--report-json", default=None)
    p.add_argument("--clear-scene", action="store_true")
    p.add_argument("--setup-preview", action="store_true")
    return p


def add_label(text: str, loc: tuple[float, float, float], col: bpy.types.Collection) -> None:
    try:
        bpy.ops.object.text_add(location=loc, rotation=(math.radians(70), 0, 0))
        obj = bpy.context.object
        obj.name = sanitize_name("label_" + text[:30])
        obj.data.body = text[:100]
        obj.data.align_x = "CENTER"
        obj.data.align_y = "CENTER"
        obj.data.size = 0.12
        link_to_collection(obj, col)
    except Exception:
        pass


def setup_preview(count: int, grid_cols: int, sx: float, sy: float) -> None:
    rows = max(1, math.ceil(count / max(grid_cols, 1)))
    width = max(grid_cols - 1, 1) * sx + 3.0
    depth = max(rows - 1, 1) * sy + 3.0
    cx = (grid_cols - 1) * sx * 0.5
    cy = (rows - 1) * sy * 0.5
    col = get_collection("preview")
    floor_mat = make_material("preview_floor", (0.62, 0.64, 0.64, 1), "plastic")
    cube_obj("preview_floor", (cx, cy, -0.015), (width, depth, 0.03), floor_mat, col, bevel=0)
    # grid lines
    line_mat = make_material("preview_grid_line", (0.38, 0.38, 0.38, 1), "plastic")
    for i in range(grid_cols + 1):
        x = i * sx - sx * 0.5
        cube_obj("preview_grid_x", (x, cy, 0.002), (0.01, depth, 0.004), line_mat, col, bevel=0)
    for j in range(rows + 1):
        y = j * sy - sy * 0.5
        cube_obj("preview_grid_y", (cx, y, 0.002), (width, 0.01, 0.004), line_mat, col, bevel=0)
    bpy.ops.object.light_add(type="AREA", location=(cx, cy - depth * 0.45, 8.0))
    light = bpy.context.object
    light.name = "preview_area_light"
    light.data.energy = 900
    light.data.size = max(width, depth) * 0.65
    bpy.ops.object.camera_add(location=(cx, cy - depth * 0.95, 9.0), rotation=(math.radians(62), 0, 0))
    cam = bpy.context.object
    cam.name = "preview_camera"
    bpy.context.scene.camera = cam
    cam.data.lens = 28
    bpy.context.scene.render.resolution_x = 1800
    bpy.context.scene.render.resolution_y = 1200


def all_subclasses() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for base, subs in TAXONOMY.items():
        for sub in subs:
            out.append((base, sub))
    return out


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = build_cli().parse_args(script_argv())
    if args.clear_scene:
        clear_scene()
    report: list[dict[str, Any]] = []
    col = get_collection("procedural_taxonomy_objects")

    if args.catalog_json:
        data = read_json(args.catalog_json)
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise RuntimeError("catalog JSON must contain items[]")
        items = [x for x in items if isinstance(x, dict)][: max(1, args.catalog_limit)]
        if args.setup_preview:
            setup_preview(len(items), args.grid_cols, args.spacing_x, args.spacing_y)
        for i, item in enumerate(items):
            loc = ((i % args.grid_cols) * args.spacing_x, (i // args.grid_cols) * args.spacing_y, 0.0)
            before = set(bpy.data.objects)
            build_from_catalog_item(item, loc=loc)
            created = [o for o in bpy.data.objects if o not in before]
            subclass = str(created[0].get("taxonomy_subclass", "unknown")) if created else "unknown"
            base_type = str(created[0].get("taxonomy_base_type", "unknown")) if created else "unknown"
            add_label(f"{i+1}. {base_type}\n{subclass}", (loc[0], loc[1] - 1.0, 0.05), col)
            report.append({"index": i, "mode": "catalog", "subclass": subclass, "base_type": base_type, "created_objects": len(created), "title": item.get("title")})
    elif args.build_all:
        pairs = all_subclasses()
        if args.limit and args.limit > 0:
            pairs = pairs[: args.limit]
        if args.setup_preview:
            setup_preview(len(pairs), args.grid_cols, args.spacing_x, args.spacing_y)
        for i, (base_type, subclass) in enumerate(pairs):
            loc = ((i % args.grid_cols) * args.spacing_x, (i // args.grid_cols) * args.spacing_y, 0.0)
            before = set(bpy.data.objects)
            build_taxonomy_object(subclass, base_type=base_type, color=args.color, material=args.material, loc=loc)
            created = [o for o in bpy.data.objects if o not in before]
            add_label(f"{i+1}. {base_type}\n{subclass}", (loc[0], loc[1] - 1.0, 0.05), col)
            report.append({"index": i, "mode": "taxonomy", "subclass": subclass, "base_type": base_type, "created_objects": len(created)})
    else:
        subclass = args.subclass or "l_shaped_sofa"
        base_type = args.base_type or SUBCLASS_TO_BASE.get(subclass)
        before = set(bpy.data.objects)
        build_taxonomy_object(subclass, base_type=base_type, width=args.width, depth=args.depth, height=args.height, color=args.color, material=args.material)
        created = [o for o in bpy.data.objects if o not in before]
        report.append({"index": 0, "mode": "single", "subclass": subclass, "base_type": base_type, "created_objects": len(created)})
        if args.setup_preview:
            setup_preview(1, 1, args.spacing_x, args.spacing_y)

    out = Path(args.out_blend).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))

    if args.report_json:
        write_json(args.report_json, {
            "schema": "procedural_object_factory_report/v1",
            "taxonomy_schema": "interior_object_taxonomy/v2",
            "out_blend": str(out),
            "object_batches": len(report),
            "items": report,
        })
    print(f"saved_blend = {out}")
    print(f"items = {len(report)}")


if __name__ == "__main__":
    main()
