from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Density = Literal["normal", "high", "very_high"]


@dataclass(frozen=True)
class ObjectSpec:
    category: str
    name: str
    size_m: tuple[float, float, float]
    layer: str
    mount_type: str = "floor"
    replace_with_supplier: bool = True
    allow_collision: bool = False
    requires_access: bool = False
    support_surface: bool = False
    front_target_hint: str | None = None


BEDROOM_SPECS: dict[str, ObjectSpec] = {
    "single_bed": ObjectSpec("bed", "Single bed", (0.9, 2.0, 0.55), "primary", requires_access=True, support_surface=True, front_target_hint="room_center"),
    "compact_double_bed": ObjectSpec("bed", "One-and-a-half bed", (1.4, 2.0, 0.55), "primary", requires_access=True, support_surface=True, front_target_hint="room_center"),
    "double_bed": ObjectSpec("bed", "Double bed", (1.8, 2.05, 0.55), "primary", requires_access=True, support_surface=True, front_target_hint="room_center"),
    "queen_bed": ObjectSpec("bed", "Large double bed", (1.8, 2.1, 0.55), "primary", requires_access=True, support_surface=True, front_target_hint="room_center"),
    "headboard": ObjectSpec("headboard", "Headboard", (1.8, 0.08, 1.1), "secondary"),
    "nightstand": ObjectSpec("nightstand", "Nightstand", (0.45, 0.42, 0.55), "secondary", support_surface=True),
    "nightstand_compact": ObjectSpec("nightstand", "Compact nightstand", (0.32, 0.34, 0.52), "secondary", support_surface=True),
    "wardrobe_module": ObjectSpec("wardrobe_module", "Wardrobe module", (0.6, 0.62, 2.25), "storage", requires_access=True, front_target_hint="room_center"),
    "wardrobe_narrow": ObjectSpec("wardrobe_module", "Narrow wardrobe module", (0.48, 0.48, 2.1), "storage", requires_access=True, front_target_hint="room_center"),
    "dresser": ObjectSpec("dresser", "Dresser", (1.15, 0.45, 0.85), "storage", support_surface=True, requires_access=True, front_target_hint="room_center"),
    "console_dresser": ObjectSpec("dresser", "Small console dresser", (0.85, 0.32, 0.82), "storage", support_surface=True, requires_access=True, front_target_hint="room_center"),
    "desk": ObjectSpec("desk", "Writing desk", (1.15, 0.58, 0.75), "secondary", support_surface=True, requires_access=True, front_target_hint="chair"),
    "chair": ObjectSpec("chair", "Chair", (0.5, 0.5, 0.85), "secondary"),
    "bench": ObjectSpec("bench", "Bed bench", (1.2, 0.42, 0.45), "secondary"),
    "stool_pouf": ObjectSpec("stool", "Round pouf stool", (0.46, 0.46, 0.42), "secondary", support_surface=True),
    "rug": ObjectSpec("rug", "Large bedroom rug", (1.8, 2.5, 0.03), "textile", replace_with_supplier=True, allow_collision=True),
    "table_lamp": ObjectSpec("table_lamp", "Table lamp", (0.28, 0.28, 0.55), "lighting", allow_collision=True),
    "floor_lamp": ObjectSpec("floor_lamp", "Floor lamp", (0.35, 0.35, 1.55), "lighting"),
    "plant": ObjectSpec("plant", "Indoor plant", (0.42, 0.42, 1.2), "decor"),
    "wall_art": ObjectSpec("wall_art", "Wall art", (0.8, 0.05, 0.6), "wall_decor", mount_type="wall", allow_collision=True),
    "mural": ObjectSpec("wall_art", "Soft Italian garden mural", (2.15, 0.04, 1.45), "wall_decor", mount_type="wall", allow_collision=True, replace_with_supplier=False),
    "curtains": ObjectSpec("curtain", "Pale green curtains", (2.2, 0.06, 2.15), "wall_decor", mount_type="wall", allow_collision=True, replace_with_supplier=False),
    "sconce": ObjectSpec("wall_light", "Wall sconce", (0.22, 0.08, 0.32), "lighting", mount_type="wall", allow_collision=True),
    "pillow": ObjectSpec("pillow", "Decorative pillow", (0.45, 0.16, 0.18), "soft_decor", allow_collision=True),
    "blanket": ObjectSpec("blanket", "Bed blanket", (1.4, 1.0, 0.06), "soft_decor", allow_collision=True),
    "decor_books": ObjectSpec("decor_books", "Decorative books", (0.28, 0.2, 0.08), "decor", allow_collision=True),
    "decor_vase": ObjectSpec("decor_vase", "Decorative vase", (0.18, 0.18, 0.32), "decor", allow_collision=True),
    "decor_box": ObjectSpec("decor_box", "Storage box", (0.32, 0.24, 0.16), "decor", allow_collision=True),
}

LIVING_ROOM_SPECS: dict[str, ObjectSpec] = {
    "sofa_2": ObjectSpec("sofa", "Two-seat sofa", (1.65, 0.88, 0.85), "primary", requires_access=True, support_surface=True, front_target_hint="tv"),
    "sofa_3": ObjectSpec("sofa", "Three-seat sofa", (2.25, 0.92, 0.85), "primary", requires_access=True, support_surface=True, front_target_hint="tv"),
    "sectional_sofa": ObjectSpec("sofa", "Sectional sofa", (2.65, 1.65, 0.85), "primary", requires_access=True, support_surface=True, front_target_hint="tv"),
    "armchair": ObjectSpec("armchair", "Armchair", (0.82, 0.82, 0.9), "secondary"),
    "coffee_table": ObjectSpec("coffee_table", "Coffee table", (1.1, 0.62, 0.42), "primary", support_surface=True),
    "side_table": ObjectSpec("side_table", "Side table", (0.45, 0.45, 0.55), "secondary", support_surface=True),
    "tv_stand": ObjectSpec("tv_stand", "TV stand", (1.65, 0.42, 0.52), "primary", support_surface=True),
    "tv": ObjectSpec("tv", "TV", (1.25, 0.06, 0.72), "electronics", mount_type="wall", allow_collision=True),
    "bookshelf": ObjectSpec("bookshelf", "Bookshelf", (0.82, 0.36, 2.05), "storage", support_surface=True, requires_access=True),
    "console_table": ObjectSpec("console_table", "Console table", (1.15, 0.35, 0.82), "secondary", support_surface=True),
    "rug": ObjectSpec("rug", "Living room rug", (2.1, 2.8, 0.03), "textile", allow_collision=True),
    "floor_lamp": ObjectSpec("floor_lamp", "Floor lamp", (0.38, 0.38, 1.65), "lighting"),
    "table_lamp": ObjectSpec("table_lamp", "Table lamp", (0.28, 0.28, 0.55), "lighting", allow_collision=True),
    "plant_large": ObjectSpec("plant", "Large indoor plant", (0.52, 0.52, 1.45), "decor"),
    "plant_small": ObjectSpec("plant", "Small indoor plant", (0.25, 0.25, 0.45), "decor", allow_collision=True),
    "wall_art": ObjectSpec("wall_art", "Wall art", (0.9, 0.05, 0.65), "wall_decor", mount_type="wall", allow_collision=True),
    "decor_books": ObjectSpec("decor_books", "Coffee table books", (0.32, 0.24, 0.08), "decor", allow_collision=True),
    "decor_vase": ObjectSpec("decor_vase", "Decorative vase", (0.2, 0.2, 0.34), "decor", allow_collision=True),
    "decor_tray": ObjectSpec("decor_tray", "Decorative tray", (0.42, 0.28, 0.05), "decor", allow_collision=True),
    "pillow": ObjectSpec("pillow", "Sofa pillow", (0.42, 0.15, 0.18), "soft_decor", allow_collision=True),
    "blanket": ObjectSpec("blanket", "Sofa blanket", (0.8, 0.5, 0.06), "soft_decor", allow_collision=True),
    "dining_table": ObjectSpec("dining_table", "Dining table", (1.4, 0.85, 0.75), "secondary", requires_access=True, support_surface=True),
    "dining_chair": ObjectSpec("dining_chair", "Dining chair", (0.48, 0.52, 0.9), "secondary"),
}

CORRIDOR_SPECS: dict[str, ObjectSpec] = {
    "shoe_cabinet": ObjectSpec("shoe_cabinet", "Shoe cabinet", (0.85, 0.32, 0.9), "storage", support_surface=True),
    "bench": ObjectSpec("entry_bench", "Entry bench", (1.0, 0.36, 0.45), "secondary"),
    "coat_rack": ObjectSpec("coat_rack", "Coat rack", (0.65, 0.35, 1.8), "storage"),
    "wardrobe_narrow": ObjectSpec("wardrobe", "Narrow wardrobe", (0.85, 0.48, 2.2), "storage"),
    "mirror": ObjectSpec("mirror", "Full-height mirror", (0.65, 0.05, 1.8), "wall_decor", mount_type="wall", allow_collision=True),
    "console_table": ObjectSpec("console_table", "Narrow console table", (0.95, 0.30, 0.8), "secondary", support_surface=True),
    "runner_rug": ObjectSpec("runner_rug", "Runner rug", (0.75, 2.6, 0.03), "textile", allow_collision=True),
    "umbrella_stand": ObjectSpec("umbrella_stand", "Umbrella stand", (0.25, 0.25, 0.55), "decor"),
    "wall_hooks": ObjectSpec("wall_hooks", "Wall hooks", (0.8, 0.05, 0.25), "wall_decor", mount_type="wall", allow_collision=True),
    "wall_art": ObjectSpec("wall_art", "Wall art", (0.55, 0.05, 0.45), "wall_decor", mount_type="wall", allow_collision=True),
    "storage_basket": ObjectSpec("storage_basket", "Storage basket", (0.34, 0.28, 0.28), "decor", allow_collision=True),
    "key_tray": ObjectSpec("decor_tray", "Key tray", (0.30, 0.18, 0.05), "decor", allow_collision=True),
    "small_plant": ObjectSpec("plant", "Small plant", (0.23, 0.23, 0.42), "decor", allow_collision=True),
}

BATHROOM_SPECS: dict[str, ObjectSpec] = {
    "bathtub": ObjectSpec("bathtub", "Bathtub", (1.65, 0.72, 0.58), "primary", requires_access=True),
    "compact_bathtub": ObjectSpec("bathtub", "Compact bathtub", (1.35, 0.68, 0.58), "primary", requires_access=True),
    "shower": ObjectSpec("shower", "Shower cabin", (0.9, 0.9, 2.1), "primary", requires_access=True),
    "compact_shower": ObjectSpec("shower", "Compact shower cabin", (0.75, 0.75, 2.05), "primary", requires_access=True),
    "sink": ObjectSpec("sink", "Bathroom sink", (0.62, 0.46, 0.85), "primary", requires_access=True, support_surface=True, front_target_hint="door"),
    "compact_sink": ObjectSpec("sink", "Compact bathroom sink", (0.45, 0.34, 0.82), "primary", requires_access=True, support_surface=True, front_target_hint="door"),
    "vanity": ObjectSpec("vanity", "Bathroom vanity", (0.75, 0.48, 0.86), "storage", requires_access=True, support_surface=True),
    "toilet": ObjectSpec("toilet", "Toilet", (0.38, 0.68, 0.78), "primary", requires_access=True, front_target_hint="door"),
    "compact_toilet": ObjectSpec("toilet", "Compact toilet", (0.36, 0.60, 0.76), "primary", requires_access=True, front_target_hint="door"),
    "washing_machine": ObjectSpec("washing_machine", "Washing machine", (0.6, 0.6, 0.85), "appliance", requires_access=True, support_surface=True),
    "towel_rack": ObjectSpec("towel_rack", "Towel rack", (0.55, 0.06, 0.18), "wall_decor", mount_type="wall", allow_collision=True),
    "mirror": ObjectSpec("mirror", "Bathroom mirror", (0.6, 0.05, 0.8), "wall_decor", mount_type="wall", allow_collision=True),
    "wall_shelf": ObjectSpec("shelf", "Wall shelf", (0.55, 0.16, 0.12), "wall_decor", mount_type="wall", allow_collision=True, support_surface=True),
    "laundry_basket": ObjectSpec("laundry_basket", "Laundry basket", (0.38, 0.38, 0.58), "decor"),
    "bath_mat": ObjectSpec("bath_mat", "Bath mat", (0.75, 0.5, 0.03), "textile", allow_collision=True),
    "soap_dispenser": ObjectSpec("soap_dispenser", "Soap dispenser", (0.08, 0.08, 0.18), "decor", allow_collision=True),
    "toothbrush_cup": ObjectSpec("toothbrush_cup", "Toothbrush cup", (0.09, 0.09, 0.12), "decor", allow_collision=True),
    "shampoo_bottle": ObjectSpec("shampoo_bottle", "Shampoo bottle", (0.08, 0.08, 0.22), "decor", allow_collision=True),
}

TOILET_SPECS: dict[str, ObjectSpec] = {
    "toilet": ObjectSpec("toilet", "Toilet", (0.38, 0.68, 0.78), "primary", requires_access=True, front_target_hint="door"),
    "compact_toilet": ObjectSpec("toilet", "Compact toilet", (0.36, 0.60, 0.76), "primary", requires_access=True, front_target_hint="door"),
    "sink": ObjectSpec("sink", "Compact handwash sink", (0.42, 0.32, 0.82), "primary", requires_access=True, support_surface=True, front_target_hint="door"),
    "corner_sink": ObjectSpec("sink", "Corner handwash sink", (0.34, 0.34, 0.82), "primary", requires_access=True, support_surface=True, front_target_hint="door"),
    "toilet_paper_holder": ObjectSpec("toilet_paper_holder", "Toilet paper holder", (0.18, 0.06, 0.16), "wall_decor", mount_type="wall", allow_collision=True),
    "hygiene_shower": ObjectSpec("hygiene_shower", "Hygiene shower", (0.12, 0.06, 0.22), "wall_decor", mount_type="wall", allow_collision=True),
    "mirror": ObjectSpec("mirror", "Small mirror", (0.45, 0.04, 0.55), "wall_decor", mount_type="wall", allow_collision=True),
    "wall_shelf": ObjectSpec("shelf", "Small wall shelf", (0.45, 0.14, 0.12), "wall_decor", mount_type="wall", allow_collision=True, support_surface=True),
    "air_freshener": ObjectSpec("air_freshener", "Air freshener", (0.08, 0.08, 0.18), "decor", allow_collision=True),
    "small_bin": ObjectSpec("small_bin", "Small bin", (0.24, 0.24, 0.32), "decor"),
    "soap_dispenser": ObjectSpec("soap_dispenser", "Soap dispenser", (0.08, 0.08, 0.18), "decor", allow_collision=True),
}


def normalize_density(value: str | None) -> Density:
    raw = (value or "high").strip().lower().replace("-", "_")
    if raw in {"low", "minimal", "normal"}:
        return "normal"
    if raw in {"very_high", "veryhigh", "dense", "max", "maximum"}:
        return "very_high"
    return "high"


def density_rank(density: Density) -> int:
    return {"normal": 1, "high": 2, "very_high": 3}[density]
