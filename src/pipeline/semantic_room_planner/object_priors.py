from __future__ import annotations

from copy import deepcopy
from typing import Any


def _p(category: str, placement: str, dims: tuple[float, float, float], limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]], supports=None, anchors=None, front="+Y") -> dict[str, Any]:
    return {
        "category": category,
        "placement_type": placement,
        "default_dimensions_m": {"width": dims[0], "depth": dims[1], "height": dims[2]},
        "dimension_limits_m": {"width": list(limits[0]), "depth": list(limits[1]), "height": list(limits[2])},
        "front_axis_local": front,
        "valid_supports": list(supports or []),
        "default_anchors": list(anchors or []),
    }


FLOOR = ((0.2, 4.0), (0.2, 4.0), (0.05, 3.0))
SMALL = ((0.03, 0.8), (0.03, 0.8), (0.01, 0.8))
WALL = ((0.1, 3.0), (0.01, 0.25), (0.1, 2.5))

LABELS: dict[str, tuple[str, str]] = {
    "bed": ("кровать", "bed"),
    "desk": ("письменный стол", "work desk"),
    "office_chair": ("рабочее кресло", "office chair"),
    "chair": ("стул", "chair"),
    "dining_chair": ("обеденный стул", "dining chair"),
    "pillow": ("подушка", "pillow"),
    "blanket": ("одеяло", "blanket"),
    "nightstand": ("прикроватная тумба", "nightstand"),
    "table_lamp": ("настольная лампа", "table lamp"),
    "wardrobe": ("шкаф", "wardrobe"),
    "dresser": ("комод", "dresser"),
    "shelf": ("стеллаж", "shelf"),
    "bookcase": ("стеллаж", "bookcase"),
    "mug": ("кружка кофе", "coffee mug"),
    "water_bottle": ("бутылка воды", "water bottle"),
    "laptop": ("ноутбук", "laptop"),
    "monitor": ("монитор", "monitor"),
    "keyboard": ("клавиатура", "keyboard"),
    "mouse": ("компьютерная мышь", "computer mouse"),
    "notebook": ("блокнот", "notebook"),
    "desk_organizer": ("органайзер для стола", "desk organizer"),
    "plant": ("декоративное растение", "decorative plant"),
    "potted_plant": ("растение в горшке", "potted plant"),
    "small_potted_plant": ("маленькое растение в горшке", "small potted plant"),
    "hanging_planter": ("подвесное кашпо", "hanging planter"),
    "plant_stand": ("подставка для растения", "plant stand"),
    "plant_pot": ("горшок с растением", "plant pot"),
    "rug": ("ковёр", "rug"),
    "wall_art": ("настенный постер", "wall art"),
    "mirror": ("зеркало", "mirror"),
    "storage_box": ("коробка для хранения", "storage box"),
    "book": ("книга", "book"),
    "phone": ("телефон", "phone"),
    "toy_car": ("игрушечная машинка", "toy car"),
    "car_model": ("модель автомобиля", "car model"),
    "car_poster": ("постер с машиной", "car poster"),
    "racing_wall_art": ("постер с гоночной машиной", "racing car poster"),
    "racing_rug": ("ковёр с дорожным рисунком", "racing road rug"),
    "road_play_mat": ("игровой коврик дорога", "road play mat"),
    "toy_storage_box": ("коробка для машинок", "toy car storage box"),
    "storage_box_for_toys": ("коробка для машинок", "toy car storage box"),
    "car_decor": ("автомобильный декор", "car decor"),
    "kitchen_counter": ("кухонная рабочая поверхность", "kitchen counter"),
    "kitchen_cabinet": ("кухонный шкаф", "kitchen cabinet"),
    "fridge": ("холодильник", "fridge"),
    "stove": ("плита", "stove"),
    "oven": ("духовой шкаф", "oven"),
    "range_hood": ("вытяжка", "range hood"),
    "kitchen_sink": ("кухонная мойка", "kitchen sink"),
    "kettle": ("чайник", "kettle"),
    "cutting_board": ("разделочная доска", "cutting board"),
    "fruit_bowl": ("ваза с фруктами", "fruit bowl"),
    "cookbook": ("кулинарная книга", "cookbook"),
    "pan": ("сковорода", "pan"),
    "pot": ("кастрюля", "pot"),
    "towel": ("полотенце", "towel"),
    "hand_towel": ("полотенце для рук", "hand towel"),
    "bath_mat": ("коврик для ванной", "bath mat"),
    "towel_rack": ("полотенцедержатель", "towel rack"),
    "toilet_paper_holder": ("держатель туалетной бумаги", "toilet paper holder"),
    "toilet_brush": ("ершик для туалета", "toilet brush"),
    "laundry_basket": ("корзина для белья", "laundry basket"),
    "shampoo_bottle": ("бутылка шампуня", "shampoo bottle"),
}

STYLE_DEFAULTS: dict[str, dict[str, str]] = {
    "bed": {"color": "beige fabric", "material": "fabric upholstery, wood"},
    "desk": {"color": "light oak with black metal", "material": "wood veneer, powder-coated metal"},
    "office_chair": {"color": "black", "material": "fabric, plastic, metal"},
    "chair": {"color": "black", "material": "wood, fabric"},
    "dining_chair": {"color": "black", "material": "wood, fabric"},
    "wardrobe": {"color": "warm white", "material": "laminated board"},
    "dresser": {"color": "warm white and light oak", "material": "laminated board, wood veneer"},
    "shelf": {"color": "light oak", "material": "wood veneer"},
    "bookcase": {"color": "light oak", "material": "wood veneer"},
    "nightstand": {"color": "light oak", "material": "wood"},
    "table_lamp": {"color": "black metal, warm white shade", "material": "metal, fabric"},
    "mug": {"color": "white", "material": "ceramic"},
    "water_bottle": {"color": "transparent blue", "material": "plastic"},
    "plant": {"color": "green leaves, white pot", "material": "ceramic, plant"},
    "potted_plant": {"color": "green leaves, terracotta pot", "material": "plant, soil, ceramic pot"},
    "small_potted_plant": {"color": "green leaves, white ceramic pot", "material": "plant, soil, ceramic pot"},
    "hanging_planter": {"color": "green trailing leaves, white hanging pot", "material": "plant, soil, ceramic pot, cord"},
    "plant_stand": {"color": "black metal", "material": "powder-coated metal"},
    "plant_pot": {"color": "green leaves, terracotta pot", "material": "plant, soil, ceramic pot"},
    "wall_art": {"color": "muted green and beige", "material": "paper print, thin frame"},
    "pillow": {"color": "warm white", "material": "cotton textile"},
    "blanket": {"color": "beige", "material": "soft textile"},
    "rug": {"color": "muted green and beige", "material": "woven textile"},
    "laptop": {"color": "silver", "material": "aluminum, glass"},
    "monitor": {"color": "black", "material": "plastic, glass"},
    "keyboard": {"color": "black", "material": "plastic"},
    "mouse": {"color": "black", "material": "plastic"},
    "notebook": {"color": "warm white", "material": "paper"},
    "desk_organizer": {"color": "black", "material": "metal mesh"},
    "storage_box": {"color": "beige fabric", "material": "fabric, cardboard"},
    "mirror": {"color": "black thin frame", "material": "glass, metal"},
    "book": {"color": "muted green cover", "material": "paper"},
    "phone": {"color": "black", "material": "glass, metal"},
    "toy_car": {"color": "red and blue", "material": "painted metal, plastic"},
    "car_model": {"color": "red", "material": "painted metal"},
    "car_poster": {"color": "muted green, beige, racing red", "material": "paper print, thin frame"},
    "racing_wall_art": {"color": "muted green, beige, racing red", "material": "paper print, thin frame"},
    "racing_rug": {"color": "gray road pattern with muted green", "material": "woven textile"},
    "road_play_mat": {"color": "gray road pattern with muted green", "material": "soft play textile"},
    "toy_storage_box": {"color": "warm white with racing accent", "material": "fabric, cardboard"},
    "storage_box_for_toys": {"color": "warm white with racing accent", "material": "fabric, cardboard"},
    "car_decor": {"color": "racing red accent", "material": "painted wood, plastic"},
    "kitchen_counter": {"color": "light oak and warm white", "material": "laminated board, composite worktop"},
    "kitchen_cabinet": {"color": "warm white", "material": "laminated board"},
    "fridge": {"color": "white", "material": "painted metal, plastic"},
    "stove": {"color": "black glass", "material": "glass, metal"},
    "oven": {"color": "black glass and steel", "material": "glass, metal"},
    "range_hood": {"color": "steel", "material": "metal"},
    "kitchen_sink": {"color": "stainless steel", "material": "metal"},
    "kettle": {"color": "black", "material": "metal, plastic"},
    "cutting_board": {"color": "light wood", "material": "wood"},
    "fruit_bowl": {"color": "white bowl with fruit colors", "material": "ceramic, fruit"},
    "cookbook": {"color": "warm white cover", "material": "paper"},
    "pan": {"color": "black", "material": "metal"},
    "pot": {"color": "steel", "material": "metal"},
    "towel": {"color": "warm white", "material": "cotton textile"},
    "hand_towel": {"color": "warm white", "material": "cotton textile"},
    "bath_mat": {"color": "beige", "material": "cotton textile"},
    "towel_rack": {"color": "black metal", "material": "metal"},
    "toilet_paper_holder": {"color": "white and chrome", "material": "paper, metal"},
    "toilet_brush": {"color": "white", "material": "plastic, metal"},
    "laundry_basket": {"color": "beige woven", "material": "fabric, wicker"},
    "shampoo_bottle": {"color": "muted green", "material": "plastic"},
}


CATEGORY_BY_SUBCLASS: dict[str, str] = {
    **{k: "furniture" for k in {"bed", "desk", "office_chair", "chair", "dining_chair", "wardrobe", "dresser", "shelf", "bookcase", "nightstand", "sofa", "armchair", "coffee_table", "side_table", "dining_table", "kitchen_table", "tv_stand", "kitchen_counter", "kitchen_cabinet"}},
    **{k: "appliance" for k in {"fridge", "stove", "oven", "range_hood", "kitchen_sink"}},
    **{k: "electronics" for k in {"laptop", "monitor", "keyboard", "mouse", "phone", "tv", "tv_projector_screen", "remote"}},
    **{k: "textile" for k in {"pillow", "blanket", "rug", "racing_rug", "road_play_mat", "towel", "hand_towel", "bath_mat"}},
    **{k: "decor_accessory" for k in {"mug", "cup", "water_bottle", "book", "notebook", "plant", "potted_plant", "small_potted_plant", "hanging_planter", "plant_stand", "plant_pot", "vase", "wall_art", "car_poster", "racing_wall_art", "toy_car", "car_model", "car_decor", "mirror", "plate", "bowl", "desk_organizer", "kettle", "cutting_board", "fruit_bowl", "cookbook", "pan", "pot", "shampoo_bottle"}},
    **{k: "storage" for k in {"storage_box", "toy_storage_box", "storage_box_for_toys"}},
    **{k: "lighting" for k in {"table_lamp", "desk_lamp", "floor_lamp"}},
    **{k: "sanitary" for k in {"toilet", "sink", "bathtub", "shower", "soap_dispenser", "toothbrush_cup", "towel_rack", "toilet_paper_holder", "toilet_brush", "laundry_basket"}},
}


subclass_priors: dict[str, dict[str, Any]] = {
    "desk": _p("furniture", "floor", (1.2, 0.6, 0.75), ((0.7, 2.0), (0.4, 1.0), (0.65, 0.9))),
    "office_chair": _p("furniture", "floor", (0.55, 0.55, 0.9), ((0.4, 0.8), (0.4, 0.8), (0.7, 1.2))),
    "chair": _p("furniture", "floor", (0.45, 0.5, 0.85), ((0.35, 0.7), (0.35, 0.7), (0.7, 1.1))),
    "dining_chair": _p("furniture", "floor", (0.45, 0.5, 0.85), ((0.35, 0.7), (0.35, 0.7), (0.7, 1.1))),
    "bed": _p("furniture", "floor", (1.4, 2.0, 0.55), ((0.8, 2.0), (1.7, 2.3), (0.35, 0.9))),
    "nightstand": _p("furniture", "floor", (0.45, 0.4, 0.55), ((0.3, 0.8), (0.25, 0.7), (0.35, 0.8))),
    "wardrobe": _p("furniture", "floor", (1.2, 0.6, 2.1), ((0.6, 2.5), (0.35, 0.8), (1.5, 2.6))),
    "dresser": _p("furniture", "floor", (1.0, 0.45, 0.9), ((0.5, 1.8), (0.3, 0.7), (0.6, 1.4))),
    "shelf": _p("furniture", "floor", (0.9, 0.35, 1.8), ((0.4, 2.0), (0.25, 0.6), (0.8, 2.5))),
    "bookcase": _p("furniture", "floor", (0.9, 0.35, 1.8), ((0.4, 2.0), (0.25, 0.6), (0.8, 2.5))),
    "sofa": _p("furniture", "floor", (1.8, 0.85, 0.85), ((1.0, 3.0), (0.65, 1.1), (0.65, 1.1))),
    "armchair": _p("furniture", "floor", (0.8, 0.85, 0.85), ((0.55, 1.1), (0.55, 1.1), (0.65, 1.1))),
    "coffee_table": _p("furniture", "floor", (0.9, 0.55, 0.4), ((0.45, 1.5), (0.35, 0.9), (0.25, 0.55))),
    "side_table": _p("furniture", "floor", (0.45, 0.45, 0.5), ((0.3, 0.8), (0.3, 0.8), (0.35, 0.75))),
    "dining_table": _p("furniture", "floor", (1.2, 0.8, 0.75), ((0.7, 2.2), (0.6, 1.4), (0.65, 0.9))),
    "kitchen_table": _p("furniture", "floor", (1.0, 0.7, 0.75), ((0.6, 1.8), (0.5, 1.2), (0.65, 0.9))),
    "kitchen_counter": _p("furniture", "floor", (1.8, 0.6, 0.9), ((0.8, 3.2), (0.45, 0.75), (0.85, 0.95))),
    "kitchen_cabinet": _p("furniture", "floor", (1.2, 0.45, 2.1), ((0.5, 2.4), (0.3, 0.65), (1.4, 2.5))),
    "fridge": _p("appliance", "floor", (0.65, 0.65, 1.85), ((0.5, 0.85), (0.5, 0.85), (1.4, 2.2))),
    "stove": _p("appliance", "support", (0.6, 0.52, 0.08), ((0.45, 0.8), (0.35, 0.7), (0.04, 0.18)), supports=["kitchen_counter"]),
    "oven": _p("appliance", "floor", (0.6, 0.55, 0.75), ((0.45, 0.8), (0.4, 0.7), (0.55, 0.9))),
    "range_hood": _p("appliance", "wall", (0.65, 0.25, 0.35), WALL, front="-Y"),
    "kitchen_sink": _p("appliance", "support", (0.55, 0.42, 0.12), ((0.4, 0.8), (0.3, 0.6), (0.08, 0.2)), supports=["kitchen_counter"]),
    "tv": _p("electronics", "wall", (1.0, 0.06, 0.6), WALL, front="-Y"),
    "tv_projector_screen": _p("electronics", "wall", (1.6, 0.05, 0.9), WALL, front="-Y"),
    "tv_stand": _p("furniture", "floor", (1.3, 0.4, 0.45), ((0.7, 2.2), (0.3, 0.7), (0.25, 0.7))),
    "floor_lamp": _p("lighting", "floor", (0.35, 0.35, 1.55), ((0.2, 0.6), (0.2, 0.6), (1.0, 2.0))),
    "rug": _p("decor", "floor", (1.6, 1.1, 0.03), ((0.5, 3.5), (0.5, 3.0), (0.005, 0.08))),
    "mirror": _p("decor", "wall", (0.6, 0.04, 0.9), WALL, front="-Y"),
    "wall_art": _p("decor", "wall", (0.7, 0.04, 0.5), WALL, front="-Y"),
    "plant": _p("decor", "floor", (0.35, 0.35, 0.75), ((0.12, 0.8), (0.12, 0.8), (0.15, 1.6))),
    "potted_plant": _p("decor", "floor", (0.38, 0.38, 0.85), ((0.16, 0.8), (0.16, 0.8), (0.2, 1.6))),
    "plant_stand": _p("decor", "floor", (0.36, 0.36, 0.7), ((0.25, 0.65), (0.25, 0.65), (0.4, 1.1))),
    "hanging_planter": _p("decor", "wall", (0.28, 0.18, 0.45), WALL, front="-Y"),
    "storage_box": _p("storage", "support", (0.34, 0.28, 0.22), ((0.18, 0.7), (0.16, 0.6), (0.12, 0.45)), supports=["wardrobe", "shelf", "bookcase", "dresser"]),
    "toy_storage_box": _p("storage", "support", (0.36, 0.3, 0.24), ((0.2, 0.7), (0.18, 0.6), (0.12, 0.45)), supports=["wardrobe", "shelf", "bookcase", "dresser"]),
    "storage_box_for_toys": _p("storage", "support", (0.36, 0.3, 0.24), ((0.2, 0.7), (0.18, 0.6), (0.12, 0.45)), supports=["wardrobe", "shelf", "bookcase", "dresser"]),
    "toilet": _p("sanitary", "floor", (0.38, 0.65, 0.78), ((0.3, 0.6), (0.5, 0.9), (0.6, 1.0))),
    "sink": _p("sanitary", "floor", (0.55, 0.45, 0.85), ((0.35, 0.9), (0.3, 0.7), (0.7, 1.0))),
    "bathtub": _p("sanitary", "floor", (1.7, 0.75, 0.6), ((1.2, 2.0), (0.6, 0.95), (0.45, 0.8))),
    "shower": _p("sanitary", "floor", (0.9, 0.9, 2.0), ((0.7, 1.2), (0.7, 1.2), (1.8, 2.4))),
    "towel_rack": _p("sanitary", "wall", (0.55, 0.05, 0.16), WALL, front="-Y"),
    "laundry_basket": _p("sanitary", "floor", (0.38, 0.38, 0.55), ((0.25, 0.7), (0.25, 0.7), (0.35, 0.8))),
}

for name, dims in {
    "laptop": (0.32, 0.22, 0.03), "monitor": (0.55, 0.18, 0.42), "keyboard": (0.42, 0.14, 0.03),
    "mouse": (0.07, 0.11, 0.04), "mug": (0.09, 0.09, 0.1), "cup": (0.08, 0.08, 0.1),
    "water_bottle": (0.08, 0.08, 0.25), "book": (0.22, 0.16, 0.04), "notebook": (0.24, 0.18, 0.03),
    "phone": (0.08, 0.16, 0.01), "remote": (0.05, 0.18, 0.03), "table_lamp": (0.25, 0.25, 0.45),
    "pillow": (0.55, 0.35, 0.12), "blanket": (1.2, 1.6, 0.08), "vase": (0.16, 0.16, 0.3),
    "plate": (0.25, 0.25, 0.03), "bowl": (0.18, 0.18, 0.08), "soap_dispenser": (0.08, 0.08, 0.18),
    "toothbrush_cup": (0.08, 0.08, 0.12), "desk_lamp": (0.25, 0.25, 0.45), "desk_organizer": (0.25, 0.16, 0.12),
    "small_potted_plant": (0.18, 0.18, 0.28), "plant_pot": (0.2, 0.2, 0.32),
    "kettle": (0.22, 0.18, 0.24), "cutting_board": (0.32, 0.22, 0.03), "fruit_bowl": (0.28, 0.28, 0.14),
    "cookbook": (0.25, 0.19, 0.04), "pan": (0.28, 0.28, 0.08), "pot": (0.26, 0.26, 0.18),
    "towel": (0.36, 0.18, 0.04), "hand_towel": (0.28, 0.16, 0.03), "bath_mat": (0.75, 0.5, 0.025),
    "toilet_paper_holder": (0.16, 0.12, 0.14), "toilet_brush": (0.12, 0.12, 0.45), "shampoo_bottle": (0.08, 0.08, 0.22),
}.items():
    placement = "floor" if name == "bath_mat" else "support"
    subclass_priors[name] = _p("accessory" if name not in {"monitor", "laptop", "keyboard", "mouse", "table_lamp", "desk_lamp"} else "electronics", placement, dims, SMALL, supports=["desk", "nightstand", "dresser", "shelf", "bed", "sink", "kitchen_counter", "dining_table", "coffee_table", "kitchen_table"])

for name, dims in {
    "toy_car": (0.12, 0.06, 0.05),
    "car_model": (0.16, 0.07, 0.06),
    "car_decor": (0.18, 0.08, 0.08),
}.items():
    subclass_priors[name] = _p("decor_accessory", "support", dims, SMALL, supports=["desk", "shelf", "bookcase", "storage_box", "toy_storage_box"])

subclass_priors["car_poster"] = _p("decor_accessory", "wall", (0.7, 0.04, 0.5), WALL, front="-Y")
subclass_priors["racing_wall_art"] = _p("decor_accessory", "wall", (0.7, 0.04, 0.5), WALL, front="-Y")
subclass_priors["racing_rug"] = _p("textile", "floor", (1.5, 1.0, 0.03), ((0.6, 3.2), (0.5, 2.6), (0.005, 0.08)))
subclass_priors["road_play_mat"] = _p("textile", "floor", (1.4, 0.9, 0.025), ((0.6, 3.0), (0.5, 2.4), (0.005, 0.08)))

subclass_priors["pillow"] = _p("decor", "support", (0.55, 0.35, 0.12), ((0.25, 0.8), (0.18, 0.6), (0.05, 0.25)), supports=["bed", "sofa"])
subclass_priors["blanket"] = _p("decor", "support", (1.2, 1.6, 0.08), ((0.6, 2.0), (0.6, 2.2), (0.02, 0.18)), supports=["bed", "sofa"])


ALIASES = {
    "desk_lamp": "table_lamp",
    "computer_chair": "office_chair",
    "bookshelf": "bookcase",
    "bottle": "water_bottle",
    "small_plant": "plant",
    "decorative_plant": "potted_plant",
    "plant": "potted_plant",
    "orchid": "potted_plant",
    "desktop_plant": "small_potted_plant",
    "tabletop_plant": "small_potted_plant",
    "hanging_plant": "hanging_planter",
    "car_wall_art": "car_poster",
    "poster_car": "car_poster",
    "storage_box_for_toys": "toy_storage_box",
    "counter": "kitchen_counter",
    "kitchen_countertop": "kitchen_counter",
    "refrigerator": "fridge",
    "hob": "stove",
    "cooktop": "stove",
    "range": "stove",
    "kitchen_sink": "kitchen_sink",
    "bathroom_sink": "sink",
}


def normalize_subclass(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return ALIASES.get(key, key if key in subclass_priors else key)


def get_prior(subclass: str) -> dict[str, Any]:
    return deepcopy(subclass_priors.get(normalize_subclass(subclass), _p("decor", "floor", (0.4, 0.4, 0.4), FLOOR)))


def default_labels(subclass: str) -> tuple[str, str]:
    key = normalize_subclass(subclass)
    return LABELS.get(key, (key.replace("_", " "), key.replace("_", " ")))


def default_style(subclass: str) -> dict[str, str]:
    return deepcopy(STYLE_DEFAULTS.get(normalize_subclass(subclass), {"color": "warm white", "material": "mixed material"}))


def default_category(subclass: str) -> str:
    key = normalize_subclass(subclass)
    return CATEGORY_BY_SUBCLASS.get(key, get_prior(key).get("category", "decor_accessory"))
