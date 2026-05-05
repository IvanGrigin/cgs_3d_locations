from __future__ import annotations

SELECTION_MODES: tuple[str, ...] = ("cheapest", "optimal", "best_match")

KITCHEN_ROLES: tuple[str, ...] = (
    "facade_sheet",
    "board_sheet",
    "countertop_slab",
    "premium_countertop_slab",
    "backsplash_panel",
    "edge_band",
    "accent_edge_band",
    "countertop_wall_plinth",
    "joint_profile",
    "end_profile",
    "corner_profile",
    "ventilation_grille",
    "unknown",
)

KITCHEN_DIMENSIONS_MM: dict[str, int] = {
    "base_depth": 560,
    "countertop_depth": 600,
    "base_body_height": 720,
    "plinth_height": 100,
    "countertop_thickness": 38,
    "upper_depth": 320,
    "upper_height": 720,
    "hood_cabinet_height": 360,
    "backsplash_height": 600,
    "backsplash_thickness": 4,
    "fridge_width": 600,
    "fridge_depth": 650,
    "fridge_height": 1900,
    "appliance_width": 600,
    "entry_handwash_width": 400,
    "entry_handwash_depth": 400,
    "minimum_storage_width": 300,
}

STORAGE_FILL_WIDTHS_MM: tuple[int, ...] = (800, 600, 500, 450, 400, 300)
DISHWASHER_WIDTHS_MM: tuple[int, ...] = (600, 450)

MATERIAL_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "facade": ("facade_sheet", "board_sheet"),
    "body": ("board_sheet", "facade_sheet"),
    "countertop": ("countertop_slab", "premium_countertop_slab"),
    "backsplash": ("backsplash_panel",),
    "edge_band": ("edge_band", "accent_edge_band"),
    "wall_plinth": ("countertop_wall_plinth",),
    "joint_profile": ("joint_profile",),
    "end_profile": ("end_profile",),
    "corner_profile": ("corner_profile",),
}

ROLE_PRICE_DEFAULTS_RUB: dict[str, float] = {
    "facade_sheet": 7800.0,
    "board_sheet": 5200.0,
    "countertop_slab": 8500.0,
    "premium_countertop_slab": 14000.0,
    "backsplash_panel": 4500.0,
    "edge_band": 35.0,
    "accent_edge_band": 90.0,
    "countertop_wall_plinth": 1500.0,
    "joint_profile": 900.0,
    "end_profile": 650.0,
    "corner_profile": 650.0,
    "ventilation_grille": 700.0,
    "unknown": 1000.0,
}

MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "cheapest": {
        "role_score": 0.30,
        "color_score": 0.15,
        "style_score": 0.10,
        "pattern_score": 0.05,
        "finish_score": 0.05,
        "dimension_score": 0.10,
        "durability_score": 0.05,
        "availability_score": 0.05,
        "price_score": 0.25,
        "compatibility_score": 0.00,
    },
    "optimal": {
        "role_score": 0.22,
        "color_score": 0.20,
        "style_score": 0.14,
        "pattern_score": 0.12,
        "finish_score": 0.08,
        "dimension_score": 0.08,
        "durability_score": 0.06,
        "availability_score": 0.04,
        "price_score": 0.06,
        "compatibility_score": 0.10,
    },
    "best_match": {
        "role_score": 0.12,
        "color_score": 0.24,
        "style_score": 0.20,
        "pattern_score": 0.18,
        "finish_score": 0.12,
        "dimension_score": 0.04,
        "durability_score": 0.04,
        "availability_score": 0.02,
        "price_score": 0.00,
        "compatibility_score": 0.14,
    },
}

COLOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "white": ("white", "белый", "белая", "крем", "кремовый", "ivory", "айвори", "молочный", "warm white"),
    "beige": ("beige", "беж", "бежевый", "песочный", "sand", "taupe", "тауп", "linen", "лен"),
    "gray": ("gray", "grey", "серый", "серо", "графит", "graphite", "платина", "антрацит", "anthracite"),
    "black": ("black", "черный", "черная", "черное", "чёрный", "чёрная", "чёрное", "charcoal", "угольный"),
    "brown": ("brown", "коричневый", "шоколад", "каштан", "табак", "орех", "walnut", "венге"),
    "light_wood": ("light oak", "дуб светлый", "светлый дуб", "дуб сонома", "сонома", "ясень", "клен", "клён", "бук", "натуральный дуб", "pale wood"),
    "dark_wood": ("dark oak", "дуб темный", "дуб тёмный", "темный дуб", "тёмный дуб", "темное дерево", "тёмное дерево", "венге"),
    "wood": ("wood", "дерево", "древес", "дуб", "oak", "орех", "walnut", "бук", "beech", "ясень", "ash", "клен", "клён"),
    "green": ("green", "зеленый", "зелёный", "олив", "olive", "sage", "шалфей"),
    "blue": ("blue", "синий", "голубой", "navy", "индиго"),
    "red": ("red", "красный", "бордо", "терракот", "terracotta"),
    "stone": ("stone", "камень", "гранит", "granite", "мрамор", "marble", "травертин", "travertine", "терраццо", "terrazzo"),
}

PATTERN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "plain": ("однотон", "plain", "solid", "матовый", "глянец", "белый", "серый", "черный", "чёрный"),
    "wood": ("wood", "дерево", "древес", "дуб", "oak", "орех", "walnut", "бук", "beech", "ясень", "ash"),
    "marble": ("marble", "мрамор", "каррара", "cipollino", "чипполино", "сиена"),
    "stone": ("stone", "камень", "гранит", "granite", "травертин", "travertine", "пьетра", "pietra"),
    "concrete": ("concrete", "бетон", "цемент"),
    "terrazzo": ("terrazzo", "терраццо", "терраццио"),
    "metal": ("metal", "металл", "сталь", "steel", "бронза", "bronze", "алюминий"),
}

STYLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "modern": ("modern", "соврем", "contemporary", "minimal", "минимал", "гладкий", "plain", "graphite", "stone", "marble", "concrete"),
    "scandinavian": ("scandinavian", "сканди", "light oak", "светлый дуб", "сонома", "white", "warm white", "beige", "wood", "матовый"),
    "japandi": ("japandi", "джапанди", "wood", "light oak", "beige", "warm", "natural", "минимал", "matte"),
    "loft": ("loft", "лофт", "concrete", "бетон", "black", "metal", "graphite", "dark wood"),
    "classic": ("classic", "класс", "cream", "крем", "wood", "орех", "филен", "филён"),
}
