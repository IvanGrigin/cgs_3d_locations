from __future__ import annotations

from collections.abc import Iterable


PROMPT_OBJECT_TO_SEMANTIC: dict[str, str] = {
    "bed": "Bed",
    "single bed": "Bed",
    "double bed": "Bed",
    "queen bed": "Bed",
    "king bed": "Bed",
    "nightstand": "SideTable",
    "side table": "SideTable",
    "bedside table": "SideTable",
    "wardrobe": "Storage",
    "cabinet": "Storage",
    "kitchen cabinet": "Storage",
    "dresser": "Storage",
    "bookshelf": "Storage",
    "bookcase": "Storage",
    "shelf": "Storage",
    "low storage": "LowStorage",
    "tall storage": "TallStorage",
    "sink": "Sink",
    "kitchen sink": "Sink",
    "toilet": "Toilet",
    "bathtub": "Bathtub",
    "bath": "Bathtub",
    "shower": "Shower",
    "kitchen counter": "KitchenCounter",
    "countertop": "KitchenCounter",
    "lamp": "Lighting",
    "ceiling light": "CeilingLight",
    "floor lamp": "FloorLamp",
    "desk lamp": "Lighting",
    "table lamp": "Lighting",
    "chair": "Chair",
    "armchair": "Chair",
    "accent chair": "Chair",
    "desk": "Desk",
    "rug": "Rug",
    "mirror": "Mirror",
    "plant": "LargePlant",
    "large plant": "LargePlant",
    "decor": "Decor",
}


FACTORY_TO_SEMANTIC: dict[str, str] = {
    "BedFactory": "Bed",
    "SideTableFactory": "SideTable",
    "SidetableDeskFactory": "SideTable",
    "SimpleBookcaseFactory": "TallStorage",
    "CellShelfFactory": "TallStorage",
    "LargeShelfFactory": "TallStorage",
    "KitchenCabinetFactory": "TallStorage",
    "SingleCabinetFactory": "LowStorage",
    "StandingSinkFactory": "Sink",
    "SinkFactory": "Sink",
    "ToiletFactory": "Toilet",
    "BathtubFactory": "Bathtub",
    "ShowerFactory": "Shower",
    "LargePlantContainerFactory": "LargePlant",
    "FloorLampFactory": "FloorLamp",
    "CeilingLightFactory": "CeilingLight",
    "LampFactory": "Lighting",
    "DeskLampFactory": "Lighting",
    "SimpleDeskFactory": "Desk",
    "DeskFactory": "Desk",
    "CoffeeTableFactory": "Table",
    "DiningTableFactory": "Table",
    "ChairFactory": "Chair",
    "ArmChairFactory": "Chair",
    "MirrorFactory": "Mirror",
    "RugFactory": "Rug",
    "WallArtFactory": "WallDecoration",
    "BookStackFactory": "Decor",
    "BookColumnFactory": "Decor",
    "NatureShelfTrinketsFactory": "Decor",
}


SEMANTIC_TO_ALLOWED_FACTORIES: dict[str, list[str]] = {
    "Bed": ["BedFactory"],
    "SideTable": ["SideTableFactory", "SidetableDeskFactory"],
    "Storage": [
        "SimpleBookcaseFactory",
        "CellShelfFactory",
        "LargeShelfFactory",
        "KitchenCabinetFactory",
        "SingleCabinetFactory",
    ],
    "LowStorage": ["SingleCabinetFactory"],
    "TallStorage": [
        "SimpleBookcaseFactory",
        "CellShelfFactory",
        "LargeShelfFactory",
        "KitchenCabinetFactory",
    ],
    "Sink": ["StandingSinkFactory", "SinkFactory"],
    "Toilet": ["ToiletFactory"],
    "Bathtub": ["BathtubFactory"],
    "Shower": ["ShowerFactory"],
    "Lighting": ["LampFactory", "DeskLampFactory", "CeilingLightFactory", "FloorLampFactory"],
    "CeilingLight": ["CeilingLightFactory"],
    "FloorLamp": ["FloorLampFactory"],
    "Chair": ["ChairFactory", "ArmChairFactory"],
    "Desk": ["SimpleDeskFactory", "DeskFactory"],
    "Table": ["CoffeeTableFactory", "DiningTableFactory"],
    "LargePlant": ["LargePlantContainerFactory"],
    "Mirror": ["MirrorFactory"],
    "Rug": ["RugFactory"],
    "Decor": ["MirrorFactory", "RugFactory"],
}


PRIMARY_SEMANTICS: set[str] = {
    "Bed",
    "Storage",
    "SideTable",
    "Table",
    "Chair",
    "Seating",
    "LoungeSeating",
    "KitchenCounter",
    "KitchenAppliance",
    "Sink",
    "Toilet",
    "Bathtub",
    "Shower",
    "WallDecoration",
    "Lighting",
    "CeilingLight",
    "Furniture",
}

SECONDARY_SEMANTICS: set[str] = {
    "Dishware",
    "Cookware",
    "Utensils",
    "FoodPantryItem",
    "TableDisplayItem",
    "OfficeShelfItem",
    "KitchenCounterItem",
    "BathroomItem",
    "ClothDrapeItem",
    "HandheldItem",
    "Rug",
    "Mirror",
    "LargePlant",
    "Decor",
    "LowStorage",
    "TallStorage",
    "FloorLamp",
}


def _normalized_tokens(value: str) -> list[str]:
    low = (value or "").strip().lower().replace("-", " ")
    return [" ".join(low.split()), low.replace(" ", "")]


def normalize_prompt_object(name: str) -> str | None:
    for token in _normalized_tokens(name):
        if token in PROMPT_OBJECT_TO_SEMANTIC:
            return PROMPT_OBJECT_TO_SEMANTIC[token]
    return None


def factory_to_semantic(factory_name: str) -> str | None:
    return FACTORY_TO_SEMANTIC.get(str(factory_name or "").strip())


def semantic_to_factory_family(semantic: str) -> list[str]:
    return list(SEMANTIC_TO_ALLOWED_FACTORIES.get(str(semantic or "").strip(), []))


def expand_semantics_to_factories(semantics: Iterable[str]) -> list[str]:
    out: list[str] = []
    for semantic in semantics:
        out.extend(semantic_to_factory_family(semantic))
    return sorted(set(out))


NON_CORE_FACTORY_NAMES: set[str] = {
    "BookStackFactory",
    "BookColumnFactory",
    "NatureShelfTrinketsFactory",
}


TECHNICAL_FACTORY_PREFIXES: tuple[str, ...] = (
    "hoof_parent_temp",
    "beziercurve",
)


def is_technical_factory_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    return any(low.startswith(prefix) for prefix in TECHNICAL_FACTORY_PREFIXES)


def is_core_furniture_factory(name: str) -> bool:
    if is_technical_factory_name(name):
        return False
    factory_name = str(name or "").strip()
    if factory_name in NON_CORE_FACTORY_NAMES:
        return False
    semantic = factory_to_semantic(factory_name)
    return semantic not in {None, "", "Decor"}


POLICY_TO_INFINIGEN_SEMANTICS: dict[str, list[str]] = {
    "Bed": ["Bed"],
    "Storage": ["Storage"],
    "LowStorage": ["Storage"],
    "TallStorage": ["Storage"],
    "SideTable": ["SideTable"],
    "Table": ["Table"],
    "Desk": ["Table"],
    "Chair": ["Chair"],
    "Lighting": ["Lighting"],
    "CeilingLight": ["CeilingLight"],
    "FloorLamp": ["Lighting"],
    "Mirror": ["WallDecoration"],
    "Rug": [],
    "LargePlant": [],
    "Decor": [],
}


def policy_semantic_to_infinigen(semantic: str) -> list[str]:
    return list(POLICY_TO_INFINIGEN_SEMANTICS.get(str(semantic or "").strip(), []))


def expand_policy_semantics_to_infinigen(semantics: Iterable[str]) -> list[str]:
    out: list[str] = []
    for semantic in semantics:
        out.extend(policy_semantic_to_infinigen(semantic))
    return sorted(set(out))
