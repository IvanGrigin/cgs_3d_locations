from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .kitchen_text_features import normalize_text

APPLIANCE_TARGETS: dict[str, dict[str, Any]] = {
    "fridge": {
        "module_types": {"fridge_slot"},
        "category_terms": ("refrigerator_freezer", "fridge", "refrigerator", "холодиль"),
        "title_terms": ("fridge", "refrigerator", "холодиль", "мини холодильник", "минихолодильник"),
        "target_dims_cm": (60.0, 65.0, 190.0),
        "min_dims_cm": (45.0, 45.0, 120.0),
        "colors": ("white", "gray", "black", "белый", "серый", "черный", "чёрный"),
    },
    "washing_machine": {
        "module_types": {"washing_machine_slot"},
        "category_terms": ("washing_machine", "washer", "стирал"),
        "title_terms": ("washing machine", "washer", "стиральная", "стирал"),
        "target_dims_cm": (60.0, 56.0, 85.0),
        "colors": ("white", "gray", "белый", "серый"),
    },
    "dishwasher": {
        "module_types": {"dishwasher_slot"},
        "category_terms": ("dishwasher", "посудомо", "посудомоеч"),
        "title_terms": ("dishwasher", "посудомоечная", "посудомойка"),
        "target_dims_cm": (60.0, 56.0, 85.0),
        "colors": ("white", "gray", "black", "белый", "серый", "черный", "чёрный"),
    },
    "oven": {
        "module_types": {"oven_cabinet"},
        "category_terms": ("oven", "духов"),
        "title_terms": ("oven", "духов", "hansa", "bosch"),
        "target_dims_cm": (60.0, 56.0, 60.0),
        "colors": ("black", "gray", "white", "черный", "чёрный", "серый", "белый"),
    },
    "cooktop": {
        "module_types": {"oven_cabinet"},
        "category_terms": ("cooktop_hob", "cooktop", "hob", "вароч"),
        "title_terms": ("cooktop", "hob", "варочная", "индукционная"),
        "target_dims_cm": (58.0, 52.0, 5.0),
        "min_dims_cm": (45.0, 30.0, 2.0),
        "colors": ("black", "white", "gray", "черный", "чёрный", "белый", "серый"),
    },
    "hood": {
        "module_types": {"hood_cabinet", "hood_wall_mounted", "hood_compact_wall"},
        "category_terms": ("extractor_hood", "hood", "rangehood", "вытяж"),
        "title_terms": ("hood", "rangehood", "вытяж", "miele", "teka"),
        "target_dims_cm": (60.0, 35.0, 45.0),
        "min_dims_cm": (50.0, 20.0, 20.0),
        "colors": ("black", "gray", "white", "steel", "черный", "чёрный", "серый", "белый", "сталь"),
    },
    "sink": {
        "module_types": {"sink_cabinet"},
        "category_terms": ("kitchen_sink", "sink", "мойк"),
        "title_terms": ("kitchen sink", "мойка", "florentina", "смеситель", "abber", "emar"),
        "target_dims_cm": (56.0, 50.0, 18.0),
        "colors": ("gray", "black", "white", "серый", "черный", "чёрный", "белый"),
    },
    "faucet": {
        "module_types": {"sink_cabinet"},
        "category_terms": ("kitchen_faucet", "faucet", "смеситель"),
        "title_terms": ("kitchen faucet", "смеситель", "faucet"),
        "target_dims_cm": (22.0, 28.0, 36.0),
        "colors": ("gray", "chrome", "steel", "серый", "хром", "сталь", "нержав"),
    },
    "microwave": {
        "module_types": set(),
        "category_terms": ("microwave", "микровол", "свч"),
        "title_terms": ("microwave", "микроволновая", "свч", "gorenje"),
        "target_dims_cm": (45.0, 33.0, 26.0),
        "colors": ("white", "gray", "black", "белый", "серый", "черный", "чёрный"),
    },
    "small_kitchen_appliance": {
        "module_types": set(),
        "category_terms": ("small_kitchen_appliance", "kitchenware"),
        "title_terms": ("кофемашина", "чайник", "kettle", "coffee", "bosch", "philips"),
        "target_dims_cm": (24.0, 24.0, 28.0),
        "colors": ("white", "gray", "black", "белый", "серый", "черный", "чёрный"),
    },
    "flowers_vase": {
        "module_types": set(),
        "category_terms": ("plant_planter_vase", "plant", "vase", "decorative_set", "sculpture_decor_set"),
        "title_terms": ("flower vase", "flower bouquet", "bouquet", "vase", "plant", "букет", "цвет", "ваза", "растение", "ландыши", "розы"),
        "target_dims_cm": (28.0, 28.0, 48.0),
        "colors": ("white", "green", "pink", "gray", "белый", "зелен", "роз", "серый"),
        "prefer_fbx": True,
    },
    "oil_bottles_decor": {
        "module_types": set(),
        "category_terms": ("decorative_set", "food_drink", "kitchenware"),
        "title_terms": ("olive and oil", "oil decorative", "decanters and bottles", "bottle", "decanter", "jar", "бутыл", "масл", "графин", "банка"),
        "target_dims_cm": (24.0, 18.0, 34.0),
        "colors": ("green", "brown", "clear", "glass", "black", "зелен", "корич", "стекл", "черн"),
        "prefer_fbx": True,
    },
    "decorative_kitchen_set": {
        "module_types": set(),
        "category_terms": ("kitchenware", "food_drink", "decorative_set", "small_kitchen_appliance"),
        "title_terms": ("kitchen accessories", "kitchen decor", "tableware", "fruit", "bread", "посуда", "тарел", "чаш", "миска", "фрукт", "хлеб"),
        "target_dims_cm": (32.0, 22.0, 18.0),
        "colors": ("white", "gray", "wood", "green", "белый", "серый", "дерево", "зелен"),
        "prefer_fbx": True,
    },
}

ASSET_IMPORT_OVERRIDES: dict[str, dict[str, Any]] = {
    "3ddd::url::https://3ddd.ru/3dmodels/show/kholodil_nik_aeg_s98392cmx2": {
        "rotation_z_deg_by_layout": {"x": 0.0, "y": -90.0},
        "avoid_for_role": "fridge",
        "note": "Imports reliably, but has only Base/Door meshes and reads too much like a procedural block.",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/kholodilnik_hofmann_rf564cdbs_hf": {
        "rotation_z_deg_by_layout": {"x": 0.0, "y": -90.0},
        "preferred_for_role": "fridge",
        "note": "Detailed FBX refrigerator with many mesh parts and better visible door detail.",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/kholodilnik_s_displeem_atlant_khm_4424_nd": {
        "rotation_z_deg_by_layout": {"x": 0.0, "y": -90.0},
        "preferred_for_role": "fridge",
        "note": "Detailed 60 cm wide FBX refrigerator; keeps realistic tall proportions in a standard kitchen slot.",
    },
    "zeelproject::id::2538": {
        "rotation_z_deg_by_layout": {"x": 0.0, "y": -90.0},
        "avoid_for_role": "hood",
        "note": "Valid asset, but the imported local axes make it easy to face sideways in wall scenes.",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/miele_rangehood": {
        "rotation_z_deg_by_layout": {"x": 180.0, "y": 90.0},
        "preferred_for_role": "hood",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/mikrovolnovaia_pech_hofmann_mw720dhss_hf_1": {
        "rotation_z_deg_by_layout": {"x": 180.0, "y": 90.0},
        "preferred_for_role": "microwave",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/moika_florentina_lipsi_460_chernyi_i_smesitel_florentina_vita_av": {
        "rotation_z_deg_by_layout": {"x": 180.0, "y": 90.0},
        "avoid_for_roles": ("sink", "faucet"),
        "note": "Combined OBJ sink+faucet set imports as fragmented helper geometry in the current kitchen inset pipeline.",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/om-moika-emar-emq-emb-560-top-pvd-1": {
        "rotation_z_deg_by_layout": {"x": 0.0, "y": -90.0},
        "preferred_for_role": "sink",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/kukhonnaia_moika_abber_wasser_kreis_af2194": {
        "rotation_z_deg_by_layout": {"x": 0.0, "y": -90.0},
        "avoid_for_role": "sink",
        "note": "FBX contains several sink variants in one file, which is less stable for automatic inset fitting.",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/om-tumba-c-rakovinoi-napolnaia-lago-80-2d": {
        "avoid_for_role": "sink",
        "note": "Bathroom vanity cabinet with basin, not a standalone kitchen sink for countertop insertion.",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/om-tumba-c-rakovinoi-podvesnaia-lago-80-2y": {
        "avoid_for_role": "sink",
        "note": "Bathroom vanity cabinet with basin, not a standalone kitchen sink for countertop insertion.",
    },
    "3ddd::url::https://3ddd.ru/3dmodels/show/kitchen_faucet_8": {
        "rotation_z_deg_by_layout": {"x": 180.0, "y": 90.0},
        "preferred_for_role": "faucet",
    },
}


def load_supplier_catalog(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.get("items") or []
    else:
        items = data
    return [item for item in items if isinstance(item, dict)]


def _dims_cm(item: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    dims = item.get("dimensions_cm") if isinstance(item.get("dimensions_cm"), dict) else {}

    def get(key: str) -> float | None:
        value = dims.get(key) if dims else item.get(f"{key}_cm")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return None

    return get("width"), get("depth"), get("height")


def _asset_path(item: dict[str, Any]) -> str | None:
    raw = item.get("asset_local_path")
    if not raw:
        return None
    path = Path(str(raw))
    if path.exists() and _is_importable_asset_path(path):
        return str(path)
    fixed = Path(str(raw).replace("\\", "/"))
    if fixed.exists() and _is_importable_asset_path(fixed):
        return str(fixed)
    return None


def _is_importable_asset_path(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in {".fbx", ".obj", ".glb", ".gltf"}:
        return False
    if suffix == ".fbx":
        try:
            return path.read_bytes()[:20].startswith(b"Kaydara FBX Binary")
        except Exception:
            return False
    return True


def _text(item: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            str(item.get(key) or "")
            for key in (
                "title",
                "category_raw",
                "category_norm",
                "description",
                "materials",
                "color",
                "vlm_description_summary",
                "vlm_description_text",
            )
        )
    )


def _color_score(item: dict[str, Any], desired: tuple[str, ...]) -> float:
    text = _text(item)
    color_features = item.get("image_color_features") if isinstance(item.get("image_color_features"), dict) else {}
    tokens = set(color_features.get("color_tokens") or [])
    desired_norm = {normalize_text(x) for x in desired}
    if desired_norm.intersection(tokens):
        return 1.0
    if any(color in text for color in desired_norm):
        return 0.85
    return 0.45


def _category_score(item: dict[str, Any], target: dict[str, Any]) -> float:
    category = normalize_text(item.get("category_norm"))
    title = normalize_text(item.get("title"))
    text = _text(item)
    category_terms = tuple(normalize_text(term) for term in target["category_terms"])
    title_terms = tuple(normalize_text(term) for term in target["title_terms"])
    if "kitchen_sink" in category_terms:
        if "kitchen_sink" in category or "кухон" in title and "мойк" in title or "мойк" in title:
            return 1.0
        return 0.0
    if any(term in category for term in category_terms):
        return 1.0
    if any(term in title for term in title_terms):
        return 0.9
    return 0.0


def _prompt_preference_score(item: dict[str, Any], role: str, user_prompt: str | None) -> float:
    if not user_prompt:
        return 0.5

    prompt = normalize_text(user_prompt)
    text = _text(item)

    if role == "fridge":
        wants_light = any(term in prompt for term in ("светл", "white", "бел", "сканди", "warm white"))
        unique_key = str(item.get("unique_key") or "")
        title = normalize_text(item.get("title"))
        if "hofmann_rf564cdbs" in unique_key or "hofmann rf564cdbs" in title:
            return 0.9
        if "atlant_khm_4424" in unique_key or "atlant" in title:
            return 1.0
        if "aeg_s98392cmx2" in unique_key or "aeg s98392cmx2" in title:
            return 0.42
        if any(term in title for term in ("hofmann", "gaggenau")):
            return 0.88
        if wants_light:
            title_and_color = normalize_text(f"{item.get('title') or ''} {item.get('color') or ''}")
            if any(term in title_and_color for term in ("бел", "white")):
                return 1.0
            if any(term in title_and_color for term in ("сер", "gray", "grey", "stainless")):
                return 0.82
            if any(term in title_and_color for term in ("черн", "черный", "чёрный", "black")):
                return 0.12
        return 0.5

    if role != "cooktop":
        if role == "hood":
            title = normalize_text(item.get("title"))
            if "miele" in title or "rangehood" in title:
                return 1.0
            if "teka" in title:
                return 0.35
        if role == "sink":
            title = normalize_text(item.get("title"))
            if "florentina" in title and ("смеситель" in title or "mixer" in title):
                return 1.0
            if "florentina" in title:
                return 0.95
            if title == "кухонная мойка" or "кухонная мойка" in title:
                return 0.9
            if "abber" in title:
                return 0.72
            if "emar" in title:
                return 0.55
        if role == "small_kitchen_appliance":
            if any(term in text for term in ("кофемашина", "coffee", "bosch", "чайник", "kettle", "philips")):
                return 0.9
        if role == "microwave":
            if any(term in text for term in ("gorenje", "микроволновая", "microwave")):
                return 0.85
        if role == "faucet":
            if any(term in text for term in ("kitchen faucet", "смеситель для кухни", "нержав", "chrome")):
                return 0.9
        return 0.5

    wants_induction = any(term in prompt for term in ("индукц", "induction", "индукционная"))
    wants_gas = any(term in prompt for term in ("газов", "gas"))
    wants_electric = any(term in prompt for term in ("электр", "electric", "miele"))

    if wants_induction:
        if any(term in text for term in ("индукц", "induction", "miele")):
            return 1.0
        if any(term in text for term in ("газов", "gas", "gorenje")):
            return 0.15
        return 0.45

    if wants_gas:
        if any(term in text for term in ("газов", "gas", "gorenje")):
            return 1.0
        return 0.35

    if wants_electric:
        if any(term in text for term in ("электр", "electric", "miele")):
            return 0.9
        if any(term in text for term in ("газов", "gas")):
            return 0.3

    return 0.5


def _dimension_score(item: dict[str, Any], target_dims: tuple[float, float, float]) -> float:
    dims = _dims_cm(item)
    if not all(dims):
        return 0.45
    ratios = []
    for value, target in zip(dims, target_dims):
        if value is None:
            continue
        ratios.append(min(float(value), target) / max(float(value), target))
    if not ratios:
        return 0.45
    return sum(ratios) / len(ratios)


def _passes_min_dimensions(item: dict[str, Any], target: dict[str, Any]) -> bool:
    min_dims = target.get("min_dims_cm")
    if not min_dims:
        return True
    dims = _dims_cm(item)
    if not all(dims):
        return True
    return all(float(value) >= float(limit) for value, limit in zip(dims, min_dims) if value is not None)


def _candidate_record(item: dict[str, Any], role: str, score: float, breakdown: dict[str, float]) -> dict[str, Any]:
    unique_key = item.get("unique_key")
    import_override = ASSET_IMPORT_OVERRIDES.get(str(unique_key or ""), {})
    price = item.get("price") if item.get("price") is not None else item.get("price_value")
    return {
        "role": role,
        "unique_key": unique_key,
        "title": item.get("title"),
        "source_site": item.get("source_site"),
        "category_norm": item.get("category_norm"),
        "color": item.get("color"),
        "price": price,
        "price_currency": item.get("price_currency") or "RUB",
        "dimensions_cm": {
            "width": _dims_cm(item)[0],
            "depth": _dims_cm(item)[1],
            "height": _dims_cm(item)[2],
        },
        "asset_local_path": _asset_path(item),
        "asset_format": item.get("asset_format") or item.get("model_format"),
        "preview_local_path": item.get("preview_local_path"),
        "product_url": item.get("product_url") or item.get("source_url"),
        "blender_import": import_override,
        "score": round(score, 6),
        "score_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
    }


def _is_forbidden_for_role(item: dict[str, Any], role: str) -> bool:
    title = normalize_text(item.get("title"))
    category = normalize_text(item.get("category_norm"))
    unique_key = str(item.get("unique_key") or "")
    override = ASSET_IMPORT_OVERRIDES.get(unique_key, {})
    forbidden_roles = set(override.get("avoid_for_roles") or ())
    if override.get("avoid_for_role"):
        forbidden_roles.add(str(override["avoid_for_role"]))
    if role in forbidden_roles:
        return True
    if role == "faucet" and "мойк" in title and "смесител" in title and "faucet" not in category:
        return True
    return False


def select_kitchen_appliance_assets(
    supplier_catalog: str | Path | list[dict[str, Any]],
    layout_plan: dict[str, Any],
    required_appliances: dict[str, Any],
    only_local_assets: bool = True,
    top_n: int = 5,
    user_prompt: str | None = None,
) -> dict[str, Any]:
    items = load_supplier_catalog(supplier_catalog) if isinstance(supplier_catalog, (str, Path)) else supplier_catalog
    present_roles: set[str] = set()

    for module in layout_plan.get("base_modules") or []:
        appliance = module.get("appliance")
        if appliance:
            present_roles.add(str(appliance))
        if "sink" in module.get("cutouts", []):
            present_roles.add("sink")
        if "cooktop" in module.get("cutouts", []):
            present_roles.add("cooktop")
            if required_appliances.get("hood", True):
                present_roles.add("hood")
        if "sink" in module.get("cutouts", []):
            present_roles.add("faucet")

    if required_appliances.get("microwave"):
        present_roles.add("microwave")
    present_roles.add("small_kitchen_appliance")
    for item in layout_plan.get("decor_items") or []:
        item_type = str(item.get("type") or "")
        if item_type in APPLIANCE_TARGETS:
            present_roles.add(item_type)

    result: dict[str, Any] = {"appliances": {}, "warnings": [], "unavailable_assets": {}}

    for role, target in APPLIANCE_TARGETS.items():
        if role not in present_roles and not required_appliances.get(role):
            continue

        scored: list[dict[str, Any]] = []
        unavailable_scored: list[dict[str, Any]] = []
        for item in items:
            if _is_forbidden_for_role(item, role):
                continue
            category = _category_score(item, target)
            if category <= 0:
                continue
            if not _passes_min_dimensions(item, target):
                continue
            asset = _asset_path(item)
            if target.get("prefer_fbx") and asset and not str(asset).lower().endswith(".fbx"):
                continue
            color = _color_score(item, target["colors"])
            dims = _dimension_score(item, target["target_dims_cm"])
            prompt = _prompt_preference_score(item, role, user_prompt)
            if asset and str(asset).lower().endswith(".fbx"):
                asset_score = 1.0
            elif asset:
                asset_score = 0.62
            else:
                asset_score = 0.35
            override = ASSET_IMPORT_OVERRIDES.get(str(item.get("unique_key") or ""), {})
            override_score = 0.0
            if override.get("preferred_for_role") == role:
                override_score = 0.12
            if override.get("avoid_for_role") == role:
                override_score = -0.28
            breakdown = {
                "category_score": category,
                "color_score": color,
                "dimension_score": dims,
                "prompt_score": prompt,
                "asset_score": asset_score,
                "override_score": override_score,
            }
            score = 0.34 * category + 0.22 * color + 0.18 * dims + 0.16 * prompt + 0.08 * asset_score + override_score
            record = _candidate_record(item, role, score, breakdown)
            if only_local_assets and not asset:
                unavailable_scored.append(record)
                continue
            scored.append(record)

        scored.sort(key=lambda x: x["score"], reverse=True)
        unavailable_scored.sort(key=lambda x: x["score"], reverse=True)
        if unavailable_scored:
            result["unavailable_assets"][role] = unavailable_scored[:top_n]
        if not scored:
            result["warnings"].append(f"no_appliance_asset_for_role:{role}")
            continue
        result["appliances"][role] = {
            "chosen_asset": scored[0],
            "top_candidates": scored[:top_n],
        }

    return result
