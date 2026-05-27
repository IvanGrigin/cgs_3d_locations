#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enrich Savlukov canonical catalog cards from current public product pages."""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


CATALOG = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
SAVLUKOV_ROOT = Path("data/sourse/suppliers/savlukov")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml"}


PRODUCT_URLS = {
    "alaska_pryamoy": "https://savlukov.by/product/pryamie-divany/380-priamoy-alaska",
    "alaska_uglovoy": "https://savlukov.by/product/uglovye-divany/412-uglovoy-alaska",
    "bonta": "https://savlukov.by/product/pryamie-divany/53-priamoy-bonta",
    "bonta_sleep": "https://savlukov.by/product/pryamie-divany/347-priamoy-bonta-sleep",
    "chicago": "https://savlukov.by/product/pryamie-divany/280-priamoy-chicago",
    "city": "https://savlukov.by/product/kresla/438-armchair-city",
    "corfu_pryamoy": "https://savlukov.by/product/pryamie-divany/171-priamoy-corfu",
    "corfu_uglovoy": "https://savlukov.by/product/modulnye-divany/186-uglovoy-corfu",
    "cosmo": "https://savlukov.by/product/modulnye-divany/466-priamoy-cosmo",
    "dolce": "https://savlukov.by/product/pryamie-divany/82-priamoy-dolce",
    "ego": "https://savlukov.by/product/pryamie-divany/78-priamoy-ego",
    "galaxy": "https://savlukov.by/product/pryamie-divany/353-priamoy-galaxy",
    "hilton_pryamoy": "https://savlukov.by/product/pryamie-divany/255-priamoy-hilton",
    "hilton_uglovoy": "https://savlukov.by/product/uglovye-divany/270-uglovoy-hilton",
    "manhattan": "https://savlukov.by/product/uglovye-divany/407-uglovoy-manhattan",
    "mercury_pryamoy": "https://savlukov.by/product/pryamie-divany/211-priamoy-mercury",
    "mercury_uglovoy": "https://savlukov.by/product/uglovye-divany/364-uglovoy-mercury",
    "oscar_pryamoy": "https://savlukov.by/product/pryamie-divany/249-priamoy-oscar",
    "oscar_uglovoy": "https://savlukov.by/product/uglovye-divany/252-uglovoy-oscar",
    "ostin_pryamoy": "https://savlukov.by/product/pryamie-divany/296-priamoy-ostin",
    "ostin_uglovoy": "https://savlukov.by/product/uglovye-divany/297-uglovoi-ostin",
    "riviera": "https://savlukov.by/product/uglovye-divany/229-uglovoy-riviera",
    "skandinavia_pryamoy": "https://savlukov.by/product/pryamie-divany/227-pryamoy-skandinavia",
    "skandinavia_uglovoy": "https://savlukov.by/product/uglovye-divany/234-uglovoy-skandinavia",
    "tavola": "https://savlukov.by/product/pryamie-divany/41-priamoy-tavola",
    "texas_pryamoy": "https://savlukov.by/product/pryamie-divany/275-priamoy-texas",
    "texas_uglovoy": "https://savlukov.by/product/uglovye-divany/401-uglovoy-texas",
    "tuscan_pryamoy": "https://savlukov.by/product/pryamie-divany/7-priamoy-tuscan",
    "tuscan_uglovoy": "https://savlukov.by/product/uglovye-divany/10-uglovoy-tuscan",
    "twist": "https://savlukov.by/product/kresla/239-kreslo-twist",
    "twist_maxi": "https://savlukov.by/product/kresla/406-armchairtwistmaxi",
    "vegas": "https://savlukov.by/product/pryamie-divany/136-priamoy-vegas",
    "yoga_pryamoy": "https://savlukov.by/product/pryamie-divany/37-priamoy-yoga",
    "yoga_uglovoy": "https://savlukov.by/product/uglovye-divany/30-uglovoy-yoga",
}


LEGACY_NAMES = {
    "прага": "Диван Прага",
    "прованс": "Диван Прованс",
    "соната": "Диван Соната",
    "boston": "Диван Boston",
    "kanzas": "Диван Kanzas",
    "oksford": "Диван Oksford",
}


def clean_text(value: Any) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_mm(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:[,.]\d+)?)", value)
    if not match:
        return None
    return round(float(match.group(1).replace(",", ".")) / 10.0, 2)


def parse_product_page(url: str, session: requests.Session) -> dict[str, Any]:
    resp = session.get(url, timeout=(10, 45))
    resp.raise_for_status()
    text = resp.text
    product: dict[str, Any] = {}
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', text, re.S):
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("@type") == "Product":
            product = obj
            break

    params: dict[str, str] = {}
    for match in re.finditer(
        r'<div class="row param">\s*<div[^>]*title">(.*?)</div>\s*<div[^>]*value">(.*?)</div>',
        text,
        re.S,
    ):
        params[clean_text(match.group(1))] = clean_text(match.group(2))

    dims: dict[str, str] = {}
    for match in re.finditer(
        r'<div class="[^"]*image-param[^"]*">\s*<p>(.*?)</p>.*?<p class="value">(.*?)</p>',
        text,
        re.S,
    ):
        dims[clean_text(match.group(1))] = clean_text(match.group(2))

    offers = product.get("offers") if isinstance(product.get("offers"), dict) else {}
    images = product.get("image") if isinstance(product.get("image"), list) else []
    images = [urljoin(url, str(x)) for x in images if str(x).strip()]
    return {
        "url": url,
        "name": clean_text(product.get("name")),
        "description": clean_text(product.get("description")),
        "sku": clean_text(product.get("sku")),
        "brand": clean_text((product.get("brand") or {}).get("name") if isinstance(product.get("brand"), dict) else product.get("brand")),
        "category": clean_text(product.get("category")),
        "price": float(offers["price"]) if str(offers.get("price") or "").strip() else None,
        "currency": clean_text(offers.get("priceCurrency")) or None,
        "availability": clean_text(offers.get("availability")) or None,
        "images": images,
        "params": params,
        "dims": dims,
        "fetched_at_unix": time.time(),
    }


def load_or_fetch_products(cache_path: Path, refresh: bool) -> dict[str, dict[str, Any]]:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update(HEADERS)
    out: dict[str, dict[str, Any]] = {}
    for key, url in PRODUCT_URLS.items():
        try:
            out[key] = parse_product_page(url, session)
            print(f"[product] {key} {out[key].get('price')} {out[key].get('currency')} {out[key].get('name')}")
        except Exception as exc:
            out[key] = {"url": url, "error": f"{type(exc).__name__}: {exc}", "fetched_at_unix": time.time()}
            print(f"[product] ERROR {key}: {exc}")
        time.sleep(0.08)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def image_index() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    raw = SAVLUKOV_ROOT / "raw"
    for path in sorted(raw.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        key = path.parent.name.lower()
        out.setdefault(key, []).append(str(path))
    return out


def product_key_for_item(item: dict[str, Any]) -> str | None:
    key = str(item.get("unique_key") or "").lower()
    asset = str(item.get("asset_local_path") or "").lower()
    text = f"{key} {asset}"

    if "oscar" in text:
        return "oscar_uglovoy" if "yglovoi" in text or "uglov" in text else "oscar_pryamoy"
    if "yoga" in text:
        return "yoga_uglovoy" if "uglovoi" in text or "yglovoi" in text else "yoga_pryamoy"
    if "vegas" in text or "вегас" in text:
        return "vegas"
    if "manhattan" in text or "манхэттен" in text:
        return "manhattan"
    if "mercury" in text or "меркури" in text:
        return "mercury_uglovoy" if "3750" in text or "2250" in text else "mercury_pryamoy"
    if "остин" in text or "ostin" in text:
        return "ostin_uglovoy" if "uglov" in text or "углов" in text else "ostin_pryamoy"
    if "скандинавия" in text or "skandinavia" in text:
        return "skandinavia_uglovoy" if "uglov" in text or "углов" in text else "skandinavia_pryamoy"
    if "texas" in text or "техас" in text:
        return "texas_uglovoy" if "corner" in text or "uglov" in text else "texas_pryamoy"
    if "hilton" in text or "хилтон" in text:
        return "hilton_uglovoy" if "uglov" in text or "углов" in text else "hilton_pryamoy"
    if "chicago" in text or "чикаго" in text:
        return "chicago"
    if "bonta_sleep" in text or "bonta sleep" in text:
        return "bonta_sleep"
    if "bonta" in text:
        return "bonta"
    if "sity" in text or "city" in text or "сити" in text:
        return "city"
    if "cosmo" in text:
        return "cosmo"
    if "dolce" in text:
        return "dolce"
    if "ego" in text:
        return "ego"
    if "galaxy" in text:
        return "galaxy"
    if "rivera" in text or "riviera" in text:
        return "riviera"
    if "tavola" in text:
        return "tavola"
    if "tuscan_2630x1850" in text:
        return "tuscan_uglovoy"
    if "tuscan" in text:
        return "tuscan_pryamoy"
    if "twist_max" in text:
        return "twist_maxi"
    if "twist" in text:
        return "twist"
    if "alaska" in text or "аляска" in text:
        return "alaska_uglovoy" if "uglov" in text or "углов" in text else "alaska_pryamoy"
    if "korfy" in text or "corfu" in text or "корфу" in text:
        return "corfu_uglovoy" if "3600" in text or "uglov" in text else "corfu_pryamoy"
    return None


def legacy_status_for_item(item: dict[str, Any]) -> str | None:
    text = f"{item.get('unique_key','')} {item.get('asset_local_path','')}".lower()
    for key in LEGACY_NAMES:
        if key in text:
            return key
    return None


def local_images_for_item(item: dict[str, Any], images_by_folder: dict[str, list[str]]) -> list[str]:
    asset = Path(str(item.get("asset_local_path") or ""))
    folder = asset.parent.name.lower()
    candidates: list[str] = []
    for key, paths in images_by_folder.items():
        if key == folder or folder in key or key in folder:
            candidates.extend(paths)
    if not candidates:
        collection = str(item.get("collection") or "").lower()
        for key, paths in images_by_folder.items():
            if collection and (collection in key or key in collection):
                candidates.extend(paths)
    seen: set[str] = set()
    out: list[str] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out[:8]


def apply_product(item: dict[str, Any], product_key: str, product: dict[str, Any], local_images: list[str]) -> None:
    params = product.get("params") or {}
    dims = product.get("dims") or {}
    name = product.get("name") or item.get("title")
    item["title"] = name
    item["brand"] = product.get("brand") or "Savlukov-Mebel"
    item["source_url"] = product.get("url")
    item["product_url"] = product.get("url")
    item["model_vendor_url"] = product.get("url")
    item["model_page_url"] = "https://designers.savlukov.by/models"
    item["model_download_landing_url"] = "https://disk.yandex.ru/d/8Jqha8s5btjZmg"
    item["external_id"] = product.get("sku") or item.get("external_id")
    item["category_raw"] = product.get("category") or item.get("category_raw")
    category = item["category_raw"]
    if "Крес" in str(category):
        item["category_norm"] = "armchair"
    elif "Углов" in str(category) or "Модуль" in str(category):
        item["category_norm"] = "sectional_sofa"
    else:
        item["category_norm"] = "sofa"
    item["price_value"] = product.get("price")
    item["price_currency"] = product.get("currency") or "BYN"
    item["style"] = params.get("Стиль") or item.get("style")
    material_bits = []
    for param_key in ("Каркас", "Наполнение подушек", "Наполнение", "Материал опор"):
        if params.get(param_key):
            material_bits.append(f"{param_key}: {params[param_key]}")
    item["materials"] = "; ".join(material_bits) or item.get("materials")
    item["availability"] = "InStock" if "InStock" in str(product.get("availability")) else "public"
    item["description"] = product.get("description") or item.get("description")
    images = product.get("images") or []
    item["images"] = images + [x for x in local_images if x not in images]
    if local_images:
        item["preview_local_path"] = local_images[0]
    item["dimensions_cm"] = {
        "width": parse_mm(dims.get("Длина")),
        "depth": parse_mm(dims.get("Ширина")),
        "height": parse_mm(dims.get("Высота")),
        "weight_kg": None,
        "package_width": None,
        "package_depth": None,
        "package_height": None,
        "packed_weight_kg": None,
        "volume_m3": None,
    }
    item["tags"] = sorted(set([*(item.get("tags") or []), "Savlukov", ".fbx", item["category_norm"], item.get("price_currency") or "BYN"]))
    extra = item.setdefault("extra", {})
    extra["savlukov_product_enrichment"] = {
        "status": "matched_current_site",
        "product_key": product_key,
        "product_url": product.get("url"),
        "price_currency_note": "Savlukov site default currency is Belarusian ruble (BYN).",
        "params": params,
        "dims_raw": dims,
        "fetched_at_unix": product.get("fetched_at_unix"),
    }
    comp = item.setdefault("completeness", {})
    comp.update(
        {
            "has_title": bool(item.get("title")),
            "has_price": item.get("price_value") is not None,
            "has_full_dimensions": all(item.get("dimensions_cm", {}).get(k) is not None for k in ("width", "depth", "height")),
            "has_description": bool(item.get("description")),
            "has_category": bool(item.get("category_norm")),
            "has_brand": bool(item.get("brand")),
            "has_model_link": bool(item.get("asset_local_path")),
            "rich_card": item.get("price_value") is not None and bool(item.get("description")),
        }
    )


def apply_legacy(item: dict[str, Any], legacy_key: str, local_images: list[str]) -> None:
    text = str(item.get("asset_local_path") or "").lower()
    title = LEGACY_NAMES[legacy_key]
    if legacy_key == "прага":
        if "armchair" in text:
            title = "Кресло Прага"
        elif "puff" in text:
            title = "Пуф Прага"
        elif "sofa" in text:
            title = "Диван Прага"
    item["title"] = title
    item["brand"] = "Savlukov-Mebel"
    item["source_url"] = "https://designers.savlukov.by/models"
    item["product_url"] = None
    item["model_vendor_url"] = "https://savlukov.by/"
    item["model_page_url"] = "https://designers.savlukov.by/models"
    item["model_download_landing_url"] = "https://disk.yandex.ru/d/8Jqha8s5btjZmg"
    if "armchair" in text or "kreslo" in text:
        item["category_raw"] = "Кресла"
        item["category_norm"] = "armchair"
    elif "puff" in text or "puf" in text:
        item["category_raw"] = "Пуфы"
        item["category_norm"] = "ottoman"
    else:
        item["category_raw"] = "Диваны"
        item["category_norm"] = "sofa"
    item["price_value"] = None
    item["price_currency"] = None
    if local_images:
        item["preview_local_path"] = local_images[0]
        item["images"] = local_images
    item["description"] = (
        f"{item['title']}: 3D-модель из публичной библиотеки Savlukov для дизайнеров. "
        "Актуальная товарная страница и цена не найдены в текущем публичном каталоге Savlukov."
    )
    extra = item.setdefault("extra", {})
    extra["savlukov_product_enrichment"] = {
        "status": "not_found_on_current_site",
        "searched_current_catalog": True,
        "price_currency_note": "No current public BYN price was found; price intentionally left null.",
    }
    comp = item.setdefault("completeness", {})
    comp.update(
        {
            "has_title": True,
            "has_price": False,
            "has_full_dimensions": False,
            "has_description": True,
            "has_category": True,
            "has_brand": True,
            "has_model_link": True,
            "rich_card": False,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default=str(CATALOG))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    products = load_or_fetch_products(SAVLUKOV_ROOT / "savlukov_product_enrichment.json", args.refresh)
    images_by_folder = image_index()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    updated = 0
    matched = 0
    legacy = 0
    for item in payload.get("items", []):
        if item.get("source_site") != "savlukov":
            continue
        local_images = local_images_for_item(item, images_by_folder)
        product_key = product_key_for_item(item)
        if product_key and product_key in products and not products[product_key].get("error"):
            apply_product(item, product_key, products[product_key], local_images)
            matched += 1
        else:
            legacy_key = legacy_status_for_item(item)
            if legacy_key:
                apply_legacy(item, legacy_key, local_images)
                legacy += 1
            else:
                item.setdefault("extra", {})["savlukov_product_enrichment"] = {"status": "unmatched"}
        updated += 1

    meta = payload.setdefault("meta", {})
    meta["item_count"] = len(payload.get("items", []))
    meta.setdefault("manual_merges", []).append(
        {
            "source": "savlukov_enrich_catalog_cards",
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "updated_savlukov_items": updated,
            "matched_current_site": matched,
            "legacy_not_found_on_current_site": legacy,
            "currency_rule": "Use BYN from Savlukov Product JSON-LD; do not convert to RUB.",
        }
    )
    tmp = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(catalog_path)
    print(f"[catalog] updated={updated} matched={matched} legacy_not_found={legacy}")


if __name__ == "__main__":
    main()
