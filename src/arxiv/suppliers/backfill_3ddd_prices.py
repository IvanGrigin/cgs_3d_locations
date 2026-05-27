# -*- coding: utf-8 -*-
"""
Backfill approximate supplier prices for manually ingested 3ddd cards.

The input 3ddd pages often contain manufacturer links but no prices. This
script records researched supplier/retail prices in the normal product price
columns and stores an audit trail in extra_json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.suppliers.db import PRODUCT_COLUMNS
from src.suppliers.models import ProductRecord
from src.suppliers.runner import save_metadata_json
from src.suppliers.utils import now_utc_iso


PRICE_DATA: dict[str, dict[str, Any]] = {
    "Ванна GRECA": {
        "price": 473100,
        "currency": "RUB",
        "source": "https://salini-srl.com/vanny/otdelnostoyashchie/greca/180/",
        "method": "manufacturer_page",
        "confidence": "exact",
        "note": "Salini GRECA 180 listed at 473100 RUB.",
    },
    "GEMELLI унитаз": {
        "price": 124700,
        "currency": "RUB",
        "source": "https://salini-srl.com/unitazy-i-bide/unitazy-gemelli/",
        "method": "manufacturer_page",
        "confidence": "exact",
        "note": "Salini GEMELLI toilet listed at 124700 RUB.",
    },
    "Зеркало OMBRA H": {
        "price": 94700,
        "currency": "RUB",
        "source": "https://santehmoll.ru/product/zerkalo-100x70-sm-salini-ombra-27m011070bh/",
        "method": "retailer_page",
        "confidence": "variant_exact",
        "note": "Salini Ombra 100x70 27M011070BH mirror listed at 94700 RUB.",
    },
    "Зеркало OMBRA U": {
        "price": 94700,
        "currency": "RUB",
        "source": "https://santehmoll.ru/product/zerkalo-100x70-sm-salini-ombra-27m011070bh/",
        "method": "retailer_variant_proxy",
        "confidence": "estimate",
        "note": "Used the closest Salini OMBRA 100x70 LED mirror price.",
    },
    "ОМ Тумба c раковиной напольная Lago 80.2D": {
        "price": 14075,
        "currency": "RUB",
        "source": "https://www.vseinstrumenti.ru/product/tumba-equil-lago-80-napolnaya-2-dv-belyj-derevo-pod-rakovinu-miranda-80-tnlago80-2d-04-21278630/",
        "method": "retailer_page",
        "confidence": "variant_exact",
        "note": "EQUIL Lago 80.2D cabinet under Miranda 80 sink listed at 14075 RUB.",
    },
    "ОМ Тумба c раковиной подвесная Lago 80.2Y": {
        "price": 17448,
        "currency": "RUB",
        "source": "https://www.vseinstrumenti.ru/product/tumba-equil-lago-80-podvesnaya-2-yasch-belyj-derevo-pod-rakovinu-miranda-80-tplago80-2y-04-21278648/",
        "method": "retailer_page",
        "confidence": "variant_exact",
        "note": "EQUIL Lago 80.2Y wall-mounted cabinet under Miranda 80 sink listed at 17448 RUB.",
    },
    "ОМ Пенал напольный Lago 35": {
        "price": 17572,
        "currency": "RUB",
        "source": "https://mosplitka.ru/product/shkaf-penal-equil-lago-35-sm-pnlago35-04-belyy-svetloe-derevo/",
        "method": "retailer_page",
        "confidence": "variant_exact",
        "note": "EQUIL Lago 35 tall cabinet listed at 17572 RUB.",
    },
    "ОМ Душевая стойка BOND B06-9500": {
        "price": 18860,
        "old_price": 25790,
        "currency": "RUB",
        "source": "https://www.termokit.ru/product/dushevaya_kolonna_so_smesitelem_bond_kub_b06_9500.htm",
        "method": "retailer_page",
        "confidence": "exact",
        "note": "BOND B06-9500 listed at 18860 RUB, old price 25790 RUB.",
    },
    "ОМ Душевая кабина DIWO Новгород 100х100 средний поддон": {
        "price": 37193,
        "old_price": 41999,
        "currency": "RUB",
        "source": "https://diwo.ru/product/dushevaya-kabina-diwo-novgorod-100kh100-sredniy-poddon-gradient-73718/",
        "method": "manufacturer_page",
        "confidence": "variant_exact",
        "note": "DIWO Novgorod 100x100 middle tray gradient listed at 37193 RUB.",
    },
    "Душевой уголок ABBER Schwarzer Diamant AG01090": {
        "price": 30800,
        "currency": "RUB",
        "source": "https://abber-shop.ru/dushevye-ugolki/dushevoy-ugolok-abber-schwarzer-diamant-ag01090/",
        "method": "official_shop_page",
        "confidence": "exact",
        "note": "ABBER AG01090 listed at 30800 RUB.",
    },
    "Душевой уголок ABBER Sonnenstrand AG0407": {
        "price": 20740,
        "currency": "RUB",
        "source": "https://santehmoll.ru/product/ag04070/",
        "method": "retailer_page",
        "confidence": "variant_exact",
        "note": "ABBER Sonnenstrand AG04070 70 cm door listed at 20740 RUB.",
    },
    "Ванна из искусственного камня ABBER Stein AS9651": {
        "price": 165000,
        "currency": "RUB",
        "source": "https://gemy-russia.ru/",
        "method": "supplier_catalog_estimate",
        "confidence": "estimate",
        "note": "Estimated from premium ABBER artificial stone bathtub market range.",
    },
    "OM Унитаз-компакт AVS Хорда безободковый": {
        "price": 15390,
        "currency": "RUB",
        "source": "https://akvamir.online/product/unitaz-napolnyy-s-sidenem-khorda-avs-813-0007-gw/",
        "method": "supplier_product_url",
        "confidence": "estimate",
        "note": "Approximate price for AVS 813-0007-GW compact toilet.",
    },
    "Полотенцесушитель Asti pulsante": {
        "price": 18500,
        "currency": "RUB",
        "source": "https://meduzza.ru/",
        "method": "supplier_catalog_estimate",
        "confidence": "estimate",
        "note": "Approximate price for designer water/electric towel warmer class.",
    },
    "OM Мойка EMAR EMQ-EMB-560 (TOP) PVD": {
        "price": 23500,
        "currency": "RUB",
        "source": "https://emar.su/",
        "method": "supplier_catalog_estimate",
        "confidence": "estimate",
        "note": "Approximate EMAR EMB/EMQ 560 PVD kitchen sink price.",
    },
    "ОМ Смеситель для кухни из нержавеющей стали Stainless, с поворотным Г изливо": {
        "price": 12990,
        "currency": "RUB",
        "source": "https://sancos.su/catalog/dlya_kukhni/smesitel_dlya_kukhni_iz_nerzhaveyushchey_stali_stainless_s_povorotnym_g_izlivom_sc12002ss/",
        "method": "supplier_product_url",
        "confidence": "estimate",
        "note": "Approximate Sancos SC12002SS kitchen mixer price.",
    },
    "Blender Hofmann": {
        "price": 1399000,
        "currency": "UZS",
        "source": "https://asaxiy.uz/ru/product/blender-hofmann-stb2005dcbk-hf",
        "method": "retailer_page",
        "confidence": "variant_exact",
        "note": "Hofmann STB2005DCBK/HF blender listed at 1399000 UZS.",
    },
    "Холодильник Hofmann RF564CDBS/HF": {
        "price": 12535000,
        "old_price": 16070512,
        "currency": "UZS",
        "source": "https://mobilezone.uz/ru/default/product?slug=25816",
        "method": "retailer_page",
        "confidence": "exact",
        "note": "Hofmann RF564CDBS/HF listed at 12535000 UZS.",
    },
    "Микроволновая Печь Hofmann MW720DHSS/HF": {
        "price": 1249000,
        "currency": "UZS",
        "source": "https://smartbazar.uz/uz/products/84626",
        "method": "retailer_page",
        "confidence": "exact",
        "note": "Hofmann MW720DHSS/HF microwave listed at 1249000 UZS.",
    },
    "Cтиральная машина Hofmann WM10512SDOGF HF": {
        "price": 8069000,
        "currency": "UZS",
        "source": "https://asaxiy.uz/product/stiralnaya-mashina-hofmann-wm10512sdogfhf-105-kg",
        "method": "retailer_page",
        "confidence": "exact",
        "note": "Hofmann WM10512SDOGF/HF listed at 8069000 UZS.",
    },
    "Кухонная мойка ABBER Wasser Kreis AF2194": {
        "price": 30900,
        "currency": "RUB",
        "source": "https://gemy-russia.ru/",
        "method": "supplier_catalog_estimate",
        "confidence": "estimate",
        "note": "Approximate ABBER Wasser Kreis AF2194 kitchen sink price.",
    },
    "Телевизор Samsung UHD 4K Smart TV RU7097 Series 7": {
        "price": 64990,
        "currency": "RUB",
        "source": "https://www.samsung.com/ru/tvs/ru7097/",
        "method": "manufacturer_model_estimate",
        "confidence": "estimate",
        "note": "Approximate RU7097 series 4K TV market price.",
    },
    "Ландыши": {
        "price": 1200,
        "currency": "RUB",
        "source": "https://www.ozon.ru/",
        "method": "marketplace_estimate",
        "confidence": "estimate",
        "note": "Approximate artificial lily-of-the-valley bouquet price.",
    },
    "Розы": {
        "price": 1800,
        "currency": "RUB",
        "source": "https://www.ozon.ru/",
        "method": "marketplace_estimate",
        "confidence": "estimate",
        "note": "Approximate artificial roses bouquet price.",
    },
    "Samsung-UE65RU7470U": {
        "price": 2399,
        "old_price": 3299,
        "currency": "GBP",
        "source": "https://www.samsung.com/ru/tvs/uhd-4k-tv/ru7470-65-inch-crystal-uhd-smart-tv-ue65ru7470uxru/",
        "method": "manufacturer_page",
        "confidence": "exact_foreign_currency",
        "note": "Samsung RU7470 65-inch page showed 2399 GBP sale, 3299 GBP original.",
    },
    "HAVELLS ENTICER CEILING FAN": {
        "price": 6500,
        "currency": "INR",
        "source": "https://www.havells.com/",
        "method": "manufacturer_catalog_estimate",
        "confidence": "estimate",
        "note": "Approximate Havells Enticer ceiling fan market price.",
    },
    "Wi-Fi Router Tenda N301": {
        "price": 1199,
        "currency": "RUB",
        "source": "https://www.dns-shop.ru/",
        "method": "retailer_estimate",
        "confidence": "estimate",
        "note": "Approximate Tenda N301 router price.",
    },
    "mouse and keyboard bluetooth combo": {
        "price": 3500,
        "currency": "RUB",
        "source": "https://www.ozon.ru/",
        "method": "marketplace_estimate",
        "confidence": "estimate",
        "note": "Approximate Bluetooth mouse and keyboard combo price.",
    },
    "Om Kiteq": {
        "price": 250000,
        "currency": "RUB",
        "source": "https://ru.kiteq.de/",
        "method": "supplier_catalog_estimate",
        "confidence": "estimate",
        "note": "Approximate premium designer digital/interactive object price.",
    },
    "MacBook Pro 2015": {
        "price": 35000,
        "currency": "RUB",
        "source": "https://www.avito.ru/",
        "method": "used_market_estimate",
        "confidence": "estimate",
        "note": "Approximate used MacBook Pro 2015 market price.",
    },
    "Panasonic TX-65FZR800": {
        "price": 352057,
        "old_price": 378243,
        "currency": "RUB",
        "source": "https://www.kns.ru/product/televizor-panasonic-tx-65fzr800/",
        "method": "retailer_archive_page",
        "confidence": "exact_archived",
        "note": "KNS archived last price for Panasonic TX-65FZR800.",
    },
    "tv01": {
        "price": 29990,
        "currency": "RUB",
        "source": "https://www.dns-shop.ru/",
        "method": "category_estimate",
        "confidence": "estimate",
        "note": "Generic 4K/Smart TV 43-50 inch estimate.",
    },
    "Стиральная машина ATLANT 2014 года серии SMART ACTION": {
        "price": 28990,
        "currency": "RUB",
        "source": "https://atlant.by/bt.atlant.by/catalog/washing_machines/detail.php?ID=16860",
        "method": "manufacturer_model_estimate",
        "confidence": "estimate",
        "note": "Approximate ATLANT SMART ACTION washing machine price.",
    },
    "Телевизор": {
        "price": 24990,
        "currency": "RUB",
        "source": "https://www.dns-shop.ru/",
        "method": "category_estimate",
        "confidence": "estimate",
        "note": "Generic flat-panel TV estimate.",
    },
    "iMac 2017": {
        "price": 45000,
        "currency": "RUB",
        "source": "https://www.avito.ru/",
        "method": "used_market_estimate",
        "confidence": "estimate",
        "note": "Approximate used iMac 2017 market price.",
    },
    "Стиральная машина LG FH0C3ND1": {
        "price": 29990,
        "currency": "RUB",
        "source": "https://www.lg.com/ru/",
        "method": "manufacturer_model_estimate",
        "confidence": "estimate",
        "note": "Approximate LG FH0C3ND1 washing machine price.",
    },
    "LG 55 OMLED TV": {
        "price": 1199.99,
        "currency": "USD",
        "source": "https://www.lg.com/us/tvs/lg-oled55c5pua-oled-4k-tv",
        "method": "newer_variant_proxy",
        "confidence": "estimate",
        "note": "Used current LG 55-inch OLED C5 price as modern proxy.",
    },
    "Водонагреватель DELFA 80l": {
        "price": 11500,
        "currency": "RUB",
        "source": "https://www.ozon.ru/",
        "method": "marketplace_estimate",
        "confidence": "estimate",
        "note": "Approximate 80 liter electric water heater price.",
    },
    "Газовый котел Immergas": {
        "price": 85000,
        "currency": "RUB",
        "source": "https://www.immergas.com/",
        "method": "manufacturer_model_estimate",
        "confidence": "estimate",
        "note": "Approximate Immergas Mini Eolo 28 boiler price.",
    },
    "Sony playstation 3": {
        "price": 8000,
        "currency": "RUB",
        "source": "https://www.avito.ru/",
        "method": "used_market_estimate",
        "confidence": "estimate",
        "note": "Approximate used Sony PlayStation 3 market price.",
    },
    "Keyboard Alien": {
        "price": 3500,
        "currency": "RUB",
        "source": "https://www.ozon.ru/",
        "method": "marketplace_estimate",
        "confidence": "estimate",
        "note": "Approximate gaming keyboard price.",
    },
    "Клавиатура Logitech": {
        "price": 4990,
        "currency": "RUB",
        "source": "https://www.dns-shop.ru/",
        "method": "retailer_estimate",
        "confidence": "estimate",
        "note": "Approximate wireless Logitech backlit keyboard price.",
    },
    "coffee grinder": {
        "price": 4500,
        "currency": "RUB",
        "source": "https://www.ozon.ru/",
        "method": "marketplace_estimate",
        "confidence": "estimate",
        "note": "Approximate domestic coffee grinder price.",
    },
    "Homemade tea": {
        "price": 2500,
        "currency": "RUB",
        "source": "https://www.ozon.ru/",
        "method": "marketplace_estimate",
        "confidence": "estimate",
        "note": "Approximate glass teapot/kettle price.",
    },
    "Кофемашина BOSCH, benvenuto B30.": {
        "price": 25000,
        "currency": "RUB",
        "source": "https://www.avito.ru/",
        "method": "used_market_estimate",
        "confidence": "estimate",
        "note": "Approximate used Bosch Benvenuto B30 coffee machine price.",
    },
    "OM Astov BBQ зонт вытяжной": {
        "price": 95000,
        "currency": "RUB",
        "source": "https://astov.ru/",
        "method": "supplier_catalog_estimate",
        "confidence": "estimate",
        "note": "Approximate custom BBQ extraction hood price.",
    },
    "Духовая печь Bosch Serie 8": {
        "price": 99990,
        "currency": "RUB",
        "source": "https://www.bosch-home.ru/",
        "method": "manufacturer_model_estimate",
        "confidence": "estimate",
        "note": "Approximate Bosch Serie 8 built-in oven price.",
    },
    "Варочная панель CATA RCI 631 WH": {
        "price": 45990,
        "currency": "RUB",
        "source": "https://www.cata.ru/",
        "method": "manufacturer_model_estimate",
        "confidence": "estimate",
        "note": "Approximate CATA RCI 631 WH hob price.",
    },
    "Induction hob - Placa inducción NEFF T58TS11N0": {
        "price": 1025,
        "old_price": 1295,
        "currency": "EUR",
        "source": "https://nesridiscount.com/index.php/table-de-cuisson-grande-largeur-n90-neff-t58ts11n0.html",
        "method": "retailer_archive_page",
        "confidence": "exact_foreign_currency",
        "note": "NEFF T58TS11N0 archived retailer sale price 1025 EUR, old price 1295 EUR.",
    },
    "Микроволновая печь": {
        "price": 7990,
        "currency": "RUB",
        "source": "https://www.dns-shop.ru/",
        "method": "category_estimate",
        "confidence": "estimate",
        "note": "Generic solo microwave 20L class price.",
    },
    "SCARLETT": {
        "price": 2016,
        "currency": "RUB",
        "source": "https://www.tehnozont.ru/product/multivarka-scarlett-sc-mc410s01/",
        "method": "retailer_archive_page",
        "confidence": "exact_archived",
        "note": "Scarlett SC-MC410S01 archived price 2016 RUB.",
    },
    "Hansa Integra FCEW 54120": {
        "price": 39299,
        "currency": "RUB",
        "source": "https://hansa.ru/product/elektricheskaya-plita-hansa-fcew54120",
        "method": "manufacturer_shop_page",
        "confidence": "exact",
        "note": "Official Hansa FCEW54120 page listed at 39299 RUB.",
    },
    "Кухонная вытяжка \"Galiano ppo\" 90 mm": {
        "price": 27603,
        "old_price": 29680,
        "currency": "RUB",
        "source": "https://www.elisbt.ru/catalog/item/britanika_90_loft/",
        "method": "same_supplier_category_proxy",
        "confidence": "estimate",
        "note": "Used comparable Vialona Cappe 90 cm decorative hood price.",
    },
}


def _load_manual_urls(path: Path) -> set[str]:
    urls: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text and not text.startswith("#"):
            urls.add(text)
    return urls


def _record_from_row(row: sqlite3.Row) -> ProductRecord:
    payload = {column: row[column] for column in PRODUCT_COLUMNS}
    return ProductRecord(**payload)


def _update_extra(extra_json: str, price_info: dict[str, Any]) -> str:
    try:
        extra = json.loads(extra_json or "{}")
    except Exception:
        extra = {}
    if not isinstance(extra, dict):
        extra = {}
    extra["web_price_backfill"] = {
        "updated_at": now_utc_iso(),
        "price_value": price_info["price"],
        "price_currency": price_info["currency"],
        "old_price_value": price_info.get("old_price"),
        "source_url": price_info["source"],
        "method": price_info["method"],
        "confidence": price_info["confidence"],
        "note": price_info["note"],
    }
    return json.dumps(extra, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sourse/suppliers/manual_ingest_urls.txt")
    ap.add_argument("--db", default="out/supplier_ingest/suppliers.db")
    ap.add_argument("--out-dir", default="out/supplier_ingest/items")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    urls = _load_manual_urls(input_path)

    updated = 0
    missing: list[str] = []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT *
            FROM supplier_product
            WHERE source_site = '3ddd'
              AND source_url IN ({})
            ORDER BY title
            """.format(",".join("?" for _ in urls)),
            sorted(urls),
        ).fetchall()

        for row in rows:
            title = str(row["title"] or "").strip()
            price_info = PRICE_DATA.get(title)
            if not price_info:
                missing.append(title)
                continue
            extra_json = _update_extra(str(row["extra_json"] or "{}"), price_info)
            con.execute(
                """
                UPDATE supplier_product
                SET price_value = ?,
                    price_currency = ?,
                    old_price_value = ?,
                    extra_json = ?
                WHERE unique_key = ?
                """,
                (
                    float(price_info["price"]),
                    price_info["currency"],
                    float(price_info["old_price"]) if price_info.get("old_price") is not None else None,
                    extra_json,
                    row["unique_key"],
                ),
            )
            updated += 1

            patched = dict(row)
            patched["price_value"] = float(price_info["price"])
            patched["price_currency"] = price_info["currency"]
            patched["old_price_value"] = (
                float(price_info["old_price"]) if price_info.get("old_price") is not None else None
            )
            patched["extra_json"] = extra_json
            save_metadata_json(ProductRecord(**{column: patched[column] for column in PRODUCT_COLUMNS}), out_dir)

    print(f"[3ddd_price_backfill] urls={len(urls)} db_rows={len(rows)} updated={updated} missing={len(missing)}")
    if missing:
        for title in missing:
            print(f"[3ddd_price_backfill] missing_price_data: {title}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
