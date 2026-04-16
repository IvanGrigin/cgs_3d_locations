# -*- coding: utf-8 -*-
"""
This script probes a single supplier asset page and download candidate.
It is used for debugging extraction quality without running a full batch crawl.
The code prints detailed diagnostics about links, files, and model formats.
It intentionally keeps the logic isolated from the main acquisition path.
Keep this script easy to run and safe to discard outputs from.
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.suppliers.utils import DEFAULT_HEADERS

MODEL_EXTENSIONS = [
    ".glb",
    ".gltf",
    ".blend",
    ".fbx",
    ".obj",
    ".zip",
    ".rar",
    ".7z",
    ".3ds",
    ".max",
    ".dae",
    ".skp",
    ".step",
    ".stp",
    ".dwg",
    ".rvt",
    ".ifc",
]


@dataclass
class ProbeResult:
    supplier: str
    product_url: str

    title: Optional[str] = None
    brand: Optional[str] = None
    collection: Optional[str] = None
    related_items: list[dict[str, Any]] | None = None

    category_raw: Optional[str] = None
    description: Optional[str] = None

    width_m: Optional[float] = None
    depth_m: Optional[float] = None
    height_m: Optional[float] = None
    weight_kg: Optional[float] = None

    price_value: Optional[float] = None
    price_currency: Optional[str] = None
    price_type: Optional[str] = None

    materials: list[str] | None = None

    download_url: Optional[str] = None
    download_filename: Optional[str] = None
    download_content_type: Optional[str] = None
    downloaded_ok: bool = False
    downloaded_size_bytes: int = 0

    candidate_formats: list[str] | None = None
    preview_images: list[str] | None = None

    required_fields_ok: bool = False
    missing_fields: list[str] | None = None

    notes: list[str] | None = None
    raw_meta: dict[str, Any] | None = None


def fetch_html(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def text_or_none(node) -> Optional[str]:
    if node is None:
        return None
    txt = node.get_text(" ", strip=True)
    return txt or None


def uniq_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tag in soup.select('script[type="application/ld+json"]'):
        raw = tag.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    out.append(item)
    return out


def first_jsonld_of_type(items: list[dict[str, Any]], expected_type: str) -> Optional[dict[str, Any]]:
    expected_type = expected_type.lower()
    for item in items:
        t = item.get("@type")
        if isinstance(t, str) and t.lower() == expected_type:
            return item
        if isinstance(t, list):
            for x in t:
                if isinstance(x, str) and x.lower() == expected_type:
                    return item
    return None


def parse_ru_float(value: str) -> Optional[float]:
    value = value.strip().replace(",", ".")
    value = re.sub(r"[^\d\.]+", "", value)
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None


def cm_to_m(text: str) -> Optional[float]:
    value = parse_ru_float(text)
    if value is None:
        return None
    return value / 100.0


WEIGHT_PATTERNS = [
    re.compile(r"вес\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*кг", re.IGNORECASE),
    re.compile(r"weight\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*kg", re.IGNORECASE),
]


def parse_weight_kg(text: str) -> Optional[float]:
    for pattern in WEIGHT_PATTERNS:
        m = pattern.search(text)
        if m:
            return parse_ru_float(m.group(1))
    return None


def extract_preview_images(soup: BeautifulSoup, base_url: str) -> list[str]:
    out: list[str] = []
    for img in soup.select("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        abs_url = urljoin(base_url, src)
        low = abs_url.lower()
        if any(x in low for x in ["logo", "sprite", "icon", "banner", "adv/"]):
            continue
        out.append(abs_url)
    return uniq_keep_order(out)[:20]


def infer_formats_from_text(text: str) -> list[str]:
    low = f" {text.lower()} "
    out: list[str] = []

    mapping = {
        "glb": [".glb", " glb ", " gltf ", ".gltf"],
        "blend": [".blend", " blend "],
        "fbx": [".fbx", " fbx "],
        "obj": [".obj", " obj "],
        "3ds": [".3ds", " 3ds "],
        "max": [".max", " max ", "3ds max"],
        "dae": [".dae", " collada "],
        "skp": [".skp", " sketchup "],
        "zip": [".zip"],
        "rar": [".rar"],
        "7z": [".7z"],
        "dwg": [".dwg", " autocad "],
        "rvt": [".rvt", " revit "],
        "ifc": [".ifc", " ifc "],
    }

    for fmt, needles in mapping.items():
        if any(n in low for n in needles):
            out.append(fmt)

    return uniq_keep_order(out)


def guess_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    if name:
        return name
    return "download.bin"


def download_file(url: str, out_path: Path, timeout: int = 60) -> tuple[bool, int, str | None]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")

        total = 0
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)

    return total > 0, total, content_type


def choose_download_url_for_imodern(
    soup: BeautifulSoup,
    html: str,
    base_url: str,
) -> tuple[Optional[str], list[str]]:
    notes: list[str] = []
    candidates: list[str] = []

    # 1. Явные ссылки
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        low = abs_url.lower()

        if href == "#model":
            notes.append("found #model anchor in href")
        elif any(low.endswith(ext) for ext in MODEL_EXTENSIONS):
            candidates.append(abs_url)

    # 2. data-* атрибуты
    for tag in soup.find_all(True):
        for attr in ["data-href", "data-url", "data-file", "data-download", "data-model", "data-src"]:
            value = tag.attrs.get(attr)
            if not value or not isinstance(value, str):
                continue
            abs_url = urljoin(base_url, value.strip())
            low = abs_url.lower()
            if any(low.endswith(ext) for ext in MODEL_EXTENSIONS):
                candidates.append(abs_url)

    # 3. onclick в любом теге — это то, что у тебя на скрине
    for tag in soup.find_all(True):
        onclick = tag.attrs.get("onclick")
        if not onclick or not isinstance(onclick, str):
            continue

        # window.location.href='/upload/...zip'
        m = re.search(r"""window\.location\.href\s*=\s*['"]([^'"]+)['"]""", onclick)
        if m:
            abs_url = urljoin(base_url, m.group(1))
            candidates.append(abs_url)

        # Любой URL с модельным расширением внутри onclick
        for found in re.findall(r"""https?://[^\s'"]+|/[^'"\s)]+""", onclick):
            abs_url = urljoin(base_url, found)
            low = abs_url.lower()
            if any(low.endswith(ext) for ext in MODEL_EXTENSIONS):
                candidates.append(abs_url)

    # 4. inline scripts
    for script in soup.select("script"):
        script_text = script.get_text(" ", strip=False)
        if not script_text:
            continue

        if "3d" not in script_text.lower() and "model" not in script_text.lower() and "upload" not in script_text.lower():
            continue

        for found in re.findall(r"""https?://[^\s'"]+|/upload/[^\s'"]+""", script_text):
            abs_url = urljoin(base_url, found)
            low = abs_url.lower()
            if any(low.endswith(ext) for ext in MODEL_EXTENSIONS):
                candidates.append(abs_url)

    candidates = uniq_keep_order(candidates)

    preferred_ext_order = [".glb", ".gltf", ".blend", ".fbx", ".obj", ".zip", ".rar", ".7z", ".max", ".3ds"]
    for ext in preferred_ext_order:
        for c in candidates:
            if c.lower().endswith(ext):
                return c, notes

    return (candidates[0], notes) if candidates else (None, notes)


def extract_imodern_fields(
    soup: BeautifulSoup,
    page_text: str,
    product_ld: dict[str, Any],
) -> dict[str, Any]:
    title = None
    h1 = soup.select_one("h1")
    if h1:
        title = text_or_none(h1)
    if not title:
        title = product_ld.get("name")

    description = None
    desc_node = soup.select_one('[itemprop="description"]')
    if desc_node:
        description = re.sub(r"\s+", " ", desc_node.get_text(" ", strip=True)).strip()

    width_m = depth_m = height_m = None

    m = re.search(r"Ширина:\s*([0-9.,]+)\s*см", page_text, re.IGNORECASE)
    if m:
        width_m = cm_to_m(m.group(1))

    m = re.search(r"Глубина:\s*([0-9.,]+)\s*см", page_text, re.IGNORECASE)
    if m:
        depth_m = cm_to_m(m.group(1))

    m = re.search(r"Высота(?:\s+общая)?:\s*([0-9.,]+)\s*-\s*([0-9.,]+)\s*см", page_text, re.IGNORECASE)
    if m:
        height_m = cm_to_m(m.group(2))
    else:
        m = re.search(r"Высота(?:\s+общая)?:\s*([0-9.,]+)\s*см", page_text, re.IGNORECASE)
        if m:
            height_m = cm_to_m(m.group(1))

    materials: list[str] = []
    for label in ["Обивка", "Сиденье и спинка", "Каркас", "Материал", "Корпус", "Ножки"]:
        mm = re.search(rf"{re.escape(label)}:\s*([^\n<]+)", page_text, re.IGNORECASE)
        if mm:
            materials.append(f"{label}: {mm.group(1).strip()}")
    materials = uniq_keep_order(materials)

    price_value = None
    price_currency = None
    price_type = "unknown"

    # Цена из HTML-блока карточки
    m = re.search(r"(\d[\d\s]{3,})\s*руб", page_text, re.IGNORECASE)
    if m:
        try:
            price_value = float(re.sub(r"\s+", "", m.group(1)))
            price_currency = "RUB"
            price_type = "explicit"
        except Exception:
            pass

    brand = "Imodern"
    collection = None
    category_raw = None

    # category из meta
    meta_cat = soup.find("meta", attrs={"itemprop": "category"})
    if meta_cat and meta_cat.get("content"):
        category_raw = meta_cat["content"].strip()

    return {
        "title": title,
        "brand": brand,
        "collection": collection,
        "category_raw": category_raw,
        "description": description,
        "width_m": width_m,
        "depth_m": depth_m,
        "height_m": height_m,
        "materials": materials,
        "price_value": price_value,
        "price_currency": price_currency,
        "price_type": price_type,
    }


def extract_imodern_related_items(soup: BeautifulSoup) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for node in soup.select(".sim_prod"):
        raw = node.get("data-product")
        if not raw:
            continue
        try:
            payload = json.loads(html_lib.unescape(raw))
        except Exception:
            continue

        item = {
            "name": payload.get("name"),
            "id": payload.get("id"),
            "price": payload.get("price"),
            "brand": payload.get("brand"),
            "category": payload.get("category"),
            "list": payload.get("list"),
            "position": payload.get("position"),
        }
        out.append(item)

    return out[:20]


def probe_single_asset(
    supplier: str,
    product_url: str,
    out_dir: Path,
    explicit_download_url: Optional[str] = None,
) -> ProbeResult:
    html = fetch_html(product_url)
    soup = soup_from_html(html)
    page_text = soup.get_text("\n", strip=True)
    jsonlds = extract_json_ld(soup)
    product_ld = (
        first_jsonld_of_type(jsonlds, "Product")
        or first_jsonld_of_type(jsonlds, "IndividualProduct")
        or {}
    )

    notes: list[str] = []
    supplier_l = supplier.lower()

    title = None
    brand = None
    collection = None
    category_raw = None
    description = None
    width_m = depth_m = height_m = None
    materials: list[str] = []
    price_value = None
    price_currency = None
    price_type = "unknown"
    related_items: list[dict[str, Any]] = []

    if supplier_l == "imodern":
        fields = extract_imodern_fields(soup, page_text, product_ld)
        title = fields["title"]
        brand = fields["brand"]
        collection = fields["collection"]
        category_raw = fields["category_raw"]
        description = fields["description"]
        width_m = fields["width_m"]
        depth_m = fields["depth_m"]
        height_m = fields["height_m"]
        materials = fields["materials"]
        price_value = fields["price_value"]
        price_currency = fields["price_currency"]
        price_type = fields["price_type"]
        related_items = extract_imodern_related_items(soup)
    else:
        h1 = soup.select_one("h1")
        if h1:
            title = text_or_none(h1)
        if not title:
            title = product_ld.get("name")

        brand_value = product_ld.get("brand")
        if isinstance(brand_value, dict):
            brand = brand_value.get("name")
        elif isinstance(brand_value, str):
            brand = brand_value

        category_raw = product_ld.get("category")
        description = product_ld.get("description")

    weight_kg = parse_weight_kg(page_text)
    preview_images = extract_preview_images(soup, product_url)

    if explicit_download_url:
        download_url = explicit_download_url
    elif supplier_l == "imodern":
        download_url, dl_notes = choose_download_url_for_imodern(soup, html, product_url)
        notes.extend(dl_notes)
    else:
        download_url = None

    candidate_formats = infer_formats_from_text((download_url or "") + " " + page_text)

    downloaded_ok = False
    downloaded_size_bytes = 0
    download_filename = None
    download_content_type = None

    if download_url:
        try:
            download_filename = guess_filename_from_url(download_url)
            target = out_dir / download_filename
            downloaded_ok, downloaded_size_bytes, download_content_type = download_file(download_url, target)

            ext = Path(download_filename).suffix.lower()
            is_model_ext = ext in MODEL_EXTENSIONS
            is_html = download_content_type and "text/html" in download_content_type.lower()

            if is_html:
                downloaded_ok = False
                notes.append("downloaded file is html page, not a model")
            elif not is_model_ext:
                notes.append(f"downloaded file extension is suspicious: {ext or '<none>'}")
        except Exception as e:
            notes.append(f"download failed: {type(e).__name__}: {e}")
    else:
        notes.append("direct model url not found")

    required_fields = {
        "title": title,
        "brand": brand,
        "width_m": width_m,
        "depth_m": depth_m,
        "height_m": height_m,
        "price_value": price_value,
        "price_currency": price_currency,
        "download_url": download_url,
        "downloaded_ok": downloaded_ok if download_url else None,
    }

    missing_fields = [k for k, v in required_fields.items() if v in (None, "", [])]
    required_fields_ok = len(missing_fields) == 0

    return ProbeResult(
        supplier=supplier,
        product_url=product_url,
        title=title,
        brand=brand,
        collection=collection,
        related_items=related_items,
        category_raw=category_raw,
        description=description,
        width_m=width_m,
        depth_m=depth_m,
        height_m=height_m,
        weight_kg=weight_kg,
        price_value=price_value,
        price_currency=price_currency,
        price_type=price_type,
        materials=materials or None,
        download_url=download_url,
        download_filename=download_filename,
        download_content_type=download_content_type,
        downloaded_ok=downloaded_ok,
        downloaded_size_bytes=downloaded_size_bytes,
        candidate_formats=candidate_formats,
        preview_images=preview_images,
        required_fields_ok=required_fields_ok,
        missing_fields=missing_fields,
        notes=notes,
        raw_meta={
            "jsonld_product": product_ld,
        },
    )


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplier", required=True)
    parser.add_argument("--product-url", required=True)
    parser.add_argument("--download-url", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    started = time.time()
    result = probe_single_asset(
        supplier=args.supplier,
        product_url=args.product_url,
        out_dir=out_dir,
        explicit_download_url=args.download_url,
    )
    elapsed = time.time() - started

    payload = asdict(result)
    payload["elapsed_sec"] = round(elapsed, 3)

    save_json(report_path, payload)

    print(f"report: {report_path}")
    print(f"title: {result.title}")
    print(f"download_url: {result.download_url}")
    print(f"downloaded_ok: {result.downloaded_ok}")
    print(f"downloaded_size_bytes: {result.downloaded_size_bytes}")
    print(f"download_content_type: {result.download_content_type}")
    print(f"missing_fields: {result.missing_fields}")
    print(f"notes: {result.notes}")

    if not result.required_fields_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
