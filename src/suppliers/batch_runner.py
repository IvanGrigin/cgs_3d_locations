# -*- coding: utf-8 -*-
"""
This module runs batched supplier crawling across supported sites.
It discovers product URLs, applies per-site crawl plans, and stores results.
The code coordinates adapters, fallback enrichment, and persistence.
It is the main entrypoint for scalable catalog collection.
Keep site discovery rules isolated and deterministic.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from src.suppliers.db import init_db, insert_fetch_log, upsert_products
from src.suppliers.registry import build_adapters
from src.suppliers.runner import coerce_product_record, save_metadata_json
from src.suppliers.utils import DEFAULT_HEADERS


NON_HTML_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".svg",
    ".pdf",
    ".zip",
    ".rar",
    ".7z",
    ".obj",
    ".fbx",
    ".3ds",
    ".max",
    ".blend",
    ".glb",
    ".gltf",
    ".css",
    ".js",
    ".xml",
    ".ico",
}


@dataclass(frozen=True)
class SiteBatchPlan:
    site_name: str
    root_url: str
    seed_urls: tuple[str, ...]
    product_path_markers: tuple[str, ...]
    category_path_markers: tuple[str, ...]
    deny_path_markers: tuple[str, ...] = ()
    discovery_mode: str = "generic"


SITE_BATCH_PLANS = {
    "homeconcept": SiteBatchPlan(
        site_name="homeconcept",
        root_url="https://homeconcept.ru/",
        seed_urls=(
            "https://homeconcept.ru/3d-models/",
        ),
        product_path_markers=("/catalog/product/",),
        category_path_markers=("/catalog/", "/3d-models/"),
        deny_path_markers=("/brands/", "/stores/", "/delivery/", "/fitting/", "/news/"),
        discovery_mode="homeconcept_library",
    ),
    "imodern": SiteBatchPlan(
        site_name="imodern",
        root_url="https://imodern.ru/",
        seed_urls=(
            "https://imodern.ru/sitemap.xml",
        ),
        product_path_markers=("/product/",),
        category_path_markers=("/catalog/", "/mebel/", "/svet/", "/decor/", "/sale/"),
        deny_path_markers=("/brands/", "/blog/", "/news/", "/about/", "/contacts/", "/delivery/"),
        discovery_mode="sitemap",
    ),
    "loftdesigne": SiteBatchPlan(
        site_name="loftdesigne",
        root_url="https://loftdesigne.ru/",
        seed_urls=(
            "https://loftdesigne.ru/catalog/stulya/",
            "https://loftdesigne.ru/catalog/stoly/",
            "https://loftdesigne.ru/catalog/divany-i-kresla/",
            "https://loftdesigne.ru/catalog/svet/",
            "https://loftdesigne.ru/catalog/",
        ),
        product_path_markers=("/catalog/products/",),
        category_path_markers=("/catalog/",),
        deny_path_markers=("/brands/", "/blog/", "/news/", "/contacts/", "/delivery/", "/information/"),
    ),
    "3ddd": SiteBatchPlan(
        site_name="3ddd",
        root_url="https://3ddd.ru/",
        seed_urls=(
            "https://3ddd.ru/3dmodels",
        ),
        product_path_markers=("/3dmodels/show/",),
        category_path_markers=("/3dmodels",),
        deny_path_markers=("/users/", "/auth/", "/finance", "/jobs/", "/gallery/", "/blog/", "/forum/"),
        discovery_mode="3ddd_api",
    ),
    "sancos": SiteBatchPlan(
        site_name="sancos",
        root_url="https://sancos.su/",
        seed_urls=(
            "https://sancos.su/sitemap.xml",
        ),
        product_path_markers=("/catalog/",),
        category_path_markers=("/catalog/", "/collections/"),
        deny_path_markers=("/wheretobuy/", "/news/", "/about/", "/contacts/", "/delivery/", "/download/"),
        discovery_mode="sitemap",
    ),
    "timotrader": SiteBatchPlan(
        site_name="timotrader",
        root_url="https://timotrader.ru/",
        seed_urls=(
            "https://timotrader.ru/3d-modeli",
        ),
        product_path_markers=("/katalog/",),
        category_path_markers=("/3d-modeli",),
        deny_path_markers=("/assets/", "/contacts/", "/dostavka/", "/servis/", "/o-magazine/"),
        discovery_mode="timotrader_3d_listing",
    ),
    "cersanit": SiteBatchPlan(
        site_name="cersanit",
        root_url="https://cersanit.ru/",
        seed_urls=(
            "https://cersanit.ru/catalog/mito/3d-be/",
        ),
        product_path_markers=("/catalog/mito/3d-be/",),
        category_path_markers=("/catalog/mito/3d-be/",),
        deny_path_markers=("/download/", "/catalog/pdf/", "/upload/", "/blog/", "/gde-kupit/"),
        discovery_mode="cersanit_collection",
    ),
}


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path or "/"
    host = parsed.netloc.lower()

    query = ""
    if path.lower() == "/3dmodels":
        allowed_query_keys = {"cat", "subcat", "types", "page", "order", "query"}
        filtered_query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=False) if key in allowed_query_keys]
        query = urlencode(filtered_query, doseq=True)
    elif parsed.netloc.lower() in {"cersanit.ru", "www.cersanit.ru"}:
        allowed_query_keys = {"PAGEN_1"}
        filtered_query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=False) if key in allowed_query_keys]
        query = urlencode(filtered_query, doseq=True)

    normalized = parsed._replace(query=query, fragment="")
    if path != "/" and path.endswith("/") and host not in {"sancos.su", "www.sancos.su"}:
        normalized = normalized._replace(path=path.rstrip("/"))
    return urlunparse(normalized)


def same_host(url: str, root_url: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(root_url).netloc.lower()


def is_non_html_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in NON_HTML_EXTENSIONS)


def is_product_url(url: str, plan: SiteBatchPlan) -> bool:
    path = urlparse(url).path.lower()
    return any(marker in path for marker in plan.product_path_markers)


def is_category_url(url: str, plan: SiteBatchPlan) -> bool:
    path = urlparse(url).path.lower()
    if is_non_html_url(url):
        return False
    if is_product_url(url, plan):
        return False
    if any(marker in path for marker in plan.deny_path_markers):
        return False
    if path in {"", "/"}:
        return True
    return any(marker in path for marker in plan.category_path_markers)


def extract_links(html: str, base_url: str, root_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        if not same_host(absolute, root_url):
            continue
        if is_non_html_url(absolute):
            continue
        out.add(absolute)

    return out


def extract_sitemap_locs(xml_text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r"<loc>\s*(.*?)\s*</loc>", xml_text, flags=re.IGNORECASE | re.DOTALL):
        text = (match.group(1) or "").strip()
        if not text:
            continue
        normalized = normalize_url(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    if out:
        return out

    soup = BeautifulSoup(xml_text, "html.parser")
    for loc in soup.find_all("loc"):
        text = (loc.get_text(strip=True) or "").strip()
        if not text:
            continue
        normalized = normalize_url(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    return out


def extract_homeconcept_library_product_urls(html: str, base_url: str, plan: SiteBatchPlan) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()

    for card in soup.select(".items-list-3d-models .item"):
        link = card.select_one(".item-link-image a[href]")
        if not link or not link.get("href"):
            continue
        absolute = normalize_url(urljoin(base_url, link["href"]))
        if not same_host(absolute, plan.root_url):
            continue
        if not is_product_url(absolute, plan):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        out.append(absolute)

    return out


def extract_homeconcept_library_fallback_map(html: str, base_url: str, plan: SiteBatchPlan) -> dict[str, dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, dict[str, str]] = {}

    for card in soup.select(".items-list-3d-models .item"):
        link = card.select_one(".item-link-image a[href]")
        download = card.select_one(".item-price a[href]")
        title_node = card.select_one(".item-name")

        if not link or not link.get("href"):
            continue

        product_url = normalize_url(urljoin(base_url, link["href"]))
        if not same_host(product_url, plan.root_url):
            continue
        if not is_product_url(product_url, plan):
            continue

        title = title_node.get_text(" ", strip=True) if title_node else ""
        title = title.strip()

        model_download_url = ""
        if download and download.get("href"):
            model_download_url = normalize_url(urljoin(base_url, download["href"]))

        out[product_url] = {
            "title": title,
            "model_download_url": model_download_url,
        }

    return out


def discover_product_urls_homeconcept_library(
    adapter,
    plan: SiteBatchPlan,
    limit: int,
) -> list[str]:
    library_url = normalize_url(plan.seed_urls[0])
    log(f"[{plan.site_name}] library listing: {library_url}")

    try:
        html, final_url = adapter.fetch_html(library_url)
    except Exception as exc:
        log(f"[{plan.site_name}] library error: {library_url} -> {type(exc).__name__}: {exc}")
        return []

    product_urls = extract_homeconcept_library_product_urls(html, final_url, plan)
    log(f"[{plan.site_name}] library discovered products: {len(product_urls)}")
    return product_urls[:limit]


def extract_timotrader_listing_product_urls(html: str, base_url: str, plan: SiteBatchPlan) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()

    for card in soup.select("#products .tm-product-item"):
        link = card.select_one(".tm-media-box[href]") or card.select_one(".tm-product-card-body a[href]")
        if not link or not link.get("href"):
            continue

        absolute = normalize_url(urljoin(base_url, link["href"]))
        if not same_host(absolute, plan.root_url):
            continue
        if not is_product_url(absolute, plan):
            continue
        if absolute in seen:
            continue

        seen.add(absolute)
        out.append(absolute)

    return out


def extract_timotrader_listing_page_numbers(html: str) -> set[int]:
    soup = BeautifulSoup(html, "html.parser")
    pages = {1}
    for a in soup.select("ul.uk-pagination a[href], .uk-pagination a[href]"):
        text = a.get_text(" ", strip=True)
        href = str(a.get("href") or "")

        for value in (text, dict(parse_qsl(urlparse(href).query)).get("page") or ""):
            try:
                page = int(str(value).strip())
            except Exception:
                continue
            if page > 0:
                pages.add(page)

    return pages


def timotrader_listing_page_url(seed_url: str, page: int) -> str:
    if page <= 1:
        return seed_url

    parsed = urlparse(seed_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=False))
    query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))


def discover_product_urls_timotrader_3d_listing(
    adapter,
    plan: SiteBatchPlan,
    limit: int,
    max_listing_pages: int,
) -> list[str]:
    seed_url = plan.seed_urls[0]
    seen_products: set[str] = set()
    discovered_products: list[str] = []
    pages_to_visit: deque[int] = deque([1])
    queued_pages: set[int] = {1}
    visited_pages: set[int] = set()

    while pages_to_visit and len(visited_pages) < max_listing_pages and len(discovered_products) < limit:
        page = pages_to_visit.popleft()
        if page in visited_pages:
            continue

        visited_pages.add(page)
        listing_url = timotrader_listing_page_url(seed_url, page)
        log(f"[{plan.site_name}] 3d listing page {page}: {listing_url}")

        try:
            html, final_url = adapter.fetch_html(listing_url)
        except Exception as exc:
            log(f"[{plan.site_name}] listing error: {listing_url} -> {type(exc).__name__}: {exc}")
            continue

        added_products = 0
        for product_url in extract_timotrader_listing_product_urls(html, final_url, plan):
            if product_url in seen_products:
                continue
            seen_products.add(product_url)
            discovered_products.append(product_url)
            added_products += 1
            if len(discovered_products) >= limit:
                break

        for next_page in sorted(extract_timotrader_listing_page_numbers(html)):
            if next_page in queued_pages or next_page in visited_pages:
                continue
            queued_pages.add(next_page)
            pages_to_visit.append(next_page)

        log(
            f"[{plan.site_name}] +products={added_products} "
            f"queued_pages={len(pages_to_visit)} total_products={len(discovered_products)}"
        )

    return discovered_products[:limit]


def extract_cersanit_collection_product_urls(html: str, base_url: str, plan: SiteBatchPlan) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()

    for a in soup.select("a.catalog-list-item__info[href], a.catalog-list-item__pic-area[href]"):
        href = str(a.get("href") or "").strip()
        if not href:
            continue

        absolute = normalize_url(urljoin(base_url, href))
        if not same_host(absolute, plan.root_url):
            continue
        if not absolute.startswith(plan.seed_urls[0].rstrip("/")):
            continue
        if absolute in seen:
            continue

        seen.add(absolute)
        out.append(absolute)

    return out


def extract_cersanit_collection_page_urls(html: str, base_url: str, plan: SiteBatchPlan) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    seen: set[str] = set()

    for a in soup.select("a[href*='PAGEN_1=']"):
        href = str(a.get("href") or "").strip()
        if not href:
            continue

        absolute = normalize_url(urljoin(base_url, href))
        if not same_host(absolute, plan.root_url):
            continue
        if not absolute.startswith(plan.seed_urls[0].rstrip("/")):
            continue
        if absolute in seen:
            continue

        seen.add(absolute)
        out.append(absolute)

    return out


def discover_product_urls_cersanit_collection(
    adapter,
    plan: SiteBatchPlan,
    limit: int,
    max_listing_pages: int,
) -> list[str]:
    seed_url = normalize_url(plan.seed_urls[0])
    pages_to_visit: deque[str] = deque([seed_url])
    queued_pages: set[str] = {seed_url}
    visited_pages: set[str] = set()
    seen_products: set[str] = set()
    discovered_products: list[str] = []

    while pages_to_visit and len(visited_pages) < max_listing_pages and len(discovered_products) < limit:
        current_url = pages_to_visit.popleft()
        if current_url in visited_pages:
            continue

        visited_pages.add(current_url)
        log(f"[{plan.site_name}] collection page {len(visited_pages)}/{max_listing_pages}: {current_url}")

        try:
            html, final_url = adapter.fetch_html(current_url)
        except Exception as exc:
            log(f"[{plan.site_name}] collection error: {current_url} -> {type(exc).__name__}: {exc}")
            continue

        added_products = 0
        for product_url in extract_cersanit_collection_product_urls(html, final_url, plan):
            if product_url in seen_products:
                continue
            seen_products.add(product_url)
            discovered_products.append(product_url)
            added_products += 1
            if len(discovered_products) >= limit:
                break

        for page_url in extract_cersanit_collection_page_urls(html, final_url, plan):
            if page_url in visited_pages or page_url in queued_pages:
                continue
            queued_pages.add(page_url)
            pages_to_visit.append(page_url)

        log(
            f"[{plan.site_name}] +products={added_products} "
            f"queued_pages={len(pages_to_visit)} total_products={len(discovered_products)}"
        )

    return discovered_products[:limit]


def build_site_fallback_map(adapter, plan: SiteBatchPlan) -> dict[str, dict[str, str]]:
    if plan.discovery_mode != "homeconcept_library":
        return {}

    library_url = normalize_url(plan.seed_urls[0])
    try:
        html, final_url = adapter.fetch_html(library_url)
    except Exception as exc:
        log(f"[{plan.site_name}] fallback map error: {library_url} -> {type(exc).__name__}: {exc}")
        return {}

    fallback_map = extract_homeconcept_library_fallback_map(html, final_url, plan)
    log(f"[{plan.site_name}] fallback map built: {len(fallback_map)} entries")
    return fallback_map


def discover_product_urls_from_sitemap(
    adapter,
    plan: SiteBatchPlan,
    limit: int,
    max_listing_pages: int,
) -> list[str]:
    queue: deque[str] = deque(normalize_url(url) for url in plan.seed_urls)
    visited_sitemaps: set[str] = set()
    seen_products: set[str] = set()
    discovered_products: list[str] = []

    while queue and len(visited_sitemaps) < max_listing_pages and len(discovered_products) < limit:
        sitemap_url = queue.popleft()
        if sitemap_url in visited_sitemaps:
            continue

        visited_sitemaps.add(sitemap_url)
        log(f"[{plan.site_name}] sitemap {len(visited_sitemaps)}/{max_listing_pages}: {sitemap_url}")

        try:
            xml_text, final_url = adapter.fetch_html(sitemap_url)
        except Exception as exc:
            log(f"[{plan.site_name}] sitemap error: {sitemap_url} -> {type(exc).__name__}: {exc}")
            continue

        locs = extract_sitemap_locs(xml_text)
        added_products = 0
        queued_sitemaps = 0

        for loc in locs:
            if not same_host(loc, plan.root_url):
                continue

            path = urlparse(loc).path.lower()
            if path.endswith(".xml"):
                if loc not in visited_sitemaps:
                    queue.append(loc)
                    queued_sitemaps += 1
                continue

            if not is_product_url(loc, plan):
                continue

            if loc in seen_products:
                continue

            seen_products.add(loc)
            discovered_products.append(loc)
            added_products += 1

            if len(discovered_products) >= limit:
                break

        log(
            f"[{plan.site_name}] sitemap discovered products: +{added_products}, "
            f"queued sitemaps: +{queued_sitemaps}, total products: {len(discovered_products)}"
        )

    return discovered_products[:limit]


def build_3ddd_listing_payload(listing_url: str) -> dict[str, object]:
    params = parse_qsl(urlparse(listing_url).query, keep_blank_values=False)
    payload: dict[str, object] = {}

    categories = [value for key, value in params if key == "subcat" and value]
    if categories:
        payload["categories"] = categories

    types = [value for key, value in params if key == "types" and value]
    if types:
        payload["types"] = types

    for source_key, target_key in (
        ("query", "query"),
        ("order", "order"),
    ):
        value = next((v for k, v in params if k == source_key and v), None)
        if value:
            payload[target_key] = value

    page_value = next((v for k, v in params if k == "page" and v), None)
    try:
        payload["page"] = max(1, int(page_value)) if page_value else 1
    except Exception:
        payload["page"] = 1

    return payload


def discover_product_urls_3ddd_api(
    adapter,
    plan: SiteBatchPlan,
    limit: int,
    max_listing_pages: int,
) -> list[str]:
    discovered_products: list[str] = []
    seen_products: set[str] = set()
    pages_fetched = 0

    for listing_url in plan.seed_urls:
        if len(discovered_products) >= limit or pages_fetched >= max_listing_pages:
            break

        base_payload = build_3ddd_listing_payload(listing_url)
        page = int(base_payload.get("page", 1) or 1)
        total_pages: int | None = None

        while len(discovered_products) < limit and pages_fetched < max_listing_pages:
            request_payload = dict(base_payload)
            request_payload["page"] = page
            pages_fetched += 1

            log(f"[{plan.site_name}] api listing {pages_fetched}/{max_listing_pages}: page={page} payload={request_payload}")

            try:
                response = requests.post(
                    urljoin(plan.root_url, "/api/models"),
                    headers={
                        **DEFAULT_HEADERS,
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=(10, adapter.timeout),
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                log(f"[{plan.site_name}] api listing error: page={page} -> {type(exc).__name__}: {exc}")
                break

            data = payload.get("data") if isinstance(payload, dict) else None
            models = data.get("models") if isinstance(data, dict) else None
            if not isinstance(models, list) or not models:
                log(f"[{plan.site_name}] api listing returned no models on page={page}")
                break

            added_products = 0
            for model in models:
                if not isinstance(model, dict):
                    continue

                slug = str(model.get("slug") or "").strip()
                if not slug:
                    continue

                product_url = normalize_url(urljoin(plan.root_url, f"/3dmodels/show/{slug}"))
                if product_url in seen_products:
                    continue

                seen_products.add(product_url)
                discovered_products.append(product_url)
                added_products += 1

                if len(discovered_products) >= limit:
                    break

            per_page = data.get("per_page") if isinstance(data, dict) else None
            total_value = data.get("total_value") if isinstance(data, dict) else None
            try:
                per_page_int = max(1, int(per_page))
                total_value_int = int(total_value)
                total_pages = max(1, (total_value_int + per_page_int - 1) // per_page_int)
            except Exception:
                total_pages = None

            log(
                f"[{plan.site_name}] api discovered products: +{added_products} "
                f"total_products={len(discovered_products)} total_pages={total_pages or '?'}"
            )

            page += 1
            if total_pages is not None and page > total_pages:
                break

    return discovered_products[:limit]


def discover_product_urls(
    adapter,
    plan: SiteBatchPlan,
    limit: int,
    max_listing_pages: int,
    max_depth: int,
) -> list[str]:
    if plan.discovery_mode == "homeconcept_library":
        return discover_product_urls_homeconcept_library(
            adapter=adapter,
            plan=plan,
            limit=limit,
        )

    if plan.discovery_mode == "timotrader_3d_listing":
        return discover_product_urls_timotrader_3d_listing(
            adapter=adapter,
            plan=plan,
            limit=limit,
            max_listing_pages=max_listing_pages,
        )

    if plan.discovery_mode == "cersanit_collection":
        return discover_product_urls_cersanit_collection(
            adapter=adapter,
            plan=plan,
            limit=limit,
            max_listing_pages=max_listing_pages,
        )

    if plan.discovery_mode == "sitemap":
        return discover_product_urls_from_sitemap(
            adapter=adapter,
            plan=plan,
            limit=limit,
            max_listing_pages=max_listing_pages,
        )

    if plan.discovery_mode == "3ddd_api":
        return discover_product_urls_3ddd_api(
            adapter=adapter,
            plan=plan,
            limit=limit,
            max_listing_pages=max_listing_pages,
        )

    queue: deque[tuple[str, int]] = deque((normalize_url(url), 0) for url in plan.seed_urls)
    queued_pages: set[str] = {normalize_url(url) for url in plan.seed_urls}
    visited_pages: set[str] = set()
    seen_products: set[str] = set()
    discovered_products: list[str] = []

    while queue and len(discovered_products) < limit and len(visited_pages) < max_listing_pages:
        current_url, depth = queue.popleft()
        visited_pages.add(current_url)
        log(f"[{plan.site_name}] listing {len(visited_pages)}/{max_listing_pages}: {current_url}")

        try:
            html, final_url = adapter.fetch_html(current_url)
        except Exception as exc:
            log(f"[{plan.site_name}] listing error: {current_url} -> {type(exc).__name__}: {exc}")
            continue

        links = extract_links(html, final_url, plan.root_url)
        product_candidates = []
        category_candidates = []

        for link in links:
            if is_product_url(link, plan):
                if link not in seen_products:
                    seen_products.add(link)
                    product_candidates.append(link)
                continue

            if depth >= max_depth:
                continue

            if is_category_url(link, plan) and link not in queued_pages and link not in visited_pages:
                category_candidates.append(link)

        for product_url in product_candidates:
            discovered_products.append(product_url)
            if len(discovered_products) >= limit:
                break

        for category_url in category_candidates:
            queued_pages.add(category_url)
            queue.append((category_url, depth + 1))

        log(
            f"[{plan.site_name}] +products={len(product_candidates)} "
            f"+categories={len(category_candidates)} total_products={len(discovered_products)}"
        )

    return discovered_products[:limit]


def compute_discovery_limit(requested_limit: int) -> int:
    return requested_limit + max(10, min(50, requested_limit))


def prioritize_product_urls(plan: SiteBatchPlan, product_urls: list[str]) -> list[str]:
    def priority(url: str) -> tuple[int, str]:
        path = urlparse(url).path.lower()

        if plan.site_name == "imodern":
            score = 100

            if "/product/stul-" in path:
                score = 0
            elif "/product/kreslo-" in path:
                score = 1
            elif "/product/divan-" in path:
                score = 2
            elif "/product/krovat-" in path:
                score = 3
            elif "/product/" in path:
                score = 10

            if "/product/komplekt-iz-" in path:
                score += 100
            if "/product/zapchast-" in path:
                score += 200

            return score, url

        return 0, url

    return sorted(product_urls, key=priority)


def load_existing_product_urls(db_path: Path, site_name: str) -> set[str]:
    if not db_path.exists():
        return set()

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT product_url
            FROM supplier_product
            WHERE source_site = ?
              AND product_url IS NOT NULL
              AND product_url != ''
            """,
            (site_name,),
        ).fetchall()

    return {normalize_url(row[0]) for row in rows if row and row[0]}


def merge_extra_json(raw_json: str, patch: dict) -> str:
    base = {}
    if raw_json:
        try:
            loaded = json.loads(raw_json)
            if isinstance(loaded, dict):
                base = loaded
        except Exception:
            base = {}
    base.update(patch)
    return json.dumps(base, ensure_ascii=False)


def apply_fallback_to_product(product, fallback_entry: dict[str, str] | None) -> None:
    if not fallback_entry:
        return

    fallback_title = (fallback_entry.get("title") or "").strip()
    fallback_model_url = (fallback_entry.get("model_download_url") or "").strip()
    used_fallback = False

    if (not product.title) and fallback_title:
        product.title = fallback_title
        used_fallback = True

    if (not product.model_download_url) and fallback_model_url:
        product.model_download_url = fallback_model_url
        product.model_link_type = product.model_link_type or "direct_file"
        product.model_page_url = product.model_page_url or product.product_url or product.source_url
        product.model_vendor_url = product.model_vendor_url or product.product_url or product.source_url
        product.model_download_filename = product.model_download_filename or adapter_filename_from_url(fallback_model_url)
        product.model_format = product.model_format or adapter_ext_from_url(fallback_model_url)
        method = product.model_extraction_method or "product_page"
        if "library_fallback" not in method:
            product.model_extraction_method = f"{method}+library_fallback"
        used_fallback = True

    if used_fallback:
        product.extra_json = merge_extra_json(
            product.extra_json,
            {
                "library_fallback_used": True,
                "library_fallback_title": fallback_title or None,
                "library_fallback_model_download_url": fallback_model_url or None,
            },
        )


def adapter_filename_from_url(url: str | None) -> str | None:
    from src.suppliers.adapters.base import SupplierAdapter

    return SupplierAdapter.filename_from_url(url)


def adapter_ext_from_url(url: str | None) -> str | None:
    from src.suppliers.adapters.base import SupplierAdapter

    return SupplierAdapter.ext_from_url(url)


def process_single_product(
    adapter,
    site_name: str,
    url: str,
    db_path: Path,
    out_dir: Path,
    fallback_map: dict[str, dict[str, str]] | None = None,
) -> tuple[int, int]:
    try:
        html, final_url = adapter.fetch_html(url)
        raw_items = adapter.parse(url, html, final_url)
        products = [coerce_product_record(item, adapter, url, final_url) for item in raw_items]

        if not products:
            if getattr(adapter, "empty_parse_is_skip", False):
                insert_fetch_log(
                    db_path=db_path,
                    source_site=site_name,
                    source_url=url,
                    fetched_at=adapter.now_utc_iso(),
                    ok=True,
                    error="skip: empty adapter result",
                )
                log(f"[{site_name}] skipped (no downloadable model): {url}")
                return 0, 0
            raise ValueError("adapter returned zero records for product page")

        if fallback_map:
            normalized_product_url = normalize_url(final_url)
            fallback_entry = fallback_map.get(normalized_product_url)
            for product in products:
                product_url = normalize_url(product.product_url or final_url)
                apply_fallback_to_product(product, fallback_map.get(product_url) or fallback_entry)

        upsert_products(db_path, products)
        insert_fetch_log(
            db_path=db_path,
            source_site=site_name,
            source_url=url,
            fetched_at=products[0].parsed_at,
            ok=True,
            error=None,
        )

        meta_paths = [save_metadata_json(product, out_dir) for product in products]
        log(
            f"[{site_name}] saved {len(products)} record(s); "
            f"title={products[0].title!r}; metadata={meta_paths[0]}"
        )
        return len(products), 0
    except Exception as exc:
        insert_fetch_log(
            db_path=db_path,
            source_site=site_name,
            source_url=url,
            fetched_at=adapter.now_utc_iso(),
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        log(f"[{site_name}] product error: {url} -> {type(exc).__name__}: {exc}")
        return 0, 1


def process_product_urls_parallel(
    adapter,
    site_name: str,
    product_urls: Iterable[str],
    db_path: Path,
    out_dir: Path,
    workers: int,
    fallback_map: dict[str, dict[str, str]] | None = None,
    limit_success: int | None = None,
) -> tuple[int, int]:
    product_urls = list(product_urls)
    saved = 0
    failed = 0

    for index, url in enumerate(product_urls, start=1):
        log(f"[{site_name}] queue product {index}/{len(product_urls)}: {url}")

    if limit_success is not None:
        for url in product_urls:
            current_saved, current_failed = process_single_product(
                adapter,
                site_name,
                url,
                db_path,
                out_dir,
                fallback_map,
            )
            saved += current_saved
            failed += current_failed

            if saved >= limit_success:
                break

        return saved, failed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        in_flight: dict[object, str] = {}
        iterator = iter(product_urls)

        while len(in_flight) < max(1, workers):
            try:
                next_url = next(iterator)
            except StopIteration:
                break
            future = executor.submit(process_single_product, adapter, site_name, next_url, db_path, out_dir, fallback_map)
            in_flight[future] = next_url

        while in_flight:
            for future in as_completed(list(in_flight.keys()), timeout=None):
                in_flight.pop(future, None)
                current_saved, current_failed = future.result()
                saved += current_saved
                failed += current_failed

                if limit_success is not None and saved >= limit_success:
                    return saved, failed

                try:
                    next_url = next(iterator)
                except StopIteration:
                    next_url = None

                if next_url is not None:
                    next_future = executor.submit(
                        process_single_product,
                        adapter,
                        site_name,
                        next_url,
                        db_path,
                        out_dir,
                        fallback_map,
                    )
                    in_flight[next_future] = next_url
                break

    return saved, failed


def build_adapter_map() -> dict[str, object]:
    return {adapter.site_name: adapter for adapter in build_adapters()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="homeconcept,imodern,loftdesigne,3ddd,sancos,timotrader,cersanit")
    ap.add_argument("--limit-per-site", type=int, default=500)
    ap.add_argument("--max-listing-pages", type=int, default=24)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing-products", action="store_true")
    ap.add_argument("--db", default="out/supplier_ingest/suppliers.db")
    ap.add_argument("--out-dir", default="out/supplier_ingest/items")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    init_db(db_path)

    adapter_map = build_adapter_map()
    requested_sites = [site.strip() for site in args.sites.split(",") if site.strip()]

    total_saved = 0
    total_failed = 0

    for site_name in requested_sites:
        plan = SITE_BATCH_PLANS.get(site_name)
        adapter = adapter_map.get(site_name)

        if plan is None or adapter is None:
            log(f"[{site_name}] skipped: no batch plan or adapter")
            continue

        log(f"[{site_name}] start")

        try:
            fallback_map = build_site_fallback_map(adapter, plan)
            discovery_limit = compute_discovery_limit(args.limit_per_site)
            product_urls = discover_product_urls(
                adapter=adapter,
                plan=plan,
                limit=discovery_limit,
                max_listing_pages=args.max_listing_pages,
                max_depth=args.max_depth,
            )
            product_urls = prioritize_product_urls(plan, product_urls)
            log(f"[{site_name}] total discovered product urls: {len(product_urls)}")

            if not product_urls:
                log(f"[{site_name}] no product urls discovered")
                continue

            if args.skip_existing_products:
                existing_urls = load_existing_product_urls(db_path, site_name)
                before_count = len(product_urls)
                product_urls = [url for url in product_urls if normalize_url(url) not in existing_urls]
                log(
                    f"[{site_name}] skip existing products: "
                    f"before={before_count}, existing={before_count - len(product_urls)}, left={len(product_urls)}"
                )

            if not product_urls:
                log(f"[{site_name}] nothing left after skip-existing-products")
                continue

            saved, failed = process_product_urls_parallel(
                adapter=adapter,
                site_name=site_name,
                product_urls=product_urls,
                db_path=db_path,
                out_dir=out_dir,
                workers=args.workers,
                fallback_map=fallback_map,
                limit_success=args.limit_per_site,
            )
            total_saved += saved
            total_failed += failed
            log(f"[{site_name}] done: saved={saved}, failed={failed}")
        except Exception as exc:
            total_failed += 1
            log(f"[{site_name}] fatal site error: {type(exc).__name__}: {exc}")

    log(f"all sites done: saved={total_saved}, failed={total_failed}")


if __name__ == "__main__":
    main()
