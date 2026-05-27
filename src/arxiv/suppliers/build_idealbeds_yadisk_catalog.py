#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script builds a supplier catalog from the IdealBeds public Yandex Disk.
It matches archives to 3ddd cards, enriches metadata, and stores audit trails.
The output mirrors the same schema used by other supplier sources.
It is intentionally conservative about unresolved archive-to-model matches.
Keep matching heuristics inspectable and reversible.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests

from src.suppliers.adapters.three_ddd import ThreeDDDAdapter
from src.suppliers.db import init_db, insert_fetch_log, upsert_products
from src.suppliers.models import ProductRecord
from src.suppliers.runner import save_metadata_json
from src.suppliers.utils import DEFAULT_HEADERS


PUBLIC_RESOURCES_API = "https://cloud-api.yandex.net/v1/disk/public/resources"
DEFAULT_PUBLIC_KEY = "https://disk.360.yandex.ru/d/k0ShSJqr-IvwoA?w=1"
THREEDDD_SEARCH_API = "https://3ddd.ru/api/models"

GENERIC_ARCHIVE_TOKENS = {
    "fabric",
    "velvet",
    "studio",
    "corner",
    "l-corner",
    "angle",
    "open",
}
GENERIC_CATEGORY_TOKENS = {
    "bed",
    "sofa",
    "chair",
    "armchair",
    "table",
    "desk",
    "daybed",
    "couch",
    "headboard",
    "footstool",
    "ottoman",
    "banquette",
    "bench",
    "kyshetka",
}

MANUAL_QUERY_ALIASES = {
    "cortona slim arm": ["cortona sectional", "rh cortona chaise sectional sofa"],
    "elliot deco bed": ["om elliot", "elliot deco bed"],
    "emilio big and small": ["emilio idealbeds", "om emilio"],
    "grappolo": ["grappolo bed", "idealbeds grappolo bed"],
    "king's cave": ["king cave idealbeds", "om kings cave"],
    "kings cave": ["king cave idealbeds", "om kings cave"],
    "maddox sectional": ["maddox idealbeds", "om maddox"],
    "martna bed": ["martina bed idealbeds", "om martina"],
    "melony": ["melony idealbeds", "melani bed"],
    "memphis big and small": ["memphis idealbeds", "om memphis"],
    "san bernardo": ["san bernardo idealbeds bed", "san bernardo bed"],
    "blake": ["blake idealbeds", "om blake"],
    "lawrence": ["lawrence idealbeds", "om lawrence"],
    "marble": ["marble idealbeds", "om marble"],
    "well": ["well idealbeds", "om well"],
}


@dataclass
class ArchiveItem:
    archive_name: str
    archive_stem: str
    ext: str
    md5: str | None
    sha256: str | None
    size_bytes: int
    download_url: str
    public_key: str
    public_folder_url: str | None
    antivirus_status: str | None
    resource_id: str | None
    created: str | None
    modified: str | None
    inferred_category: str | None
    query_variants: list[str]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).strip().lower()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _norm(text))


def _strip_archive_name(name: str) -> tuple[str, str]:
    p = Path(name)
    stem = p.stem
    stem = re.sub(r"idealbeds", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", " ", stem).strip(" _-")
    return stem.strip(), p.suffix.lower()


def _infer_archive_category(stem: str) -> str | None:
    s = _norm(stem)
    rules: list[tuple[str, list[str]]] = [
        ("bed", [" bed ", " headboard "]),
        ("sofa", [" sofa ", " couch "]),
        ("armchair", [" armchair "]),
        ("chair", [" chair "]),
        ("table", [" table ", " desk "]),
        ("sofa", [" daybed ", " kyshetka "]),
        ("ottoman", [" footstool ", " ottoman ", " banquette ", " bench "]),
        ("mirror", [" mirror "]),
        ("wardrobe", [" wardrobe ", " closet ", " cabinet "]),
    ]
    padded = f" {s} "
    for category, keys in rules:
        if any(key in padded for key in keys):
            return category
    return None


def _build_query_variants(stem: str) -> list[str]:
    raw_tokens = _tokenize(stem)
    if not raw_tokens:
        return []

    primary = [t for t in raw_tokens if t not in GENERIC_ARCHIVE_TOKENS]
    base = " ".join(primary).strip()

    variants: list[str] = []
    if base:
        variants.append(base)

    no_category = [t for t in primary if t not in GENERIC_CATEGORY_TOKENS]
    if no_category:
        variants.append(" ".join(no_category))

    if len(primary) >= 2:
        variants.append(" ".join(primary[:2]))
    if len(no_category) >= 2:
        variants.append(" ".join(no_category[:2]))

    if base:
        variants.append(f"{base} idealbeds")
        variants.append(f"idealbeds {base}")

    alias_key = _norm(stem)
    for key, alias_queries in MANUAL_QUERY_ALIASES.items():
        if alias_key == key:
            variants.extend(alias_queries)
            break

    # Deduplicate, keep meaningful queries only.
    out: list[str] = []
    seen: set[str] = set()
    for q in variants:
        q = q.strip()
        if not q:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def fetch_yadisk_listing(public_key: str, limit: int = 500, timeout: int = 45) -> list[ArchiveItem]:
    resp = requests.get(
        PUBLIC_RESOURCES_API,
        headers={**DEFAULT_HEADERS, "Accept": "application/json"},
        params={"public_key": public_key, "limit": limit},
        timeout=(10, timeout),
    )
    resp.raise_for_status()
    payload = resp.json()
    embedded = payload.get("_embedded") or {}
    items = embedded.get("items") or []
    out: list[ArchiveItem] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "file":
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        stem, ext = _strip_archive_name(name)
        out.append(
            ArchiveItem(
                archive_name=name,
                archive_stem=stem,
                ext=ext,
                md5=item.get("md5"),
                sha256=item.get("sha256"),
                size_bytes=int(item.get("size") or 0),
                download_url=str(item.get("file") or "").strip(),
                public_key=str(item.get("public_key") or public_key),
                public_folder_url=str(payload.get("public_url") or "").strip() or None,
                antivirus_status=item.get("antivirus_status"),
                resource_id=item.get("resource_id"),
                created=item.get("created"),
                modified=item.get("modified"),
                inferred_category=_infer_archive_category(stem),
                query_variants=_build_query_variants(stem),
            )
        )
    return out


def _string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _candidate_category(model: dict[str, Any]) -> str | None:
    category = model.get("category") or {}
    title = str((category.get("title_en") or category.get("title") or "")).strip().lower()
    if not title:
        return None
    mapping: list[tuple[str, str]] = [
        ("table lamp", "lamp"),
        ("wall light", "lamp"),
        ("pendant light", "lamp"),
        ("sconce", "lamp"),
        ("sideboard", "sideboard"),
        ("chest of drawer", "sideboard"),
        ("console", "console_table"),
        ("wardrobe", "wardrobe"),
        ("display cabinet", "wardrobe"),
        ("office furniture", "chair"),
        ("arm chair", "armchair"),
        ("chair", "chair"),
        ("table", "table"),
        ("sofa", "sofa"),
        ("bed", "bed"),
        ("mirror", "mirror"),
    ]
    for key, value in mapping:
        if key in title:
            return value
    return None


def _candidate_parent_category(model: dict[str, Any]) -> str | None:
    parent = model.get("category_parent") or {}
    title = str((parent.get("title_en") or parent.get("title") or "")).strip().lower()
    return title or None


def _score_candidate(archive: ArchiveItem, model: dict[str, Any], query: str) -> tuple[float, dict[str, Any]]:
    archive_text = _norm(archive.archive_stem)
    archive_tokens = [t for t in _tokenize(archive.archive_stem) if t not in GENERIC_ARCHIVE_TOKENS]

    title = str(model.get("title") or "")
    title_en = str(model.get("title_en") or model.get("titleEn") or "")
    slug = str(model.get("slug") or "")
    candidate_text = " ".join(x for x in [title, title_en, slug] if x)
    candidate_norm = _norm(candidate_text)
    candidate_tokens = _tokenize(candidate_text)

    ratio = max(
        _string_similarity(archive_text, _norm(title)),
        _string_similarity(archive_text, _norm(title_en)),
        _string_similarity(archive_text, _norm(slug)),
    )
    jaccard = _token_jaccard(archive_tokens, candidate_tokens)

    query_ratio = max(
        _string_similarity(_norm(query), _norm(title)),
        _string_similarity(_norm(query), _norm(title_en)),
        _string_similarity(_norm(query), _norm(slug)),
    )
    exact_token_bonus = 0.0
    archive_token_set = set(archive_tokens)
    candidate_token_set = set(candidate_tokens)
    if archive_token_set and archive_token_set.issubset(candidate_token_set):
        exact_token_bonus += 0.10
        if len(archive_token_set) <= 2:
            exact_token_bonus += 0.10

    starts_bonus = 0.08 if candidate_norm.startswith(_norm(query)) else 0.0
    contains_bonus = 0.05 if _norm(query) in candidate_norm else 0.0

    category_bonus = 0.0
    cand_category = _candidate_category(model)
    parent_category = _candidate_parent_category(model)
    if archive.inferred_category and cand_category == archive.inferred_category:
        category_bonus += 0.18
    elif archive.inferred_category and cand_category is not None and cand_category != archive.inferred_category:
        category_bonus -= 0.12

    if parent_category == "furniture":
        category_bonus += 0.08
    elif parent_category:
        # This source is furniture-first; lighting/decor false positives are common.
        category_bonus -= 0.32

    type_bonus = 0.03 if str(model.get("model_type") or "").lower() in {"om", "free", "pro"} else 0.0

    score = (
        0.45 * ratio +
        0.28 * jaccard +
        0.16 * query_ratio +
        exact_token_bonus +
        starts_bonus +
        contains_bonus +
        category_bonus +
        type_bonus
    )

    debug = {
        "ratio": round(ratio, 4),
        "jaccard": round(jaccard, 4),
        "query_ratio": round(query_ratio, 4),
        "exact_token_bonus": round(exact_token_bonus, 4),
        "starts_bonus": round(starts_bonus, 4),
        "contains_bonus": round(contains_bonus, 4),
        "category_bonus": round(category_bonus, 4),
        "type_bonus": round(type_bonus, 4),
        "candidate_category": cand_category,
        "candidate_parent_category": parent_category,
    }
    return score, debug


def search_3ddd_models(query: str, timeout: int = 45) -> list[dict[str, Any]]:
    resp = requests.post(
        THREEDDD_SEARCH_API,
        headers={**DEFAULT_HEADERS, "Accept": "application/json", "Content-Type": "application/json"},
        json={"page": 1, "query": query},
        timeout=(10, timeout),
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or {}
    models = data.get("models") or []
    return [m for m in models if isinstance(m, dict)]


def fetch_3ddd_show_payload(slug: str, timeout: int = 45) -> dict[str, Any]:
    resp = requests.post(
        ThreeDDDAdapter.product_api_url,
        headers={**DEFAULT_HEADERS, "Accept": "application/json", "Content-Type": "application/json"},
        json={"slug": slug},
        timeout=(10, timeout),
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise RuntimeError(f"Invalid 3ddd show payload for slug={slug}")
    return payload


def _match_source_bonus(payload: dict[str, Any], slug: str) -> tuple[float, dict[str, Any]]:
    adapter = ThreeDDDAdapter()
    product_url = f"https://3ddd.ru/3dmodels/show/{slug}"
    record = adapter._build_record_from_json(
        url=product_url,
        final_url=product_url,
        raw_payload=json.dumps(payload, ensure_ascii=False),
        payload=payload,
    )
    extra = json.loads(record.extra_json or "{}")
    author = str(extra.get("author") or record.brand or "").strip().lower()
    source_product_url = str(extra.get("source_product_url") or "").strip().lower()

    bonus = 0.0
    if "idealbeds" in author:
        bonus += 0.30
    if "idealbeds.ru" in source_product_url or source_product_url.rstrip("/") == "https://idealbeds.ru":
        bonus += 0.30
    if slug.startswith("om_"):
        bonus += 0.08

    return bonus, {
        "author": extra.get("author") or record.brand,
        "source_product_url": extra.get("source_product_url"),
        "slug_prefix_bonus": 0.08 if slug.startswith("om_") else 0.0,
        "source_bonus": round(bonus, 4),
    }


def match_archive_to_3ddd(
    archive: ArchiveItem,
    timeout: int = 45,
    min_score: float = 0.58,
    query_delay_sec: float = 0.0,
) -> dict[str, Any]:
    seen_slugs: set[str] = set()
    ranked: list[dict[str, Any]] = []

    for query in archive.query_variants:
        if query_delay_sec > 0:
            time.sleep(query_delay_sec)
        try:
            models = search_3ddd_models(query, timeout=timeout)
        except Exception as exc:
            ranked.append({
                "query": query,
                "error": f"{type(exc).__name__}: {exc}",
                "results": [],
            })
            continue

        results: list[dict[str, Any]] = []
        for model in models[:20]:
            slug = str(model.get("slug") or "").strip()
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            score, debug = _score_candidate(archive, model, query)
            results.append(
                {
                    "slug": slug,
                    "title": model.get("title"),
                    "title_en": model.get("title_en") or model.get("titleEn"),
                    "category_title": (model.get("category") or {}).get("title_en") or (model.get("category") or {}).get("title"),
                    "model_type": model.get("model_type"),
                    "score": round(score, 4),
                    "debug": debug,
                    "model": model,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        ranked.append({"query": query, "results": results[:10]})

    flat = [res for pack in ranked for res in pack.get("results", [])]
    flat.sort(key=lambda x: x["score"], reverse=True)

    show_payload = None
    rescored: list[dict[str, Any]] = []
    rescored_candidates: list[dict[str, Any]] = []
    rescored_slugs: set[str] = set()

    for candidate in flat[:5]:
        rescored_candidates.append(candidate)
        rescored_slugs.add(candidate["slug"])

    for pack in ranked:
        query = str(pack.get("query") or "")
        if "idealbeds" not in query:
            continue
        for candidate in (pack.get("results") or [])[:3]:
            if candidate["slug"] in rescored_slugs:
                continue
            rescored_candidates.append(candidate)
            rescored_slugs.add(candidate["slug"])

    for candidate in rescored_candidates:
        payload = fetch_3ddd_show_payload(candidate["slug"], timeout=timeout)
        source_bonus, source_debug = _match_source_bonus(payload, candidate["slug"])
        rescored.append(
            {
                **candidate,
                "show_payload": payload,
                "source_debug": source_debug,
                "reranked_score": round(candidate["score"] + source_bonus, 4),
            }
        )

    if rescored:
        rescored.sort(key=lambda x: x["reranked_score"], reverse=True)
        best = rescored[0]
        show_payload = best["show_payload"]
        matched = best["reranked_score"] >= min_score
    else:
        best = flat[0] if flat else None
        matched = bool(best and best["score"] >= min_score)

    return {
        "archive_name": archive.archive_name,
        "archive_stem": archive.archive_stem,
        "inferred_category": archive.inferred_category,
        "matched": matched,
        "best_score": round(best["score"], 4) if best else None,
        "best_reranked_score": round(best["reranked_score"], 4) if best and best.get("reranked_score") is not None else None,
        "best_slug": best["slug"] if best else None,
        "best_title": best["title"] if best else None,
        "best_source_debug": best.get("source_debug") if best else None,
        "search_traces": ranked,
        "rescored_traces": [
            {
                "slug": x["slug"],
                "title": x["title"],
                "score": x["score"],
                "reranked_score": x["reranked_score"],
                "source_debug": x["source_debug"],
            }
            for x in rescored
        ],
        "show_payload": show_payload,
    }


def _record_from_match(archive: ArchiveItem, match: dict[str, Any]) -> ProductRecord:
    if not match.get("matched") or not match.get("show_payload"):
        raise ValueError(f"Archive is unmatched: {archive.archive_name}")

    payload = match["show_payload"]
    adapter = ThreeDDDAdapter()
    slug = str(match["best_slug"])
    product_url = f"https://3ddd.ru/3dmodels/show/{slug}"
    base_record = adapter._build_record_from_json(
        url=product_url,
        final_url=product_url,
        raw_payload=json.dumps(payload, ensure_ascii=False),
        payload=payload,
    )

    extra = json.loads(base_record.extra_json or "{}")
    category_norm = (
        base_record.category_norm
        or adapter.classify_category(base_record.category_raw)
        or _candidate_category((payload.get("data") or {}))
        or archive.inferred_category
    )
    extra.update(
        {
            "archive_source": "idealbeds_yadisk",
            "archive_name": archive.archive_name,
            "archive_stem": archive.archive_stem,
            "archive_ext": archive.ext,
            "archive_size_bytes": archive.size_bytes,
            "archive_md5": archive.md5,
            "archive_sha256": archive.sha256,
            "archive_download_url": archive.download_url,
            "archive_public_key": archive.public_key,
            "archive_public_folder_url": archive.public_folder_url,
            "archive_antivirus_status": archive.antivirus_status,
            "archive_resource_id": archive.resource_id,
            "archive_created": archive.created,
            "archive_modified": archive.modified,
            "archive_inferred_category": archive.inferred_category,
            "match_score": match.get("best_score"),
            "match_slug": match.get("best_slug"),
            "match_title": match.get("best_title"),
            "match_queries": archive.query_variants,
            "match_search_traces": match.get("search_traces"),
        }
    )

    return ProductRecord(
        unique_key=f"idealbeds_yadisk::archive::{archive.md5 or archive.archive_name}",
        source_site="idealbeds_yadisk",
        source_url=archive.public_folder_url or DEFAULT_PUBLIC_KEY,
        parsed_at=adapter.now_utc_iso(),
        external_id=slug,
        category_raw=base_record.category_raw,
        category_norm=category_norm,
        title=base_record.title or archive.archive_stem,
        brand=base_record.brand,
        collection=base_record.collection,
        product_url=product_url,
        model_link_type="direct_file",
        model_page_url=product_url,
        model_download_url=archive.download_url,
        model_download_landing_url=archive.public_folder_url or DEFAULT_PUBLIC_KEY,
        model_vendor_url=product_url,
        model_extraction_method="idealbeds_yadisk_matched_to_3ddd",
        model_download_filename=archive.archive_name,
        model_format=archive.ext.lstrip(".") or base_record.model_format,
        price_value=base_record.price_value,
        price_currency=base_record.price_currency,
        old_price_value=base_record.old_price_value,
        style=base_record.style,
        color=base_record.color,
        description=base_record.description,
        width_cm=base_record.width_cm,
        depth_cm=base_record.depth_cm,
        height_cm=base_record.height_cm,
        weight_kg=base_record.weight_kg,
        room=base_record.room,
        materials=base_record.materials,
        availability=base_record.availability,
        country_brand=base_record.country_brand,
        production_country=base_record.production_country,
        tags_json=base_record.tags_json,
        images_json=base_record.images_json,
        related_json=base_record.related_json,
        extra_json=json.dumps(extra, ensure_ascii=False),
        raw_html=base_record.raw_html,
    )


def _write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "archive_name",
        "archive_stem",
        "archive_ext",
        "archive_size_mb",
        "inferred_category",
        "matched",
        "best_score",
        "best_reranked_score",
        "best_slug",
        "best_title",
        "query_variants",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build IdealBeds Yandex Disk catalog matched to 3ddd model pages")
    ap.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--audit-json", default=None)
    ap.add_argument("--audit-csv", default=None)
    ap.add_argument("--only-unmatched-from-audit", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N archives after listing")
    ap.add_argument("--min-score", type=float, default=0.58)
    ap.add_argument("--query-delay-sec", type=float, default=0.15)
    ap.add_argument("--timeout", type=int, default=45)
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    audit_json_path = Path(args.audit_json).expanduser().resolve() if args.audit_json else None
    audit_csv_path = Path(args.audit_csv).expanduser().resolve() if args.audit_csv else None
    only_unmatched_audit = Path(args.only_unmatched_from_audit).expanduser().resolve() if args.only_unmatched_from_audit else None

    init_db(db_path)
    archives = fetch_yadisk_listing(args.public_key, limit=500, timeout=args.timeout)
    if only_unmatched_audit:
        payload = json.loads(only_unmatched_audit.read_text(encoding="utf-8"))
        rows = payload.get("rows") or []
        wanted = {str(r.get("archive_name")) for r in rows if not r.get("matched")}
        archives = [a for a in archives if a.archive_name in wanted]
    if args.limit > 0:
        archives = archives[: args.limit]

    matched_records: list[ProductRecord] = []
    audit_rows: list[dict[str, Any]] = []
    matched_count = 0
    unmatched_count = 0

    for index, archive in enumerate(archives, start=1):
        print(f"[{index}/{len(archives)}] match archive: {archive.archive_name}", flush=True)
        try:
            match = match_archive_to_3ddd(
                archive=archive,
                timeout=args.timeout,
                min_score=args.min_score,
                query_delay_sec=args.query_delay_sec,
            )
            if match.get("matched"):
                record = _record_from_match(archive, match)
                matched_records.append(record)
                matched_count += 1
                save_metadata_json(record, out_dir)
                insert_fetch_log(
                    db_path=db_path,
                    source_site="idealbeds_yadisk",
                    source_url=archive.download_url,
                    fetched_at=record.parsed_at,
                    ok=True,
                    error=None,
                )
            else:
                unmatched_count += 1
                insert_fetch_log(
                    db_path=db_path,
                    source_site="idealbeds_yadisk",
                    source_url=archive.download_url,
                    fetched_at=ThreeDDDAdapter.now_utc_iso(),
                    ok=False,
                    error=f"unmatched: best_score={match.get('best_score')}",
                )

            audit_rows.append(
                {
                    "archive_name": archive.archive_name,
                    "archive_stem": archive.archive_stem,
                    "archive_ext": archive.ext,
                    "archive_size_mb": round(archive.size_bytes / 1024 / 1024, 2),
                    "inferred_category": archive.inferred_category,
                    "matched": bool(match.get("matched")),
                    "best_score": match.get("best_score"),
                    "best_reranked_score": match.get("best_reranked_score"),
                    "best_slug": match.get("best_slug"),
                    "best_title": match.get("best_title"),
                    "query_variants": " | ".join(archive.query_variants),
                    "search_traces": match.get("search_traces"),
                    "rescored_traces": match.get("rescored_traces"),
                }
            )
        except Exception as exc:
            unmatched_count += 1
            audit_rows.append(
                {
                    "archive_name": archive.archive_name,
                    "archive_stem": archive.archive_stem,
                    "archive_ext": archive.ext,
                    "archive_size_mb": round(archive.size_bytes / 1024 / 1024, 2),
                    "inferred_category": archive.inferred_category,
                    "matched": False,
                    "best_score": None,
                    "best_reranked_score": None,
                    "best_slug": None,
                    "best_title": None,
                    "query_variants": " | ".join(archive.query_variants),
                    "search_traces": [{"error": f"{type(exc).__name__}: {exc}"}],
                    "rescored_traces": [],
                }
            )
            insert_fetch_log(
                db_path=db_path,
                source_site="idealbeds_yadisk",
                source_url=archive.download_url,
                fetched_at=ThreeDDDAdapter.now_utc_iso(),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    if matched_records:
        upsert_products(db_path, matched_records)

    summary = {
        "public_key": args.public_key,
        "db_path": str(db_path),
        "out_dir": str(out_dir),
        "archive_count": len(archives),
        "matched_count": matched_count,
        "unmatched_count": unmatched_count,
        "min_score": args.min_score,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if audit_json_path:
        audit_json_path.parent.mkdir(parents=True, exist_ok=True)
        audit_json_path.write_text(
            json.dumps({"summary": summary, "rows": audit_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if audit_csv_path:
        _write_audit_csv(audit_csv_path, audit_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
