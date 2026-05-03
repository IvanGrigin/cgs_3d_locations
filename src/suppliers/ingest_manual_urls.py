# -*- coding: utf-8 -*-
"""
Manual supplier URL ingest entrypoint.

Use this when product pages are chosen by hand and need to be parsed through
the same adapter -> ProductRecord -> DB/items path as the regular pipeline.
The input file can be:
  - .txt/.list/.urls: one URL per line, '#' comments allowed
  - .jsonl: one JSON string or object per line
  - .json: list of strings or objects

It can also auto-link pre-downloaded local 3ddd archives by matching archive
filenames to 3ddd image basenames returned by the product API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.suppliers.db import insert_fetch_log, upsert_products
from src.suppliers.db_core import init_db, upsert_asset
from src.suppliers.registry import find_adapter
from src.suppliers.runner import coerce_product_record, save_metadata_json
from src.suppliers.acquire_site_assets import (
    CONVERTIBLE_EXTS,
    PREFERRED_EXPORT_EXTS,
    SEARCH_EXTS,
    build_asset_record,
    build_blender_job_spec,
    inspect_archive,
    slugify,
    _pick_model_from_extracted_dir,
)
from src.suppliers.utils import json_loads_or


def _read_text_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        rows.append({"url": text, "_line_no": line_no})
    return rows


def _read_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, str):
            rows.append({"url": payload, "_line_no": line_no})
        elif isinstance(payload, dict):
            row = dict(payload)
            row["_line_no"] = line_no
            rows.append(row)
        else:
            raise TypeError(f"Unsupported JSONL row type at line {line_no}: {type(payload).__name__}")
    return rows


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected JSON list in {path}, got {type(payload).__name__}")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            rows.append({"url": item, "_line_no": index})
        elif isinstance(item, dict):
            row = dict(item)
            row["_line_no"] = index
            rows.append(row)
        else:
            raise TypeError(f"Unsupported JSON item type at index {index}: {type(item).__name__}")
    return rows


def load_manual_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".list", ".urls"}:
        return _read_text_lines(path)
    if suffix == ".jsonl":
        return _read_jsonl_lines(path)
    if suffix == ".json":
        return _read_json_list(path)
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _merge_manual_metadata(product, row: dict[str, Any], input_path: Path, final_url: str) -> None:
    try:
        extra = json.loads(product.extra_json or "{}")
    except Exception:
        extra = {}
    if not isinstance(extra, dict):
        extra = {}

    manual_meta: dict[str, Any] = {
        "ingest_mode": "manual_urls",
        "input_file": str(input_path),
        "input_line": int(row.get("_line_no") or 0),
    }

    for key in (
        "label",
        "notes",
        "local_archive_path",
        "downloaded_at",
        "source",
        "room",
        "category_hint",
    ):
        value = row.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value not in (None, "", [], {}):
            manual_meta[key] = value

    source_url = _clean_text(row.get("url"))
    if source_url:
        manual_meta["input_url"] = source_url
    if final_url and final_url != source_url:
        manual_meta["final_url"] = final_url

    local_archive_path = _clean_text(row.get("local_archive_path"))
    if local_archive_path:
        archive_name = Path(local_archive_path).name
        archive_ext = Path(local_archive_path).suffix.lower()
        if archive_name and not product.model_download_filename:
            product.model_download_filename = archive_name
        if archive_ext and not product.model_format:
            product.model_format = archive_ext

    extra["manual_ingest"] = manual_meta
    product.extra_json = json.dumps(extra, ensure_ascii=False)


def _normalized_archive_stem(path: Path) -> str:
    raw_name = path.name if path.is_dir() else path.stem
    stem = re.sub(r"\s+\(\d+\)$", "", raw_name.strip())
    return _path_token(stem)


def _title_token(value: Any) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return re.sub(r"_+", "_", token).strip("_")


def _path_token(value: Any) -> str:
    token = re.sub(r"[^\w]+", "_", str(value or "").strip().casefold(), flags=re.UNICODE)
    return re.sub(r"_+", "_", token).strip("_")


ARCHIVE_EXTS = {".zip", ".rar", ".7z"}


def _build_archive_index(archive_dir: Path) -> list[Path]:
    asset_paths = [
        path
        for path in archive_dir.iterdir()
        if (
            (path.is_file() and path.suffix.lower() in {*ARCHIVE_EXTS, *SEARCH_EXTS})
            or (path.is_dir() and not path.name.startswith("."))
        )
    ]
    asset_paths.sort(
        key=lambda path: (
            _normalized_archive_stem(path),
            0 if re.search(r"\s+\(\d+\)$", path.stem) is None else 1,
            -path.stat().st_mtime,
        )
    )
    return asset_paths


def _product_image_basenames(product) -> list[str]:
    basenames: list[str] = []
    seen: set[str] = set()
    for item in json_loads_or(product.images_json, []):
        url = str(item or "").strip()
        if not url:
            continue
        base = Path(urlparse(url).path).stem.strip()
        if base and base not in seen:
            seen.add(base)
            basenames.append(base)
    return basenames


def _archive_match_candidates(product) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    def add_name_tokens(value: str | None) -> None:
        add(_title_token(value))
        path_token = _path_token(value)
        add(path_token)

    for base in _product_image_basenames(product):
        add(base)
        prefix = base.split(".", 1)[0].strip()
        if prefix.isdigit():
            add(prefix)

    filename = str(getattr(product, "model_download_filename", "") or "").strip()
    if filename:
        add(Path(filename).stem)

    external_id = str(getattr(product, "external_id", "") or "").strip()
    if external_id:
        add_name_tokens(external_id)

    title = str(getattr(product, "title", "") or "").strip()
    if title:
        add_name_tokens(title)

    product_url = str(getattr(product, "product_url", "") or "").strip()
    path_slug = Path(urlparse(product_url).path).stem
    path_slug = re.sub(r"^\d+[-_]+", "", path_slug).strip()
    if path_slug:
        add_name_tokens(path_slug)

    return candidates


def _match_local_archive(product, archive_paths: list[Path]) -> tuple[Path | None, list[str]]:
    notes: list[str] = []
    if not archive_paths:
        return None, ["auto_archive_scan:no_archives_found"]

    candidates = _archive_match_candidates(product)
    if not candidates:
        return None, ["auto_archive_scan:no_product_candidates"]

    notes.append(f"auto_archive_candidates:{candidates}")
    exact_matches: list[Path] = []
    prefix_matches: list[Path] = []
    token_matches: list[Path] = []

    for archive_path in archive_paths:
        stem_norm = _normalized_archive_stem(archive_path)
        for candidate in candidates:
            candidate_norm = candidate.casefold()
            if not candidate_norm:
                continue
            if stem_norm == candidate_norm:
                exact_matches.append(archive_path)
                break
            if stem_norm.startswith(candidate_norm + "_") or stem_norm.startswith(candidate_norm + "."):
                prefix_matches.append(archive_path)
                break
            if len(candidate_norm) >= 8 and candidate_norm in stem_norm:
                token_matches.append(archive_path)
                break

    for label, matches in (
        ("exact", exact_matches),
        ("prefix", prefix_matches),
        ("token", token_matches),
    ):
        if matches:
            chosen = matches[0]
            notes.append(
                f"auto_archive_match:{label}:{chosen.name}:candidates={len(matches)}"
            )
            if len(matches) > 1:
                notes.append(
                    "auto_archive_match_others:" + ",".join(path.name for path in matches[1:5])
                )
            return chosen, notes

    notes.append("auto_archive_match:none")
    return None, notes


def _link_local_archive_asset(
    product,
    archive_path: Path,
    assets_root: Path,
    delete_archive_after_extract: bool = False,
) -> tuple[Any, list[str]]:
    item_dir = (assets_root / product.source_site / slugify(product.title or product.unique_key)).resolve()
    item_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.is_dir():
        selected_path, selected_ext, notes = _pick_model_from_extracted_dir(archive_path, record=product)
        if selected_path is None:
            extracted_dir = archive_path / "extracted"
            if extracted_dir.is_dir():
                selected_path, selected_ext, extracted_notes = _pick_model_from_extracted_dir(extracted_dir, record=product)
                notes.extend([f"local_dir_extracted:{extracted_dir}", *extracted_notes])
        notes = [f"local_dir:{archive_path}", *notes]

        extra = json_loads_or(product.extra_json, {})
        if not isinstance(extra, dict):
            extra = {}
        extra["manual_local_dir"] = {
            "dir_path": str(archive_path),
            "selected_path": selected_path,
            "selected_ext": selected_ext,
        }

        if selected_path and selected_ext in PREFERRED_EXPORT_EXTS:
            asset = build_asset_record(
                record=product,
                status="local_dir_preferred",
                asset_format=selected_ext.lstrip("."),
                asset_source_url=product.product_url,
                asset_local_path=selected_path,
                preview_local_path=None,
                blender_job_path=None,
                notes=notes,
                extra=extra,
            )
            return asset, notes

        if selected_path and selected_ext:
            blender_job_path = build_blender_job_spec(
                record=product,
                item_dir=item_dir,
                reason="manual_dir_requires_conversion",
                source_path=selected_path,
                preview_path=None,
            )
            asset = build_asset_record(
                record=product,
                status="needs_blender_rebuild",
                asset_format=selected_ext.lstrip("."),
                asset_source_url=product.product_url,
                asset_local_path=selected_path,
                preview_local_path=None,
                blender_job_path=blender_job_path,
                notes=notes,
                extra=extra,
            )
            return asset, notes

        asset = build_asset_record(
            record=product,
            status="local_dir_no_supported_model",
            asset_format=None,
            asset_source_url=product.product_url,
            asset_local_path=None,
            preview_local_path=None,
            blender_job_path=None,
            notes=notes,
            extra=extra,
        )
        return asset, notes

    direct_ext = archive_path.suffix.lower()
    if direct_ext in SEARCH_EXTS:
        notes = [f"local_model:{archive_path}", f"local_model_selected:{direct_ext}"]
        extra = json_loads_or(product.extra_json, {})
        if not isinstance(extra, dict):
            extra = {}
        extra["manual_local_model"] = {
            "model_path": str(archive_path),
            "selected_path": str(archive_path),
            "selected_ext": direct_ext,
        }

        status = "local_model_preferred" if direct_ext in PREFERRED_EXPORT_EXTS else "needs_blender_rebuild"
        blender_job_path = None
        if direct_ext in CONVERTIBLE_EXTS and direct_ext not in PREFERRED_EXPORT_EXTS:
            blender_job_path = build_blender_job_spec(
                record=product,
                item_dir=item_dir,
                reason="manual_model_requires_conversion",
                source_path=str(archive_path),
                preview_path=None,
            )

        asset = build_asset_record(
            record=product,
            status=status,
            asset_format=direct_ext.lstrip("."),
            asset_source_url=product.product_url,
            asset_local_path=str(archive_path),
            preview_local_path=None,
            blender_job_path=blender_job_path,
            notes=notes,
            extra=extra,
        )
        return asset, notes

    extract_dir = item_dir / f"extracted_{slugify(archive_path.stem)}"
    selected_path, selected_ext, notes = inspect_archive(archive_path, extract_dir, record=product)
    notes = [f"local_archive:{archive_path}", *notes]
    if (
        delete_archive_after_extract
        and selected_path
        and selected_ext
        and archive_path.suffix.lower() in ARCHIVE_EXTS
    ):
        try:
            archive_path.unlink()
            notes.append("local_archive_deleted_after_extract")
        except Exception as exc:
            notes.append(f"local_archive_delete_failed:{type(exc).__name__}:{exc}")

    extra = json_loads_or(product.extra_json, {})
    if not isinstance(extra, dict):
        extra = {}
    extra["manual_local_archive"] = {
        "archive_path": str(archive_path),
        "extract_dir": str(extract_dir),
        "selected_path": selected_path,
        "selected_ext": selected_ext,
    }

    if selected_path and selected_ext in PREFERRED_EXPORT_EXTS:
        asset = build_asset_record(
            record=product,
            status="archive_extracted_preferred",
            asset_format=selected_ext.lstrip("."),
            asset_source_url=product.product_url,
            asset_local_path=selected_path,
            preview_local_path=None,
            blender_job_path=None,
            notes=notes,
            extra=extra,
        )
        return asset, notes

    if selected_path and selected_ext:
        blender_job_path = build_blender_job_spec(
            record=product,
            item_dir=item_dir,
            reason="manual_archive_requires_conversion",
            source_path=selected_path,
            preview_path=None,
        )
        asset = build_asset_record(
            record=product,
            status="needs_blender_rebuild",
            asset_format=selected_ext.lstrip("."),
            asset_source_url=product.product_url,
            asset_local_path=selected_path,
            preview_local_path=None,
            blender_job_path=blender_job_path,
            notes=notes,
            extra=extra,
        )
        return asset, notes

    asset = build_asset_record(
        record=product,
        status="archive_extract_failed",
        asset_format=archive_path.suffix.lower().lstrip(".") or None,
        asset_source_url=product.product_url,
        asset_local_path=None,
        preview_local_path=None,
        blender_job_path=None,
        notes=notes,
        extra=extra,
    )
    return asset, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sourse/suppliers/manual_ingest_urls.txt")
    ap.add_argument("--db", default="out/supplier_ingest/manual/suppliers.db")
    ap.add_argument("--out-dir", default="out/supplier_ingest/manual/items")
    ap.add_argument("--archive-dir", default=None, help="Optional local archive directory for auto-linking 3ddd downloads")
    ap.add_argument("--assets-root", default="data/sourse/suppliers/manual_assets", help="Workspace directory for extracted local manual assets")
    ap.add_argument("--delete-archives-after-extract", action="store_true")
    ap.add_argument("--replace-assets-for-input", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    archive_dir = Path(args.archive_dir).expanduser().resolve() if str(args.archive_dir or "").strip() else None
    assets_root = Path(args.assets_root).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if archive_dir is not None and not archive_dir.is_dir():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    rows = load_manual_rows(input_path)
    init_db(db_path)
    archive_paths = _build_archive_index(archive_dir) if archive_dir else []

    total_ok = 0
    total_failed = 0
    total_records = 0
    total_assets = 0

    for row in rows:
        url = _clean_text(row.get("url"))
        line_no = int(row.get("_line_no") or 0)
        if not url:
            total_failed += 1
            print(f"[manual_ingest] line={line_no} skipped: empty url", flush=True)
            continue

        try:
            adapter = find_adapter(url)
            html, final_url = adapter.fetch_html(url)
            raw_items = adapter.parse(url, html, final_url)
            products = [coerce_product_record(item, adapter, url, final_url) for item in raw_items]

            if not products:
                if getattr(adapter, "empty_parse_is_skip", False):
                    insert_fetch_log(
                        db_path=db_path,
                        source_site=adapter.site_name,
                        source_url=url,
                        fetched_at=adapter.now_utc_iso(),
                        ok=True,
                        error="skip: empty adapter result",
                    )
                    print(f"[manual_ingest] line={line_no} site={adapter.site_name} records=0 status=skipped_empty_result", flush=True)
                    continue
                raise ValueError(f"Adapter {adapter.site_name} returned zero records")

            assets: list[Any] = []
            for product in products:
                if args.replace_assets_for_input:
                    with sqlite3.connect(db_path) as con:
                        con.execute("DELETE FROM supplier_asset WHERE unique_key = ?", (product.unique_key,))

                local_archive_notes: list[str] = []
                local_archive_path = _clean_text(row.get("local_archive_path"))

                if not local_archive_path and archive_paths:
                    matched_archive, local_archive_notes = _match_local_archive(product, archive_paths)
                    if matched_archive is not None:
                        local_archive_path = str(matched_archive)
                        row["local_archive_path"] = local_archive_path

                if local_archive_notes:
                    merged_notes = _clean_text(row.get("notes"))
                    notes_text = "; ".join(local_archive_notes)
                    row["notes"] = f"{merged_notes}; {notes_text}".strip("; ").strip()

                _merge_manual_metadata(product, row, input_path, final_url)
                if local_archive_path:
                    asset, _asset_notes = _link_local_archive_asset(
                        product=product,
                        archive_path=Path(local_archive_path).expanduser().resolve(),
                        assets_root=assets_root,
                        delete_archive_after_extract=args.delete_archives_after_extract,
                    )
                    assets.append(asset)

            upsert_products(db_path, products)
            insert_fetch_log(
                db_path=db_path,
                source_site=adapter.site_name,
                source_url=url,
                fetched_at=products[0].parsed_at,
                ok=True,
                error=None,
            )

            meta_paths = [save_metadata_json(product, out_dir) for product in products]
            for asset in assets:
                upsert_asset(db_path, asset)
                total_assets += 1
            total_ok += 1
            total_records += len(products)

            print(
                f"[manual_ingest] line={line_no} site={adapter.site_name} records={len(products)} "
                f"title={products[0].title!r} metadata={meta_paths[0]} "
                f"asset_status={getattr(assets[0], 'asset_status', None) if assets else None!r} "
                f"archive={_clean_text(row.get('local_archive_path')) or None!r}",
                flush=True,
            )
        except Exception as exc:
            total_failed += 1
            try:
                adapter = find_adapter(url)
                source_site = adapter.site_name
                fetched_at = adapter.now_utc_iso()
            except Exception:
                source_site = "unknown"
                fetched_at = ""

            if fetched_at:
                insert_fetch_log(
                    db_path=db_path,
                    source_site=source_site,
                    source_url=url,
                    fetched_at=fetched_at,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

            print(f"[manual_ingest] line={line_no} failed: {type(exc).__name__}: {exc}", flush=True)
            if args.strict:
                raise

    print(
        f"[manual_ingest] done input={input_path} urls_ok={total_ok} "
        f"urls_failed={total_failed} records={total_records} assets={total_assets}",
        flush=True,
    )


if __name__ == "__main__":
    main()
