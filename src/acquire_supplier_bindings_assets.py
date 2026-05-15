#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from .pipeline_artifacts import read_json, write_json
    from .suppliers.acquire_site_assets import acquire_asset_for_record, asset_is_ready, now_utc_iso
    from .suppliers.db_core import init_db, upsert_asset, upsert_product
    from .suppliers.models import ProductRecord
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from pipeline_artifacts import read_json, write_json
    from src.suppliers.acquire_site_assets import acquire_asset_for_record, asset_is_ready, now_utc_iso
    from src.suppliers.db_core import init_db, upsert_asset, upsert_product
    from src.suppliers.models import ProductRecord


SUPPORTED_LOCAL_ASSET_EXTS = {"obj", "fbx", "glb", "gltf"}
LOW_QUALITY_ASSET_STATUSES = {"proxy_generated_with_blender", "needs_blender_rebuild"}
SELECTED_BINDING_STATUSES = {
    "heuristic_top1_selected",
    "heuristic_first_acceptable_selected",
    "llm_reranked_top1_selected",
    "llm_reranked_first_acceptable_selected",
}


def _infer_asset_format(candidate: dict[str, Any]) -> str | None:
    asset_format = str(candidate.get("asset_format") or "").strip().lower()
    if asset_format:
        return asset_format
    local_path = str(candidate.get("asset_local_path") or "").strip()
    if local_path:
        return Path(local_path).suffix.lower().lstrip(".") or None
    return None


def _candidate_has_ready_local_asset(candidate: dict[str, Any]) -> bool:
    local_path = str(candidate.get("asset_local_path") or "").strip()
    if not local_path:
        return False
    path = Path(local_path).expanduser()
    if not path.is_file():
        return False
    return (_infer_asset_format(candidate) or "") in SUPPORTED_LOCAL_ASSET_EXTS


def _candidate_has_real_local_asset(candidate: dict[str, Any]) -> bool:
    if not _candidate_has_ready_local_asset(candidate):
        return False
    status = str(candidate.get("asset_status") or "").strip().lower()
    return status not in LOW_QUALITY_ASSET_STATUSES


def _asset_payload_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    status = str(candidate.get("asset_status") or "").strip().lower()
    local_path = str(candidate.get("asset_local_path") or "").strip()
    payload = {
        "asset_status": candidate.get("asset_status") or "ready_existing_local_asset",
        "asset_source_url": candidate.get("asset_source_url"),
        "preview_local_path": candidate.get("preview_local_path"),
        "blender_job_path": candidate.get("blender_job_path"),
        "asset_notes_json": candidate.get("asset_notes_json"),
        "asset_extra_json": candidate.get("asset_extra_json"),
    }
    if status in LOW_QUALITY_ASSET_STATUSES:
        payload["asset_low_quality_local_path"] = str(Path(local_path).expanduser().resolve()) if local_path else None
    else:
        payload["asset_format"] = _infer_asset_format(candidate)
        payload["asset_local_path"] = str(Path(local_path).expanduser().resolve()) if local_path else None
    return payload


def _asset_payload_from_acquired_asset(asset) -> dict[str, Any]:
    status = str(getattr(asset, "asset_status", "") or "").strip().lower()
    payload = {
        "asset_status": asset.asset_status,
        "asset_source_url": asset.asset_source_url,
        "preview_local_path": asset.preview_local_path,
        "blender_job_path": asset.blender_job_path,
        "asset_notes_json": asset.notes_json,
        "asset_extra_json": asset.extra_json,
    }
    if status in LOW_QUALITY_ASSET_STATUSES:
        payload["asset_low_quality_local_path"] = asset.asset_local_path
    else:
        payload["asset_format"] = asset.asset_format
        payload["asset_local_path"] = asset.asset_local_path
    return payload


def _candidate_is_acceptable(candidate: dict[str, Any]) -> bool:
    acceptability = candidate.get("acceptability")
    if not isinstance(acceptability, dict):
        return True
    return bool(acceptability.get("accepted"))


def _binding_candidate_pool(binding: dict[str, Any]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in [binding.get("chosen_candidate"), *(binding.get("top_candidates") or [])]:
        if not isinstance(candidate, dict):
            continue
        unique_key = str(candidate.get("unique_key") or "").strip()
        if not unique_key or unique_key in seen:
            continue
        seen.add(unique_key)
        pool.append(candidate)
    return pool


def _catalog_row_by_unique_key(catalog_json_paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for json_path in catalog_json_paths:
        data = read_json(json_path)
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list) and isinstance(data, dict) and data.get("unique_key"):
            items = [data]
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            unique_key = str(item.get("unique_key") or "").strip()
            if not unique_key:
                continue
            dims = item.get("dimensions_cm") or {}
            rows[unique_key] = {
                "unique_key": unique_key,
                "source_site": item.get("source_site"),
                "source_url": item.get("source_url"),
                "parsed_at": item.get("parsed_at"),
                "external_id": item.get("external_id"),
                "category_raw": item.get("category_raw"),
                "category_norm": item.get("category_norm"),
                "title": item.get("title"),
                "brand": item.get("brand"),
                "collection": item.get("collection"),
                "product_url": item.get("product_url"),
                "model_link_type": item.get("model_link_type"),
                "model_page_url": item.get("model_page_url"),
                "model_download_url": item.get("model_download_url"),
                "model_download_landing_url": item.get("model_download_landing_url"),
                "model_vendor_url": item.get("model_vendor_url"),
                "model_extraction_method": item.get("model_extraction_method"),
                "model_download_filename": item.get("model_download_filename"),
                "model_format": item.get("model_format"),
                "price_value": item.get("price_value"),
                "price_currency": item.get("price_currency"),
                "old_price_value": item.get("old_price_value"),
                "style": item.get("style"),
                "color": item.get("color"),
                "description": item.get("description"),
                "width_cm": item.get("width_cm", dims.get("width")),
                "depth_cm": item.get("depth_cm", dims.get("depth")),
                "height_cm": item.get("height_cm", dims.get("height")),
                "weight_kg": item.get("weight_kg", dims.get("weight_kg")),
                "room": item.get("room"),
                "materials": item.get("materials"),
                "availability": item.get("availability"),
                "images_json": json.dumps(item.get("images") or [], ensure_ascii=False),
                "extra_json": json.dumps(item.get("extra") or {}, ensure_ascii=False),
                "asset_status": item.get("asset_status"),
                "asset_format": item.get("asset_format"),
                "asset_local_path": item.get("asset_local_path"),
            }
    return rows


def _merge_catalog_fields(candidate: dict[str, Any], catalog_row: dict[str, Any] | None) -> dict[str, Any]:
    if not catalog_row:
        return deepcopy(candidate)
    merged = deepcopy(catalog_row)
    catalog_asset_status = str(catalog_row.get("asset_status") or "").strip().lower()
    preserve_catalog_asset = catalog_asset_status == "trellis_generated_local_asset"
    for key, value in candidate.items():
        if preserve_catalog_asset and key in {
            "asset_status",
            "asset_format",
            "asset_local_path",
            "asset_source_url",
            "asset_notes_json",
            "asset_extra_json",
        }:
            continue
        if value is not None:
            merged[key] = deepcopy(value)
    return merged


def _candidate_to_product_record(candidate: dict[str, Any]) -> ProductRecord:
    unique_key = str(candidate.get("unique_key") or "").strip()
    source_site = str(candidate.get("source_site") or "").strip()
    if not unique_key or not source_site:
        raise RuntimeError("chosen_candidate must have unique_key and source_site")

    source_url = (
        str(candidate.get("source_url") or "").strip()
        or str(candidate.get("product_url") or "").strip()
        or str(candidate.get("model_page_url") or "").strip()
        or str(candidate.get("model_download_url") or "").strip()
        or str(candidate.get("model_download_landing_url") or "").strip()
    )

    return ProductRecord(
        unique_key=unique_key,
        source_site=source_site,
        source_url=source_url,
        parsed_at=str(candidate.get("parsed_at") or now_utc_iso()),
        external_id=candidate.get("external_id"),
        category_raw=candidate.get("category_raw"),
        category_norm=candidate.get("category_norm"),
        title=candidate.get("title"),
        brand=candidate.get("brand"),
        collection=candidate.get("collection"),
        product_url=candidate.get("product_url"),
        model_link_type=candidate.get("model_link_type"),
        model_page_url=candidate.get("model_page_url"),
        model_download_url=candidate.get("model_download_url"),
        model_download_landing_url=candidate.get("model_download_landing_url"),
        model_vendor_url=candidate.get("model_vendor_url"),
        model_extraction_method=candidate.get("model_extraction_method"),
        model_download_filename=candidate.get("model_download_filename"),
        model_format=candidate.get("model_format"),
        price_value=candidate.get("price_value"),
        price_currency=candidate.get("price_currency"),
        old_price_value=candidate.get("old_price_value"),
        style=candidate.get("style"),
        color=candidate.get("color"),
        description=candidate.get("description"),
        width_cm=candidate.get("width_cm"),
        depth_cm=candidate.get("depth_cm"),
        height_cm=candidate.get("height_cm"),
        weight_kg=candidate.get("weight_kg"),
        room=candidate.get("room"),
        materials=candidate.get("materials"),
        availability=candidate.get("availability"),
        images_json=str(candidate.get("images_json") or "[]"),
        extra_json=str(candidate.get("extra_json") or "{}"),
    )


def _apply_asset_payload(candidate: dict[str, Any], payload: dict[str, Any]) -> None:
    if str(payload.get("asset_status") or "").strip().lower() in LOW_QUALITY_ASSET_STATUSES:
        for key in ("asset_format", "asset_local_path"):
            candidate.pop(key, None)
    for key, value in payload.items():
        if value is not None:
            candidate[key] = value


def _apply_payload_to_binding(binding: dict[str, Any], unique_key: str, payload: dict[str, Any]) -> None:
    chosen_candidate = binding.get("chosen_candidate")
    if isinstance(chosen_candidate, dict) and str(chosen_candidate.get("unique_key") or "").strip() == unique_key:
        _apply_asset_payload(chosen_candidate, payload)
    for top_candidate in binding.get("top_candidates") or []:
        if not isinstance(top_candidate, dict):
            continue
        if str(top_candidate.get("unique_key") or "").strip() != unique_key:
            continue
        _apply_asset_payload(top_candidate, payload)


def acquire_assets_for_bindings_data(
    bindings_data: dict[str, Any],
    *,
    db_path: Path,
    out_dir: Path,
    blender_bin: str | None = None,
    catalog_rows_by_key: dict[str, dict[str, Any]] | None = None,
    keep_unresolved_candidates: bool = False,
) -> dict[str, Any]:
    out = deepcopy(bindings_data)
    bindings = out.get("bindings") or []
    if not isinstance(bindings, list):
        raise RuntimeError("Некорректный supplier_bindings JSON: нет bindings")

    init_db(db_path)
    catalog_rows_by_key = catalog_rows_by_key or {}

    selected_binding_count = 0
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        chosen = binding.get("chosen_candidate")
        if not isinstance(chosen, dict):
            continue
        if str(binding.get("selection_status") or "") not in SELECTED_BINDING_STATUSES:
            continue
        selected_binding_count += 1

    ready_before_count = 0
    downloaded_ready_count = 0
    unresolved_count = 0
    failed_count = 0
    cached_payload_by_key: dict[str, dict[str, Any]] = {}
    ready_before_keys: set[str] = set()
    downloaded_ready_keys: set[str] = set()
    unresolved_keys: set[str] = set()
    failed_keys: set[str] = set()

    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        if str(binding.get("selection_status") or "") not in SELECTED_BINDING_STATUSES:
            continue
        chosen = binding.get("chosen_candidate")
        if not isinstance(chosen, dict):
            continue

        selected_real_candidate: dict[str, Any] | None = None
        selected_rank: int | None = None

        for rank_idx, candidate in enumerate(_binding_candidate_pool(binding), start=1):
            if not _candidate_is_acceptable(candidate):
                continue
            unique_key = str(candidate.get("unique_key") or "").strip()
            if not unique_key:
                continue
            merged_candidate = _merge_catalog_fields(candidate, catalog_rows_by_key.get(unique_key))

            payload = cached_payload_by_key.get(unique_key)
            if payload is None:
                if _candidate_has_ready_local_asset(merged_candidate):
                    payload = _asset_payload_from_candidate(merged_candidate)
                    if _candidate_has_real_local_asset(merged_candidate):
                        ready_before_keys.add(unique_key)
                    else:
                        unresolved_keys.add(unique_key)
                else:
                    try:
                        record = _candidate_to_product_record(merged_candidate)
                        upsert_product(db_path, record)
                        asset = acquire_asset_for_record(record, db_path=db_path, out_dir=out_dir, blender_bin=blender_bin)
                        upsert_asset(db_path, asset)
                        payload = _asset_payload_from_acquired_asset(asset)
                        candidate_after = deepcopy(merged_candidate)
                        _apply_asset_payload(candidate_after, payload)
                        if _candidate_has_real_local_asset(candidate_after):
                            downloaded_ready_keys.add(unique_key)
                        else:
                            unresolved_keys.add(unique_key)
                    except Exception as exc:
                        payload = {
                            "asset_status": f"asset_acquire_failed:{type(exc).__name__}",
                            "asset_error": f"{type(exc).__name__}: {exc}",
                        }
                        failed_keys.add(unique_key)
                cached_payload_by_key[unique_key] = payload

            _apply_payload_to_binding(binding, unique_key, payload)
            _apply_asset_payload(merged_candidate, payload)

            if _candidate_has_real_local_asset(merged_candidate):
                selected_real_candidate = merged_candidate
                selected_rank = rank_idx
                break

        if selected_real_candidate is not None:
            binding["chosen_candidate"] = deepcopy(selected_real_candidate)
            notes = list(binding.get("selection_notes") or [])
            notes.append(f"asset_acquisition_selected_real_candidate_rank:{selected_rank}")
            binding["selection_notes"] = notes
        elif keep_unresolved_candidates:
            notes = list(binding.get("selection_notes") or [])
            notes.append("asset_acquisition_no_real_asset_found_keep_candidate_for_fallback")
            binding["selection_notes"] = notes
        else:
            binding["chosen_candidate"] = None
            binding["selection_status"] = "no_real_asset_after_acquisition"
            notes = list(binding.get("selection_notes") or [])
            notes.append("asset_acquisition_no_real_asset_found_keep_generated")
            binding["selection_notes"] = notes
            provenance = binding.get("provenance")
            if isinstance(provenance, dict):
                provenance["final_asset_source"] = "generated"

    meta = deepcopy(out.get("meta") or {})
    meta["asset_acquisition"] = {
        "selected_binding_count": selected_binding_count,
        "unique_candidate_count": len(cached_payload_by_key),
        "ready_before_count": len(ready_before_keys),
        "downloaded_ready_count": len(downloaded_ready_keys),
        "unresolved_count": len(unresolved_keys),
        "failed_count": len(failed_keys),
        "keep_unresolved_candidates": bool(keep_unresolved_candidates),
        "db_path": str(db_path.resolve()),
        "out_dir": str(out_dir.resolve()),
    }
    out["meta"] = meta
    return out


def acquire_assets_for_bindings_json(
    *,
    bindings_json_path: str | Path,
    output_json_path: str | Path,
    db_path: str | Path,
    out_dir: str | Path,
    blender_bin: str | None = None,
    catalog_json_paths: list[str | Path] | None = None,
    keep_unresolved_candidates: bool = False,
) -> Path:
    bindings_data = read_json(bindings_json_path)
    catalog_rows_by_key = _catalog_row_by_unique_key(
        [Path(x).expanduser().resolve() for x in (catalog_json_paths or []) if str(x).strip()]
    )
    out = acquire_assets_for_bindings_data(
        bindings_data,
        db_path=Path(db_path).expanduser().resolve(),
        out_dir=Path(out_dir).expanduser().resolve(),
        blender_bin=blender_bin,
        catalog_rows_by_key=catalog_rows_by_key,
        keep_unresolved_candidates=keep_unresolved_candidates,
    )
    output_path = Path(output_json_path).expanduser().resolve()
    write_json(output_path, out)
    return output_path


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Acquire assets only for chosen supplier candidates from supplier_bindings.")
    ap.add_argument("--bindings-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--blender", default=None)
    ap.add_argument("--catalog-json", action="append", default=[])
    return ap


def main() -> None:
    args = build_cli().parse_args()
    out_path = acquire_assets_for_bindings_json(
        bindings_json_path=args.bindings_json,
        output_json_path=args.out,
        db_path=args.db,
        out_dir=args.out_dir,
        blender_bin=args.blender,
        catalog_json_paths=args.catalog_json,
    )
    data = read_json(out_path)
    summary = ((data.get("meta") or {}).get("asset_acquisition") or {})
    print(f"saved = {out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
