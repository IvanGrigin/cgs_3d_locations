# -*- coding: utf-8 -*-
"""
This module turns parsed supplier products into local asset records and files.
It resolves download links, unpacks archives, inspects model payloads,
and prepares Blender-ready or fallback asset metadata for placement.
The script is intentionally conservative about asset quality.
Keep archive inspection and status reporting deterministic.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import zipfile

from src.suppliers.adapters.base import SupplierAdapter
from src.suppliers.batch_runner import (
    SITE_BATCH_PLANS,
    apply_fallback_to_product,
    build_adapter_map,
    build_site_fallback_map,
    compute_discovery_limit,
    discover_product_urls,
    prioritize_product_urls,
)
from src.suppliers.db_core import init_db, insert_download, upsert_asset, upsert_product
from src.suppliers.fetch_product_and_model import (
    download_binary,
    download_binary_for_record,
    resolve_yadisk_public_download,
)
from src.suppliers.runner import coerce_product_record, save_metadata_json
from src.suppliers.site_models import SupplierAssetRecord
from src.suppliers.utils import json_loads_or, now_utc_iso


PREFERRED_EXPORT_EXTS = [".glb", ".fbx"]
CONVERTIBLE_EXTS = [".glb", ".gltf", ".fbx", ".obj", ".blend"]
ARCHIVE_EXTS = [".zip", ".rar", ".7z"]
SEARCH_EXTS = [".glb", ".gltf", ".fbx", ".obj", ".blend", ".3ds", ".max", ".skp"]
BLENDER_HELPER = Path("src/tools/blender_supplier_asset.py")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text.strip())
    value = "_".join(part for part in value.split("_") if part)
    return value[:120] or "item"


def _normalize_model_name_token(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _candidate_name_tokens(record) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        token = _normalize_model_name_token(value)
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)

    external_id = str(getattr(record, "external_id", "") or "").strip()
    add(external_id)
    add(external_id.replace(" ", "_"))
    add(str(getattr(record, "title", "") or "").strip())

    for part in re.split(r"[^a-zA-Z0-9]+", external_id):
        if len(part) >= 4:
            add(part)

    return tokens


def ensure_direct_model_url(record) -> tuple[str | None, list[str]]:
    notes: list[str] = []
    if record.model_download_url:
        if record.source_site == "sancos":
            marker = "https://sancos.su/upload/3D/"
            if record.model_download_url.count(marker) > 1:
                record.model_download_url = marker + record.model_download_url.rsplit(marker, 1)[-1]
                notes.append("normalized_duplicate_sancos_3d_url")
        return record.model_download_url, notes

    if record.source_site == "loftdesigne" and record.model_download_landing_url:
        try:
            direct_url, filename = resolve_yadisk_public_download(record.model_download_landing_url)
            record.model_download_url = direct_url
            record.model_download_filename = filename or record.model_download_filename
            record.model_format = record.model_format or SupplierAdapter.ext_from_url(filename or direct_url)
            record.model_extraction_method = "yadisk_public_api"
            notes.append("resolved_yadisk_public_download")
            return record.model_download_url, notes
        except Exception as exc:  # pragma: no cover
            notes.append(f"yadisk_resolution_failed:{type(exc).__name__}:{exc}")  # pragma: no cover

    return None, notes


def download_preview_image(preview_urls: list[str], item_dir: Path) -> tuple[str | None, list[str]]:
    notes: list[str] = []
    if not preview_urls:
        return None, notes  # pragma: no cover

    preview_dir = item_dir / "preview"
    for index, url in enumerate(preview_urls[:3], start=1):
        result = download_binary(url, preview_dir, filename_hint=f"preview_{index}.jpg")
        if result.ok and result.local_path:
            notes.append("preview_downloaded")
            return result.local_path, notes
        notes.append(f"preview_download_failed:{index}")

    return None, notes  # pragma: no cover


def _pick_model_from_extracted_dir(extract_dir: Path, record=None) -> tuple[str | None, str | None, list[str]]:
    notes: list[str] = []
    paths = [path for path in extract_dir.rglob("*") if path.is_file() and path.suffix.lower() in SEARCH_EXTS]
    preferred_tokens = _candidate_name_tokens(record) if record is not None else []

    if preferred_tokens:
        notes.append(f"archive_preferred_name_tokens:{preferred_tokens}")
        for ext in PREFERRED_EXPORT_EXTS + [".gltf", ".obj", ".blend", ".3ds", ".max", ".skp"]:
            ext_paths = [path for path in paths if path.suffix.lower() == ext]
            for token in preferred_tokens:
                for path in ext_paths:
                    stem = _normalize_model_name_token(path.stem)
                    full = _normalize_model_name_token(path.name)
                    if token == stem or token == full or token in stem or token in full:
                        notes.append(f"archive_selected_by_name:{token}:{path.suffix.lower()}")
                        return str(path), path.suffix.lower(), notes

    for ext in PREFERRED_EXPORT_EXTS + [".gltf", ".obj", ".blend", ".3ds", ".max", ".skp"]:
        for path in paths:
            if path.suffix.lower() == ext:
                notes.append(f"archive_selected:{path.suffix.lower()}")
                return str(path), path.suffix.lower(), notes

    notes.append("archive_has_no_supported_model")
    return None, None, notes


def _extract_archive_once(archive_path: Path, extract_dir: Path) -> tuple[bool, list[str]]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    ext = archive_path.suffix.lower()

    if ext == ".zip":
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)
        except Exception as exc:
            notes.append(f"zip_extract_failed:{type(exc).__name__}:{exc}")
            return False, notes
        notes.append("archive_extracted_with:zipfile")
        return True, notes

    extract_commands = [
        ["unar", "-f", "-o", str(extract_dir), str(archive_path)],
        ["7z", "x", "-y", str(archive_path), f"-o{extract_dir}"],
        ["bsdtar", "-xf", str(archive_path), "-C", str(extract_dir)],
    ]

    for cmd in extract_commands:
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            notes.append(f"archive_extract_runner_failed:{cmd[0]}:{type(exc).__name__}:{exc}")
            continue

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            notes.append(f"archive_extract_failed:{cmd[0]}:{completed.returncode}")
            if stderr:
                notes.append(f"archive_extract_stderr:{stderr[:300]}")
            continue

        notes.append(f"archive_extracted_with:{cmd[0]}")
        return True, notes

    notes.append(f"archive_extract_unsupported:{ext}")
    return False, notes


def _extract_nested_archives(root_dir: Path, *, max_depth: int = 2) -> list[str]:
    notes: list[str] = []
    seen: set[Path] = set()

    for depth in range(max_depth):
        archives = [
            path
            for path in root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ARCHIVE_EXTS and path.resolve() not in seen
        ]
        if not archives:
            break

        expanded_any = False
        for archive in archives:
            seen.add(archive.resolve())
            nested_dir = archive.parent / f"{archive.stem}__extracted"
            ok, nested_notes = _extract_archive_once(archive, nested_dir)
            notes.extend([f"nested_archive:{archive.name}"] + nested_notes)
            expanded_any = expanded_any or ok

        if not expanded_any:
            notes.append(f"nested_archive_stop_at_depth:{depth}")  # pragma: no cover
            break  # pragma: no cover

    return notes


def inspect_archive(archive_path: Path, extract_dir: Path, record=None) -> tuple[str | None, str | None, list[str]]:
    notes: list[str] = []
    ok, extract_notes = _extract_archive_once(archive_path, extract_dir)
    notes.extend(extract_notes)
    if not ok:
        return None, None, notes  # pragma: no cover

    selected_path, selected_ext, pick_notes = _pick_model_from_extracted_dir(extract_dir, record=record)
    notes.extend(pick_notes)
    if selected_path:
        return selected_path, selected_ext, notes

    notes.extend(_extract_nested_archives(extract_dir))
    selected_path, selected_ext, pick_notes = _pick_model_from_extracted_dir(extract_dir, record=record)
    notes.extend(pick_notes)
    return selected_path, selected_ext, notes


def build_blender_job_spec(record, item_dir: Path, reason: str, source_path: str | None, preview_path: str | None) -> str:
    images = json_loads_or(record.images_json, [])
    payload = {
        "unique_key": record.unique_key,
        "source_site": record.source_site,
        "product_url": record.product_url,
        "title": record.title,
        "reason": reason,
        "source_asset_path": source_path,
        "preview_local_path": preview_path,
        "preview_images": images,
        "dimensions_cm": {
            "width": record.width_cm,
            "depth": record.depth_cm,
            "height": record.height_cm,
        },
        "materials": record.materials,
        "description": record.description,
        "desired_formats": ["glb", "fbx"],
    }
    path = item_dir / "blender_job.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def run_blender_helper(
    blender_bin: str,
    mode: str,
    input_path: str | None,
    out_glb: Path,
    out_fbx: Path,
    preview_path: str | None = None,
    width_m: float | None = None,
    depth_m: float | None = None,
    height_m: float | None = None,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    helper_path = BLENDER_HELPER.expanduser().resolve()
    if not helper_path.is_file():
        return False, [f"blender_helper_missing:{helper_path}"]

    cmd = [str(Path(blender_bin).expanduser().resolve()), "--background"]
    if mode == "convert" and input_path and Path(input_path).suffix.lower() == ".blend":
        cmd.append(str(Path(input_path).expanduser().resolve()))
    cmd += [
        "--python",
        str(helper_path),
        "--",
        "--mode",
        mode,
        "--out-glb",
        str(out_glb),
        "--out-fbx",
        str(out_fbx),
    ]

    if input_path:
        cmd += ["--input", str(Path(input_path).expanduser().resolve())]
    if preview_path:
        cmd += ["--texture", str(Path(preview_path).expanduser().resolve())]
    if width_m is not None:
        cmd += ["--width-m", str(width_m)]
    if depth_m is not None:
        cmd += ["--depth-m", str(depth_m)]
    if height_m is not None:
        cmd += ["--height-m", str(height_m)]

    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return False, [f"blender_run_failed:{type(exc).__name__}:{exc}"]

    if completed.returncode != 0:
        notes.append(f"blender_returncode:{completed.returncode}")
        if completed.stderr.strip():
            notes.append(f"blender_stderr:{completed.stderr.strip()[:500]}")
        return False, notes

    notes.append(f"blender_{mode}_ok")
    return True, notes


def build_asset_record(
    record,
    status: str,
    asset_format: str | None,
    asset_source_url: str | None,
    asset_local_path: str | None,
    preview_local_path: str | None,
    blender_job_path: str | None,
    notes: list[str],
    extra: dict,
) -> SupplierAssetRecord:
    return SupplierAssetRecord(
        unique_key=record.unique_key,
        updated_at=now_utc_iso(),
        source_site=record.source_site,
        product_url=record.product_url,
        title=record.title,
        asset_status=status,
        asset_format=asset_format,
        asset_source_url=asset_source_url,
        asset_local_path=asset_local_path,
        preview_local_path=preview_local_path,
        blender_job_path=blender_job_path,
        notes_json=json.dumps(notes, ensure_ascii=False),
        extra_json=json.dumps(extra, ensure_ascii=False),
    )


def asset_is_ready(asset: SupplierAssetRecord) -> bool:
    return bool(asset.asset_local_path and asset.asset_format in {"glb", "fbx"})


def log_product_card(record) -> None:
    dimensions = " x ".join(
        str(int(value) if float(value).is_integer() else value)
        for value in [record.width_cm, record.depth_cm, record.height_cm]
        if value is not None
    ) or "-"
    price = f"{record.price_value} {record.price_currency or ''}".strip() if record.price_value is not None else "-"

    log(
        "[card] "
        f"title={record.title!r}; "
        f"category={record.category_raw!r}; "
        f"brand={record.brand!r}; "
        f"collection={record.collection!r}; "
        f"price={price!r}; "
        f"old_price={record.old_price_value!r}; "
        f"color={record.color!r}; "
        f"style={record.style!r}; "
        f"materials={record.materials!r}; "
        f"dimensions_cm={dimensions!r}; "
        f"product_url={record.product_url!r}; "
        f"model_page_url={record.model_page_url!r}; "
        f"model_download_url={record.model_download_url!r}; "
        f"model_download_landing_url={record.model_download_landing_url!r}; "
        f"model_vendor_url={record.model_vendor_url!r}"
    )


def acquire_asset_for_record(record, db_path: Path, out_dir: Path, blender_bin: str | None) -> SupplierAssetRecord:
    site_dir = out_dir / record.source_site
    item_dir = site_dir / slugify(record.title or record.unique_key)
    item_dir.mkdir(parents=True, exist_ok=True)

    meta_path = save_metadata_json(record, item_dir)
    preview_urls = json_loads_or(record.images_json, [])
    preview_local_path, preview_notes = download_preview_image(preview_urls, item_dir)
    model_url, resolve_notes = ensure_direct_model_url(record)

    notes = [f"metadata:{meta_path}", *preview_notes, *resolve_notes]
    extra_base = json_loads_or(record.extra_json, {})
    if not isinstance(extra_base, dict):
        extra_base = {}

    if model_url or record.source_site == "3ddd":
        download_dir = item_dir / "download"
        result = download_binary_for_record(
            record,
            download_dir,
            filename_hint=record.model_download_filename,
        )
        model_url = record.model_download_url or model_url
        extra = {
            **extra_base,
            "model_page_url": record.model_page_url,
            "model_download_url": record.model_download_url,
            "model_download_landing_url": record.model_download_landing_url,
        }
        insert_download(
            db_path=db_path,
            unique_key=record.unique_key,
            downloaded_at=now_utc_iso(),
            final_url=result.final_url,
            local_path=result.local_path,
            filename=result.filename,
            content_type=result.content_type,
            ok=result.ok,
            size_bytes=result.size_bytes,
            error=result.error,
        )

        if not result.ok or not result.local_path:
            notes.append(f"download_failed:{result.error}")
        else:
            local_path = Path(result.local_path).resolve()
            ext = local_path.suffix.lower()
            notes.append(f"downloaded:{ext or 'none'}")

            if ext in PREFERRED_EXPORT_EXTS:
                return build_asset_record(
                    record=record,
                    status="downloaded_preferred",
                    asset_format=ext.lstrip("."),
                    asset_source_url=result.final_url or model_url,
                    asset_local_path=str(local_path),
                    preview_local_path=preview_local_path,
                    blender_job_path=None,
                    notes=notes,
                    extra=extra,
                )

            selected_path = None
            selected_ext = None
            if ext in ARCHIVE_EXTS:
                selected_path, selected_ext, archive_notes = inspect_archive(local_path, item_dir / "extracted", record=record)
                notes.extend(archive_notes)
                if selected_path and selected_ext in PREFERRED_EXPORT_EXTS:
                    return build_asset_record(
                        record=record,
                        status="archive_extracted_preferred",
                        asset_format=selected_ext.lstrip("."),
                        asset_source_url=result.final_url or model_url,
                        asset_local_path=selected_path,
                        preview_local_path=preview_local_path,
                        blender_job_path=None,
                        notes=notes,
                        extra=extra,
                    )
            elif ext in CONVERTIBLE_EXTS:
                selected_path = str(local_path)
                selected_ext = ext

            if blender_bin and selected_path and selected_ext in {".obj", ".blend", ".gltf", ".glb", ".fbx"}:
                built_dir = item_dir / "built"
                out_glb = built_dir / "model.glb"
                out_fbx = built_dir / "model.fbx"
                ok, blender_notes = run_blender_helper(
                    blender_bin=blender_bin,
                    mode="convert",
                    input_path=selected_path,
                    out_glb=out_glb,
                    out_fbx=out_fbx,
                    preview_path=preview_local_path,
                    width_m=(record.width_cm / 100.0) if record.width_cm else None,
                    depth_m=(record.depth_cm / 100.0) if record.depth_cm else None,
                    height_m=(record.height_cm / 100.0) if record.height_cm else None,
                )
                notes.extend(blender_notes)
                if ok:
                    chosen = out_glb if out_glb.is_file() else out_fbx if out_fbx.is_file() else None
                    if chosen:
                        return build_asset_record(
                            record=record,
                            status="converted_with_blender",
                            asset_format=chosen.suffix.lower().lstrip("."),
                            asset_source_url=result.final_url or model_url,
                            asset_local_path=str(chosen.resolve()),
                            preview_local_path=preview_local_path,
                            blender_job_path=None,
                            notes=notes,
                            extra=extra,
                        )

            blender_job_path = build_blender_job_spec(
                record=record,
                item_dir=item_dir,
                reason="downloaded_asset_is_not_glb_or_fbx",
                source_path=selected_path or result.local_path,
                preview_path=preview_local_path,
            )
            return build_asset_record(
                record=record,
                status="needs_blender_rebuild",
                asset_format=(selected_ext or ext).lstrip(".") if (selected_ext or ext) else None,
                asset_source_url=result.final_url or model_url,
                asset_local_path=selected_path or result.local_path,
                preview_local_path=preview_local_path,
                blender_job_path=blender_job_path,
                notes=notes,
                extra=extra,
            )

    extra = {
        **extra_base,
        "model_page_url": record.model_page_url,
        "model_download_url": record.model_download_url,
        "model_download_landing_url": record.model_download_landing_url,
    }

    if blender_bin and record.width_cm and record.depth_cm and record.height_cm:
        built_dir = item_dir / "built"
        out_glb = built_dir / "proxy.glb"
        out_fbx = built_dir / "proxy.fbx"
        ok, blender_notes = run_blender_helper(
            blender_bin=blender_bin,
            mode="proxy",
            input_path=None,
            out_glb=out_glb,
            out_fbx=out_fbx,
            preview_path=preview_local_path,
            width_m=record.width_cm / 100.0,
            depth_m=record.depth_cm / 100.0,
            height_m=record.height_cm / 100.0,
        )
        notes.extend(blender_notes)
        if ok:
            chosen = out_glb if out_glb.is_file() else out_fbx if out_fbx.is_file() else None
            if chosen:
                return build_asset_record(
                    record=record,
                    status="proxy_generated_with_blender",
                    asset_format=chosen.suffix.lower().lstrip("."),
                    asset_source_url=None,
                    asset_local_path=str(chosen.resolve()),
                    preview_local_path=preview_local_path,
                    blender_job_path=None,
                    notes=notes,
                    extra=extra,
                )

    blender_job_path = build_blender_job_spec(
        record=record,
        item_dir=item_dir,
        reason="no_direct_model_url",
        source_path=None,
        preview_path=preview_local_path,
    )
    return build_asset_record(
        record=record,
        status="needs_blender_rebuild",
        asset_format=None,
        asset_source_url=None,
        asset_local_path=None,
        preview_local_path=preview_local_path,
        blender_job_path=blender_job_path,
        notes=notes,
        extra=extra,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="homeconcept,imodern,loftdesigne,3ddd")
    ap.add_argument("--count-per-site", type=int, default=2)
    ap.add_argument("--max-listing-pages", type=int, default=12)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--db", default="out/supplier_ingest/site_assets/site_assets.db")
    ap.add_argument("--out-dir", default="out/supplier_ingest/site_assets/assets")
    ap.add_argument("--blender", default=None)
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    init_db(db_path)

    adapter_map = build_adapter_map()
    site_names = [part.strip() for part in args.sites.split(",") if part.strip()]

    for site_name in site_names:
        plan = SITE_BATCH_PLANS.get(site_name)
        adapter = adapter_map.get(site_name)
        if not plan or not adapter:
            log(f"[{site_name}] skipped: no plan or adapter")
            continue

        log(f"[{site_name}] start asset acquisition")
        fallback_map = build_site_fallback_map(adapter, plan)
        discovery_limit = compute_discovery_limit(args.count_per_site)
        product_urls = discover_product_urls(
            adapter=adapter,
            plan=plan,
            limit=discovery_limit,
            max_listing_pages=args.max_listing_pages,
            max_depth=args.max_depth,
        )
        product_urls = prioritize_product_urls(plan, product_urls)
        log(f"[{site_name}] discovered urls: {len(product_urls)}")

        ready_count = 0
        for index, url in enumerate(product_urls, start=1):
            if ready_count >= args.count_per_site:
                break

            log(f"[{site_name}] asset candidate {index}/{len(product_urls)}: {url}")
            try:
                html, final_url = adapter.fetch_html(url)
                raw_items = adapter.parse(url, html, final_url)
                if not raw_items:
                    if getattr(adapter, "empty_parse_is_skip", False):
                        log(f"[{site_name}] asset skipped (no downloadable model): {url}")
                        continue
                    raise ValueError("adapter returned zero records")  # pragma: no cover
                record = coerce_product_record(raw_items[0], adapter, url, final_url)
                apply_fallback_to_product(record, fallback_map.get(record.product_url or final_url))
                log_product_card(record)
                upsert_product(db_path, record)
                asset = acquire_asset_for_record(record, db_path=db_path, out_dir=out_dir, blender_bin=args.blender)
                upsert_asset(db_path, asset)
                log(
                    f"[{site_name}] asset status={asset.asset_status} "
                    f"format={asset.asset_format!r} path={asset.asset_local_path!r}"
                )
                if asset_is_ready(asset):
                    ready_count += 1
            except Exception as exc:
                log(f"[{site_name}] asset error: {url} -> {type(exc).__name__}: {exc}")

        log(f"[{site_name}] ready assets: {ready_count}/{args.count_per_site}")


if __name__ == "__main__":
    main()  # pragma: no cover
