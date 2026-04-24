#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe supplier model URLs and record whether the downloadable payload contains FBX.

The script operates on distinct model_download_url values, downloads each payload once,
inspects direct files or archive contents, then writes the result back into:
  1. supplier_product.extra_json in SQLite
  2. per-item metadata JSON files under data/sourse/suppliers/items

It is designed to be resumable. Re-running with --skip-checked avoids re-probing
URLs that already have a recorded result in extra_json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from src.suppliers.db import PRODUCT_COLUMNS, upsert_products
from src.suppliers.fetch_product_and_model import download_binary
from src.suppliers.models import ProductRecord
from src.suppliers.runner import save_metadata_json
from src.suppliers.utils import json_loads_or, now_utc_iso

ARCHIVE_EXTS = {".zip", ".rar", ".7z"}


@dataclass
class ProbeResult:
    url: str
    checked_at: str
    download_ok: bool
    final_url: str | None
    filename: str | None
    content_type: str | None
    size_bytes: int
    source_ext: str | None
    is_archive: bool
    has_fbx: bool | None
    has_obj: bool | None
    has_max: bool | None
    has_blend: bool | None
    entry_count: int | None
    entry_sample: list[str]
    error: str | None


def _guess_ext(filename: str | None, url: str | None) -> str | None:
    for raw in (filename, urlparse(url or "").path):
        if not raw:
            continue
        suffix = Path(unquote(str(raw))).suffix.lower().strip()
        if suffix:
            return suffix
    return None


def _list_archive_members_with_7z(archive_path: Path) -> list[str]:
    runner = shutil.which("7z")
    if not runner:
        raise RuntimeError("7z not found")

    completed = subprocess.run(
        [runner, "l", "-slt", str(archive_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"7z list failed rc={completed.returncode}: {stderr[:300]}")

    members: list[str] = []
    archive_markers = {
        str(archive_path),
        archive_path.name,
        archive_path.resolve().as_posix(),
    }
    first_path_seen = False
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("Path = "):
            continue
        value = line[len("Path = "):].strip()
        if not first_path_seen:
            first_path_seen = True
            continue
        if value in archive_markers:
            continue
        if value:
            members.append(value)
    return members


def _list_archive_members(archive_path: Path) -> list[str]:
    try:
        members = _list_archive_members_with_7z(archive_path)
        if members:
            return members
    except Exception:
        pass

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            return [name for name in zf.namelist() if name]

    raise RuntimeError(f"unsupported or unreadable archive: {archive_path.name}")


def _summarize_members(members: list[str]) -> tuple[bool, bool, bool, bool]:
    lowered = [name.lower() for name in members]
    return (
        any(name.endswith(".fbx") for name in lowered),
        any(name.endswith(".obj") for name in lowered),
        any(name.endswith(".max") for name in lowered),
        any(name.endswith(".blend") for name in lowered),
    )


def _probe_single_url(url: str, temp_root: Path) -> ProbeResult:
    checked_at = now_utc_iso()
    with tempfile.TemporaryDirectory(prefix="supplier_model_probe_", dir=temp_root) as tmp_dir:
        download_dir = Path(tmp_dir)
        download = download_binary(url, download_dir)
        if not download.ok or not download.local_path:
            return ProbeResult(
                url=url,
                checked_at=checked_at,
                download_ok=False,
                final_url=download.final_url,
                filename=download.filename,
                content_type=download.content_type,
                size_bytes=download.size_bytes,
                source_ext=_guess_ext(download.filename, download.final_url or url),
                is_archive=False,
                has_fbx=None,
                has_obj=None,
                has_max=None,
                has_blend=None,
                entry_count=None,
                entry_sample=[],
                error=download.error,
            )

        local_path = Path(download.local_path)
        source_ext = _guess_ext(local_path.name, download.final_url or url)
        is_archive = source_ext in ARCHIVE_EXTS

        try:
            if is_archive:
                members = _list_archive_members(local_path)
                has_fbx, has_obj, has_max, has_blend = _summarize_members(members)
                entry_sample = members[:20]
                entry_count = len(members)
            else:
                lowered_name = local_path.name.lower()
                has_fbx = lowered_name.endswith(".fbx")
                has_obj = lowered_name.endswith(".obj")
                has_max = lowered_name.endswith(".max")
                has_blend = lowered_name.endswith(".blend")
                entry_sample = [local_path.name]
                entry_count = 1

            return ProbeResult(
                url=url,
                checked_at=checked_at,
                download_ok=True,
                final_url=download.final_url,
                filename=download.filename,
                content_type=download.content_type,
                size_bytes=download.size_bytes,
                source_ext=source_ext,
                is_archive=is_archive,
                has_fbx=has_fbx,
                has_obj=has_obj,
                has_max=has_max,
                has_blend=has_blend,
                entry_count=entry_count,
                entry_sample=entry_sample,
                error=None,
            )
        except Exception as exc:
            return ProbeResult(
                url=url,
                checked_at=checked_at,
                download_ok=True,
                final_url=download.final_url,
                filename=download.filename,
                content_type=download.content_type,
                size_bytes=download.size_bytes,
                source_ext=source_ext,
                is_archive=is_archive,
                has_fbx=None,
                has_obj=None,
                has_max=None,
                has_blend=None,
                entry_count=None,
                entry_sample=[],
                error=f"{type(exc).__name__}: {exc}",
            )


def _load_site_records(db_path: Path, site: str) -> dict[str, list[ProductRecord]]:
    sql = f"""
        SELECT {", ".join(PRODUCT_COLUMNS)}
        FROM supplier_product
        WHERE source_site = ?
          AND model_download_url IS NOT NULL
          AND trim(model_download_url) != ''
        ORDER BY model_download_url, title, unique_key
    """
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, (site,)).fetchall()

    grouped: dict[str, list[ProductRecord]] = defaultdict(list)
    for row in rows:
        payload = {column: row[column] for column in PRODUCT_COLUMNS}
        record = ProductRecord(**payload)
        grouped[str(record.model_download_url or "").strip()].append(record)
    return dict(grouped)


def _extra_patch(result: ProbeResult) -> dict[str, Any]:
    return {
        "model_probe_version": "fbx_presence_v1",
        "model_probe_checked_at": result.checked_at,
        "model_probe_url": result.url,
        "model_probe_final_url": result.final_url,
        "model_probe_download_ok": result.download_ok,
        "model_probe_error": result.error,
        "model_probe_filename": result.filename,
        "model_probe_content_type": result.content_type,
        "model_probe_size_bytes": result.size_bytes,
        "model_probe_source_ext": result.source_ext,
        "model_probe_is_archive": result.is_archive,
        "model_probe_has_fbx": result.has_fbx,
        "model_probe_has_obj": result.has_obj,
        "model_probe_has_max": result.has_max,
        "model_probe_has_blend": result.has_blend,
        "model_probe_entry_count": result.entry_count,
        "model_probe_entry_sample": result.entry_sample,
        "model_contains_fbx": result.has_fbx,
    }


def _apply_probe_to_record(record: ProductRecord, result: ProbeResult) -> ProductRecord:
    extra = json_loads_or(record.extra_json, {})
    if not isinstance(extra, dict):
        extra = {}
    extra.update(_extra_patch(result))
    record.extra_json = json.dumps(extra, ensure_ascii=False)
    return record


def _group_is_already_checked(records: list[ProductRecord]) -> bool:
    if not records:
        return True
    for record in records:
        extra = json_loads_or(record.extra_json, {})
        if not isinstance(extra, dict):
            return False
        if not extra.get("model_probe_checked_at"):
            return False
    return True


def _write_group_records(db_path: Path, out_dir: Path, records: list[ProductRecord]) -> None:
    upsert_products(db_path, records)
    for record in records:
        save_metadata_json(record, out_dir)


def _build_summary(results: list[ProbeResult], total_record_count: int) -> dict[str, Any]:
    by_ext: Counter[str] = Counter()
    for result in results:
        by_ext[str(result.source_ext or "unknown")] += 1

    return {
        "schema": "supplier_model_probe_report/v1",
        "generated_at": now_utc_iso(),
        "distinct_url_count": len(results),
        "total_record_count": total_record_count,
        "download_ok_count": sum(1 for r in results if r.download_ok),
        "error_count": sum(1 for r in results if r.error),
        "has_fbx_count": sum(1 for r in results if r.has_fbx is True),
        "has_obj_count": sum(1 for r in results if r.has_obj is True),
        "has_max_count": sum(1 for r in results if r.has_max is True),
        "has_blend_count": sum(1 for r in results if r.has_blend is True),
        "by_source_ext": dict(sorted(by_ext.items())),
    }


def _make_logger(progress_log: Path | None):
    def _log(message: str) -> None:
        print(message, flush=True)
        if progress_log is None:
            return
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        with progress_log.open("a", encoding="utf-8") as fh:
            fh.write(message)
            fh.write("\n")
    return _log


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe supplier model URLs for FBX presence.")
    ap.add_argument("--db", default="data/sourse/suppliers/suppliers.db")
    ap.add_argument("--out-dir", default="data/sourse/suppliers/items")
    ap.add_argument("--site", default="sancos")
    ap.add_argument("--temp-root", default="/tmp")
    ap.add_argument("--report-json", default="data/sourse/suppliers/sancos_model_probe_report.json")
    ap.add_argument("--progress-log", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-checked", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    temp_root = Path(args.temp_root).expanduser().resolve()
    report_json = Path(args.report_json).expanduser().resolve()
    progress_log = Path(args.progress_log).expanduser().resolve() if args.progress_log else None
    log = _make_logger(progress_log)

    grouped = _load_site_records(db_path, args.site)
    total_record_count = sum(len(records) for records in grouped.values())
    urls = sorted(grouped)
    if args.skip_checked:
        urls = [url for url in urls if not _group_is_already_checked(grouped[url])]
    if args.limit and args.limit > 0:
        urls = urls[: args.limit]

    log(f"[{args.site}] record_count={total_record_count}")
    log(f"[{args.site}] distinct_model_urls={len(grouped)}")
    log(f"[{args.site}] probe_queue={len(urls)}")

    results: list[ProbeResult] = []
    if not urls:
        report = {
            "summary": _build_summary(results, total_record_count),
            "items": [],
        }
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[{args.site}] nothing to do; report={report_json}")
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_url = {executor.submit(_probe_single_url, url, temp_root): url for url in urls}
        for index, future in enumerate(as_completed(future_to_url), start=1):
            url = future_to_url[future]
            result = future.result()
            results.append(result)

            records = [_apply_probe_to_record(record, result) for record in grouped[url]]
            _write_group_records(db_path, out_dir, records)

            status = "ok" if result.error is None else "error"
            log(
                f"[{args.site}] {index}/{len(urls)} {status} "
                f"ext={result.source_ext or 'unknown'} "
                f"fbx={result.has_fbx} "
                f"url={url}"
            )
            if result.error:
                log(f"[{args.site}]   error={result.error}")

    results.sort(key=lambda item: item.url)
    report = {
        "summary": _build_summary(results, total_record_count),
        "items": [asdict(item) for item in results],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    log(
        f"[{args.site}] done "
        f"urls={summary['distinct_url_count']} "
        f"download_ok={summary['download_ok_count']} "
        f"has_fbx={summary['has_fbx_count']} "
        f"errors={summary['error_count']}"
    )
    log(f"[{args.site}] report={report_json}")


if __name__ == "__main__":
    main()
