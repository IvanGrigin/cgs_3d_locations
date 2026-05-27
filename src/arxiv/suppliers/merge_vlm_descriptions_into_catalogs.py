#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Safely merge VLM descriptions into supplier JSON catalogs and SQLite DBs.

Rows are matched only by unique_key. A row is updated only when the target key is
unique and title/source_site match the VLM row exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VLM_FIELDS = [
    "vlm_description_summary",
    "vlm_description_text",
    "vlm_description_json",
    "vlm_description_model",
    "vlm_description_status",
    "vlm_description_processed_at_unix",
]


def load_vlm_rows(path: Path) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    duplicate_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            unique_key = row.get("unique_key")
            if not unique_key:
                continue
            if unique_key in by_key:
                duplicate_keys.add(str(unique_key))
                continue
            by_key[str(unique_key)] = row
    if duplicate_keys:
        raise RuntimeError(f"VLM JSONL has duplicate unique_key values: {sorted(duplicate_keys)[:10]}")
    return by_key


def description_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("status") != "ok":
        return None
    desc = row.get("vlm_description")
    if not isinstance(desc, dict) or not desc.get("detailed_description"):
        return None
    return {
        "vlm_description_summary": desc.get("object_summary") or "",
        "vlm_description_text": desc.get("detailed_description") or "",
        "vlm_description_json": json.dumps(desc, ensure_ascii=False, sort_keys=True),
        "vlm_description_model": row.get("model") or "",
        "vlm_description_status": row.get("status") or "",
        "vlm_description_processed_at_unix": row.get("processed_at_unix"),
    }


def validate_match(target: dict[str, Any], row: dict[str, Any]) -> str:
    if (target.get("title") or "") != (row.get("title") or ""):
        return "title_mismatch"
    if (target.get("source_site") or "") != (row.get("source_site") or ""):
        return "source_site_mismatch"
    return "ok"


def backup_path(path: Path, stamp: str) -> Path:
    return path.with_name(f"{path.name}.bak_vlm_desc_{stamp}")


def merge_json_catalog(path: Path, vlm_by_key: dict[str, dict[str, Any]], dry_run: bool, stamp: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise RuntimeError(f"JSON catalog does not contain items list: {path}")

    target_keys = Counter(str(item.get("unique_key") or "") for item in items)
    stats = Counter()
    audit_rows: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        unique_key = str(item.get("unique_key") or "")
        if not unique_key:
            stats["target_missing_unique_key"] += 1
            continue
        if target_keys[unique_key] != 1:
            stats["target_duplicate_unique_key"] += 1
            continue
        row = vlm_by_key.get(unique_key)
        if not row:
            stats["no_vlm_row"] += 1
            continue
        match_status = validate_match(item, row)
        if match_status != "ok":
            stats[match_status] += 1
            audit_rows.append(
                {
                    "target": str(path),
                    "target_type": "json",
                    "index": index,
                    "unique_key": unique_key,
                    "status": match_status,
                    "target_title": item.get("title"),
                    "vlm_title": row.get("title"),
                    "target_source_site": item.get("source_site"),
                    "vlm_source_site": row.get("source_site"),
                }
            )
            continue
        payload = description_payload(row)
        if not payload:
            stats["vlm_without_description"] += 1
            continue
        stats["updated"] += 1
        if not dry_run:
            item.update(payload)

    if not dry_run:
        shutil.copy2(path, backup_path(path, stamp))
        if isinstance(data, dict):
            meta = data.setdefault("meta", {})
            meta["vlm_description_merge"] = {
                "source_jsonl": str(Path(args.vlm_jsonl)),
                "merged_at_unix": time.time(),
                "match_key": "unique_key",
                "updated": int(stats["updated"]),
            }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"target": str(path), "type": "json", "stats": dict(stats), "audit_rows": audit_rows}


def table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"pragma table_info({table})")}


def add_columns_if_needed(con: sqlite3.Connection, table: str) -> None:
    cols = table_columns(con, table)
    definitions = {
        "vlm_description_summary": "TEXT",
        "vlm_description_text": "TEXT",
        "vlm_description_json": "TEXT",
        "vlm_description_model": "TEXT",
        "vlm_description_status": "TEXT",
        "vlm_description_processed_at_unix": "REAL",
    }
    for name, sql_type in definitions.items():
        if name not in cols:
            con.execute(f"alter table {table} add column {name} {sql_type}")


def merge_sqlite_table(
    path: Path,
    table: str,
    vlm_by_key: dict[str, dict[str, Any]],
    dry_run: bool,
    stamp: str,
) -> dict[str, Any]:
    if not path.is_file():
        return {"target": str(path), "type": "sqlite", "table": table, "stats": {"missing_db": 1}, "audit_rows": []}

    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"select rowid as _rowid, unique_key, title, source_site from {table}").fetchall()
    key_counts = Counter(str(row["unique_key"] or "") for row in rows)
    stats = Counter()
    audit_rows: list[dict[str, Any]] = []
    updates: list[tuple[Any, ...]] = []

    for row in rows:
        unique_key = str(row["unique_key"] or "")
        if not unique_key:
            stats["target_missing_unique_key"] += 1
            continue
        if key_counts[unique_key] != 1:
            stats["target_duplicate_unique_key"] += 1
            continue
        vlm_row = vlm_by_key.get(unique_key)
        if not vlm_row:
            stats["no_vlm_row"] += 1
            continue
        target = {"title": row["title"], "source_site": row["source_site"]}
        match_status = validate_match(target, vlm_row)
        if match_status != "ok":
            stats[match_status] += 1
            audit_rows.append(
                {
                    "target": str(path),
                    "target_type": "sqlite",
                    "table": table,
                    "rowid": row["_rowid"],
                    "unique_key": unique_key,
                    "status": match_status,
                    "target_title": row["title"],
                    "vlm_title": vlm_row.get("title"),
                    "target_source_site": row["source_site"],
                    "vlm_source_site": vlm_row.get("source_site"),
                }
            )
            continue
        payload = description_payload(vlm_row)
        if not payload:
            stats["vlm_without_description"] += 1
            continue
        stats["updated"] += 1
        updates.append(
            (
                payload["vlm_description_summary"],
                payload["vlm_description_text"],
                payload["vlm_description_json"],
                payload["vlm_description_model"],
                payload["vlm_description_status"],
                payload["vlm_description_processed_at_unix"],
                row["_rowid"],
            )
        )

    if not dry_run:
        con.close()
        shutil.copy2(path, backup_path(path, stamp))
        con = sqlite3.connect(path, timeout=30)
        add_columns_if_needed(con, table)
        con.executemany(
            f"""
            update {table}
            set vlm_description_summary = ?,
                vlm_description_text = ?,
                vlm_description_json = ?,
                vlm_description_model = ?,
                vlm_description_status = ?,
                vlm_description_processed_at_unix = ?
            where rowid = ?
            """,
            updates,
        )
        con.commit()
    con.close()
    return {"target": str(path), "type": "sqlite", "table": table, "stats": dict(stats), "audit_rows": audit_rows}


def write_audit(out_dir: Path, results: list[dict[str, Any]], dry_run: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dry_run": dry_run,
        "results": [
            {k: v for k, v in result.items() if k != "audit_rows"}
            for result in results
        ],
    }
    (out_dir / "vlm_description_merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(result.get("audit_rows") or [])
    with (out_dir / "vlm_description_merge_mismatches.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "target",
            "target_type",
            "table",
            "index",
            "rowid",
            "unique_key",
            "status",
            "target_title",
            "vlm_title",
            "target_source_site",
            "vlm_source_site",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-jsonl", required=True)
    parser.add_argument("--out-dir", default="reports/vlm_description_merge")
    parser.add_argument("--apply", action="store_true", help="Write changes. Without this, only audit.")
    parser.add_argument(
        "--json-catalog",
        action="append",
        default=[
            "data/sourse/suppliers/supplier_catalog_canonical.json",
        ],
    )
    parser.add_argument(
        "--sqlite-target",
        action="append",
        default=[],
    )
    return parser


def main() -> None:
    global args
    args = build_parser().parse_args()
    dry_run = not bool(args.apply)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    vlm_by_key = load_vlm_rows(Path(args.vlm_jsonl))
    results: list[dict[str, Any]] = []

    for target in args.json_catalog:
        results.append(merge_json_catalog(Path(target), vlm_by_key, dry_run=dry_run, stamp=stamp))
    for spec in args.sqlite_target:
        db_path, table = spec.split(":", 1)
        results.append(merge_sqlite_table(Path(db_path), table, vlm_by_key, dry_run=dry_run, stamp=stamp))

    write_audit(Path(args.out_dir), results, dry_run=dry_run)
    for result in results:
        label = result["target"]
        if result.get("table"):
            label += f":{result['table']}"
        print(label, result["stats"])


if __name__ == "__main__":
    main()
