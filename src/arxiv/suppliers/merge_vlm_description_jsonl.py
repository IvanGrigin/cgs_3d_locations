#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Merge base VLM description JSONL with retry rows, preferring successful retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def row_id(row: dict[str, Any]) -> str:
    value = row.get("id")
    if value in (None, ""):
        raise RuntimeError(f"row without id: {row}")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--retry-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    base_rows = load_rows(Path(args.base_jsonl))
    retry_rows = load_rows(Path(args.retry_jsonl))
    by_id = {row_id(row): row for row in base_rows}
    replaced = 0
    retry_ok = 0
    retry_not_ok = 0
    for row in retry_rows:
        if row.get("status") != "ok":
            retry_not_ok += 1
            continue
        retry_ok += 1
        rid = row_id(row)
        if rid in by_id:
            replaced += 1
        by_id[rid] = row

    ordered = [by_id[row_id(row)] for row in base_rows if row_id(row) in by_id]
    seen = {row_id(row) for row in ordered}
    ordered.extend(row for rid, row in by_id.items() if rid not in seen)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote={out_path} rows={len(ordered)} retry_ok={retry_ok} "
        f"retry_not_ok={retry_not_ok} replaced={replaced}"
    )


if __name__ == "__main__":
    main()
