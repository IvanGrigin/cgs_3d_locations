#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner:
- iterates imodern_item rows
- finds local OBJ folder by bx_id in folder name
- runs BlenderOrientItem.py (interactive)
- Blender script writes results directly into SQLite (imodern_item), not JSON
- non-crashing: per-item errors are recorded and batch continues
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".exr", ".webp"}

# ----------------------------- time / db -----------------------------

def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def _connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA busy_timeout=5000;")
    return con

def _cols(con: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in con.execute(f"PRAGMA table_info({table});").fetchall()]

def _add_col_if_missing(con: sqlite3.Connection, table: str, col: str, col_def: str) -> None:
    if col in set(_cols(con, table)):
        return
    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def};")

def ensure_schema_imodern_item(con: sqlite3.Connection) -> None:
    t = "imodern_item"

    # paths / status
    _add_col_if_missing(con, t, "local_obj_path", "TEXT")
    _add_col_if_missing(con, t, "local_model_dir", "TEXT")
    _add_col_if_missing(con, t, "orient_status", "TEXT")        # ok / no_obj / blender_failed / exception / ...
    _add_col_if_missing(con, t, "orient_error", "TEXT")
    _add_col_if_missing(con, t, "orient_updated_at", "TEXT")

    # textures
    _add_col_if_missing(con, t, "has_base_textures", "INTEGER DEFAULT 0")
    _add_col_if_missing(con, t, "texture_files", "TEXT")        # JSON list (paths)
    _add_col_if_missing(con, t, "texture_apply_error", "TEXT")

    # numeric coefficients
    _add_col_if_missing(con, t, "scale_power10_k", "INTEGER")
    _add_col_if_missing(con, t, "scale_factor", "REAL")
    _add_col_if_missing(con, t, "unit_guess", "TEXT")

    _add_col_if_missing(con, t, "bbox_raw_x", "REAL")
    _add_col_if_missing(con, t, "bbox_raw_y", "REAL")
    _add_col_if_missing(con, t, "bbox_raw_z", "REAL")

    _add_col_if_missing(con, t, "bbox_m_x", "REAL")
    _add_col_if_missing(con, t, "bbox_m_y", "REAL")
    _add_col_if_missing(con, t, "bbox_m_z", "REAL")

    _add_col_if_missing(con, t, "wall_contact_required", "INTEGER")
    _add_col_if_missing(con, t, "floor_contact_required", "INTEGER")
    _add_col_if_missing(con, t, "ceiling_contact_required", "INTEGER")
    _add_col_if_missing(con, t, "front_clearance_m", "REAL")

    _add_col_if_missing(con, t, "wall_prob_back", "REAL")
    _add_col_if_missing(con, t, "wall_prob_left", "REAL")
    _add_col_if_missing(con, t, "wall_prob_right", "REAL")
    _add_col_if_missing(con, t, "wall_prob_front", "REAL")

    _add_col_if_missing(con, t, "world_matrix_4x4", "TEXT")     # JSON 4x4

    con.commit()

def _update_item(con: sqlite3.Connection, item_id: int, **fields) -> None:
    if not fields:
        return
    keys = list(fields.keys())
    sql = "UPDATE imodern_item SET " + ", ".join([f"{k}=?" for k in keys]) + " WHERE id=?"
    con.execute(sql, [fields[k] for k in keys] + [item_id])

# ----------------------------- model discovery -----------------------------

def _iter_dirs_with_bx(root: Path) -> Sequence[Path]:
    if not root.exists():
        return []
    out = []
    try:
        for p in root.rglob("*"):
            if p.is_dir():
                out.append(p)
    except Exception:
        return []
    return out

def build_index(model_roots: List[Path]) -> Dict[str, List[Path]]:
    """
    Map bx_id -> list of directories where bx_id appears in name.
    """
    idx: Dict[str, List[Path]] = {}
    bx_pat = re.compile(r"(bx_\d+_\d+)")
    for root in model_roots:
        if not root.exists():
            continue
        # shallow scan first (fast)
        try:
            for p in root.iterdir():
                if p.is_dir():
                    m = bx_pat.search(p.name)
                    if m:
                        idx.setdefault(m.group(1), []).append(p)
        except Exception:
            pass
    return idx

def _choose_largest_obj(dir_path: Path) -> Optional[Path]:
    objs = list(dir_path.rglob("*.obj"))
    if not objs:
        return None
    objs.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return objs[0]

def find_obj_for_bx(model_roots: List[Path], idx: Dict[str, List[Path]], bx_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Returns (model_dir, obj_path)
    """
    # 1) from index (fast)
    dirs = idx.get(bx_id, [])
    for d in dirs:
        obj = _choose_largest_obj(d)
        if obj:
            return d, obj

    # 2) fallback: search by substring across roots (slower)
    for root in model_roots:
        if not root.exists():
            continue
        try:
            for d in root.rglob(f"*{bx_id}*"):
                if d.is_dir():
                    obj = _choose_largest_obj(d)
                    if obj:
                        return d, obj
        except Exception:
            continue

    return None, None

# ----------------------------- blender run -----------------------------

def run_blender(
    blender_bin: Path,
    blender_script: Path,
    db_path: Path,
    item_id: int,
    bx_id: str,
    obj_path: Path,
) -> Tuple[bool, str]:
    """
    Blender script writes to DB. We only check exit code.
    """
    cmd = [
        str(blender_bin),
        "--factory-startup",
        "--python",
        str(blender_script),
        "--",
        "--db", str(db_path),
        "--item-id", str(item_id),
        "--bx-id", str(bx_id),
        "--obj", str(obj_path),
    ]
    try:
        p = subprocess.run(cmd, check=False)
        if p.returncode != 0:
            return False, f"blender returncode={p.returncode}"
        return True, ""
    except Exception as e:
        return False, f"subprocess exception: {e!r}"

# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="src/data/sourse/imodern.db")
    ap.add_argument("--table", default="imodern_item")  # фиксируем, но оставим параметр
    ap.add_argument("--model-root", action="append", default=[], help="Repeatable. Roots where extracted models live")
    ap.add_argument("--blender-bin", default="/Applications/Blender.app/Contents/MacOS/Blender")
    ap.add_argument("--blender-script", default="src/Plasement/BlenderOrientItem.py")
    ap.add_argument("--where", default=None, help="Optional SQL WHERE (without 'WHERE')")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-ok", action="store_true", help="Skip rows where orient_status='ok'")
    ap.add_argument("--log", default="batch_orient_to_db.log")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    blender_bin = Path(args.blender_bin).expanduser().resolve()
    blender_script = Path(args.blender_script).expanduser().resolve()

    # default model roots if not provided
    model_roots = [Path(p).expanduser().resolve() for p in (args.model_root or [])]
    if not model_roots:
        # безопасные дефолты под твой репозиторий
        model_roots = [
            Path("extracted").resolve(),
            Path("src/data/sourse/imodern").resolve(),  # если у тебя там есть распаковка
            Path("src/data/sourse/imodern/extracted").resolve(),
        ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.log, encoding="utf-8")],
    )

    if not db_path.exists():
        logging.error("DB not found: %s", db_path)
        return 2
    if not blender_bin.exists():
        logging.error("Blender bin not found: %s", blender_bin)
        return 2
    if not blender_script.exists():
        logging.error("Blender script not found: %s", blender_script)
        return 2

    con = _connect(db_path)
    try:
        ensure_schema_imodern_item(con)

        idx = build_index(model_roots)
        logging.info("Model roots: %s", ", ".join(str(p) for p in model_roots))
        logging.info("Indexed bx dirs: %d", len(idx))

        where_parts = []
        if args.where:
            where_parts.append(f"({args.where})")
        if args.skip_ok:
            where_parts.append("(orient_status IS NULL OR orient_status <> 'ok')")

        where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        limit_sql = f" LIMIT {int(args.limit)}" if args.limit and args.limit > 0 else ""

        rows = con.execute(
            f"SELECT id, bx_id FROM imodern_item{where_sql} ORDER BY id{limit_sql};"
        ).fetchall()

        logging.info("Selected rows: %d", len(rows))

        for i, r in enumerate(rows, 1):
            item_id = int(r["id"])
            bx_id = (r["bx_id"] or "").strip()
            if not bx_id:
                _update_item(
                    con, item_id,
                    orient_status="skipped",
                    orient_error="bx_id is NULL/empty",
                    orient_updated_at=_utc_now_iso(),
                )
                con.commit()
                logging.warning("[%d/%d] id=%d: empty bx_id -> skipped", i, len(rows), item_id)
                continue

            logging.info("[%d/%d] %s (id=%d): locating OBJ", i, len(rows), bx_id, item_id)
            try:
                model_dir, obj_path = find_obj_for_bx(model_roots, idx, bx_id)
                if not obj_path:
                    _update_item(
                        con, item_id,
                        orient_status="no_obj",
                        orient_error="local OBJ not found",
                        orient_updated_at=_utc_now_iso(),
                        local_model_dir=str(model_dir) if model_dir else None,
                        local_obj_path=None,
                    )
                    con.commit()
                    logging.warning("[%d/%d] %s: no obj found", i, len(rows), bx_id)
                    continue

                # write paths early
                _update_item(
                    con, item_id,
                    local_model_dir=str(model_dir),
                    local_obj_path=str(obj_path),
                    orient_status="pending",
                    orient_error=None,
                    orient_updated_at=_utc_now_iso(),
                )
                con.commit()

                ok, err = run_blender(blender_bin, blender_script, db_path, item_id, bx_id, obj_path)
                if not ok:
                    _update_item(
                        con, item_id,
                        orient_status="blender_failed",
                        orient_error=err,
                        orient_updated_at=_utc_now_iso(),
                    )
                    con.commit()
                    logging.error("[%d/%d] %s: blender failed: %s", i, len(rows), bx_id, err)
                    continue

                # Blender script should mark orient_status='ok' itself on successful save.
                logging.info("[%d/%d] %s: done", i, len(rows), bx_id)

            except Exception as e:
                _update_item(
                    con, item_id,
                    orient_status="exception",
                    orient_error=repr(e),
                    orient_updated_at=_utc_now_iso(),
                )
                con.commit()
                logging.exception("[%d/%d] %s: exception", i, len(rows), bx_id)

        logging.info("All done.")
        return 0
    finally:
        con.close()

if __name__ == "__main__":
    raise SystemExit(main())