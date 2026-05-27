#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/tools/export_angle_pairs_csv.py

Экспорт пар предметов (между категориями) из 3D-FRONT *.mini.json в один CSV.

Фильтры:
- Использовать только те комнаты, где объектов строго больше 3.
- Строим полный граф по объектам, но сохраняем только пары с разными категориями (cat1 != cat2).
- Порядок категорий в строке: алфавитный (cat1 <= cat2).
- Угол всегда считается от cat1 к cat2 (после упорядочивания).

Геометрия:
- Угол: theta = atan2(z2-z1, x2-x1) в градусах, нормировка в [0, 360).
- dist: расстояние между центрами (x,z).
- dist_norm: dist / D_room, где D_room — диагональ bbox комнаты.

Квантование:
- angle_q30: округление угла до ближайших 30°
- angle_q60: до 60°
- angle_q90: до 90°
- dist_q40: округление dist до ближайших 0.4 метра (40 см)
- dist_norm_q40: dist_q40 / D_room

Запуск:
  python -m src.tools.export_angle_pairs_csv \
    --input_glob "data/sourse/3D-FRONT/3D-FRONT-processed-mini/*.mini.json" \
    --out_csv "data/input/graph_stat/pairs.csv" \
    --limit 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import glob


# -----------------------------
# JSON helpers
# -----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


# -----------------------------
# Geometry
# -----------------------------

def bbox_from_poly(poly_xz: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(poly_xz, list) or len(poly_xz) < 3:
        return None
    xs: List[float] = []
    zs: List[float] = []
    for v in poly_xz:
        if not isinstance(v, dict):
            continue
        x = safe_float(v.get("x"))
        z = safe_float(v.get("z"))
        if x is None or z is None:
            continue
        xs.append(x)
        zs.append(z)
    if len(xs) < 3:
        return None
    return (min(xs), max(xs), min(zs), max(zs))


def room_diag_bbox(poly_xz: List[Dict[str, Any]]) -> Optional[float]:
    bb = bbox_from_poly(poly_xz)
    if bb is None:
        return None
    xmin, xmax, zmin, zmax = bb
    dx = xmax - xmin
    dz = zmax - zmin
    d = math.sqrt(dx * dx + dz * dz)
    if not math.isfinite(d) or d <= 1e-12:
        return None
    return d


def angle_deg_0_360(x1: float, z1: float, x2: float, z2: float) -> float:
    dx = x2 - x1
    dz = z2 - z1
    a = math.degrees(math.atan2(dz, dx))
    return (a + 360.0) % 360.0


def dist(x1: float, z1: float, x2: float, z2: float) -> float:
    dx = x2 - x1
    dz = z2 - z1
    return math.sqrt(dx * dx + dz * dz)


# -----------------------------
# Quantization
# -----------------------------

def quantize_angle_deg(angle_deg: float, step_deg: float) -> float:
    """
    Округление угла в градусах до ближайшего шага step_deg.
    Результат в [0, 360).
    Пример: step=30 -> 0,30,60,...,330
    """
    a = angle_deg % 360.0
    k = int(round(a / step_deg))
    q = (k * step_deg) % 360.0
    # защита от -0.0
    if abs(q) < 1e-12:
        q = 0.0
    return float(q)


def quantize_dist_m(dist_m: float, step_m: float) -> float:
    """
    Округление расстояния до ближайшего шага (в метрах).
    step_m=0.4 -> кратность 40 см.
    """
    if dist_m < 0:
        dist_m = 0.0
    k = int(round(dist_m / step_m))
    q = float(k * step_m)
    if q < 0.0:
        q = 0.0
    return q


# -----------------------------
# Object category extraction
# -----------------------------

def obj_category(obj: Dict[str, Any]) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    for k in ("name", "label"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    sc = obj.get("super-category")
    c = obj.get("category")
    if isinstance(sc, str) and sc.strip() and isinstance(c, str) and c.strip():
        return f"{sc.strip()}/{c.strip()}"

    if isinstance(c, str) and c.strip():
        return c.strip()

    return None


def obj_pos_xz(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pos = obj.get("pos")
    if not isinstance(pos, dict):
        return None
    x = safe_float(pos.get("x"))
    z = safe_float(pos.get("z"))
    if x is None or z is None:
        return None
    return (x, z)


# -----------------------------
# Main export
# -----------------------------

def iter_mini_json_files(input_glob: str) -> List[Path]:
    files = glob.glob(input_glob, recursive=True)
    files = [f for f in files if f.endswith(".mini.json")]
    files.sort()
    return [Path(f) for f in files]


def export_pairs_to_csv(input_glob: str, out_csv: Path, limit_files: int = 0) -> None:
    files = iter_mini_json_files(input_glob)
    if limit_files and limit_files > 0:
        files = files[:limit_files]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "cat1",
        "cat2",
        "angle_deg",
        "angle_q30",
        "angle_q60",
        "angle_q90",
        "dist",
        "dist_q40",
        "dist_norm",
        "dist_norm_q40",
        "source_file",
        "room_id",
    ]

    total_files = 0
    total_rooms = 0
    total_pairs = 0
    skipped_rooms_small = 0
    skipped_rooms_badgeom = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for fp in files:
            if not fp.exists():
                continue
            total_files += 1

            try:
                data = load_json(fp)
            except Exception:
                continue

            rooms = data.get("rooms", [])
            if not isinstance(rooms, list):
                continue

            for room in rooms:
                if not isinstance(room, dict):
                    continue

                objs = room.get("objects", [])
                if not isinstance(objs, list) or len(objs) <= 3:
                    skipped_rooms_small += 1
                    continue

                poly = room.get("floor_polygon_xz", [])
                D = room_diag_bbox(poly)
                if D is None:
                    skipped_rooms_badgeom += 1
                    continue

                # валидные объекты: категория + позиция
                items: List[Tuple[str, float, float]] = []
                for obj in objs:
                    cat = obj_category(obj)
                    p = obj_pos_xz(obj)
                    if cat is None or p is None:
                        continue
                    x, z = p
                    items.append((cat, x, z))

                if len(items) <= 3:
                    skipped_rooms_small += 1
                    continue

                room_id = room.get("id")
                room_id_str = str(room_id) if room_id is not None else ""

                n = len(items)
                total_rooms += 1

                for i in range(n):
                    cat_i, xi, zi = items[i]
                    for j in range(i + 1, n):
                        cat_j, xj, zj = items[j]
                        if cat_i == cat_j:
                            continue

                        # алфавитный порядок в строке
                        if cat_i < cat_j:
                            c1, x1, z1 = cat_i, xi, zi
                            c2, x2, z2 = cat_j, xj, zj
                        else:
                            c1, x1, z1 = cat_j, xj, zj
                            c2, x2, z2 = cat_i, xi, zi

                        a = angle_deg_0_360(x1, z1, x2, z2)
                        d = dist(x1, z1, x2, z2)
                        dn = d / D

                        a30 = quantize_angle_deg(a, 30.0)
                        a60 = quantize_angle_deg(a, 60.0)
                        a90 = quantize_angle_deg(a, 90.0)

                        dq = quantize_dist_m(d, 0.4)      # 40 см
                        dnq = dq / D

                        w.writerow([
                            c1,
                            c2,
                            f"{a:.6f}",
                            f"{a30:.6f}",
                            f"{a60:.6f}",
                            f"{a90:.6f}",
                            f"{d:.6f}",
                            f"{dq:.6f}",
                            f"{dn:.6f}",
                            f"{dnq:.6f}",
                            str(fp),
                            room_id_str,
                        ])
                        total_pairs += 1

    print(f"[export_angle_pairs_csv] saved: {out_csv}")
    print(f"[export_angle_pairs_csv] files={total_files} rooms_used={total_rooms} pairs={total_pairs}")
    print(f"[export_angle_pairs_csv] skipped_rooms_small(<=3 objs)={skipped_rooms_small}")
    print(f"[export_angle_pairs_csv] skipped_rooms_badgeom={skipped_rooms_badgeom}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_glob",
        required=True,
        help="Glob for *.mini.json (supports **). Example: data/.../**/*.mini.json",
    )
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all files, иначе ограничить число файлов.")
    args = ap.parse_args()

    export_pairs_to_csv(
        input_glob=str(args.input_glob),
        out_csv=Path(args.out_csv),
        limit_files=int(args.limit),
    )


if __name__ == "__main__":
    main()
