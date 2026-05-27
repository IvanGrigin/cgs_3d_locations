#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/tools/export_angle_pairs_csv.py

Экспорт пар предметов (между категориями) из 3D-FRONT *.mini.json в один CSV.

Требования:
- Берём только те комнаты, где объектов строго больше 3.
- Для комнаты берём полигон пола (XZ), считаем bbox-центр комнаты C (фиксированный метод).
- Строим полный граф по объектам, но оставляем только рёбра МЕЖДУ категориями (cat1 != cat2).
- Для каждой пары считаем:
  * угол направления theta_ij = atan2(zj-zi, xj-xi) в градусах, нормировка в [0, 360)
  * расстояние между центрами предметов
  * нормированное расстояние = dist / D_room, где D_room — диагональ bbox комнаты
- Запись в CSV:
  cat1 | cat2 | angle_deg | dist | dist_norm | source_file | room_id
  Порядок предметов в строке — алфавитный по имени категории (cat1 <= cat2).
  Угол всегда считается от cat1 к cat2 (после упорядочивания).

Пример запуска:
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    """
    Возвращает (xmin, xmax, zmin, zmax) по floor_polygon_xz.
    """
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


def room_center_bbox(poly_xz: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    """
    Центр комнаты C как центр bbox полигона (фиксируем этот метод).
    """
    bb = bbox_from_poly(poly_xz)
    if bb is None:
        return None
    xmin, xmax, zmin, zmax = bb
    return (0.5 * (xmin + xmax), 0.5 * (zmin + zmax))


def room_diag_bbox(poly_xz: List[Dict[str, Any]]) -> Optional[float]:
    """
    Диагональ bbox комнаты.
    """
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
    """
    atan2(dz, dx) -> degrees in [0, 360)
    """
    dx = x2 - x1
    dz = z2 - z1
    a = math.degrees(math.atan2(dz, dx))
    a = (a + 360.0) % 360.0
    return a


def dist(x1: float, z1: float, x2: float, z2: float) -> float:
    dx = x2 - x1
    dz = z2 - z1
    return math.sqrt(dx * dx + dz * dz)


# -----------------------------
# Object category extraction
# -----------------------------

def obj_category(obj: Dict[str, Any]) -> Optional[str]:
    """
    Категория для CSV. По вашему примеру нужна строка вида 'Sofa/armchair'.
    В mini-json это поле обычно в obj["name"] или obj["label"].
    Приоритет:
      1) name
      2) label
      3) super-category + "/" + category
      4) category
    """
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
        "dist",
        "dist_norm",
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
                C = room_center_bbox(poly)  # вычисляем по ТЗ, но в CSV не пишем
                if D is None or C is None:
                    skipped_rooms_badgeom += 1
                    continue

                # собираем валидные объекты: категория + позиция
                items: List[Tuple[str, float, float]] = []
                for obj in objs:
                    cat = obj_category(obj)
                    p = obj_pos_xz(obj)
                    if cat is None or p is None:
                        continue
                    x, z = p
                    items.append((cat, x, z))

                # если после фильтрации объектов мало — пропускаем
                if len(items) <= 3:
                    skipped_rooms_small += 1
                    continue

                room_id = room.get("id")
                room_id_str = str(room_id) if room_id is not None else ""

                # полный граф по объектам, но только между категориями
                n = len(items)
                total_rooms += 1

                for i in range(n):
                    cat_i, xi, zi = items[i]
                    for j in range(i + 1, n):
                        cat_j, xj, zj = items[j]
                        if cat_i == cat_j:
                            continue  # только между категориями

                        # упорядочивание по алфавиту в строке
                        if cat_i < cat_j:
                            c1, x1, z1 = cat_i, xi, zi
                            c2, x2, z2 = cat_j, xj, zj
                        else:
                            c1, x1, z1 = cat_j, xj, zj
                            c2, x2, z2 = cat_i, xi, zi

                        a = angle_deg_0_360(x1, z1, x2, z2)
                        d = dist(x1, z1, x2, z2)
                        dn = d / D

                        w.writerow([
                            c1,
                            c2,
                            f"{a:.6f}",
                            f"{d:.6f}",
                            f"{dn:.6f}",
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
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all files, иначе ограничить число файлов (для отладки).",
    )
    args = ap.parse_args()

    export_pairs_to_csv(
        input_glob=str(args.input_glob),
        out_csv=Path(args.out_csv),
        limit_files=int(args.limit),
    )


if __name__ == "__main__":
    main()
