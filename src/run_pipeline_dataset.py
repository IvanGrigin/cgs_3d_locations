#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/run_pipeline_dataset.py

"""
Генератор датасета placement-сцен без Blender.

По умолчанию генерирует ОДИН пример и сохраняет в:
  src/data/output/<example_hash>/

Содержимое:
  room.json
  objects.json
  placement_random.json
  placement_relaxed.json
  scene_random.json
  scene_relaxed.json
  meta.json

Опционально можно генерировать N примеров (датасет), тогда будет:
  out_dir/sample_0000/...
  out_dir/sample_0001/...
  ...

Правило подбора предметов:
  15 м² -> 3 предмета, +1 за каждые +3 м²
Пул:
  стул, стол, диван, стеллаж, атаманка, тумбочка
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import secrets
import random
from pathlib import Path
from typing import Optional, Any

import run_pipeline as rp  # используем generate_objects_json, merge_room_spec_and_placements, пути/скрипты


# -----------------------------
# Геометрия / площадь комнаты
# -----------------------------
def _shoelace_area_xy(points: list[dict[str, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        s += a["x"] * b["y"] - b["x"] * a["y"]
    return abs(s) * 0.5


def room_area_m2(room_json_path: str) -> float:
    p = Path(room_json_path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    room = data.get("room", {})
    meta = room.get("meta", {})
    if isinstance(meta, dict) and isinstance(meta.get("area_m2"), (int, float)):
        return float(meta["area_m2"])
    poly = room.get("floor_polygon", [])
    if isinstance(poly, list) and poly and isinstance(poly[0], dict):
        return float(_shoelace_area_xy(poly))
    return 0.0


# -----------------------------
# Подбор предметов по площади
# -----------------------------
def items_count_by_area(area_m2: float) -> int:
    base = 3
    if area_m2 <= 15.0:
        return base
    extra = int((area_m2 - 15.0) // 3.0)
    return base + max(0, extra)


AUTO_ITEM_POOL = [
    # name, weight, max_count (None = без ограничений)
    ("стул",     5.0, None),
    ("стол",     2.0, 1),
    ("диван",    1.5, 1),
    ("стеллаж",  1.2, 1),
    ("атаманка", 1.2, 2),
    ("тумбочка", 2.0, 2),
]


def pick_items_for_area(area_m2: float, rng: random.Random) -> list[str]:
    n = items_count_by_area(area_m2)

    counts: dict[str, int] = {}
    chosen: list[str] = []

    def allowed(name: str, max_count: Optional[int]) -> bool:
        if max_count is None:
            return True
        return counts.get(name, 0) < max_count

    for _ in range(n):
        avail = [(name, w, mx) for (name, w, mx) in AUTO_ITEM_POOL if allowed(name, mx)]
        if not avail:
            break

        total_w = sum(w for _, w, _ in avail)
        r = rng.random() * total_w
        acc = 0.0
        picked = avail[-1][0]
        for name, w, _mx in avail:
            acc += w
            if r <= acc:
                picked = name
                break

        chosen.append(picked)
        counts[picked] = counts.get(picked, 0) + 1

    # если есть стол, но нет стула — гарантируем хотя бы 1 стул
    if "стол" in chosen and "стул" not in chosen and len(chosen) >= 2:
        j = rng.randrange(len(chosen))
        chosen[j] = "стул"

    return chosen


# -----------------------------
# I/O helpers
# -----------------------------
def save_copy(src_path: str, dst_path: str) -> None:
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# CubePlacement runner
# -----------------------------
def run_cubeplacement(room_path: str, mode: str) -> None:
    cube_input = f"{room_path}\n{rp.OBJECTS_JSON}\n{mode}\n"
    subprocess.run([sys.executable, rp.CUBE_SCRIPT], input=cube_input, text=True, check=True)


# -----------------------------
# Output directory logic
# -----------------------------
def make_example_out_dir(base_out_dir: Path, seed: int) -> Path:
    """
    Создаём человекочитаемый "hash"-подобный id, но без зависимостей.
    Важно: secrets - не воспроизводим, seed - воспроизводим.
    Компромисс: хешируем (seed + случайный salt) если seed не задан.
    """
    # Если seed задан — делаем детерминированный id, иначе — случайный.
    if seed is not None:
        token = f"{seed:x}".rjust(8, "0")
    else:
        token = secrets.token_hex(4)  # 8 hex chars
    return base_out_dir / f"example_{token}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate placement examples/dataset (no Blender).")
    ap.add_argument("--rooms-dir", default="src/data/input/rooms", help="Каталог с room*.json")
    # По умолчанию: output/<example_hash>/
    ap.add_argument("--out-dir", default="src/data/output", help="Базовый каталог вывода")
    ap.add_argument("--n", type=int, default=1, help="Сколько примеров сгенерировать (по умолчанию 1)")
    ap.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости (если не задан — будет случайно)")
    ap.add_argument("--max-attempts", type=int, default=20, help="Попыток на один sample (если placement падает)")
    ap.add_argument("--regen-per-attempt", action="store_true", help="Перегенерировать objects.json при каждой попытке")
    ap.add_argument("--furniture-db", default=None, help="Путь к furniture_types.json (опционально)")
    ap.add_argument("--modes", default="random,relaxed", help="Режимы через запятую (по умолчанию random,relaxed)")
    ap.add_argument("--single-dir", action="store_true",
                    help="Если указан --n>1, всё равно писать в один каталог example_* (перезаписывая файлы). Обычно НЕ нужно.")
    args = ap.parse_args()

    rooms_dir = Path(args.rooms_dir).expanduser().resolve()
    base_out_dir = Path(args.out_dir).expanduser().resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    if args.furniture_db:
        rp.FURNITURE_DB = args.furniture_db

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        modes = ["random", "relaxed"]

    # RNG: если seed не задан — используем случайный seed для выбора комнат/предметов
    seed_for_rng = args.seed if args.seed is not None else int.from_bytes(secrets.token_bytes(8), "big")
    rng = random.Random(seed_for_rng)

    room_files = sorted([p for p in rooms_dir.glob("*.json") if p.is_file()])
    if not room_files:
        raise FileNotFoundError(f"No room json files in: {rooms_dir}")

    # Вывод
    if args.n == 1:
        out_dir = make_example_out_dir(base_out_dir, args.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[gen] mode=single out_dir={out_dir}")
    else:
        if args.single_dir:
            out_dir = make_example_out_dir(base_out_dir, args.seed)
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[gen] mode=multi(single-dir) n={args.n} out_dir={out_dir}")
        else:
            out_dir = base_out_dir
            print(f"[gen] mode=dataset n={args.n} out_dir={out_dir}")

    print(f"[gen] rooms_dir={rooms_dir}")
    print(f"[gen] seed_rng={seed_for_rng} (seed_arg={args.seed}) modes={modes}")

    for i in range(args.n):
        if args.n == 1:
            sample_id = "sample_0000"
            sample_dir = out_dir
        else:
            sample_id = f"sample_{i:04d}"
            sample_dir = out_dir if args.single_dir else (out_dir / sample_id)
            sample_dir.mkdir(parents=True, exist_ok=True)

        # выбор комнаты/предметов
        room_path = str(rng.choice(room_files))
        area = room_area_m2(room_path)
        items = pick_items_for_area(area, rng)

        meta = {
            "sample_id": sample_id,
            "room_path": room_path,
            "area_m2": round(area, 3),
            "items": items,
            "modes": modes,
            "seed_global_arg": args.seed,
            "seed_rng_effective": seed_for_rng,
        }

        ok = False
        for attempt in range(1, args.max_attempts + 1):
            try:
                print(f"\n[{sample_id}] attempt={attempt}/{args.max_attempts} room={Path(room_path).name} area={area:.2f} items={items}")

                if args.regen_per_attempt or attempt == 1:
                    obj_seed = int.from_bytes(secrets.token_bytes(8), "big")
                    meta["objects_seed"] = obj_seed
                    rp.generate_objects_json(items, seed=obj_seed)
                    save_copy(rp.OBJECTS_JSON, str(sample_dir / "objects.json"))

                save_copy(room_path, str(sample_dir / "room.json"))

                for mode in modes:
                    run_cubeplacement(room_path, mode)

                    placement_dst = sample_dir / f"placement_{mode}.json"
                    save_copy(rp.PLACEMENT_JSON, str(placement_dst))

                    scene_dst = sample_dir / f"scene_{mode}.json"
                    rp.merge_room_spec_and_placements(room_path, rp.PLACEMENT_JSON, str(scene_dst))

                write_json(str(sample_dir / "meta.json"), meta)
                ok = True
                print(f"[{sample_id}] OK -> {sample_dir}")
                break

            except subprocess.CalledProcessError as e:
                print(f"[{sample_id}] FAIL mode/placement (attempt {attempt}): {e}")
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"[{sample_id}] FAIL (attempt {attempt}): {e}")
                time.sleep(0.1)
                continue

        if not ok:
            (sample_dir / "FAILED.txt").write_text("failed to generate sample\n", encoding="utf-8")
            write_json(str(sample_dir / "meta.json"), meta)
            print(f"[{sample_id}] GAVE UP -> {sample_dir}")

    print("\n[gen] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
