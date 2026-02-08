#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# src/tools/make_index_bedrooms_livingrooms.py
#
# 1) Ищет спальни и гостиные в 3D-FRONT-processed (файлы вида UUID__RoomType-xxxx.json или UUID__RoomType.json)
# 2) (опционально) Генерирует mini-json для найденных сцен через Blender (вызывает make_scene_mini_blender.py)
# 3) Записывает индекс JSON в out_dir: bedrooms_livingrooms_index.json
#
# Индекс содержит пути к mini-json (относительно out_dir) и абсолютные пути (опционально).
#

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ROOMTYPE_RE = re.compile(r"^(?P<uuid>[^_]+)__(?P<room>[^.]+)\.json$", re.IGNORECASE)


@dataclass(frozen=True)
class RoomFile:
    path: Path
    room_type: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_processed_jsons(processed_dir: Path) -> List[Path]:
    files = sorted([p for p in processed_dir.glob("*.json") if p.is_file() and p.name != "meta.json"])
    return files


def parse_room_type(filename: str) -> str:
    """
    Извлекает тип комнаты из имени файла processed:
    Примеры:
      000...__Bedroom-54672.json      -> Bedroom
      000...__MasterBedroom-5863.json -> MasterBedroom
      000...__SecondBedroom-7177.json -> SecondBedroom
      000...__LivingRoom-54780.json   -> LivingRoom
      000...__LivingDiningRoom-3474.json -> LivingDiningRoom
      000...__OtherRoom.json -> OtherRoom
    """
    m = ROOMTYPE_RE.match(filename)
    if not m:
        return ""
    room = m.group("room")

    # room может быть "Bedroom-54672" или "OtherRoom" — берём префикс до первого "-"
    base = room.split("-", 1)[0].strip()
    return base


def is_bedroom(room_type: str) -> bool:
    rt = room_type.lower()
    return "bedroom" in rt


def is_living(room_type: str) -> bool:
    rt = room_type.lower()
    # LivingRoom / LivingDiningRoom
    return rt.startswith("living")


def select_rooms(processed_dir: Path) -> Tuple[List[RoomFile], List[RoomFile]]:
    bedrooms: List[RoomFile] = []
    livings: List[RoomFile] = []
    for p in list_processed_jsons(processed_dir):
        rt = parse_room_type(p.name)
        if not rt:
            continue
        if is_bedroom(rt):
            bedrooms.append(RoomFile(path=p, room_type=rt))
        elif is_living(rt):
            livings.append(RoomFile(path=p, room_type=rt))
    return bedrooms, livings


def run_blender(
    blender_bin: Path,
    blender_script: Path,
    scene_json: Path,
    models_root: Path,
    out_json: Path,
    prefer_raw: int,
    resolve_collisions: int,
    collision_margin: float,
    wall_height: float,
    upright_idx: int,
    pretty: int,
    log_path: Path,
) -> int:
    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--scene_json",
        str(scene_json),
        "--models_root",
        str(models_root),
        "--out_json",
        str(out_json),
        "--prefer_raw",
        str(int(prefer_raw)),
        "--resolve_collisions",
        str(int(resolve_collisions)),
        "--collision_margin",
        str(float(collision_margin)),
        "--wall_height",
        str(float(wall_height)),
        "--upright_idx",
        str(int(upright_idx)),
        "--pretty",
        str(int(pretty)),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    return int(proc.returncode)


def generate_mini_for_list(
    rooms: List[RoomFile],
    out_dir: Path,
    models_root: Path,
    blender_bin: Path,
    prefer_raw: int,
    resolve_collisions: int,
    collision_margin: float,
    wall_height: float,
    upright_idx: int,
    pretty: int,
    skip_existing: int,
) -> Tuple[int, int]:
    blender_script = Path(__file__).resolve().parent / "make_scene_mini_blender.py"
    if not blender_script.exists():
        raise FileNotFoundError(f"make_scene_mini_blender.py not found: {blender_script}")

    logs_dir = out_dir / "_logs"
    ensure_dir(logs_dir)

    ok = 0
    bad = 0
    for i, rf in enumerate(rooms, 1):
        stem = rf.path.stem
        out_json = out_dir / f"{stem}.mini.json"
        log_path = logs_dir / f"{stem}.blender.log"

        if int(skip_existing) == 1 and out_json.exists() and out_json.stat().st_size > 0:
            continue

        code = run_blender(
            blender_bin=blender_bin,
            blender_script=blender_script,
            scene_json=rf.path,
            models_root=models_root,
            out_json=out_json,
            prefer_raw=prefer_raw,
            resolve_collisions=resolve_collisions,
            collision_margin=collision_margin,
            wall_height=wall_height,
            upright_idx=upright_idx,
            pretty=pretty,
            log_path=log_path,
        )

        if code == 0 and out_json.exists() and out_json.stat().st_size > 0:
            ok += 1
        else:
            bad += 1

    return ok, bad


def build_index_json(
    bedrooms: List[RoomFile],
    livings: List[RoomFile],
    out_dir: Path,
    include_absolute: int,
) -> Dict:
    def mini_relpath(rf: RoomFile) -> str:
        return f"{rf.path.stem}.mini.json"

    def mini_abspath(rf: RoomFile) -> str:
        return str((out_dir / f"{rf.path.stem}.mini.json").resolve())

    index: Dict = {
        "schema": "processed-mini/index-v1",
        "out_dir": str(out_dir.resolve()),
        "bedrooms": [mini_relpath(rf) for rf in bedrooms],
        "livingrooms": [mini_relpath(rf) for rf in livings],
    }

    if int(include_absolute) == 1:
        index["bedrooms_abs"] = [mini_abspath(rf) for rf in bedrooms]
        index["livingrooms_abs"] = [mini_abspath(rf) for rf in livings]

    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--blender_bin", required=True)

    ap.add_argument("--prefer_raw", type=int, default=1)
    ap.add_argument("--resolve_collisions", type=int, default=1)
    ap.add_argument("--collision_margin", type=float, default=0.02)
    ap.add_argument("--wall_height", type=float, default=2.7)
    ap.add_argument("--upright_idx", type=int, default=2)
    ap.add_argument("--pretty", type=int, default=1)

    ap.add_argument("--skip_existing", type=int, default=1)
    ap.add_argument("--generate_mini", type=int, default=1, help="1=догенерировать mini; 0=только индекс")
    ap.add_argument("--include_absolute", type=int, default=0, help="1=добавить абсолютные пути в индекс")

    ap.add_argument("--index_name", default="bedrooms_livingrooms_index.json")

    args = ap.parse_args()

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    blender_bin = Path(args.blender_bin).expanduser().resolve()

    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    if not models_root.exists():
        raise FileNotFoundError(f"models_root not found: {models_root}")
    if not blender_bin.exists():
        raise FileNotFoundError(f"blender_bin not found: {blender_bin}")

    ensure_dir(out_dir)

    bedrooms, livings = select_rooms(processed_dir)

    if int(args.generate_mini) == 1:
        ok1, bad1 = generate_mini_for_list(
            rooms=bedrooms,
            out_dir=out_dir,
            models_root=models_root,
            blender_bin=blender_bin,
            prefer_raw=int(args.prefer_raw),
            resolve_collisions=int(args.resolve_collisions),
            collision_margin=float(args.collision_margin),
            wall_height=float(args.wall_height),
            upright_idx=int(args.upright_idx),
            pretty=int(args.pretty),
            skip_existing=int(args.skip_existing),
        )
        ok2, bad2 = generate_mini_for_list(
            rooms=livings,
            out_dir=out_dir,
            models_root=models_root,
            blender_bin=blender_bin,
            prefer_raw=int(args.prefer_raw),
            resolve_collisions=int(args.resolve_collisions),
            collision_margin=float(args.collision_margin),
            wall_height=float(args.wall_height),
            upright_idx=int(args.upright_idx),
            pretty=int(args.pretty),
            skip_existing=int(args.skip_existing),
        )
        print(f"MINI DONE: bedrooms ok={ok1} bad={bad1}; living ok={ok2} bad={bad2}")

    index = build_index_json(
        bedrooms=bedrooms,
        livings=livings,
        out_dir=out_dir,
        include_absolute=int(args.include_absolute),
    )

    index_path = out_dir / str(args.index_name)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"INDEX WRITTEN: {index_path}")
    print(f"BEDROOMS: {len(index['bedrooms'])}  LIVINGROOMS: {len(index['livingrooms'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
