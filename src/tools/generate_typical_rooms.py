#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/generate_typical_rooms.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RoomSpec:
    room_id: str
    room_type: str
    width_m: float
    depth_m: float
    ceiling_height_m: float
    style_hint: str
    title_ru: str


def rect_floor_polygon(width_m: float, depth_m: float) -> list[dict[str, float]]:
    """
    Возвращает прямоугольный floor_polygon в формате, который уже совместим
    с вашим load_room_metrics(...): список точек [{"x": ..., "y": ...}, ...].

    Начало координат — левый нижний угол комнаты.
    """
    return [
        {"x": 0.0, "y": 0.0},
        {"x": round(width_m, 6), "y": 0.0},
        {"x": round(width_m, 6), "y": round(depth_m, 6)},
        {"x": 0.0, "y": round(depth_m, 6)},
    ]


def rect_floor_polygon_xz(width_m: float, depth_m: float) -> list[dict[str, float]]:
    """
    Дополнительное поле для тех частей пайплайна, где план комнаты ожидается в XZ.
    """
    return [
        {"x": 0.0, "z": 0.0},
        {"x": round(width_m, 6), "z": 0.0},
        {"x": round(width_m, 6), "z": round(depth_m, 6)},
        {"x": 0.0, "z": round(depth_m, 6)},
    ]


def build_room_json(spec: RoomSpec) -> dict[str, Any]:
    area_m2 = round(spec.width_m * spec.depth_m, 3)

    floor_polygon = rect_floor_polygon(spec.width_m, spec.depth_m)
    floor_polygon_xz = rect_floor_polygon_xz(spec.width_m, spec.depth_m)

    return {
        "version": "1.0",
        "room": {
            "id": spec.room_id,
            "name": spec.room_id,
            "title_ru": spec.title_ru,
            "room_type": spec.room_type,
            "style_hint": spec.style_hint,
            "width_m": round(spec.width_m, 3),
            "depth_m": round(spec.depth_m, 3),
            "area_m2": area_m2,
            "ceiling_height_m": round(spec.ceiling_height_m, 3),
            "floor_polygon": floor_polygon,
            "floor_polygon_xz": floor_polygon_xz,
            "doors": [],
            "windows": [],
            "openings": [],
            "notes": {
                "generator": "generate_typical_rooms.py",
                "comment": "Прямоугольная типовая комната без дверей и окон. Подходит как базовый room-spec."
            },
        }
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_typical_room_specs() -> list[RoomSpec]:
    """
    10 типичных комнат:
    - 4 спальни
    - 3 гостиные
    - 3 кабинета

    Размеры выбраны реалистичными для массового жилья:
    - спальни: примерно 10–18 м²
    - кабинеты: примерно 8–13 м²
    - гостиные: примерно 16–28 м²

    Это полезно, потому что генераторы мебели и расстановки ведут себя заметно
    по-разному на маленьких, средних и просторных комнатах.
    """
    return [
        RoomSpec(
            room_id="bedroom_small_01",
            room_type="bedroom",
            width_m=3.2,
            depth_m=3.4,
            ceiling_height_m=2.70,
            style_hint="compact bedroom",
            title_ru="Маленькая спальня",
        ),
        RoomSpec(
            room_id="bedroom_medium_02",
            room_type="bedroom",
            width_m=3.6,
            depth_m=4.0,
            ceiling_height_m=2.70,
            style_hint="standard bedroom",
            title_ru="Стандартная спальня",
        ),
        RoomSpec(
            room_id="bedroom_master_03",
            room_type="bedroom",
            width_m=4.2,
            depth_m=4.4,
            ceiling_height_m=2.80,
            style_hint="master bedroom",
            title_ru="Просторная спальня",
        ),
        RoomSpec(
            room_id="bedroom_narrow_04",
            room_type="bedroom",
            width_m=3.0,
            depth_m=4.6,
            ceiling_height_m=2.70,
            style_hint="narrow bedroom",
            title_ru="Узкая спальня",
        ),
        RoomSpec(
            room_id="living_compact_05",
            room_type="living",
            width_m=4.2,
            depth_m=4.3,
            ceiling_height_m=2.80,
            style_hint="compact living room",
            title_ru="Компактная гостиная",
        ),
        RoomSpec(
            room_id="living_standard_06",
            room_type="living",
            width_m=4.8,
            depth_m=5.2,
            ceiling_height_m=2.80,
            style_hint="standard living room",
            title_ru="Стандартная гостиная",
        ),
        RoomSpec(
            room_id="living_large_07",
            room_type="living",
            width_m=5.5,
            depth_m=5.1,
            ceiling_height_m=3.00,
            style_hint="large living room",
            title_ru="Большая гостиная",
        ),
        RoomSpec(
            room_id="office_small_08",
            room_type="office",
            width_m=2.8,
            depth_m=3.2,
            ceiling_height_m=2.70,
            style_hint="small office",
            title_ru="Маленький кабинет",
        ),
        RoomSpec(
            room_id="office_standard_09",
            room_type="office",
            width_m=3.2,
            depth_m=3.8,
            ceiling_height_m=2.70,
            style_hint="standard office",
            title_ru="Стандартный кабинет",
        ),
        RoomSpec(
            room_id="office_large_10",
            room_type="office",
            width_m=3.6,
            depth_m=4.2,
            ceiling_height_m=2.80,
            style_hint="large office",
            title_ru="Большой кабинет",
        ),
    ]


def build_summary(specs: list[RoomSpec]) -> dict[str, Any]:
    by_type: dict[str, list[str]] = {}
    for spec in specs:
        by_type.setdefault(spec.room_type, []).append(spec.room_id)

    return {
        "count": len(specs),
        "room_ids": [x.room_id for x in specs],
        "by_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерация 10 типичных room-spec JSON для спален, гостиных и кабинетов"
    )
    parser.add_argument(
        "--out-dir",
        default="data/input/generated_typical_rooms",
        help="Каталог, куда будут сохранены room JSON",
    )
    parser.add_argument(
        "--with-summary",
        action="store_true",
        help="Дополнительно сохранить summary.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = get_typical_room_specs()

    for spec in specs:
        room_json = build_room_json(spec)
        out_path = out_dir / f"{spec.room_id}.json"
        save_json(out_path, room_json)
        print(f"OK: {out_path}")

    if args.with_summary:
        summary_path = out_dir / "summary.json"
        save_json(summary_path, build_summary(specs))
        print(f"OK: {summary_path}")

    print(f"Generated {len(specs)} rooms into: {out_dir}")


if __name__ == "__main__":
    main()