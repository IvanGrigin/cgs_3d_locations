#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Конвертер JSON: из "полного" формата оставляет:
- schema (верхний уровень)
- upright_idx (верхний уровень, если есть)
- rooms[].id
- rooms[].floor_polygon_xz (как есть)
- rooms[].objects[]:
    - model_id
    - category
    - pos: только x,z
    - size: вычисляется из bbox_world_xy как [x_max - x_min, y_max - y_min] (если bbox_world_xy есть)
      где bbox_world_xy = [x_min, x_max, y_min, y_max]
Пропускает неизвестные поля.

Дополнительно:
- Можно включать/выключать ограничение по минимальному числу объектов.
- Если ограничение включено, сохраняются только те файлы, где суммарное число объектов
  по всем комнатам >= N.

Использование:
  python convert_json_folder.py /path/to/input_folder /path/to/output_folder
  python convert_json_folder.py /in /out --min-objects 5
  python convert_json_folder.py /in /out --min-objects 5 --no-min-objects-filter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _bbox_to_size(bbox_world_xy: Any) -> Optional[List[float]]:
    """
    bbox_world_xy ожидается как [x_min, x_max, y_min, y_max]
    size = [x_max - x_min, y_max - y_min]
    """
    if not isinstance(bbox_world_xy, list) or len(bbox_world_xy) != 4:
        return None

    x_min = _safe_float(bbox_world_xy[0])
    x_max = _safe_float(bbox_world_xy[1])
    y_min = _safe_float(bbox_world_xy[2])
    y_max = _safe_float(bbox_world_xy[3])

    if x_min is None or x_max is None or y_min is None or y_max is None:
        return None

    return [x_max - x_min, y_max - y_min]


def _convert_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "model_id" in obj:
        out["model_id"] = obj["model_id"]

    if "category" in obj:
        out["category"] = obj["category"]

    pos = obj.get("pos")
    if isinstance(pos, dict):
        pos_out: Dict[str, Any] = {}
        if "x" in pos:
            pos_out["x"] = pos["x"]
        if "z" in pos:
            pos_out["z"] = pos["z"]
        if pos_out:
            out["pos"] = pos_out

    size = _bbox_to_size(obj.get("bbox_world_xy"))
    if size is not None:
        out["size"] = size

    return out


def _convert_room(room: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "id" in room:
        out["id"] = room["id"]

    if "floor_polygon_xz" in room:
        out["floor_polygon_xz"] = room["floor_polygon_xz"]

    objects = room.get("objects")
    if isinstance(objects, list):
        out["objects"] = [_convert_object(o) for o in objects if isinstance(o, dict)]
    else:
        out["objects"] = []

    return out


def convert_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "schema" in doc:
        out["schema"] = doc["schema"]

    if "upright_idx" in doc:
        out["upright_idx"] = doc["upright_idx"]

    rooms = doc.get("rooms")
    if isinstance(rooms, list):
        out["rooms"] = [_convert_room(r) for r in rooms if isinstance(r, dict)]
    else:
        out["rooms"] = []

    return out


def count_total_objects(doc: Dict[str, Any]) -> int:
    rooms = doc.get("rooms")
    if not isinstance(rooms, list):
        return 0
    total = 0
    for r in rooms:
        if not isinstance(r, dict):
            continue
        objs = r.get("objects")
        if isinstance(objs, list):
            total += sum(1 for o in objs if isinstance(o, dict))
    return total


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert folder of scene JSONs to minimal format.")
    parser.add_argument("input_dir", type=str, help="Входная папка с JSON файлами")
    parser.add_argument("output_dir", type=str, help="Выходная папка для преобразованных JSON файлов")

    # Ограничение по минимальному числу объектов:
    # - по умолчанию фильтр ВКЛЮЧЕН (если пользователь передал --min-objects > 0).
    # - можно явно выключить флагом --no-min-objects-filter.
    parser.add_argument(
        "--min-objects",
        type=int,
        default=0,
        help="Минимальное число объектов (суммарно по всем комнатам) для сохранения файла. 0 = не ограничивать.",
    )
    parser.add_argument(
        "--min-objects-filter",
        dest="min_objects_filter",
        action="store_true",
        help="Включить фильтрацию по --min-objects (по умолчанию включено, если --min-objects > 0).",
    )
    parser.add_argument(
        "--no-min-objects-filter",
        dest="min_objects_filter",
        action="store_false",
        help="Выключить фильтрацию по --min-objects.",
    )
    parser.set_defaults(min_objects_filter=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found or not a directory: {input_dir}")

    json_files = sorted([p for p in input_dir.rglob("*.json") if p.is_file()])
    if not json_files:
        raise SystemExit(f"No .json files found in: {input_dir}")

    min_objects: int = args.min_objects
    if min_objects < 0:
        raise SystemExit("--min-objects must be >= 0")

    # Логика включения фильтра:
    # - Если пользователь явно указал --min-objects-filter / --no-min-objects-filter, используем это.
    # - Иначе: фильтр включён, если min_objects > 0.
    if args.min_objects_filter is None:
        filter_enabled = (min_objects > 0)
    else:
        filter_enabled = bool(args.min_objects_filter)

    converted_count = 0
    skipped_count = 0
    failed: List[str] = []

    for src_path in json_files:
        rel = src_path.relative_to(input_dir)
        dst_path = output_dir / rel
        try:
            raw = _load_json(src_path)
            if not isinstance(raw, dict):
                raise ValueError("Top-level JSON is not an object/dict")

            if filter_enabled and min_objects > 0:
                total_objs = count_total_objects(raw)
                if total_objs < min_objects:
                    skipped_count += 1
                    continue

            minimal = convert_doc(raw)
            _save_json(dst_path, minimal)
            converted_count += 1

        except Exception as e:
            failed.append(f"{src_path}: {e}")

    print(f"Converted: {converted_count}")
    print(f"Skipped (min-objects filter): {skipped_count}" if filter_enabled and min_objects > 0 else "Skipped: 0")
    if failed:
        print(f"Failed: {len(failed)}")
        for line in failed[:50]:
            print(f"- {line}")
        if len(failed) > 50:
            print(f"... and {len(failed) - 50} more")


if __name__ == "__main__":
    main()
