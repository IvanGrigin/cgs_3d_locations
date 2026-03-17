#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/run_ollama_layout.py

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LLMModule.ollama_client import chat_json
from LLMModule.retry_llm_json import ValidationResult, run_retry_loop


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def quantize_rot_0_90_180_270(deg: float) -> float:
    a = float(deg or 0.0) % 360.0
    allowed = (0.0, 90.0, 180.0, 270.0)
    return min(
        allowed,
        key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t),
    )


def point_in_polygon(x: float, y: float, polygon_xy: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon_xy)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / max((yj - yi), 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def polygon_bbox(poly_xy: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly_xy]
    ys = [p[1] for p in poly_xy]
    return min(xs), max(xs), min(ys), max(ys)


def aabb_from_center_size_rotation(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    sz: float,
    yaw_deg: float,
    z_floor_m: float = 0.0,
) -> Dict[str, float]:
    rot = quantize_rot_0_90_180_270(yaw_deg)
    if rot in (90.0, 270.0):
        sx, sy = sy, sx

    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": z_floor_m,
        "z_max": z_floor_m + sz,
    }


def rect_inside_polygon(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    yaw_deg: float,
    polygon_xy: List[Tuple[float, float]],
) -> bool:
    rot = quantize_rot_0_90_180_270(yaw_deg)
    if rot in (90.0, 270.0):
        sx, sy = sy, sx

    hx = sx / 2.0
    hy = sy / 2.0
    corners = [
        (cx - hx, cy - hy),
        (cx - hx, cy + hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
    ]
    return all(point_in_polygon(x, y, polygon_xy) for x, y in corners)


def rects_overlap_2d(a: Dict[str, float], b: Dict[str, float], eps: float = 1e-6) -> bool:
    return not (
        a["x_max"] <= b["x_min"] + eps
        or a["x_min"] >= b["x_max"] - eps
        or a["y_max"] <= b["y_min"] + eps
        or a["y_min"] >= b["y_max"] - eps
    )


def find_nearest_valid_position(
    target_x: float,
    target_y: float,
    sx: float,
    sy: float,
    sz: float,
    yaw_deg: float,
    polygon_xy: List[Tuple[float, float]],
    occupied: List[Dict[str, float]],
    grid_step: float = 0.10,
    max_radius_steps: int = 120,
) -> Tuple[float, float]:
    cand_aabb = aabb_from_center_size_rotation(target_x, target_y, sx, sy, sz, yaw_deg)
    if rect_inside_polygon(target_x, target_y, sx, sy, yaw_deg, polygon_xy):
        if not any(rects_overlap_2d(cand_aabb, occ) for occ in occupied):
            return target_x, target_y

    for r in range(1, max_radius_steps + 1):
        d = r * grid_step

        xs = [target_x - d + i * grid_step for i in range(2 * r + 1)]
        ys = [target_y - d + i * grid_step for i in range(2 * r + 1)]

        border_points: List[Tuple[float, float]] = []
        for x in xs:
            border_points.append((x, target_y - d))
            border_points.append((x, target_y + d))
        for y in ys[1:-1]:
            border_points.append((target_x - d, y))
            border_points.append((target_x + d, y))

        for cx, cy in border_points:
            if not rect_inside_polygon(cx, cy, sx, sy, yaw_deg, polygon_xy):
                continue
            aabb = aabb_from_center_size_rotation(cx, cy, sx, sy, sz, yaw_deg)
            if any(rects_overlap_2d(aabb, occ) for occ in occupied):
                continue
            return cx, cy

    raise ValueError("Не удалось найти допустимую позицию для объекта repair-алгоритмом")


def extract_room_polygon_xy(room: Dict[str, Any]) -> List[Tuple[float, float]]:
    root = room.get("room") if isinstance(room.get("room"), dict) else room
    poly = root.get("floor_polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        raise ValueError("В room.json не найден корректный room.floor_polygon")

    out = []
    for p in poly:
        out.append((float(p["x"]), float(p["y"])))
    return out


def extract_ceiling_height(room: Dict[str, Any]) -> float:
    root = room.get("room") if isinstance(room.get("room"), dict) else room
    return float(root.get("ceiling_height", 2.8))


def extract_objects(src: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = src.get("objects") or src.get("items") or src.get("placements") or []
    if not isinstance(items, list):
        raise ValueError("objects.json: ожидается objects/items/placements как список")
    return items


def extract_class_name(obj: Dict[str, Any]) -> str:
    for key in ("class_name", "class", "type", "name"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    asset_meta = obj.get("asset_meta") or {}
    for key in ("category", "super-category", "super_category"):
        v = asset_meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return "object"


def extract_size_m(obj: Dict[str, Any]) -> List[float]:
    for key in ("size_m", "bbox_size_m", "size"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 3:
            return [float(v[0]), float(v[1]), float(v[2])]

    for key in ("min_size_mm", "max_size_mm"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 3:
            return [float(v[0]) / 1000.0, float(v[1]) / 1000.0, float(v[2]) / 1000.0]

    asset_meta = obj.get("asset_meta") or {}
    if all(k in asset_meta for k in ("size_x", "size_y", "size_z")):
        return [
            float(asset_meta["size_x"]),
            float(asset_meta["size_y"]),
            float(asset_meta["size_z"]),
        ]

    raise ValueError(f"Не удалось определить size_m для объекта: {obj}")


def extract_mount_type(obj: Dict[str, Any]) -> str:
    constraints = obj.get("constraints") or {}
    mount_type = constraints.get("mount_type")
    if isinstance(mount_type, str) and mount_type.strip():
        return mount_type.strip().lower()
    return "floor"


def build_llm_payload(room: Dict[str, Any], objects_data: Dict[str, Any]) -> Dict[str, Any]:
    poly = extract_room_polygon_xy(room)
    x_min, x_max, y_min, y_max = polygon_bbox(poly)
    ceiling_height = extract_ceiling_height(room)

    src_items = extract_objects(objects_data)
    compact_objects = []

    for idx, obj in enumerate(src_items):
        sx, sy, sz = extract_size_m(obj)
        compact_objects.append({
            "index": idx,
            "name": extract_class_name(obj),
            "mount_type": extract_mount_type(obj),
            "size_m": [round(sx, 3), round(sy, 3), round(sz, 3)],
        })

    return {
        "room": {
            "floor_polygon_xy": [[round(x, 3), round(y, 3)] for x, y in poly],
            "bbox_xy": [round(x_min, 3), round(x_max, 3), round(y_min, 3), round(y_max, 3)],
            "ceiling_height": round(ceiling_height, 3),
        },
        "objects": compact_objects,
    }


def build_system_prompt() -> str:
    return (
        "You are a room layout planner. "
        "Return only valid JSON matching the requested format. "
        "No prose. No markdown. No comments."
    )


def build_user_prompt(payload: Dict[str, Any], mode: str) -> str:
    return (
        "Place all objects in the room. "
        "Use each object index exactly once. "
        "For floor objects, output floor XY positions. "
        "For ceiling objects, output ceiling XY positions. "
        "yaw_deg must be one of 0,90,180,270. "
        f"mode={mode}. "
        "Output format: "
        "{\"placements\":[{\"index\":0,\"x\":1.0,\"y\":2.0,\"yaw_deg\":0}]}. "
        "input="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_output_schema(n_objects: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "minItems": n_objects,
                "maxItems": n_objects,
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "yaw_deg": {"type": "integer", "enum": [0, 90, 180, 270]},
                    },
                    "required": ["index", "x", "y", "yaw_deg"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["placements"],
        "additionalProperties": False,
    }


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    if not text:
        raise ValueError("Пустой ответ модели")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))

    raise ValueError("В ответе модели не найден JSON-объект")


def validate_structure(raw_text: str, n_objects: int) -> ValidationResult[Dict[str, Any]]:
    try:
        data = extract_first_json_object(raw_text)
    except Exception as e:
        return ValidationResult(
            ok=False,
            feedback=f"Ответ не является корректным JSON. Ошибка: {e}",
        )

    placements = data.get("placements")
    if not isinstance(placements, list):
        return ValidationResult(
            ok=False,
            feedback='В JSON отсутствует поле "placements" как список.',
        )

    if len(placements) != n_objects:
        return ValidationResult(
            ok=False,
            feedback=f'Количество placements неверно: {len(placements)}. Ожидается ровно {n_objects}.',
        )

    seen = set()
    normalized: List[Dict[str, Any]] = []

    for i, p in enumerate(placements):
        if not isinstance(p, dict):
            return ValidationResult(ok=False, feedback=f"Элемент placements[{i}] должен быть объектом.")

        if "index" not in p or "x" not in p or "y" not in p or "yaw_deg" not in p:
            return ValidationResult(
                ok=False,
                feedback=f'Элемент placements[{i}] должен содержать поля "index", "x", "y", "yaw_deg".',
            )

        try:
            idx = int(p["index"])
            x = float(p["x"])
            y = float(p["y"])
            yaw_deg = int(p["yaw_deg"])
        except Exception:
            return ValidationResult(
                ok=False,
                feedback=f"Элемент placements[{i}] содержит поля неверного типа.",
            )

        if idx < 0 or idx >= n_objects:
            return ValidationResult(
                ok=False,
                feedback=f"Недопустимый index={idx}. Ожидается от 0 до {n_objects - 1}.",
            )

        if idx in seen:
            return ValidationResult(ok=False, feedback=f"Индекс {idx} встречается более одного раза.")
        seen.add(idx)

        if yaw_deg not in (0, 90, 180, 270):
            return ValidationResult(
                ok=False,
                feedback=f"Для index={idx} yaw_deg={yaw_deg}. Допустимы только 0, 90, 180, 270.",
            )

        normalized.append({
            "index": idx,
            "x": x,
            "y": y,
            "yaw_deg": yaw_deg,
        })

    normalized.sort(key=lambda z: z["index"])
    return ValidationResult(ok=True, normalized={"placements": normalized})


def normalize_and_repair_layout(
    room: Dict[str, Any],
    objects_data: Dict[str, Any],
    llm_layout: Dict[str, Any],
    llm_attempts_used: int,
) -> Dict[str, Any]:
    poly = extract_room_polygon_xy(room)
    ceiling_height = extract_ceiling_height(room)
    src_items = extract_objects(objects_data)
    placements = llm_layout["placements"]

    occupied_floor: List[Dict[str, float]] = []
    tmp_result: Dict[int, Dict[str, Any]] = {}

    floor_indices: List[int] = []
    ceiling_indices: List[int] = []

    for idx, obj in enumerate(src_items):
        if extract_mount_type(obj) == "ceiling":
            ceiling_indices.append(idx)
        else:
            floor_indices.append(idx)

    floor_indices.sort(
        key=lambda i: -(extract_size_m(src_items[i])[0] * extract_size_m(src_items[i])[1])
    )

    by_index = {int(p["index"]): p for p in placements}

    for idx in floor_indices:
        src_obj = src_items[idx]
        pred = by_index[idx]

        sx, sy, sz = extract_size_m(src_obj)
        yaw_deg = quantize_rot_0_90_180_270(float(pred["yaw_deg"]))
        target_x = float(pred["x"])
        target_y = float(pred["y"])

        fixed_x, fixed_y = find_nearest_valid_position(
            target_x=target_x,
            target_y=target_y,
            sx=sx,
            sy=sy,
            sz=sz,
            yaw_deg=yaw_deg,
            polygon_xy=poly,
            occupied=occupied_floor,
        )

        aabb = aabb_from_center_size_rotation(
            cx=fixed_x,
            cy=fixed_y,
            sx=sx,
            sy=sy,
            sz=sz,
            yaw_deg=yaw_deg,
            z_floor_m=0.0,
        )

        item = dict(src_obj)
        item["placement_source"] = "ollama_llm"
        item["rotation"] = yaw_deg
        item["yaw_deg"] = yaw_deg
        item["yaw_rad"] = math.radians(yaw_deg)
        item["position_room_xy_m"] = [fixed_x, fixed_y]
        item["z_floor_m"] = 0.0
        item["size_m"] = [sx, sy, sz]
        item["aabb"] = aabb
        item["bbox"] = dict(aabb)
        item["llm_target_position_room_xy_m"] = [target_x, target_y]
        item["llm_target_yaw_deg"] = float(pred["yaw_deg"])
        item["llm_attempts_used"] = llm_attempts_used

        occupied_floor.append(aabb)
        tmp_result[idx] = item

    for idx in ceiling_indices:
        src_obj = src_items[idx]
        pred = by_index[idx]

        sx, sy, sz = extract_size_m(src_obj)
        yaw_deg = quantize_rot_0_90_180_270(float(pred["yaw_deg"]))
        x = float(pred["x"])
        y = float(pred["y"])

        if not point_in_polygon(x, y, poly):
            x_min, x_max, y_min, y_max = polygon_bbox(poly)
            x = min(max(x, x_min), x_max)
            y = min(max(y, y_min), y_max)

        z_floor_m = max(0.0, ceiling_height - sz)

        aabb = aabb_from_center_size_rotation(
            cx=x,
            cy=y,
            sx=sx,
            sy=sy,
            sz=sz,
            yaw_deg=yaw_deg,
            z_floor_m=z_floor_m,
        )

        item = dict(src_obj)
        item["placement_source"] = "ollama_llm"
        item["rotation"] = yaw_deg
        item["yaw_deg"] = yaw_deg
        item["yaw_rad"] = math.radians(yaw_deg)
        item["position_room_xy_m"] = [x, y]
        item["z_floor_m"] = z_floor_m
        item["size_m"] = [sx, sy, sz]
        item["aabb"] = aabb
        item["bbox"] = dict(aabb)
        item["llm_target_position_room_xy_m"] = [float(pred["x"]), float(pred["y"])]
        item["llm_target_yaw_deg"] = float(pred["yaw_deg"])
        item["llm_attempts_used"] = llm_attempts_used

        tmp_result[idx] = item

    out_items = [tmp_result[i] for i in range(len(src_items))]

    return {
        "placer": "ollama_llm",
        "placements": out_items,
        "llm_raw": llm_layout,
        "llm_attempts_used": llm_attempts_used,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM/Ollama placer: room.json + objects.json -> placement.json")
    ap.add_argument("--room", required=True, help="Путь к room.json")
    ap.add_argument("--objects", required=True, help="Путь к objects.json")
    ap.add_argument("--out", required=True, help="Путь к итоговому placement_result.json")
    ap.add_argument("--mode", default="llm", help="Режим расстановки")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="База Ollama API")
    ap.add_argument("--ollama-model", default="gpt-oss:20b", help="Имя модели в Ollama")
    ap.add_argument("--timeout", type=int, default=300, help="Таймаут HTTP в секундах")
    ap.add_argument("--temperature", type=float, default=0.0, help="temperature для LLM")
    ap.add_argument("--max-llm-attempts", type=int, default=8, help="Максимум попыток перепосылки в LLM")
    args = ap.parse_args()

    room_path = Path(args.room).expanduser().resolve()
    objects_path = Path(args.objects).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not room_path.is_file():
        raise FileNotFoundError(room_path)
    if not objects_path.is_file():
        raise FileNotFoundError(objects_path)

    room = load_json(room_path)
    objects_data = load_json(objects_path)

    payload = build_llm_payload(room, objects_data)
    n_objects = len(extract_objects(objects_data))

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(payload, mode=args.mode)
    output_schema = build_output_schema(n_objects)

    def _generate(_: str) -> str:
        resp = chat_json(
            base_url=args.ollama_url,
            model=args.ollama_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=output_schema,
            timeout_sec=int(args.timeout),
            temperature=float(args.temperature),
            think="low",
        )

        message = resp.get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"Некорректный ответ Ollama /api/chat: {resp}")

        return content

    def _validate(raw_text: str) -> ValidationResult[Dict[str, Any]]:
        return validate_structure(raw_text=raw_text, n_objects=n_objects)

    debug_dir = out_path.parent / "ollama_debug"

    retry_result = run_retry_loop(
        generate_fn=_generate,
        validate_fn=_validate,
        initial_prompt=user_prompt,
        max_attempts=int(args.max_llm_attempts),
        debug_dir=str(debug_dir),
    )

    placement_result = normalize_and_repair_layout(
        room=room,
        objects_data=objects_data,
        llm_layout=retry_result.normalized,
        llm_attempts_used=retry_result.attempts_used,
    )

    save_json(out_path, placement_result)
    print(f"OK: saved placement -> {out_path}")
    print(f"OK: llm attempts used -> {retry_result.attempts_used}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise