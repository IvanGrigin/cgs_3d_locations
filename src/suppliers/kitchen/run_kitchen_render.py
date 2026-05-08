from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from .kitchen_catalog_loader import load_kitchen_material_catalog
from .kitchen_pipeline import generate_kitchen_variants


DEFAULT_MATERIAL_CATALOG = "data/floor_materials/basisrf/basisrf_surface_materials.jsonl"
DEFAULT_APPLIANCE_CATALOG = "data/sourse/suppliers/supplier_catalog_canonical.json"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_RENDER_SCRIPT = "src/tools/render_kitchen_assembly_blender.py"
KITCHEN_MIN_WIDTH_M = 1.5
KITCHEN_MAX_AUTO_WIDTH_M = 3.6
KITCHEN_DOOR_SWING_CLEARANCE_M = 1.05
KITCHEN_CABINET_DEPTH_M = 0.60


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_material_by_sku(material_catalog: str | Path, sku: str | None) -> dict[str, Any] | None:
    if not sku:
        return None
    materials = load_kitchen_material_catalog(material_catalog)
    for material in materials:
        if material.get("sku") == sku:
            return material
    raise ValueError(f"facade_sku_not_found:{sku}")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_room(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("room"), dict):
        return data["room"]
    return data if isinstance(data, dict) else {}


def _room_polygon_xy(room: dict[str, Any]) -> list[tuple[float, float]]:
    points = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    out: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        x = _float(point.get("x"), float("nan"))
        y = _float(point.get("y", point.get("z")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    if len(out) >= 3:
        return out
    width = _float(room.get("width_m") or room.get("width"), 3.2)
    depth = _float(room.get("depth_m") or room.get("depth"), 3.0)
    return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]


def _polygon_signed_area(poly: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def _wall_candidates(room: dict[str, Any]) -> list[dict[str, Any]]:
    poly = _room_polygon_xy(room)
    walls = room.get("walls") if isinstance(room.get("walls"), list) else []
    if not walls:
        walls = [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % len(poly)} for i in range(len(poly))]
    ccw = _polygon_signed_area(poly) > 0
    out: list[dict[str, Any]] = []
    for idx, wall in enumerate(walls):
        if not isinstance(wall, dict):
            continue
        a_idx = int(_float(wall.get("from_vertex"), idx))
        b_idx = int(_float(wall.get("to_vertex"), (idx + 1) % len(poly)))
        if a_idx < 0 or b_idx < 0 or a_idx >= len(poly) or b_idx >= len(poly):
            continue
        ax, ay = poly[a_idx]
        bx, by = poly[b_idx]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = (-uy, ux) if ccw else (uy, -ux)
        out.append(
            {
                "id": str(wall.get("id") or f"w{idx}"),
                "a": (ax, ay),
                "b": (bx, by),
                "u": (ux, uy),
                "n": (nx, ny),
                "length": length,
                "yaw_deg": math.degrees(math.atan2(uy, ux)),
            }
        )
    return out


def _opening_interval_on_wall(opening: dict[str, Any], wall: dict[str, Any], margin: float = 0.18) -> tuple[float, float] | None:
    if str(opening.get("wall_id") or "") != wall["id"]:
        return None
    center = opening.get("s")
    if center is None:
        return None
    width = _float(opening.get("width"), 0.8)
    s = _float(center)
    return max(0.0, s - width * 0.5 - margin), min(float(wall["length"]), s + width * 0.5 + margin)


def _wall_point_at_s(wall: dict[str, Any], s: float) -> tuple[float, float]:
    ax, ay = wall["a"]
    ux, uy = wall["u"]
    return ax + ux * s, ay + uy * s


def _project_point_to_wall_s(point: tuple[float, float], wall: dict[str, Any]) -> tuple[float, float]:
    ax, ay = wall["a"]
    ux, uy = wall["u"]
    px, py = point
    s = (px - ax) * ux + (py - ay) * uy
    closest = _wall_point_at_s(wall, max(0.0, min(float(wall["length"]), s)))
    distance = math.hypot(px - closest[0], py - closest[1])
    return s, distance


def _door_clearance_interval_on_wall(
    door: dict[str, Any],
    *,
    door_wall: dict[str, Any],
    candidate_wall: dict[str, Any],
) -> tuple[float, float] | None:
    if str(door.get("wall_id") or "") == candidate_wall["id"]:
        return None

    door_width = max(0.65, _float(door.get("width"), 0.85))
    door_s = _float(door.get("s"), float("nan"))
    if not math.isfinite(door_s):
        return None

    door_points = [
        _wall_point_at_s(door_wall, door_s),
        _wall_point_at_s(door_wall, door_s - door_width * 0.5),
        _wall_point_at_s(door_wall, door_s + door_width * 0.5),
    ]

    projected: list[float] = []
    min_distance = float("inf")
    for point in door_points:
        s, distance = _project_point_to_wall_s(point, candidate_wall)
        min_distance = min(min_distance, distance)
        projected.append(s)

    # A closed door near an adjacent wall can still swing into a base cabinet.
    # If the door leaf can reach the candidate wall plus the 600 mm cabinet
    # depth, reserve the projected sweep corridor on that wall.
    reach = door_width + KITCHEN_CABINET_DEPTH_M + 0.12
    if min_distance > reach:
        return None

    center_s = max(0.0, min(float(candidate_wall["length"]), sum(projected) / len(projected)))
    reserve = max(KITCHEN_DOOR_SWING_CLEARANCE_M, door_width + 0.25)
    return max(0.0, center_s - reserve), min(float(candidate_wall["length"]), center_s + reserve)


def _subtract_intervals(free: list[tuple[float, float]], blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    for start, end in sorted(blocked):
        next_free: list[tuple[float, float]] = []
        for a, b in free:
            if end <= a or start >= b:
                next_free.append((a, b))
            else:
                if start > a:
                    next_free.append((a, start))
                if end < b:
                    next_free.append((end, b))
        free = next_free
    return free


def _free_wall_intervals(room: dict[str, Any], wall: dict[str, Any]) -> list[tuple[float, float]]:
    wall_len = float(wall["length"])
    free = [(0.0, wall_len)]
    blocked: list[tuple[float, float]] = []
    walls_by_id = {candidate["id"]: candidate for candidate in _wall_candidates(room)}
    for key in ("doors", "windows", "openings"):
        for opening in room.get(key) or []:
            if isinstance(opening, dict):
                interval = _opening_interval_on_wall(opening, wall)
                if interval and interval[1] > interval[0]:
                    blocked.append(interval)
    for door in room.get("doors") or []:
        if not isinstance(door, dict):
            continue
        door_wall = walls_by_id.get(str(door.get("wall_id") or ""))
        if door_wall is None:
            continue
        interval = _door_clearance_interval_on_wall(door, door_wall=door_wall, candidate_wall=wall)
        if interval and interval[1] > interval[0]:
            blocked.append(interval)
    free = _subtract_intervals(free, blocked)
    return [(a, b) for a, b in free if b - a >= KITCHEN_MIN_WIDTH_M]


def _select_room_kitchen_placement(room: dict[str, Any], requested_width_m: float | None) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any], tuple[float, float]] | None = None
    for wall in _wall_candidates(room):
        for interval in _free_wall_intervals(room, wall):
            score = interval[1] - interval[0]
            if best is None or score > best[0]:
                best = (score, wall, interval)
    if best is None:
        return None
    _, wall, interval = best
    free_len = interval[1] - interval[0]
    auto_width = min(free_len, KITCHEN_MAX_AUTO_WIDTH_M)
    width = max(KITCHEN_MIN_WIDTH_M, min(requested_width_m or auto_width, free_len, KITCHEN_MAX_AUTO_WIDTH_M))
    start = interval[0] + max(0.0, (free_len - width) * 0.5)
    ax, ay = wall["a"]
    ux, uy = wall["u"]
    origin = (ax + ux * start, ay + uy * start)
    return {
        "wall_id": wall["id"],
        "available_width_mm": int(round(width * 1000.0)),
        "position": [origin[0], origin[1], 0.0],
        "rotation": [0.0, 0.0, float(wall["yaw_deg"])],
        "wall": wall,
        "start_m": start,
        "end_m": start + width,
    }


def _build_dining_items(room: dict[str, Any], placement: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not room or not placement:
        return []
    poly = _room_polygon_xy(room)
    min_x, max_x = min(x for x, _ in poly), max(x for x, _ in poly)
    min_y, max_y = min(y for _, y in poly), max(y for _, y in poly)
    wall = placement["wall"]
    ux, uy = wall["u"]
    nx, ny = wall["n"]
    width_m = placement["available_width_mm"] / 1000.0
    wall_mid = float(placement["start_m"]) + width_m * 0.5
    ax, ay = wall["a"]
    base_x = ax + ux * wall_mid
    base_y = ay + uy * wall_mid
    room_w, room_d = max_x - min_x, max_y - min_y
    compact = min(room_w, room_d) < 2.7
    table_w, table_d = ((0.72, 0.58) if compact else (1.15, 0.78))
    distance = 0.6 + (0.72 if compact else 1.05)
    cx = min(max(base_x + nx * distance, min_x + table_w * 0.5 + 0.12), max_x - table_w * 0.5 - 0.12)
    cy = min(max(base_y + ny * distance, min_y + table_d * 0.5 + 0.12), max_y - table_d * 0.5 - 0.12)
    yaw = float(wall["yaw_deg"])
    items = [{"id": "kitchen_dining_table_001", "type": "dining_table", "x_m": cx, "y_m": cy, "width_m": table_w, "depth_m": table_d, "yaw_deg": yaw}]
    chair_gap = 0.48 if compact else 0.58
    for idx, side in enumerate((-1.0, 1.0), start=1):
        items.append(
            {
                "id": f"kitchen_dining_chair_{idx:03d}",
                "type": "dining_chair",
                "x_m": min(max(cx + nx * side * chair_gap, min_x + 0.25), max_x - 0.25),
                "y_m": min(max(cy + ny * side * chair_gap, min_y + 0.25), max_y - 0.25),
                "yaw_deg": yaw + (0.0 if side < 0 else 180.0),
            }
        )
    return items


def _apply_facade_override(assembly: dict[str, Any], material: dict[str, Any]) -> None:
    binding = (assembly.get("material_bindings") or {}).get("facade")
    if binding:
        binding["chosen_material"] = material
        binding["final_score"] = 1.0
        binding["score_breakdown"] = {"manual_color_override": 1.0}

    for item in assembly.get("bill_of_materials", {}).get("items", []):
        if item.get("role") != "facade_sheet":
            continue
        item["sku"] = material.get("sku")
        item["name"] = material.get("name")
        item["kitchen_role"] = material.get("kitchen_role")
        item["unit_price"] = round(float(material.get("price") or item.get("unit_price") or 0), 2)
        item["total_price"] = round(float(item["unit_price"]) * float(item.get("quantity") or 1), 2)
        item["note"] = (item.get("note") or "") + "; manual_color_override"

    bom = assembly.get("bill_of_materials") or {}
    items = bom.get("items") or []
    total_material = round(sum(float(item.get("total_price") or 0) for item in items), 2)
    old_total = float(bom.get("total_material_price") or 0)
    delta = total_material - old_total
    bom["total_material_price"] = total_material
    bom["total_estimated_price"] = round(float(bom.get("total_estimated_price") or 0) + delta, 2)

    estimate = assembly.setdefault("price_estimate", {})
    estimate["currency"] = bom.get("currency", "RUB")
    estimate["total_material_price"] = bom["total_material_price"]
    estimate["total_estimated_price"] = bom["total_estimated_price"]
    assembly.setdefault("warnings", []).append(f"manual_facade_color_override:{material.get('sku')}")


def _build_required(args: argparse.Namespace, width_m: float) -> dict[str, Any]:
    width_mm = int(round(width_m * 1000.0))
    allow_cooktop = not args.no_cooktop and width_mm >= args.min_cooktop_width_mm
    return {
        "sink": True,
        "faucet": True,
        "cooktop": allow_cooktop,
        "oven": allow_cooktop and not args.no_oven,
        "hood": allow_cooktop and not args.no_hood,
        "fridge": bool(args.fridge),
        "dishwasher": bool(args.dishwasher),
        "washing_machine": bool(args.washing_machine),
        "microwave": True,
        "decor_accessories": bool(args.decor),
    }


def _render_with_blender(args: argparse.Namespace, json_path: Path, blend_path: Path, png_path: Path) -> None:
    cmd = [
        args.blender,
        "-b",
        "--python",
        args.render_script,
        "--",
        "--input",
        str(json_path),
        "--out-blend",
        str(blend_path),
        "--render-png",
        str(png_path),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and render a straight procedural kitchen.")
    parser.add_argument("--width-m", type=float, default=None, help="Kitchen width in meters, for example 1.5 or 4.5. Optional when --room-json is used.")
    parser.add_argument("--room-json", default=None, help="Optional room or scene JSON with floor_polygon, walls, doors and windows.")
    parser.add_argument("--prompt", required=True, help="Kitchen description.")
    parser.add_argument("--slug", default=None, help="Output filename stem. Default is derived from width.")
    parser.add_argument("--out-dir", default="out/kitchen_demo", help="Directory for JSON, Blend and PNG.")
    parser.add_argument("--mode", default="optimal", choices=("cheapest", "optimal", "best_match"))
    parser.add_argument("--facade-colors", default="", help="Comma-separated desired facade colors.")
    parser.add_argument("--countertop-colors", default="white marble,light stone,gray")
    parser.add_argument("--backsplash-colors", default="light gray,stone,white")
    parser.add_argument("--accent-colors", default="black metal")
    parser.add_argument("--facade-sku", default=None, help="Optional exact BasisRF facade material SKU override.")
    parser.add_argument("--budget", type=float, default=120000.0)
    parser.add_argument("--material-catalog", default=DEFAULT_MATERIAL_CATALOG)
    parser.add_argument("--appliance-catalog", default=DEFAULT_APPLIANCE_CATALOG)
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--render-script", default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--fridge", action="store_true")
    parser.add_argument("--dishwasher", action="store_true")
    parser.add_argument("--washing-machine", action="store_true")
    parser.add_argument("--decor", action="store_true")
    parser.add_argument("--no-cooktop", action="store_true")
    parser.add_argument("--no-oven", action="store_true")
    parser.add_argument("--no-hood", action="store_true")
    parser.add_argument("--min-cooktop-width-mm", type=int, default=1800)
    parser.add_argument("--no-render", action="store_true", help="Only write JSON, do not launch Blender.")
    parser.add_argument("--kitchen-llm-provider", choices=("none", "ollama"), default="none")
    parser.add_argument("--kitchen-ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--kitchen-ollama-model", default="gpt-oss:20b")
    parser.add_argument("--kitchen-ollama-timeout", type=int, default=180)
    parser.add_argument("--kitchen-ollama-temperature", type=float, default=0.1)
    parser.add_argument("--kitchen-ollama-num-ctx", type=int, default=8192)
    parser.add_argument("--kitchen-ollama-think", default="low")
    args = parser.parse_args(argv)

    room = _load_room(args.room_json)
    placement = _select_room_kitchen_placement(room, args.width_m) if room else None
    if args.width_m is None and placement is None:
        parser.error("--width-m is required unless --room-json contains a usable free wall")
    width_m = (placement["available_width_mm"] / 1000.0) if placement else float(args.width_m)
    width_mm = int(round(width_m * 1000.0))
    slug = args.slug or f"kitchen_{width_mm}mm"
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = _build_required(args, width_m)
    variants = generate_kitchen_variants(
        material_catalog=args.material_catalog,
        appliance_catalog=args.appliance_catalog,
        user_prompt=args.prompt,
        room=room or {"width_m": width_m, "depth_m": 2.6, "height_m": 2.7},
        kitchen_zone={
            "layout_type": "straight",
            "wall_id": placement.get("wall_id") if placement else None,
            "available_width_mm": width_mm,
        },
        required_appliances=required,
        recommended_colors={
            "facades": _split_csv(args.facade_colors),
            "countertop": _split_csv(args.countertop_colors),
            "backsplash": _split_csv(args.backsplash_colors),
            "accent": _split_csv(args.accent_colors),
        },
        budget={"total": args.budget, "currency": "RUB"},
        modes=(args.mode,),
        target_id=slug,
        llm_settings={
            "provider": args.kitchen_llm_provider,
            "ollama_url": args.kitchen_ollama_url,
            "ollama_model": args.kitchen_ollama_model,
            "ollama_timeout": args.kitchen_ollama_timeout,
            "ollama_temperature": args.kitchen_ollama_temperature,
            "ollama_num_ctx": args.kitchen_ollama_num_ctx,
            "ollama_think": args.kitchen_ollama_think,
        },
        position=placement.get("position") if placement else None,
        rotation=placement.get("rotation") if placement else None,
    )
    assembly = variants[args.mode]

    facade_override = _load_material_by_sku(args.material_catalog, args.facade_sku)
    if facade_override:
        _apply_facade_override(assembly, facade_override)

    if room:
        assembly["room_context"] = {
            "room": room,
            "kitchen_wall_id": placement.get("wall_id") if placement else None,
            "kitchen_free_interval_m": [round(float(placement["start_m"]), 4), round(float(placement["end_m"]), 4)] if placement else None,
            "dining_items": _build_dining_items(room, placement),
        }
        assembly.setdefault("warnings", []).append("room_context:full_shell_with_openings")

    json_path = out_dir / f"{slug}.json"
    blend_path = out_dir / f"{slug}.blend"
    png_path = out_dir / f"{slug}_preview.png"
    json_path.write_text(json.dumps(assembly, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_render:
        _render_with_blender(args, json_path, blend_path, png_path)

    print(f"json={json_path.resolve()}")
    if not args.no_render:
        print(f"blend={blend_path.resolve()}")
        print(f"png={png_path.resolve()}")
    print(f"price={assembly.get('price_estimate')}")
    print(f"warnings={assembly.get('warnings') or []}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
