from __future__ import annotations

"""Inspect CGS scene.v1 placements.

Usage examples:

python3 src/tools/inspect_scene_items.py \
  --scene out/test_procedural_bedroom/scene_procedural_room.standalone.v1.json

python3 src/tools/inspect_scene_items.py \
  --scene out/test_procedural_bedroom/scene_procedural_room.standalone.v1.json \
  --markdown out/test_procedural_bedroom/items_report.md

python3 src/tools/inspect_scene_items.py \
  --scene out/integration_procedural_bedroom/scene_procedural_room.base.v1.json \
  --show-aabb \
  --markdown out/integration_procedural_bedroom/items_report.md

The script is dependency-free and can be placed at:

src/tools/inspect_scene_items.py
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOFT_OR_NON_FLOOR_MOUNTS = {
    "wall",
    "ceiling",
    "on_top",
}

SOFT_LAYERS = {
    "decor",
    "soft_decor",
    "textile",
    "wall_decor",
    "lighting",
    "electronics",
}

LIKELY_SOLID_CATEGORIES = {
    "bed",
    "bench",
    "bookshelf",
    "cabinet",
    "chair",
    "coffee_table",
    "console_table",
    "desk",
    "dining_chair",
    "dining_table",
    "dresser",
    "entry_bench",
    "floor_lamp",
    "nightstand",
    "ottoman",
    "plant",
    "shoe_cabinet",
    "side_table",
    "sofa",
    "tv_stand",
    "wardrobe",
    "wardrobe_module",
}

CATEGORY_RU = {
    "armchair": "кресло",
    "bed": "кровать",
    "bench": "банкетка",
    "blanket": "плед/одеяло",
    "bookshelf": "книжный шкаф",
    "ceiling_light": "потолочный светильник",
    "coffee_table": "журнальный столик",
    "console_table": "консольный стол",
    "decor_books": "книги/стопка книг",
    "decor_box": "декоративная коробка",
    "decor_tray": "поднос/трей",
    "decor_vase": "ваза/декор",
    "dresser": "комод",
    "entry_bench": "банкетка в прихожей",
    "floor_lamp": "торшер",
    "mirror": "зеркало",
    "nightstand": "прикроватная тумба",
    "pillow": "подушка",
    "plant": "растение",
    "rug": "ковёр",
    "runner_rug": "дорожка/раннер",
    "shoe_cabinet": "обувница",
    "side_table": "приставной столик",
    "sofa": "диван",
    "storage_basket": "корзина хранения",
    "table_lamp": "настольная лампа",
    "tv": "телевизор",
    "tv_accessory": "TV-аксессуар",
    "tv_stand": "тумба под TV",
    "wall_art": "настенный декор/картина",
    "wall_hooks": "настенные крючки",
    "wall_light": "бра/настенный светильник",
    "wardrobe": "шкаф",
}


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fnum(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "?"


def vec3(value: Any, digits: int = 2) -> str:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return "?"
    return f"({fnum(value[0], digits)}, {fnum(value[1], digits)}, {fnum(value[2], digits)})"


def vec2(value: Any, digits: int = 2) -> str:
    if not isinstance(value, list | tuple) or len(value) < 2:
        return "?"
    return f"({fnum(value[0], digits)}, {fnum(value[1], digits)})"


def area_polygon(points: list[dict[str, Any]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        total += float(p.get("x", 0.0)) * float(q.get("y", 0.0))
        total -= float(q.get("x", 0.0)) * float(p.get("y", 0.0))
    return abs(total) / 2.0


def bounds_from_polygon(points: list[dict[str, Any]]) -> dict[str, float]:
    xs = [float(p.get("x", 0.0)) for p in points]
    ys = [float(p.get("y", 0.0)) for p in points]
    if not xs or not ys:
        return {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0}
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
    }


def room_dimensions(room: dict[str, Any]) -> tuple[float, float, float]:
    poly = room.get("floor_polygon") or []
    b = bounds_from_polygon(poly)
    width = float(room.get("width_m") or (b["x_max"] - b["x_min"]) or 0.0)
    depth = float(room.get("depth_m") or (b["y_max"] - b["y_min"]) or 0.0)
    height = float(room.get("height_m") or room.get("ceiling_height") or room.get("ceiling_height_m") or 0.0)
    return width, depth, height


def normalize_yaw(item: dict[str, Any]) -> float:
    raw = item.get("yaw_deg", item.get("rotation_deg", 0.0))
    try:
        return float(raw) % 360.0
    except Exception:
        return 0.0


def item_layer(item: dict[str, Any]) -> str:
    meta = item.get("meta") or {}
    constraints = item.get("constraints") or {}
    source = item.get("source") or {}
    return str(
        meta.get("density_layer")
        or meta.get("layer")
        or constraints.get("density_layer")
        or source.get("layer")
        or "unknown"
    )


def item_mount(item: dict[str, Any]) -> str:
    return str(item.get("mount_type") or (item.get("constraints") or {}).get("mount_type") or "floor")


def item_category(item: dict[str, Any]) -> str:
    return str(item.get("category") or item.get("semantic_group") or "unknown")


def item_source(item: dict[str, Any]) -> str:
    source = item.get("source") or {}
    return str(source.get("placement_source") or source.get("generator") or item.get("source") or "unknown")


def item_parent(item: dict[str, Any]) -> str:
    meta = item.get("meta") or {}
    constraints = item.get("constraints") or {}
    return str(
        meta.get("parent_id")
        or meta.get("support_id")
        or constraints.get("parent_id")
        or constraints.get("support_id")
        or ""
    )


def item_wall(item: dict[str, Any]) -> str:
    meta = item.get("meta") or {}
    constraints = item.get("constraints") or {}
    source = item.get("source") or {}
    return str(
        item.get("wall_id")
        or meta.get("wall_id")
        or constraints.get("wall_id")
        or source.get("wall_id")
        or ""
    )


def is_soft_or_non_solid(item: dict[str, Any]) -> bool:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    role = str(meta.get("physical_role") or "").strip().lower()
    if role:
        return role != "solid_floor"

    category = item_category(item)
    layer = item_layer(item)
    mount = item_mount(item)
    if mount in SOFT_OR_NON_FLOOR_MOUNTS:
        return True
    if layer in SOFT_LAYERS:
        return True
    if category not in LIKELY_SOLID_CATEGORIES:
        return True
    return False


def aabb_text(item: dict[str, Any]) -> str:
    aabb = item.get("aabb") or {}
    if not isinstance(aabb, dict) or not aabb:
        return ""
    keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    return ", ".join(f"{k}={fnum(aabb.get(k))}" for k in keys if k in aabb)


def sort_key(item: dict[str, Any]) -> tuple[str, str, float, float, str]:
    p = item.get("position_m") or [0.0, 0.0, 0.0]
    try:
        x = float(p[0])
        y = float(p[1])
    except Exception:
        x = 0.0
        y = 0.0
    return (item_layer(item), item_category(item), y, x, str(item.get("id", "")))


def table_rows(items: Iterable[dict[str, Any]], show_aabb: bool = False) -> list[list[str]]:
    rows: list[list[str]] = []
    for idx, item in enumerate(sorted(items, key=sort_key), 1):
        category = item_category(item)
        category_ru = CATEGORY_RU.get(category, "")
        position = item.get("position_m") or item.get("center") or []
        size = item.get("size_m") or item.get("size") or []
        row = [
            str(idx),
            str(item.get("id", "")),
            category,
            category_ru,
            str(item.get("name", "")),
            item_layer(item),
            item_mount(item),
            vec3(position),
            vec3(size),
            fnum(normalize_yaw(item), 1),
            item_wall(item),
            item_parent(item),
            "soft" if is_soft_or_non_solid(item) else "solid",
        ]
        if show_aabb:
            row.append(aabb_text(item))
        rows.append(row)
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        safe = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        out.append("| " + " | ".join(safe) + " |")
    return "\n".join(out)


def compact_console_table(headers: list[str], rows: list[list[str]], max_width: int = 32) -> str:
    def cut(s: str) -> str:
        s = str(s)
        if len(s) <= max_width:
            return s
        return s[: max_width - 1] + "…"

    rows2 = [[cut(c) for c in row] for row in rows]
    headers2 = [cut(h) for h in headers]
    widths = [len(h) for h in headers2]
    for row in rows2:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers2))
    sep = "  ".join("-" * widths[i] for i in range(len(widths)))
    body = ["  ".join(row[i].ljust(widths[i]) for i in range(len(widths))) for row in rows2]
    return "\n".join([line, sep, *body])


def build_report(scene_path: Path, scene: dict[str, Any], show_aabb: bool = False) -> str:
    room = scene.get("room") or {}
    items = list(scene.get("placements") or scene.get("items") or [])
    width, depth, height = room_dimensions(room)
    area = float(room.get("area_m2") or area_polygon(room.get("floor_polygon") or []) or 0.0)

    by_layer = Counter(item_layer(item) for item in items)
    by_category = Counter(item_category(item) for item in items)
    by_mount = Counter(item_mount(item) for item in items)
    solid_count = sum(1 for item in items if not is_soft_or_non_solid(item))
    soft_count = len(items) - solid_count

    source_counter = Counter(item_source(item) for item in items)
    wall_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        wall = item_wall(item)
        if wall:
            wall_groups[wall].append(item)

    out: list[str] = []
    out.append(f"# Scene items inspection")
    out.append("")
    out.append(f"- Scene: `{scene_path}`")
    out.append(f"- Room ID: `{room.get('id', '')}`")
    out.append(f"- Room type: `{room.get('room_type') or room.get('type') or ''}`")
    out.append(f"- Size: `{width:.2f} x {depth:.2f} x {height:.2f} m`")
    out.append(f"- Area: `{area:.2f} m²`")
    out.append(f"- Placement count: `{len(items)}`")
    out.append(f"- Solid floor-like items: `{solid_count}`")
    out.append(f"- Soft / wall / ceiling / on-top items: `{soft_count}`")
    out.append("")

    out.append("## Counts by layer")
    out.append("")
    for key, value in sorted(by_layer.items()):
        out.append(f"- `{key}`: {value}")
    out.append("")

    out.append("## Counts by category")
    out.append("")
    for key, value in sorted(by_category.items()):
        ru = CATEGORY_RU.get(key, "")
        suffix = f" — {ru}" if ru else ""
        out.append(f"- `{key}`: {value}{suffix}")
    out.append("")

    out.append("## Counts by mount type")
    out.append("")
    for key, value in sorted(by_mount.items()):
        out.append(f"- `{key}`: {value}")
    out.append("")

    out.append("## Counts by source")
    out.append("")
    for key, value in sorted(source_counter.items()):
        out.append(f"- `{key}`: {value}")
    out.append("")

    if wall_groups:
        out.append("## Items attached to walls")
        out.append("")
        for wall, wall_items in sorted(wall_groups.items()):
            cats = Counter(item_category(item) for item in wall_items)
            cats_text = ", ".join(f"{cat}: {count}" for cat, count in sorted(cats.items()))
            out.append(f"- `{wall}`: {len(wall_items)} items — {cats_text}")
        out.append("")

    headers = [
        "#",
        "id",
        "category",
        "ru",
        "name",
        "layer",
        "mount",
        "position_m",
        "size_m",
        "yaw_deg",
        "wall",
        "parent",
        "kind",
    ]
    if show_aabb:
        headers.append("aabb")

    out.append("## Full placement table")
    out.append("")
    out.append(markdown_table(headers, table_rows(items, show_aabb=show_aabb)))
    out.append("")
    return "\n".join(out)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect CGS scene.v1 placements.")
    parser.add_argument("--scene", required=True, help="Path to scene.v1.json / scene_procedural_room*.v1.json")
    parser.add_argument("--markdown", help="Optional path to write Markdown report")
    parser.add_argument("--show-aabb", action="store_true", help="Include AABB columns in Markdown/console output")
    parser.add_argument("--only-solid", action="store_true", help="Print only solid floor-like items in console table")
    parser.add_argument("--only-soft", action="store_true", help="Print only soft/wall/ceiling/on-top items in console table")
    parser.add_argument("--max-console-rows", type=int, default=120, help="Limit console table rows")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    scene_path = Path(args.scene)
    scene = read_json(scene_path)
    items = list(scene.get("placements") or scene.get("items") or [])

    if args.only_solid and args.only_soft:
        raise SystemExit("Use only one of --only-solid / --only-soft")
    if args.only_solid:
        items = [item for item in items if not is_soft_or_non_solid(item)]
    if args.only_soft:
        items = [item for item in items if is_soft_or_non_solid(item)]

    console_headers = [
        "#",
        "id",
        "category",
        "ru",
        "name",
        "layer",
        "mount",
        "position_m",
        "size_m",
        "yaw",
        "wall",
        "parent",
        "kind",
    ]
    if args.show_aabb:
        console_headers.append("aabb")

    rows = table_rows(items, show_aabb=args.show_aabb)
    if args.max_console_rows > 0:
        rows = rows[: args.max_console_rows]

    room = scene.get("room") or {}
    width, depth, height = room_dimensions(room)
    area = float(room.get("area_m2") or area_polygon(room.get("floor_polygon") or []) or 0.0)
    print(f"Scene: {scene_path}")
    print(f"Room: {room.get('id', '')} | type={room.get('room_type') or room.get('type') or ''} | size={width:.2f}x{depth:.2f}x{height:.2f}m | area={area:.2f}m²")
    print(f"Placements shown: {len(rows)} / total={len(scene.get('placements') or [])}")
    print()
    print(compact_console_table(console_headers, rows))

    if args.markdown:
        report = build_report(scene_path, scene, show_aabb=args.show_aabb)
        write_text(args.markdown, report)
        print()
        print(f"Markdown report written: {args.markdown}")


if __name__ == "__main__":
    main()
