#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


SANITARY_REQUIRED = ("toilet", "sink", "bath_or_shower")
SCENE_CANDIDATES = (
    "pipeline/optimal/scene_supplier.optimal.v1.flooring.v1.wall_material.v1.json",
    "pipeline/optimal/scene_supplier.optimal.v1.flooring.v1.json",
    "pipeline/optimal/scene_supplier.optimal.v1.json",
    "pipeline/optimal/scene.v1.flooring.v1.wall_material.v1.json",
    "pipeline/optimal/scene.v1.json",
)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def norm(value: Any) -> str:
    return str(value or "").replace("ё", "е").lower()


def item_text(item: dict[str, Any]) -> str:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    supplier = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    return norm(
        " ".join(
            str(x or "")
            for x in (
                item.get("id"),
                item.get("name"),
                item.get("category"),
                item.get("semantic_group"),
                source.get("supplier_unique_key"),
                supplier.get("title"),
                supplier.get("category_norm"),
                supplier.get("category_raw"),
            )
        )
    )


def classify_item(item: dict[str, Any]) -> set[str]:
    text = item_text(item)
    out: set[str] = set()
    if any(x in text for x in ("toilet", "унитаз", "wc", "watercloset")):
        out.add("toilet")
    if any(x in text for x in ("standing sink", "bathroom_sink", "washbasin", "basin", "sink", "раковин", "умывальник")):
        out.add("sink")
    if any(x in text for x in ("bathtub", "bath tub", "bathfactory", "ванн")):
        out.add("bath")
        out.add("bath_or_shower")
    if any(x in text for x in ("shower", "душ", "душев")):
        out.add("shower")
        out.add("bath_or_shower")
    if any(x in text for x in ("bedfactory", " bed", "кровать")):
        out.add("bed")
    is_lamp = any(x in text for x in ("lamp", "light", "люстр", "светиль", "ламп"))
    if (not is_lamp) and any(
        x in text
        for x in (
            "tablefactory",
            "simpledeskfactory",
            "deskfactory",
            "dining_table",
            "coffee_table",
            "side_table",
            "стол",
            "desk",
            "table",
        )
    ):
        out.add("table")
    return out


def room_items(scene: dict[str, Any]) -> list[dict[str, Any]]:
    items = scene.get("placements") or scene.get("items") or []
    return [x for x in items if isinstance(x, dict)]


def room_bounds(room: dict[str, Any]) -> tuple[float, float]:
    width = float(room.get("width_m") or 0.0)
    depth = float(room.get("depth_m") or 0.0)
    if width > 0 and depth > 0:
        return width, depth
    poly = room.get("floor_polygon") or []
    xs = [float(p.get("x", 0.0)) for p in poly if isinstance(p, dict)]
    ys = [float(p.get("y", p.get("z", 0.0))) for p in poly if isinstance(p, dict)]
    return (max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else (3.0, 3.0)


def make_box_item(
    *,
    room_id: str,
    role: str,
    index: int,
    center_xy: tuple[float, float],
    size: tuple[float, float, float],
    yaw_deg: float,
) -> dict[str, Any]:
    sx, sy, sz = size
    cx, cy = center_xy
    item_id = f"proc_{role}_{index:02d}"
    category_by_role = {
        "toilet": "ToiletFactory",
        "sink": "StandingSinkFactory",
        "shower": "ShowerFactory",
        "bath": "BathtubFactory",
        "bed": "BedFactory",
        "table": "SimpleDeskFactory",
    }
    name_by_role = {
        "toilet": "procedural toilet",
        "sink": "procedural bathroom sink",
        "shower": "procedural shower",
        "bath": "procedural bathtub",
        "bed": "procedural bed",
        "table": "procedural table",
    }
    return {
        "id": item_id,
        "name": name_by_role.get(role, f"procedural {role}"),
        "category": category_by_role.get(role, "ProceduralObject"),
        "position_m": [round(cx, 4), round(cy, 4), round(sz / 2.0, 4)],
        "size_m": [round(sx, 4), round(sy, 4), round(sz, 4)],
        "yaw_deg": round(yaw_deg, 4),
        "rotation_deg": round(yaw_deg, 4),
        "aabb": {
            "x_min": round(cx - sx / 2.0, 4),
            "x_max": round(cx + sx / 2.0, 4),
            "y_min": round(cy - sy / 2.0, 4),
            "y_max": round(cy + sy / 2.0, 4),
            "z_min": 0.0,
            "z_max": round(sz, 4),
        },
        "constraints": {"mount_type": "floor", "touch_floor": {"side": "bottom"}},
        "asset": {"kind": "procedural_requirement_proxy", "role": role},
        "source": {"asset_source": "procedural_requirement", "room_id": room_id},
        "meta": {
            "placeholder_bbox": True,
            "procedural_requirement": True,
            "required_role": role,
            "room_id": room_id,
        },
    }


def add_missing_sanitary(scene: dict[str, Any], prompt_room_type: str | None = None) -> list[dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_id = str(room.get("id") or "room")
    room_type_text = norm(" ".join([room.get("room_type") or "", room.get("source_room_type") or "", prompt_room_type or ""]))
    if not any(x in room_type_text for x in ("bathroom", "toilet", "сануз", "ванн")):
        return []

    present: set[str] = set()
    for item in room_items(scene):
        present |= classify_item(item)

    missing = [role for role in SANITARY_REQUIRED if role not in present]
    if not missing:
        return []

    width, depth = room_bounds(room)
    margin = 0.12
    specs = {
        "toilet": ((0.42, 0.68, 0.78), (min(width - 0.33, max(0.33, width * 0.28)), margin + 0.34), 0.0),
        "sink": ((0.55, 0.42, 0.85), (max(0.34, width * 0.50), min(depth - 0.21, max(0.35, depth * 0.50))), 180.0),
        "shower": ((0.82, 0.82, 2.05), (max(0.53, width - 0.53), max(0.53, depth - 0.53)), 0.0),
    }
    added: list[dict[str, Any]] = []
    for role in missing:
        actual_role = "shower" if role == "bath_or_shower" else role
        size, center, yaw = specs[actual_role]
        sx = min(size[0], max(0.25, width - 2 * margin))
        sy = min(size[1], max(0.25, depth - 2 * margin))
        cx = min(max(center[0], margin + sx / 2.0), max(margin + sx / 2.0, width - margin - sx / 2.0))
        cy = min(max(center[1], margin + sy / 2.0), max(margin + sy / 2.0, depth - margin - sy / 2.0))
        added.append(
            make_box_item(
                room_id=room_id,
                role=actual_role,
                index=len(room_items(scene)) + len(added) + 1,
                center_xy=(cx, cy),
                size=(sx, sy, size[2]),
                yaw_deg=yaw,
            )
        )
    scene.setdefault("placements", []).extend(added)
    meta = scene.setdefault("meta", {})
    meta.setdefault("requirement_postprocess", {})["added_sanitary"] = [x["id"] for x in added]
    return added


def add_apartment_required_objects(apartment_scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present: set[str] = set()
    for scene in apartment_scenes:
        for item in room_items(scene):
            present |= classify_item(item)
    added: list[dict[str, Any]] = []
    if "bed" not in present:
        target = max(apartment_scenes, key=lambda s: float(((s.get("room") or {}).get("area_m2") or 0.0)))
        room = target.get("room") or {}
        width, depth = room_bounds(room)
        item = make_box_item(
            room_id=str(room.get("id") or "room"),
            role="bed",
            index=len(room_items(target)) + 1,
            center_xy=(min(width - 1.0, max(1.0, width * 0.5)), min(depth - 0.75, max(0.75, depth * 0.5))),
            size=(min(2.0, max(1.1, width - 0.4)), 1.6, 0.65),
            yaw_deg=0.0,
        )
        target.setdefault("placements", []).append(item)
        added.append(item)
    if "table" not in present:
        candidates = sorted(
            apartment_scenes,
            key=lambda s: 0
            if norm((s.get("room") or {}).get("room_type")) in {"kitchen", "living_room", "bedroom"}
            else 1,
        )
        target = candidates[0]
        room = target.get("room") or {}
        width, depth = room_bounds(room)
        item = make_box_item(
            room_id=str(room.get("id") or "room"),
            role="table",
            index=len(room_items(target)) + 1,
            center_xy=(max(0.6, width * 0.5), max(0.45, depth * 0.5)),
            size=(1.2, 0.7, 0.75),
            yaw_deg=0.0,
        )
        target.setdefault("placements", []).append(item)
        added.append(item)
    return added


def inverse_room_frame(point: tuple[float, float], frame: dict[str, Any]) -> tuple[float, float]:
    off = frame.get("offset_xy") or [0.0, 0.0]
    origin = frame.get("origin_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or 0.0)
    x = point[0] - float(off[0])
    y = point[1] - float(off[1])
    return (
        x * math.cos(angle) - y * math.sin(angle) + float(origin[0]),
        x * math.sin(angle) + y * math.cos(angle) + float(origin[1]),
    )


def estimate_apartment_min(apartment: dict[str, Any], room_jsons: dict[str, Path]) -> tuple[float, float]:
    door_graph = (((apartment.get("room") or {}).get("meta") or {}).get("door_graph") or {})
    graph_doors = door_graph.get("doors") or []
    estimates: list[tuple[float, float]] = []
    for door in graph_doors:
        room_id = str(door.get("to") or "")
        center = door.get("center_xy")
        if room_id not in room_jsons or not isinstance(center, list) or len(center) < 2:
            continue
        room = read_json(room_jsons[room_id]).get("room") or {}
        frame = ((room.get("meta") or {}).get("coordinate_frame") or {})
        doors = room.get("doors") or []
        if not frame or not doors:
            continue
        seg = (doors[0] or {}).get("segment") or {}
        if not {"x1", "x2", "y1", "y2"} <= set(seg):
            continue
        local_center = ((float(seg["x1"]) + float(seg["x2"])) / 2.0, (float(seg["y1"]) + float(seg["y2"])) / 2.0)
        gx, gy = inverse_room_frame(local_center, frame)
        estimates.append((gx - float(center[0]), gy - float(center[1])))
    if estimates:
        return (
            sorted(x for x, _ in estimates)[len(estimates) // 2],
            sorted(y for _, y in estimates)[len(estimates) // 2],
        )
    poly = (apartment.get("room") or {}).get("floor_polygon") or []
    return (min(float(p.get("x", 0.0)) for p in poly), min(float(p.get("y", 0.0)) for p in poly)) if poly else (0.0, 0.0)


def transform_item_to_apartment(item: dict[str, Any], frame: dict[str, Any], apt_min: tuple[float, float], room_id: str) -> dict[str, Any]:
    out = deepcopy(item)
    prefix = f"{room_id}__"
    out["id"] = prefix + str(out.get("id") or "item")
    source = out.setdefault("source", {})
    source["source_room_id"] = room_id
    meta = out.setdefault("meta", {})
    meta["source_room_id"] = room_id
    angle_deg = float(frame.get("rotation_deg") or math.degrees(float(frame.get("rotation_rad") or 0.0)))

    aabb = item.get("aabb") or {}
    corners = [
        (float(aabb.get("x_min", 0.0)), float(aabb.get("y_min", 0.0))),
        (float(aabb.get("x_min", 0.0)), float(aabb.get("y_max", 0.0))),
        (float(aabb.get("x_max", 0.0)), float(aabb.get("y_min", 0.0))),
        (float(aabb.get("x_max", 0.0)), float(aabb.get("y_max", 0.0))),
    ]
    apt_pts = []
    for pt in corners:
        gx, gy = inverse_room_frame(pt, frame)
        apt_pts.append((gx - apt_min[0], gy - apt_min[1]))
    xs = [p[0] for p in apt_pts]
    ys = [p[1] for p in apt_pts]
    out["aabb"] = {
        "x_min": round(min(xs), 4),
        "x_max": round(max(xs), 4),
        "y_min": round(min(ys), 4),
        "y_max": round(max(ys), 4),
        "z_min": float(aabb.get("z_min", 0.0)),
        "z_max": float(aabb.get("z_max", 0.0)),
    }
    pos = item.get("position_m") or [
        (float(aabb.get("x_min", 0.0)) + float(aabb.get("x_max", 0.0))) / 2.0,
        (float(aabb.get("y_min", 0.0)) + float(aabb.get("y_max", 0.0))) / 2.0,
        (float(aabb.get("z_min", 0.0)) + float(aabb.get("z_max", 0.0))) / 2.0,
    ]
    gx, gy = inverse_room_frame((float(pos[0]), float(pos[1])), frame)
    out["position_m"] = [round(gx - apt_min[0], 4), round(gy - apt_min[1], 4), float(pos[2])]
    out["size_m"] = [
        round(out["aabb"]["x_max"] - out["aabb"]["x_min"], 4),
        round(out["aabb"]["y_max"] - out["aabb"]["y_min"], 4),
        round(out["aabb"]["z_max"] - out["aabb"]["z_min"], 4),
    ]
    yaw = float(item.get("yaw_deg", item.get("rotation_deg", 0.0)) or 0.0) + angle_deg
    out["yaw_deg"] = round(yaw, 4)
    out["rotation_deg"] = round(yaw, 4)
    return out


def should_skip_apartment_item(item: dict[str, Any]) -> bool:
    name = norm(item.get("name"))
    return name.startswith("room_floor_supplieroverlay") or name.startswith("room_wallpaper_supplieroverlay")


def find_room_scene(room_dir: Path) -> Path | None:
    for rel in SCENE_CANDIDATES:
        path = room_dir / rel
        if path.is_file():
            return path
    return None


def kitchen_scene_from_assembly(room_dir: Path) -> dict[str, Any] | None:
    kitchen_dir = room_dir / "kitchen"
    jsons = sorted(kitchen_dir.glob("*.json"))
    if not jsons:
        return None
    assembly = read_json(jsons[0])
    room = read_json(room_dir / "room.json").get("room") or {}
    dims = assembly.get("dimensions") or {}
    width = float(dims.get("width_m") or room.get("width_m") or 2.4)
    depth = float(dims.get("depth_m") or 0.65)
    height = float(dims.get("height_m") or 2.2)
    item = {
        "id": str(assembly.get("id") or f"{room.get('id')}_kitchen"),
        "name": "procedural kitchen set",
        "category": "kitchen_set",
        "type": "procedural_assembly",
        "assembly_type": "procedural_kitchen",
        "position_m": [width / 2.0, depth / 2.0, height / 2.0],
        "size_m": [width, depth, height],
        "yaw_deg": 0.0,
        "rotation_deg": 0.0,
        "aabb": {"x_min": 0.0, "x_max": width, "y_min": 0.0, "y_max": depth, "z_min": 0.0, "z_max": height},
        "asset": {"kind": "procedural_kitchen", "assembly_type": "procedural_kitchen"},
        "meta": {**assembly, "procedural_assembly": "kitchen"},
        "source": {"asset_source": "procedural_kitchen", "source_room_id": room.get("id")},
    }
    return {"schema": "scene.v1", "room": room, "placements": [item], "meta": {"source": str(jsons[0])}}


def process_apartment(apt_dir: Path, mode: str) -> dict[str, Any]:
    manifest_path = apt_dir / "manifest.json"
    apartment_path = apt_dir / "apartment.json"
    if not manifest_path.is_file() or not apartment_path.is_file():
        raise FileNotFoundError(f"Missing manifest/apartment json in {apt_dir}")
    manifest = read_json(manifest_path)
    apartment = read_json(apartment_path)
    rooms_meta = manifest.get("rooms") or []
    room_jsons: dict[str, Path] = {}
    loaded_scenes: list[dict[str, Any]] = []
    room_reports: list[dict[str, Any]] = []

    for room_meta in rooms_meta:
        room_id = str(room_meta.get("room_id") or "")
        room_json = Path(str(room_meta.get("room_json") or ""))
        if room_id and room_json.is_file():
            room_jsons[room_id] = room_json

    for room_meta in rooms_meta:
        room_id = str(room_meta.get("room_id") or "")
        room_dir = apt_dir / "rooms" / room_id
        scene_path = find_room_scene(room_dir)
        scene = read_json(scene_path) if scene_path else kitchen_scene_from_assembly(room_dir)
        if not isinstance(scene, dict):
            room_reports.append({"room_id": room_id, "status": "missing_scene"})
            continue
        added = add_missing_sanitary(scene, prompt_room_type=room_meta.get("prompt_room_type"))
        loaded_scenes.append(scene)
        patched_path = room_dir / "pipeline" / mode / "scene_requirements.v1.json"
        write_json(patched_path, scene)
        room_reports.append(
            {
                "room_id": room_id,
                "room_type": room_meta.get("room_type"),
                "prompt_room_type": room_meta.get("prompt_room_type"),
                "source_scene": str(scene_path) if scene_path else str(room_dir / "kitchen"),
                "requirements_scene": str(patched_path.resolve()),
                "added": [{"id": x["id"], "role": x["meta"]["required_role"]} for x in added],
            }
        )

    apartment_added = add_apartment_required_objects(loaded_scenes)
    apt_min = estimate_apartment_min(apartment, room_jsons)

    placements: list[dict[str, Any]] = []
    for scene in loaded_scenes:
        room = scene.get("room") or {}
        room_id = str(room.get("id") or "")
        frame = ((room.get("meta") or {}).get("coordinate_frame") or {})
        if not room_id or not frame:
            continue
        for item in room_items(scene):
            if should_skip_apartment_item(item):
                continue
            placements.append(transform_item_to_apartment(item, frame, apt_min, room_id))

    out_scene = {
        "schema": "scene.v1",
        "room": apartment.get("room") or {},
        "placements": placements,
        "meta": {
            "source": "ensure_apartment_requirements",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "apartment_dir": str(apt_dir.resolve()),
            "mode": mode,
            "apartment_global_min_xy": [round(apt_min[0], 6), round(apt_min[1], 6)],
            "room_reports": room_reports,
            "apartment_added": [
                {"id": x["id"], "role": x.get("meta", {}).get("required_role"), "room_id": x.get("meta", {}).get("room_id")}
                for x in apartment_added
            ],
            "requirements": {
                "sanitary_rooms": list(SANITARY_REQUIRED),
                "apartment": ["bed", "table"],
            },
        },
    }
    out_dir = apt_dir / "apartment_pipeline" / mode
    out_path = write_json(out_dir / "scene_apartment.requirements.v1.json", out_scene)
    report_path = write_json(
        out_dir / "requirements_report.json",
        {
            "apartment_dir": str(apt_dir.resolve()),
            "scene_json": str(out_path.resolve()),
            "room_reports": room_reports,
            "apartment_added": out_scene["meta"]["apartment_added"],
            "placement_count": len(placements),
        },
    )
    return {"apartment_dir": str(apt_dir), "scene_json": str(out_path), "report_json": str(report_path), "placement_count": len(placements)}


def iter_apartments(root: Path) -> list[Path]:
    if (root / "manifest.json").is_file() and (root / "apartment.json").is_file():
        return [root]
    return sorted(p for p in root.glob("*/*") if (p / "manifest.json").is_file() and (p / "apartment.json").is_file())


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ensure apartment-level required objects and assemble room scenes into one apartment scene.")
    ap.add_argument("root", help="Apartment dir or root containing project/apartment dirs.")
    ap.add_argument("--mode", default="optimal")
    ap.add_argument("--out-summary", default=None)
    return ap


def main() -> None:
    args = build_cli().parse_args()
    root = Path(args.root).expanduser().resolve()
    results = [process_apartment(apt_dir, args.mode) for apt_dir in iter_apartments(root)]
    summary_path = Path(args.out_summary).expanduser().resolve() if args.out_summary else root / "apartment_requirements_summary.json"
    write_json(summary_path, {"root": str(root), "count": len(results), "results": results})
    print(f"processed_apartments = {len(results)}")
    print(f"summary = {summary_path}")
    for result in results:
        print(f"{result['apartment_dir']} -> {result['scene_json']} ({result['placement_count']} placements)")


if __name__ == "__main__":
    main()
