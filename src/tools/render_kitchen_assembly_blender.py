from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_imports() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_mat(bpy, name: str, rgba: tuple[float, float, float, float]):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    return mat


def _cube(bpy, name: str, location: tuple[float, float, float], scale: tuple[float, float, float], mat=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


def _cube_rotated(
    bpy,
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    yaw: float,
    mat=None,
):
    obj = _cube(bpy, name, location, scale, mat)
    obj.rotation_euler[2] = yaw
    return obj


def _point_xy(point: Any) -> tuple[float, float] | None:
    if not isinstance(point, dict):
        return None
    try:
        return float(point.get("x")), float(point.get("y", point.get("z")))
    except Exception:
        return None


def _room_polygon(room: dict[str, Any]) -> list[tuple[float, float]]:
    points = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    poly = [xy for point in points if (xy := _point_xy(point)) is not None]
    if len(poly) >= 3:
        return poly
    width = float(room.get("width_m") or room.get("width") or 3.4)
    depth = float(room.get("depth_m") or room.get("depth") or 3.0)
    return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]


def _walls_for_room(room: dict[str, Any], poly: list[tuple[float, float]]) -> list[dict[str, Any]]:
    walls = room.get("walls") if isinstance(room.get("walls"), list) else []
    if not walls:
        walls = [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % len(poly)} for i in range(len(poly))]
    out = []
    for idx, wall in enumerate(walls):
        if not isinstance(wall, dict):
            continue
        a_idx = int(wall.get("from_vertex", idx))
        b_idx = int(wall.get("to_vertex", (idx + 1) % len(poly)))
        if a_idx < 0 or b_idx < 0 or a_idx >= len(poly) or b_idx >= len(poly):
            continue
        ax, ay = poly[a_idx]
        bx, by = poly[b_idx]
        length = math.hypot(bx - ax, by - ay)
        if length <= 1e-6:
            continue
        out.append({"id": str(wall.get("id") or f"w{idx}"), "a": (ax, ay), "b": (bx, by), "length": length, "yaw": math.atan2(by - ay, bx - ax)})
    return out


def _opening_interval(opening: dict[str, Any], wall: dict[str, Any]) -> tuple[float, float, float, float] | None:
    if str(opening.get("wall_id") or "") != wall["id"]:
        return None
    width = float(opening.get("width") or 0.8)
    s = opening.get("s")
    if s is None:
        return None
    center = float(s)
    z0 = float(opening.get("z0") or 0.0)
    height = float(opening.get("height") or (2.05 if opening.get("type") == "door" else 1.05))
    return max(0.0, center - width * 0.5), min(float(wall["length"]), center + width * 0.5), z0, z0 + height


def _add_floor_mesh(bpy, poly: list[tuple[float, float]], mat) -> None:
    mesh = bpy.data.meshes.new("preview_floor_mesh")
    verts = [(x, y, -0.015) for x, y in poly]
    mesh.from_pydata(verts, [], [tuple(range(len(verts)))])
    mesh.update()
    obj = bpy.data.objects.new("preview_floor", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)


def _add_wall_segment(bpy, name: str, wall: dict[str, Any], s0: float, s1: float, z0: float, z1: float, thickness: float, mat) -> None:
    if s1 - s0 <= 0.02 or z1 - z0 <= 0.02:
        return
    ax, ay = wall["a"]
    yaw = float(wall["yaw"])
    cx = ax + math.cos(yaw) * (s0 + s1) * 0.5
    cy = ay + math.sin(yaw) * (s0 + s1) * 0.5
    _cube_rotated(bpy, name, (cx, cy, (z0 + z1) * 0.5), (s1 - s0, thickness, z1 - z0), yaw, mat)


def _add_room_context(bpy, assembly: dict[str, Any]) -> None:
    context = assembly.get("room_context") if isinstance(assembly.get("room_context"), dict) else {}
    room = context.get("room") if isinstance(context.get("room"), dict) else {}
    dims = assembly.get("dimensions") or {}
    poly = _room_polygon(room)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    width = max(xs) - min(xs) if xs else float(dims.get("width_m") or 3.4)
    depth = max(ys) - min(ys) if ys else max(1.4, float(dims.get("depth_m") or 0.65) + 0.9)
    height = max(2.4, float(room.get("ceiling_height") or room.get("height_m") or room.get("height") or 2.7))

    floor_mat = _make_mat(bpy, "preview_floor_warm_gray", (0.55, 0.54, 0.50, 1.0))
    wall_mat = _make_mat(bpy, "preview_wall_off_white", (0.86, 0.84, 0.79, 1.0))
    glass_mat = _make_mat(bpy, "preview_window_glass", (0.42, 0.68, 0.92, 0.45))
    door_mat = _make_mat(bpy, "preview_door_warm_wood", (0.45, 0.28, 0.15, 1.0))
    table_mat = _make_mat(bpy, "preview_dining_wood", (0.42, 0.30, 0.18, 1.0))
    chair_mat = _make_mat(bpy, "preview_dining_chair", (0.18, 0.18, 0.17, 1.0))

    _add_floor_mesh(bpy, poly, floor_mat)
    walls = _walls_for_room(room, poly)
    wall_by_id = {wall["id"]: wall for wall in walls}
    openings_by_wall: dict[str, list[tuple[float, float, float, float, str]]] = {}
    for key, kind in (("doors", "door"), ("windows", "window"), ("openings", "opening")):
        for opening in room.get(key) or []:
            if not isinstance(opening, dict):
                continue
            wall = wall_by_id.get(str(opening.get("wall_id") or ""))
            if wall is None:
                continue
            interval = _opening_interval(opening, wall)
            if interval:
                openings_by_wall.setdefault(wall["id"], []).append((*interval, kind))

    thickness = 0.06
    for wall in walls:
        openings = sorted(openings_by_wall.get(wall["id"], []))
        cursor = 0.0
        for idx, (s0, s1, z0, z1, kind) in enumerate(openings):
            _add_wall_segment(bpy, f"preview_wall_{wall['id']}_{idx}_left", wall, cursor, s0, 0.0, height, thickness, wall_mat)
            _add_wall_segment(bpy, f"preview_wall_{wall['id']}_{idx}_below", wall, s0, s1, 0.0, z0, thickness, wall_mat)
            _add_wall_segment(bpy, f"preview_wall_{wall['id']}_{idx}_above", wall, s0, s1, z1, height, thickness, wall_mat)
            mat = glass_mat if kind == "window" else door_mat
            _add_wall_segment(bpy, f"preview_{kind}_{wall['id']}_{idx}", wall, s0 + 0.03, s1 - 0.03, z0 + 0.03, z1 - 0.03, thickness * 0.45, mat)
            cursor = max(cursor, s1)
        _add_wall_segment(bpy, f"preview_wall_{wall['id']}_tail", wall, cursor, float(wall["length"]), 0.0, height, thickness, wall_mat)

    for item in context.get("dining_items") or []:
        if not isinstance(item, dict):
            continue
        x, y = float(item.get("x_m") or 0.0), float(item.get("y_m") or 0.0)
        yaw = math.radians(float(item.get("yaw_deg") or 0.0))
        if item.get("type") == "dining_table":
            _cube_rotated(bpy, item.get("id", "dining_table_top"), (x, y, 0.75), (float(item.get("width_m") or 1.0), float(item.get("depth_m") or 0.7), 0.06), yaw, table_mat)
            _cube_rotated(bpy, f"{item.get('id', 'dining_table')}_base", (x, y, 0.36), (0.16, 0.16, 0.66), yaw, table_mat)
        elif item.get("type") == "dining_chair":
            _cube_rotated(bpy, item.get("id", "dining_chair"), (x, y, 0.24), (0.42, 0.42, 0.48), yaw, chair_mat)
            _cube_rotated(bpy, f"{item.get('id', 'dining_chair')}_back", (x, y + 0.17, 0.67), (0.42, 0.05, 0.70), yaw, chair_mat)


def _add_camera_and_lights(bpy, assembly: dict[str, Any]) -> None:
    context = assembly.get("room_context") if isinstance(assembly.get("room_context"), dict) else {}
    room = context.get("room") if isinstance(context.get("room"), dict) else {}
    dims = assembly.get("dimensions") or {}
    poly = _room_polygon(room)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x, max_x = (min(xs), max(xs)) if xs else (0.0, float(dims.get("width_m") or 3.4))
    min_y, max_y = (min(ys), max(ys)) if ys else (0.0, 3.0)
    width = max_x - min_x
    depth = max_y - min_y
    height = float(room.get("ceiling_height") or room.get("height_m") or dims.get("height_m") or 2.7)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5

    bpy.ops.object.light_add(type="AREA", location=(cx, cy, height + 0.3))
    area = bpy.context.object
    area.name = "KitchenPreview_AreaLight"
    area.data.energy = 850
    area.data.size = max(width, depth, 3.0)

    bpy.ops.object.camera_add(
        location=(cx, max_y + max(depth, width) * 0.95, height + max(depth, width) * 0.75),
        rotation=(math.radians(72), 0.0, math.radians(180)),
    )
    cam = bpy.context.object
    cam.name = "KitchenPreview_Camera"
    cam.data.lens = 20
    bpy.context.scene.camera = cam

    empty = bpy.data.objects.new("KitchenPreview_Target", None)
    empty.location = (cx, cy, min(0.95, height * 0.40))
    bpy.context.scene.collection.objects.link(empty)
    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    constraint.target = empty


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Blender preview for a procedural kitchen assembly JSON.")
    parser.add_argument("--input", default="out/kitchen_demo/kitchen_optimal.json", help="Path to kitchen assembly JSON.")
    parser.add_argument("--out-blend", default="out/kitchen_demo/kitchen_optimal.blend", help="Output .blend path.")
    parser.add_argument("--render-png", default=None, help="Optional output preview PNG.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    _ensure_repo_imports()

    import bpy  # type: ignore

    from src.suppliers.kitchen.kitchen_blender_builder import build_kitchen_assembly_in_blender

    input_path = Path(args.input).expanduser().resolve()
    out_blend = Path(args.out_blend).expanduser().resolve()
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    assembly = _load_json(input_path)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    _add_room_context(bpy, assembly)
    created = build_kitchen_assembly_in_blender(assembly)
    _add_camera_and_lights(bpy, assembly)

    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(bpy.context.scene, "eevee") else "BLENDER_EEVEE"
    bpy.context.scene.render.resolution_x = 1600
    bpy.context.scene.render.resolution_y = 1000

    if args.render_png:
        render_path = Path(args.render_png).expanduser().resolve()
        render_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"created_objects={len(created)}")
    print(f"saved_blend={out_blend}")
    if args.render_png:
        print(f"saved_render={Path(args.render_png).expanduser().resolve()}")


if __name__ == "__main__":
    main()
