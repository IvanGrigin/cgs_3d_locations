#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# src/tools/build_front_scene.py
#
# Визуализация prepared_scene_*.json в Blender с корректным upright и текстурами.
#
# Главное:
# - ref -> jid (из raw 3D-FRONT json: furniture[].uid -> furniture[].jid)
# - импорт OBJ (Blender 3.x/4.x)
# - подтягивание текстур из MTL: map_Kd (и fallback texture.png)
# - upright: 24 ориентации (осевые вращения) в WORLD-координатах комнаты
# - yaw применяется ТОЛЬКО после upright (со знаком минус из-за (x,y,z)->(x,z,y))
# - прижатие к полу (snap-to-floor) для НЕ Lighting
# - Lighting / Lamp:
#   - не участвует в размежевании (может пересекаться с другими объектами)
#   - не прижимается к полу
#   - если это потолочный/подвесной свет — поднимаем к потолку
# - Размежевание мебели:
#   - раздвигаем только НЕ Lighting/Lamp
#   - после каждого шага гарантируем, что объект остаётся внутри полигона комнаты (с зазором)
#
# Запуск:
# /Applications/Blender.app/Contents/MacOS/Blender --python src/tools/build_front_scene.py -- \
#   --scene_json data/output/prepared_scene_*.json \
#   --models_root data/sourse/3D-FRONT/3D-FUTURE-model \
#   --front_raw_json "data/sourse/3D-FRONT/3D-FRONT/....json" \
#   --room_ids LivingDiningRoom-4017,MasterBedroom-2485 \
#   --upright_idx 2 \
#   --prefer_raw 1 \
#   --resolve_collisions 1
#
"""
/Applications/Blender.app/Contents/MacOS/Blender --python src/tools/build_front_scene.py -- \
   --scene_json data/output/prepared_scene_*.json \
   --models_root data/sourse/3D-FRONT/3D-FUTURE-model \
   --front_raw_json "data/sourse/3D-FRONT/3D-FRONT/....json" \
   --room_ids LivingDiningRoom-4017,MasterBedroom-2485 \
   --upright_idx 2 \
   --prefer_raw 1 \
   --resolve_collisions 1
"""

import argparse
import json
import math
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector, Matrix


# ==========================
# Координаты: вход -> Blender
# ==========================

def to_blender_loc(x: float, y: float, z: float):
    """
    Вход: XZ плоскость, Y=высота.
    Blender: Z-up, XY плоскость.
    (x,y,z) -> (x,z,y)
    """
    return (float(x), float(z), float(y))


def yaw_deg_to_rot_z(yaw_deg: float) -> float:
    return math.radians(float(yaw_deg))


def scale_to_blender(scale_xyz):
    # вход: [sx, sy(height), sz] -> Blender: (sx, sz, sy)
    sx, sy, sz = float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])
    return (sx, sz, sy)


# ==========================
# Сцена: коллекции/камера/свет
# ==========================

def ensure_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(coll)
    return coll


def link_object_to_collection(obj: bpy.types.Object, coll: bpy.types.Collection):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def clear_scene_objects():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def compute_rooms_bbox(rooms):
    xs, ys, zs = [], [], []
    for r in rooms:
        for p in r.get("polygon", []):
            xs.append(float(p["x"]))
            ys.append(0.0)
            zs.append(float(p["z"]))
        for o in r.get("objects", []):
            pos = o.get("pos", {})
            xs.append(float(pos.get("x", 0.0)))
            ys.append(float(pos.get("y", 0.0)))
            zs.append(float(pos.get("z", 0.0)))
    if not xs:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    center = ((minx + maxx) * 0.5, (miny + maxy) * 0.5, (minz + maxz) * 0.5)
    size = (maxx - minx, maxy - miny, maxz - minz)
    return center, size


def ensure_camera_and_light(center_xyz, size_xyz):
    scene = bpy.context.scene

    cam = None
    for o in scene.objects:
        if o.type == "CAMERA":
            cam = o
            break
    if cam is None:
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)

    light = None
    for o in scene.objects:
        if o.type == "LIGHT":
            light = o
            break
    if light is None:
        light_data = bpy.data.lights.new("Sun", type="SUN")
        light = bpy.data.objects.new("Sun", light_data)
        scene.collection.objects.link(light)

    cx, cy, cz = center_xyz
    sx, sy, sz = size_xyz
    diag = max(sx, sz, 1.0)

    cam.location = to_blender_loc(cx + 0.8 * diag, 2.5 + 0.8 * diag, cz + 0.8 * diag)
    cam.rotation_euler = (math.radians(65), 0.0, math.radians(225))

    light.location = to_blender_loc(cx, 6.0 + 1.5 * diag, cz)
    light.rotation_euler = (math.radians(60), 0.0, math.radians(45))

    scene.camera = cam


# ==========================
# Комната: пол, стены, маркеры
# ==========================

def create_floor_and_walls(room_id: str, polygon_xz: list, wall_height: float, coll: bpy.types.Collection):
    if not polygon_xz or len(polygon_xz) < 3:
        return None, None

    floor_mesh = bpy.data.meshes.new(f"{room_id}_floor_mesh")
    floor_obj = bpy.data.objects.new(f"{room_id}_floor", floor_mesh)
    link_object_to_collection(floor_obj, coll)

    bm = bmesh.new()
    verts = [bm.verts.new((float(p["x"]), float(p["z"]), 0.0)) for p in polygon_xz]
    try:
        bm.faces.new(verts)
    except ValueError:
        for i in range(1, len(verts) - 1):
            try:
                bm.faces.new((verts[0], verts[i], verts[i + 1]))
            except ValueError:
                pass
    bm.normal_update()
    bm.to_mesh(floor_mesh)
    bm.free()

    walls_mesh = bpy.data.meshes.new(f"{room_id}_walls_mesh")
    walls_obj = bpy.data.objects.new(f"{room_id}_walls", walls_mesh)
    link_object_to_collection(walls_obj, coll)

    bm = bmesh.new()
    bm.from_mesh(floor_mesh)
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        res = bmesh.ops.extrude_edge_only(bm, edges=boundary_edges)
        geom = res["geom"]
        verts_extruded = [g for g in geom if isinstance(g, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, verts=verts_extruded, vec=Vector((0.0, 0.0, float(wall_height))))
    bm.normal_update()
    bm.to_mesh(walls_mesh)
    bm.free()

    return floor_obj, walls_obj


def create_marker_plane(name: str, center_xyz: dict, yaw_deg: float, size: float, coll: bpy.types.Collection):
    cx, cy, cz = float(center_xyz["x"]), float(center_xyz["y"]), float(center_xyz["z"])
    loc = to_blender_loc(cx, cy, cz)

    rot_z = yaw_deg_to_rot_z(-float(yaw_deg))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    link_object_to_collection(obj, coll)

    bm = bmesh.new()
    w = float(size) * 0.5
    h = float(size) * 0.5
    v1 = bm.verts.new((-w, 0.0, 0.0))
    v2 = bm.verts.new((w, 0.0, 0.0))
    v3 = bm.verts.new((w, 0.0, h))
    v4 = bm.verts.new((-w, 0.0, h))
    bm.faces.new((v1, v2, v3, v4))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    obj.location = loc
    obj.rotation_euler = (0.0, 0.0, rot_z)
    return obj


# ==========================
# Импорт OBJ
# ==========================

def import_obj(obj_path: Path):
    if not obj_path.exists():
        return []
    before = set(bpy.data.objects)

    if hasattr(bpy.ops.wm, "obj_import"):
        # Blender 4.x
        try:
            bpy.ops.wm.obj_import(
                filepath=str(obj_path),
                forward_axis="NEGATIVE_Z",
                up_axis="Y",
            )
        except Exception:
            return []
    elif hasattr(bpy.ops.import_scene, "obj"):
        # Blender 3.x
        try:
            bpy.ops.import_scene.obj(
                filepath=str(obj_path),
                axis_forward="-Z",
                axis_up="Y",
                use_image_search=True,
            )
        except Exception:
            return []
    else:
        return []

    after = set(bpy.data.objects)
    return [o for o in (after - before) if o.type in {"MESH", "EMPTY"}]


def join_meshes(objs):
    mesh_objs = [o for o in objs if o.type == "MESH"]
    if not mesh_objs:
        return None
    bpy.context.view_layer.objects.active = mesh_objs[0]
    for o in mesh_objs:
        o.select_set(True)
    if len(mesh_objs) > 1:
        try:
            bpy.ops.object.join()
            merged = bpy.context.view_layer.objects.active
        except Exception:
            merged = mesh_objs[0]
    else:
        merged = mesh_objs[0]
    for o in bpy.context.selected_objects:
        o.select_set(False)
    return merged


def apply_object_scale(obj: bpy.types.Object):
    if obj is None or obj.type != "MESH":
        return
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass
    finally:
        bpy.ops.object.select_all(action="DESELECT")


def import_model_from_dir(model_dir: Path, prefer_raw: bool):
    if prefer_raw:
        candidates = ["raw_model.obj", "normalized_model.obj"]
    else:
        candidates = ["normalized_model.obj", "raw_model.obj"]

    obj_path = None
    for name in candidates:
        p = model_dir / name
        if p.exists():
            obj_path = p
            break
    if obj_path is None:
        return None, None

    new_objs = import_obj(obj_path)
    merged = join_meshes(new_objs)
    return merged, obj_path


# ==========================
# Текстуры из MTL (map_Kd) + fallback texture.png
# ==========================

def find_mtllib_in_obj(obj_path: Path) -> Path | None:
    try:
        lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    for line in lines[:500]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("mtllib "):
            mtl_name = s.split(None, 1)[1].strip().strip('"')
            mtl_path = (obj_path.parent / mtl_name).resolve()
            return mtl_path if mtl_path.exists() else None
    return None


def parse_mtl_map_kd(mtl_path: Path) -> dict:
    mapping = {}
    cur = None
    try:
        lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return mapping

    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if not parts:
            continue
        key = parts[0].lower()
        if key == "newmtl" and len(parts) >= 2:
            cur = " ".join(parts[1:])
        elif key == "map_kd" and cur is not None and len(parts) >= 2:
            tex = parts[-1].strip().strip('"')
            tex_path = (mtl_path.parent / tex).resolve()
            if tex_path.exists():
                mapping[cur] = tex_path
    return mapping


def make_principled_with_image(mat: bpy.types.Material, image_path: Path):
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new(type="ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.location = (-400, 0)

    img = bpy.data.images.get(str(image_path))
    if img is None:
        img = bpy.data.images.load(str(image_path))
    tex.image = img

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])


def apply_textures_from_mtl(obj: bpy.types.Object, obj_path: Path):
    if obj is None or obj.type != "MESH":
        return

    mtl_path = find_mtllib_in_obj(obj_path)
    if mtl_path is None:
        return

    mtl_map = parse_mtl_map_kd(mtl_path)
    if not mtl_map:
        return

    mats = list(obj.data.materials)

    if not mats:
        if len(mtl_map) == 1:
            tex = next(iter(mtl_map.values()))
            mat = bpy.data.materials.new(name=f"MAT_{obj.name}")
            make_principled_with_image(mat, tex)
            obj.data.materials.append(mat)
        return

    used = False
    for mat in mats:
        if mat is None:
            continue
        tex = mtl_map.get(mat.name)
        if tex is None:
            base = mat.name.split(".")[0]
            tex = mtl_map.get(base)
        if tex is not None:
            make_principled_with_image(mat, tex)
            used = True

    if not used and len(mtl_map) == 1:
        tex = next(iter(mtl_map.values()))
        for mat in mats:
            if mat is None:
                continue
            make_principled_with_image(mat, tex)


def apply_fallback_texture_png(obj: bpy.types.Object, model_dir: Path):
    if obj is None or obj.type != "MESH":
        return
    tex = (model_dir / "texture.png").resolve()
    if not tex.exists():
        return

    if not obj.data.materials:
        mat = bpy.data.materials.new(name=f"MAT_{model_dir.name}")
        obj.data.materials.append(mat)

    mat0 = obj.data.materials[0]
    if mat0 is None:
        mat0 = bpy.data.materials.new(name=f"MAT_{model_dir.name}")
        obj.data.materials[0] = mat0

    make_principled_with_image(mat0, tex)


# ==========================
# ref -> jid из raw 3D-FRONT
# ==========================

def build_ref_to_jid(front_raw_json: Path | None):
    if front_raw_json is None or not front_raw_json.exists():
        return {}
    try:
        d = json.loads(front_raw_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    mp = {}
    for f in d.get("furniture", []):
        uid = f.get("uid")
        jid = f.get("jid")
        if uid and jid:
            mp[str(uid)] = str(jid)
    return mp


# ==========================
# Upright: 24 ориентации куба (WORLD)
# ==========================

def _rot_x90(k: int) -> Matrix:
    return Matrix.Rotation((math.pi / 2) * k, 4, "X")


def _rot_y90(k: int) -> Matrix:
    return Matrix.Rotation((math.pi / 2) * k, 4, "Y")


def _rot_z90(k: int) -> Matrix:
    return Matrix.Rotation((math.pi / 2) * k, 4, "Z")


def all_24_axis_rotations() -> list[Matrix]:
    rots = []
    seen = set()

    to_up = [
        Matrix.Identity(4),   # +Z
        _rot_x90(2),          # -Z
        _rot_x90(1),          # +Y
        _rot_x90(3),          # -Y
        _rot_y90(3),          # +X
        _rot_y90(1),          # -X
    ]

    for base in to_up:
        for kz in range(4):
            m = base @ _rot_z90(kz)
            key = tuple(round(v, 6) for row in m for v in row)
            if key not in seen:
                seen.add(key)
                rots.append(m)

    return rots


def bbox_points_local(obj: bpy.types.Object) -> list[Vector]:
    return [Vector(c) for c in obj.bound_box]


def bbox_points_world_with_world_rot(obj: bpy.types.Object, world_rot: Matrix) -> list[Vector]:
    mw = obj.matrix_world.copy()
    pivot = obj.location.copy()
    pts = []
    for lp in bbox_points_local(obj):
        p = mw @ lp
        p = pivot + (world_rot @ (p - pivot))
        pts.append(p)
    return pts


def convex_hull_area_2d(points_xy):
    pts = sorted(set((float(x), float(y)) for x, y in points_xy))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0

    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def support_metrics_for_world_rotation(obj: bpy.types.Object, world_rot: Matrix, floor_z: float = 0.0):
    pts = bbox_points_world_with_world_rot(obj, world_rot)
    zs = [p.z for p in pts]
    zmin = min(zs)

    pts_n = [Vector((p.x, p.y, p.z - zmin + floor_z)) for p in pts]

    xs = [p.x for p in pts_n]
    ys = [p.y for p in pts_n]
    zs2 = [p.z for p in pts_n]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    dz = max(zs2) - min(zs2)

    eps = max(1e-4, dz * 0.01)
    support = [(p.x, p.y) for p in pts_n if p.z <= floor_z + eps]
    support_frac = len(support) / max(1, len(pts_n))
    support_area = convex_hull_area_2d(support) if len(support) >= 3 else 0.0

    return (dx, dy, dz, support_frac, support_area)


def choose_best_upright_rotation(obj: bpy.types.Object):
    if obj is None or obj.type != "MESH":
        return Matrix.Identity(4)

    rots = all_24_axis_rotations()

    dx0, dy0, dz0, _, _ = support_metrics_for_world_rotation(obj, Matrix.Identity(4))
    scale = max(dx0, dy0, dz0, 1e-6)

    best_rot = Matrix.Identity(4)
    best_score = -1e100

    for rot in rots:
        dx, dy, dz, sfrac, sarea = support_metrics_for_world_rotation(obj, rot)

        dz_n = dz / scale
        sarea_n = sarea / (scale * scale)

        max_e = max(dx, dy, dz, 1e-9)
        flatness = dz / max_e

        if flatness < 0.25:
            score = (sfrac * 10.0) + (sarea_n * 8.0) - (dz_n * 6.0)
        else:
            score = (sarea_n * 10.0) + (dz_n * 6.0) + (sfrac * 3.0)

        score += sarea_n * 0.5

        if score > best_score:
            best_score = score
            best_rot = rot

    return best_rot


def apply_upright_rotation_world(obj: bpy.types.Object, world_rot: Matrix):
    if obj is None:
        return
    pivot = obj.location.copy()
    T1 = Matrix.Translation(pivot)
    T2 = Matrix.Translation(-pivot)
    obj.matrix_world = (T1 @ world_rot @ T2) @ obj.matrix_world
    bpy.context.view_layer.update()


def apply_yaw_after_upright(obj: bpy.types.Object, yaw_deg: float):
    if obj is None:
        return
    rz = yaw_deg_to_rot_z(-float(yaw_deg))
    Rz = Matrix.Rotation(rz, 4, "Z")
    pivot = obj.location.copy()
    T1 = Matrix.Translation(pivot)
    T2 = Matrix.Translation(-pivot)
    obj.matrix_world = (T1 @ Rz @ T2) @ obj.matrix_world
    bpy.context.view_layer.update()


def snap_object_to_floor(obj: bpy.types.Object, floor_z: float = 0.0):
    if obj is None or obj.type != "MESH":
        return
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    zmin = min(p.z for p in pts)
    obj.location.z += (floor_z - zmin)
    bpy.context.view_layer.update()


def snap_object_to_ceiling(obj: bpy.types.Object, ceiling_z: float):
    if obj is None or obj.type != "MESH":
        return
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    zmax = max(p.z for p in pts)
    obj.location.z += (ceiling_z - zmax)
    bpy.context.view_layer.update()


def is_lighting_category(cat: object) -> bool:
    """
    Классификация "свет" по категории.
    В 3D-FRONT встречаются: "Lighting", "Pendant Lamp", "Ceiling Lamp", иногда просто "... Lamp".
    """
    s = ("" if cat is None else str(cat)).strip().lower()
    if not s:
        return False
    if "lighting" in s:
        return True
    if "lamp" in s:
        return True
    if "light" in s:
        return True
    return False


def lighting_should_attach_to_ceiling(o: dict, wall_height: float) -> bool:
    """
    Решение "к потолку".
    1) Явные флаги/поля, если присутствуют.
    2) По категории: pendant/ceiling/hanging.
    3) По высоте pos.y: если заметно выше пола -> трактуем как потолочный/подвесной свет.
    """
    for k in ("attach_to_ceiling", "ceiling", "is_ceiling", "ceiling_mounted", "hang_from_ceiling"):
        if bool(o.get(k, False)):
            return True

    cat = ("" if o.get("category") is None else str(o.get("category"))).strip().lower()
    if any(t in cat for t in ("pendant", "ceiling", "hanging", "chandelier", "吊灯")):
        return True

    for k in ("anchor", "mount", "placement", "mount_type"):
        v = str(o.get(k, "")).lower().strip()
        if v in {"ceiling", "hanging", "pendant", "top", "hang"}:
            return True

    try:
        py = float(o.get("pos", {}).get("y", 0.0))
        # Практично: если это свет и y уже ~1.2м+ (как в вашем примере 1.6), считаем потолочным/подвесным.
        if py >= 1.2:
            return True
        # Резерв: если очень близко к потолку по данным
        if py >= 0.75 * float(wall_height):
            return True
    except Exception:
        pass

    return False


# ==========================
# Полигон комнаты и удержание объекта внутри
# ==========================

def room_polygon_xy(polygon_xz: list) -> list[tuple[float, float]]:
    return [(float(p["x"]), float(p["z"])) for p in polygon_xz]


def poly_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    if not poly:
        return (0.0, 0.0)
    sx = sum(p[0] for p in poly)
    sy = sum(p[1] for p in poly)
    n = float(len(poly))
    return (sx / n, sy / n)


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return True
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            if xinters > x:
                inside = not inside
    return inside


def closest_point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return (ax, ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    return (ax + t * abx, ay + t * aby)


def closest_point_on_poly(px: float, py: float, poly: list[tuple[float, float]]) -> tuple[float, float]:
    best = (px, py)
    best_d2 = 1e100
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        qx, qy = closest_point_on_segment(px, py, ax, ay, bx, by)
        dx, dy = (qx - px), (qy - py)
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = (qx, qy)
    return best


def object_world_bbox_xy(obj: bpy.types.Object) -> tuple[float, float, float, float]:
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))


def object_world_center_xy(obj: bpy.types.Object) -> tuple[float, float]:
    x0, x1, y0, y1 = object_world_bbox_xy(obj)
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def keep_object_inside_room(obj: bpy.types.Object, poly_xy: list[tuple[float, float]], margin: float):
    if obj is None or obj.type != "MESH":
        return
    if not poly_xy or len(poly_xy) < 3:
        return

    cx, cy = poly_centroid(poly_xy)
    for _ in range(8):
        x0, x1, y0, y1 = object_world_bbox_xy(obj)
        corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
        outside = [c for c in corners if not point_in_poly(c[0], c[1], poly_xy)]
        if not outside:
            return

        bx, by = outside[0]
        qx, qy = closest_point_on_poly(bx, by, poly_xy)

        vx, vy = (cx - qx), (cy - qy)
        vl = math.hypot(vx, vy)
        if vl <= 1e-9:
            vx, vy = 0.0, 0.0
        else:
            vx, vy = vx / vl, vy / vl

        tx, ty = (qx + vx * margin, qy + vy * margin)
        dx, dy = (tx - bx, ty - by)

        obj.location.x += dx
        obj.location.y += dy
        bpy.context.view_layer.update()


# ==========================
# Размежевание (только НЕ Lighting)
# ==========================

def aabb_overlap_2d(a, b, pad: float) -> tuple[bool, float, float]:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    ok = (ox > -pad) and (oy > -pad)
    return ok, ox, oy


def resolve_overlaps_in_room(
    objs: list[bpy.types.Object],
    poly_xy: list[tuple[float, float]],
    margin: float,
    max_iters: int = 40,
):
    movable = [o for o in objs if o and o.type == "MESH" and not bool(o.get("is_lighting", False))]
    if len(movable) < 2:
        return
    if not poly_xy or len(poly_xy) < 3:
        return

    for o in movable:
        keep_object_inside_room(o, poly_xy, margin=margin)

    for _ in range(max_iters):
        moved_any = False

        for i in range(len(movable)):
            A = movable[i]
            a_bb = object_world_bbox_xy(A)
            acx, acy = object_world_center_xy(A)

            for j in range(i + 1, len(movable)):
                B = movable[j]
                b_bb = object_world_bbox_xy(B)
                bcx, bcy = object_world_center_xy(B)

                ok, ox, oy = aabb_overlap_2d(a_bb, b_bb, pad=margin)
                if not ok:
                    continue

                pen_x = (margin - ox) if ox < margin else (ox + margin)
                pen_y = (margin - oy) if oy < margin else (oy + margin)

                if abs(pen_x) < abs(pen_y):
                    sign = 1.0 if (acx >= bcx) else -1.0
                    dx = 0.5 * abs(pen_x) * sign
                    A.location.x += dx
                    B.location.x -= dx
                else:
                    sign = 1.0 if (acy >= bcy) else -1.0
                    dy = 0.5 * abs(pen_y) * sign
                    A.location.y += dy
                    B.location.y -= dy

                bpy.context.view_layer.update()

                keep_object_inside_room(A, poly_xy, margin=margin)
                keep_object_inside_room(B, poly_xy, margin=margin)

                moved_any = True

        if not moved_any:
            break

    for o in movable:
        keep_object_inside_room(o, poly_xy, margin=margin)


# ==========================
# Основная сборка
# ==========================

def build(
    scene_json: Path,
    models_root: Path,
    front_raw_json: Path | None,
    room_ids: list[str] | None,
    wall_height: float,
    marker_size: float,
    upright_idx: int | None,
    prefer_raw: int,
    resolve_collisions: int,
    collision_margin: float,
):
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    rooms = data.get("rooms", [])

    if room_ids:
        want = set(room_ids)
        rooms = [r for r in rooms if r.get("id") in want]

    ref_to_jid = build_ref_to_jid(front_raw_json)
    rots24 = all_24_axis_rotations()

    clear_scene_objects()
    root = ensure_collection("PreparedScene")

    center, size = compute_rooms_bbox(rooms)
    ensure_camera_and_light(center, size)

    for r in rooms:
        rid = r.get("id", "room")
        rcoll = ensure_collection(rid, parent=root)

        poly_xy = room_polygon_xy(r.get("polygon", []))

        create_floor_and_walls(rid, r.get("polygon", []), wall_height, rcoll)

        for i, d in enumerate(r.get("doors", [])):
            create_marker_plane(f"{rid}_door_{i:02d}", d["center"], d.get("yaw_deg", 0.0), marker_size, rcoll)

        for i, w in enumerate(r.get("windows", [])):
            create_marker_plane(f"{rid}_window_{i:02d}", w["center"], w.get("yaw_deg", 0.0), marker_size, rcoll)

        imported_objs: list[bpy.types.Object] = []

        for i, o in enumerate(r.get("objects", [])):
            ref = o.get("ref", "")

            jid = o.get("jid")
            if not jid and ref:
                jid = ref_to_jid.get(ref)
            if not jid and ref:
                jid = ref.split("/")[0]

            obj = None
            obj_path = None
            model_dir = None

            if jid:
                model_dir = (models_root / str(jid)).resolve()
                obj, obj_path = import_model_from_dir(model_dir, prefer_raw=bool(prefer_raw))

            if obj is None:
                bpy.ops.mesh.primitive_cube_add(size=0.3)
                obj = bpy.context.active_object
                obj.name = f"{rid}_missing_{jid or 'nojid'}_{i:04d}"
                link_object_to_collection(obj, rcoll)

                pos = o.get("pos", {"x": 0.0, "y": 0.0, "z": 0.0})
                obj.location = to_blender_loc(pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0))
                imported_objs.append(obj)
                continue

            link_object_to_collection(obj, rcoll)
            imported_objs.append(obj)

            sc = o.get("scale")
            if isinstance(sc, (list, tuple)) and len(sc) == 3:
                obj.scale = scale_to_blender(sc)
                bpy.context.view_layer.update()
                apply_object_scale(obj)

            if obj_path is not None:
                apply_textures_from_mtl(obj, obj_path)
            if model_dir is not None:
                apply_fallback_texture_png(obj, model_dir)

            pos = o.get("pos", {"x": 0.0, "y": 0.0, "z": 0.0})
            obj.location = to_blender_loc(pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0))
            bpy.context.view_layer.update()

            cat = o.get("category")
            if cat is not None:
                obj["category"] = cat

            is_light = is_lighting_category(cat)
            obj["is_lighting"] = bool(is_light)

            if upright_idx is not None:
                k = int(upright_idx) % 24
                best_rot = rots24[k]
            else:
                best_rot = choose_best_upright_rotation(obj)

            apply_upright_rotation_world(obj, best_rot)

            if not bool(obj.get("is_lighting", False)):
                snap_object_to_floor(obj, floor_z=0.0)

            yaw = float(o.get("yaw_deg", 0.0))
            apply_yaw_after_upright(obj, yaw)

            if not bool(obj.get("is_lighting", False)):
                snap_object_to_floor(obj, floor_z=0.0)

            # Lighting: решаем по потолку ПОСЛЕ upright+yaw (чтобы bbox был корректным)
            if bool(obj.get("is_lighting", False)):
                if lighting_should_attach_to_ceiling(o, wall_height=float(wall_height)):
                    snap_object_to_ceiling(obj, ceiling_z=float(wall_height))
                else:
                    # сохраняем исходную высоту (в Blender это z)
                    desired_z = to_blender_loc(
                        pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
                    )[2]
                    obj.location.z = float(desired_z)
                    bpy.context.view_layer.update()

            # Все объекты (включая свет) должны оставаться в пределах комнаты по XY.
            keep_object_inside_room(obj, poly_xy, margin=float(collision_margin))

            if ref:
                obj["ref"] = ref
            if jid:
                obj["jid"] = str(jid)
            inst = o.get("instanceid")
            if inst:
                obj["instanceid"] = inst
            obj["valid"] = bool(o.get("valid", True))

        if int(resolve_collisions) == 1:
            # Раздвигаем только НЕ Lighting/Lamp
            resolve_overlaps_in_room(
                imported_objs,
                poly_xy=poly_xy,
                margin=float(collision_margin),
                max_iters=40,
            )

            # Пост-фиксы: пол/потолок + удержание в комнате
            for idx, obj in enumerate(imported_objs):
                if obj is None or obj.type != "MESH":
                    continue
                if bool(obj.get("is_lighting", False)):
                    # свет не раздвигаем, но потолок может понадобиться после движения других объектов
                    src_o = r.get("objects", [])[idx] if idx < len(r.get("objects", [])) else {}
                    if lighting_should_attach_to_ceiling(src_o, wall_height=float(wall_height)):
                        snap_object_to_ceiling(obj, ceiling_z=float(wall_height))
                    keep_object_inside_room(obj, poly_xy, margin=float(collision_margin))
                else:
                    snap_object_to_floor(obj, floor_z=0.0)
                    keep_object_inside_room(obj, poly_xy, margin=float(collision_margin))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_json", required=True)
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--front_raw_json", default=None, help="Исходный 3D-FRONT json для ref->jid")

    # Один id или список id через запятую
    ap.add_argument("--room_id", default=None, help="Один room id (deprecated, используйте --room_ids)")
    ap.add_argument("--room_ids", default=None, help="Список room id через запятую")

    ap.add_argument("--wall_height", type=float, default=2.7)
    ap.add_argument("--marker_size", type=float, default=1.0)
    ap.add_argument("--upright_idx", type=int, default=None, help="Принудительно выбрать ориентацию 0..23")
    ap.add_argument("--prefer_raw", type=int, default=1,
                    help="1: предпочитать raw_model.obj; 0: предпочитать normalized_model.obj")

    ap.add_argument("--resolve_collisions", type=int, default=1,
                    help="1: раздвигать пересекающиеся объекты (кроме Lighting/Lamp); 0: выключить")
    ap.add_argument("--collision_margin", type=float, default=0.02,
                    help="Зазор (метры) для раздвижения и удержания внутри комнаты")

    import sys
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    args = ap.parse_args(argv)

    room_ids = None
    if args.room_ids:
        room_ids = [s.strip() for s in str(args.room_ids).split(",") if s.strip()]
    elif args.room_id:
        room_ids = [str(args.room_id).strip()] if str(args.room_id).strip() else None

    build(
        scene_json=Path(args.scene_json),
        models_root=Path(args.models_root),
        front_raw_json=Path(args.front_raw_json) if args.front_raw_json else None,
        room_ids=room_ids,
        wall_height=args.wall_height,
        marker_size=args.marker_size,
        upright_idx=args.upright_idx,
        prefer_raw=args.prefer_raw,
        resolve_collisions=args.resolve_collisions,
        collision_margin=args.collision_margin,
    )


if __name__ == "__main__":
    main()
