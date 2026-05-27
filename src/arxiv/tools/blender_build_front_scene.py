# -*- coding: utf-8 -*-
"""
Blender script: build a full 3D-FRONT scene from processed JSON.

Run example:
blender --factory-startup --background \
  --python src/tools/blender_build_front_scene.py -- \
  --in data/sourse/3D-FRONT/3D-FRONT_processed/processed_ba4c65ee.json \
  --future-root data/sourse/3D-FRONT/3D-FUTURE-model \
  --out out/front_scene.blend \
  --build-walls

Notes:
- Input JSON is assumed Y-up (as in your processed meta).
- Blender is Z-up, so we convert coords: (x,y,z)->(x,z,y) and rotation accordingly.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector, Quaternion


# -----------------------------
# Coordinate system conversion
# -----------------------------
# Data: Y-up. Blender: Z-up.
# Map: Xb = Xd, Yb = Zd, Zb = Yd
M_D2B = Matrix(((1.0, 0.0, 0.0),
               (0.0, 0.0, 1.0),
               (0.0, 1.0, 0.0)))
M_B2D = M_D2B.inverted()


def vec_d2b(v: List[float]) -> Vector:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return M_D2B @ Vector((x, y, z))


def scale_d2b(s: List[float]) -> Vector:
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    return Vector((sx, sz, sy))


def quat_xyzw_d2b(q_xyzw: List[float]) -> Quaternion:
    # input: [x,y,z,w] in data coords; mathutils wants (w,x,y,z)
    x, y, z, w = [float(t) for t in q_xyzw]
    qd = Quaternion((w, x, y, z))
    Rd = qd.to_matrix().to_3x3()
    Rb = M_D2B @ Rd @ M_B2D
    return Rb.to_quaternion()


def yaw_d2b(yaw_about_Y: float) -> Quaternion:
    # yaw is about data +Y (up). In Blender up is +Z after mapping.
    # Equivalent rotation about Blender +Z by same angle.
    return Quaternion(Vector((0.0, 0.0, 1.0)), float(yaw_about_Y))


# -----------------------------
# Blender helpers
# -----------------------------

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0


def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if parent is None:
        if col.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(col)
    else:
        if col.name not in parent.children:
            parent.children.link(col)
    return col


def link_object_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    # unlink from all, link to col
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def set_obj_transform(obj: bpy.types.Object,
                      pos_d: List[float],
                      rot_quat_xyzw_d: Optional[List[float]],
                      yaw_d: Optional[float],
                      scale_d: Optional[List[float]]) -> None:
    obj.location = vec_d2b(pos_d)

    if rot_quat_xyzw_d is not None and len(rot_quat_xyzw_d) == 4:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = quat_xyzw_d2b(rot_quat_xyzw_d)
    elif yaw_d is not None:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = yaw_d2b(float(yaw_d))
    else:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))

    if scale_d is not None and len(scale_d) == 3:
        obj.scale = scale_d2b(scale_d)


def add_floor_polygon(room_name: str,
                      poly_xz: List[List[float]],
                      floor_y: float,
                      col: bpy.types.Collection) -> bpy.types.Object:
    """
    Build a floor mesh in Blender coordinates.
    Input poly is in data XZ plane. Convert to Blender XY, with Z=floor_y (data Y -> Blender Z).
    """
    verts = []
    for p in poly_xz:
        x = float(p[0])
        z = float(p[1])
        # data point is (x, floor_y, z)
        vb = vec_d2b([x, float(floor_y), z])
        verts.append((vb.x, vb.y, vb.z))

    # single face (ngon)
    mesh = bpy.data.meshes.new(f"{room_name}_floor_mesh")
    obj = bpy.data.objects.new(f"{room_name}_floor", mesh)
    col.objects.link(obj)

    mesh.from_pydata(verts, [], [list(range(len(verts)))])
    mesh.update()
    return obj


def add_walls_from_floor(room_name: str,
                         floor_obj: bpy.types.Object,
                         wall_height: float,
                         thickness: float,
                         col: bpy.types.Collection) -> bpy.types.Object:
    """
    Extrude floor boundary up by wall_height (in meters). Optional thickness via solidify.
    """
    bpy.context.view_layer.objects.active = floor_obj
    floor_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # extrude region along +Z (Blender up)
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, float(wall_height))})
    bpy.ops.object.mode_set(mode='OBJECT')
    floor_obj.select_set(False)

    wall_obj = floor_obj
    wall_obj.name = f"{room_name}_walls"

    # solidify for thickness
    if thickness and thickness > 0.0:
        mod = wall_obj.modifiers.new(name="Solidify", type='SOLIDIFY')
        mod.thickness = float(thickness)
        bpy.context.view_layer.objects.active = wall_obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    link_object_to_collection(wall_obj, col)
    return wall_obj


def find_model_file(future_root: Path, jid: str) -> Optional[Path]:
    """
    Best-effort lookup for 3D-FUTURE models.
    Looks for common filenames inside <future_root>/<jid>/...
    """
    if not jid:
        return None

    d = future_root / jid
    if not d.exists():
        # fallback: recursive search directory named jid
        candidates = list(future_root.rglob(jid))
        candidates = [p for p in candidates if p.is_dir()]
        if not candidates:
            return None
        d = candidates[0]

    # priority list
    pats = [
        "model.glb", "model.gltf",
        "scene.glb", "scene.gltf",
        "raw_model.obj", "model.obj", "normalized_model.obj",
        "*.glb", "*.gltf",
        "*.obj",
    ]
    for pat in pats:
        hits = list(d.glob(pat))
        if hits:
            return hits[0]
    return None


def import_model(filepath: Path) -> List[bpy.types.Object]:
    """
    Import GLB/GLTF/OBJ. Returns newly created objects.
    """
    before = set(bpy.data.objects)

    ext = filepath.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(filepath))
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=str(filepath), axis_forward='-Z', axis_up='Y')
    else:
        return []

    after = set(bpy.data.objects)
    new_objs = [o for o in after - before]
    return new_objs


def make_placeholder(name: str,
                     dx_dy_dz: Optional[List[float]],
                     col: bpy.types.Collection) -> bpy.types.Object:
    """
    Create a cube placeholder (Blender Z-up). Size is interpreted as [dx,dy,dz] in data coords.
    """
    bpy.ops.mesh.primitive_cube_add(size=1.0, enter_editmode=False)
    obj = bpy.context.active_object
    obj.name = name
    link_object_to_collection(obj, col)

    if dx_dy_dz and len(dx_dy_dz) == 3:
        dx, dy, dz = [float(x) for x in dx_dy_dz]
        # data sizes are along X,Y,Z; convert scale axes the same way:
        obj.scale = Vector((dx / 2.0, dz / 2.0, dy / 2.0))
    return obj


def ensure_basic_light_and_camera() -> None:
    # light
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = 2.0

    # camera
    bpy.ops.object.camera_add(location=(0, -15, 10), rotation=(math.radians(60), 0, 0))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam


# -----------------------------
# Build scene
# -----------------------------

def build_from_processed(data: Dict[str, Any],
                         future_root: Path,
                         build_walls: bool,
                         wall_thickness: float,
                         default_wall_height: float,
                         room_filter: Optional[str]) -> None:
    root_col = ensure_collection("3D_FRONT")

    rooms = data.get("rooms") or []
    for r in rooms:
        if not isinstance(r, dict):
            continue

        room_id = str(r.get("room_instanceid") or "unknown")
        room_type = str(r.get("room_type") or "Room")
        if room_filter and (room_filter not in room_id) and (room_filter not in room_type):
            continue

        room_col = ensure_collection(room_id, parent=root_col)

        floor = r.get("floor") or {}
        poly = floor.get("outline_xz") or []
        floor_y = float(floor.get("y", 0.0))
        ceiling_y = floor.get("ceiling_y")
        wall_h = default_wall_height
        if ceiling_y is not None:
            wall_h = max(0.1, float(ceiling_y) - floor_y)

        # floor
        if poly and len(poly) >= 3:
            floor_obj = add_floor_polygon(room_id, poly, floor_y, room_col)
            if build_walls:
                add_walls_from_floor(room_id, floor_obj, wall_h, wall_thickness, room_col)

        # objects
        for obj in (r.get("objects") or []):
            if not isinstance(obj, dict):
                continue

            inst = str(obj.get("instanceid") or "")
            title = str(obj.get("title") or "")
            ref = obj.get("ref")
            jid = None
            if isinstance(ref, str) and ref:
                jid = ref.split("/", 1)[0]

            pos = obj.get("pos") or [0.0, 0.0, 0.0]
            rotq = obj.get("rot_quat")
            yaw = obj.get("yaw")
            scl = obj.get("scale")

            size = obj.get("size") or obj.get("bbox_local")  # may be None
            # normalize nested [[...]]
            if isinstance(size, list) and len(size) == 1 and isinstance(size[0], list):
                size = size[0]

            model_file = find_model_file(future_root, jid) if jid else None

            created: List[bpy.types.Object] = []
            if model_file is not None:
                created = import_model(model_file)

            if not created:
                # fallback placeholder cube
                ph = make_placeholder(f"{inst}_placeholder", size if isinstance(size, list) else None, room_col)
                set_obj_transform(ph, pos, rotq if isinstance(rotq, list) else None, yaw, scl if isinstance(scl, list) else None)
                ph["instanceid"] = inst
                ph["title"] = title
                ph["jid"] = jid or ""
                continue

            # group imported objects under empty, then transform the empty
            bpy.ops.object.empty_add(type='PLAIN_AXES')
            empty = bpy.context.active_object
            empty.name = inst or f"{jid}_inst"
            link_object_to_collection(empty, room_col)

            for o in created:
                # parent and move to room collection
                o.parent = empty
                link_object_to_collection(o, room_col)

            set_obj_transform(empty, pos, rotq if isinstance(rotq, list) else None, yaw, scl if isinstance(scl, list) else None)

            empty["instanceid"] = inst
            empty["title"] = title
            empty["jid"] = jid or ""


def main() -> None:
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="processed JSON file (with key 'rooms')")
    ap.add_argument("--future-root", dest="future_root", required=True, help="root dir of extracted 3D-FUTURE models")
    ap.add_argument("--out", dest="out_blend", required=True, help="output .blend path")
    ap.add_argument("--build-walls", action="store_true", help="extrude walls from floor polygon")
    ap.add_argument("--wall-thickness", type=float, default=0.08, help="wall thickness (m), if build-walls")
    ap.add_argument("--default-wall-height", type=float, default=2.7, help="fallback wall height if ceiling missing")
    ap.add_argument("--room-filter", default=None, help="substring filter by room id/type")
    args = ap.parse_args(argv)

    inp = Path(args.inp)
    data = json.loads(inp.read_text(encoding="utf-8"))

    reset_scene()
    ensure_basic_light_and_camera()

    future_root = Path(args.future_root)
    build_from_processed(
        data=data,
        future_root=future_root,
        build_walls=bool(args.build_walls),
        wall_thickness=float(args.wall_thickness),
        default_wall_height=float(args.default_wall_height),
        room_filter=args.room_filter,
    )

    outp = Path(args.out_blend)
    outp.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(outp))
    print(f"[ok] saved blend: {outp}")


if __name__ == "__main__":
    main()