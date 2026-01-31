# -*- coding: utf-8 -*-
# src/Plasement/BlenderViewOBJFolder.py

import os
import sys
import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import bpy
import mathutils

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from bl_helpers import reset_scene, ensure_collection, import_mesh, world_bounds


def _pick_obj_file(folder: Path, explicit_name: Optional[str]) -> Path:
    if explicit_name:
        p = (folder / explicit_name).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--obj not found: {p}")
        return p

    objs = sorted(folder.glob("*.obj"))
    if not objs:
        raise FileNotFoundError(f"No .obj files in folder: {folder}")

    # берём самый большой по размеру файл как “главный”
    objs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return objs[0].resolve()


def _move_to_collection(objs: List[bpy.types.Object], coll: bpy.types.Collection) -> None:
    for o in objs:
        for c in list(o.users_collection):
            try:
                c.objects.unlink(o)
            except Exception:
                pass
        coll.objects.link(o)


def _make_parent_empty(name: str = "ITEM") -> bpy.types.Object:
    parent = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(parent)
    return parent


def _parent_objects(parent: bpy.types.Object, objs: List[bpy.types.Object], coll: bpy.types.Collection) -> None:
    for o in objs:
        o.parent = parent
    for c in list(parent.users_collection):
        try:
            c.objects.unlink(parent)
        except Exception:
            pass
    coll.objects.link(parent)


def _bbox_info(children: List[bpy.types.Object]) -> Tuple[mathutils.Vector, mathutils.Vector, mathutils.Vector, float]:
    bpy.context.view_layer.update()
    bmin, bmax = world_bounds(children)
    dims = (bmax - bmin)
    max_dim = float(max(dims.x, dims.y, dims.z))
    return bmin, bmax, dims, max_dim


def _power10_scale_to_fit(max_dim: float, target_max: float) -> Tuple[int, float]:
    """
    Ищем минимальное k >= 0 такое, что max_dim * 10^-k <= target_max.
    """
    if max_dim <= 0.0 or max_dim <= target_max:
        return 0, 1.0
    k = int(math.ceil(math.log10(max_dim / target_max)))
    k = max(k, 0)
    return k, 10.0 ** (-k)


def _scale_parent(parent: bpy.types.Object, children: List[bpy.types.Object], target_max: float, power10: bool) -> Tuple[int, float]:
    _, _, _, max_dim = _bbox_info(children)
    if max_dim <= 0.0:
        return 0, 1.0

    if power10:
        k, s = _power10_scale_to_fit(max_dim, target_max)
    else:
        s = (target_max / max_dim) if max_dim > target_max else 1.0
        k = 0

    if s != 1.0:
        parent.scale = (parent.scale[0] * s, parent.scale[1] * s, parent.scale[2] * s)
        bpy.context.view_layer.update()

    return k, float(s)


def _center_parent_to_origin(parent: bpy.types.Object, children: List[bpy.types.Object]) -> None:
    bpy.context.view_layer.update()
    bmin, bmax = world_bounds(children)
    center = (bmin + bmax) * 0.5
    parent.location -= center
    bpy.context.view_layer.update()


def _set_viewport_clipping(radius: float) -> None:
    wm = bpy.context.window_manager
    if not wm or not wm.windows:
        return
    win = wm.windows[0]
    if not win.screen:
        return

    clip_end = max(1000.0, radius * 50.0)
    for area in win.screen.areas:
        if area.type == "VIEW_3D":
            sp = area.spaces.active
            try:
                sp.clip_start = 0.001
                sp.clip_end = clip_end
            except Exception:
                pass


def _add_camera(center: mathutils.Vector, radius: float) -> None:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    cam.location = (center.x - 2.5 * radius, center.y - 2.5 * radius, center.z + 1.6 * radius)
    cam.data.lens = 35

    direction = mathutils.Vector(center) - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.scene.camera = cam


def _add_light(center: mathutils.Vector, radius: float) -> None:
    sun_data = bpy.data.lights.new("Sun", 'SUN')
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (center.x + 5 * radius, center.y + 5 * radius, center.z + 5 * radius)


def parse_argv(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder containing .obj")
    ap.add_argument("--obj", default=None, help="Exact .obj filename inside folder (optional)")
    ap.add_argument("--no-center", action="store_true", help="Do not center object to origin")

    # главное: автоскейл
    ap.add_argument("--fit", type=float, default=4.0, help="Fit max dimension into this many meters (default 4.0)")
    ap.add_argument("--no-scale", action="store_true", help="Disable auto scaling")
    ap.add_argument("--no-power10", action="store_true", help="Scale not by 10^-k, but by exact factor")

    ap.add_argument("--save-blend", default=None, help="Save .blend after import")
    return ap.parse_args(argv)


def main():
    args = parse_argv(sys.argv)

    folder = Path(args.dir).expanduser().resolve()
    if not folder.is_dir():
        print(f"[VIEW] ERROR: --dir is not a directory: {folder}")
        return

    obj_path = _pick_obj_file(folder, args.obj)
    print(f"[VIEW] Importing OBJ: {obj_path}")

    reset_scene()

    coll_item = ensure_collection("Item")

    imported = import_mesh(str(obj_path))
    if not imported:
        print("[VIEW] ERROR: import_mesh returned empty list.")
        return

    _move_to_collection(imported, coll_item)

    parent = _make_parent_empty("ITEM")
    _parent_objects(parent, imported, coll_item)

    # 1) Автоскейл до fit-куба
    k = 0
    s = 1.0
    if not args.no_scale:
        k, s = _scale_parent(parent, imported, target_max=float(args.fit), power10=(not args.no_power10))

    # 2) Центровка
    if not args.no_center:
        _center_parent_to_origin(parent, imported)

    # 3) Камера/свет по актуальному bbox
    bmin, bmax, dims, max_dim = _bbox_info(imported)
    center = (bmin + bmax) * 0.5
    radius = max(0.5, 0.5 * max(dims.x, dims.y, dims.z))

    _set_viewport_clipping(radius)
    _add_camera(center, radius)
    _add_light(center, radius)

    print(f"[VIEW] BBox dims (m): {dims.x:.3f} {dims.y:.3f} {dims.z:.3f}   max={max_dim:.3f}")
    if not args.no_scale:
        if args.no_power10:
            print(f"[VIEW] Applied scale: factor={s:.6g}")
        else:
            print(f"[VIEW] Applied scale: 10^-{k} (factor={s:.6g})")

    if args.save_blend:
        out = Path(args.save_blend).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out))
        print(f"[VIEW] Saved blend: {out}")


if __name__ == "__main__":
    main()