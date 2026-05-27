#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# src/tools/make_scene_mini_blender.py
#
# Blender-identical mini-json для 3D-FRONT-processed:
# - импорт OBJ (raw_model.obj/normalized_model.obj)
# - scale
# - upright (фиксированный idx, по умолчанию 2)
# - yaw после upright (со знаком минус, как в build_front_scene.py)
# - snap-to-floor для НЕ Lighting
# - Lighting: не прижимается к полу, может цепляться к потолку
# - keep-inside-room
# - resolve_collisions (AABB 2D, только НЕ Lighting)
#
# ВЫХОД:
# - Для каждого объекта: name, model_id, super-category, category, label
# - Идентификаторы: super_id, cat_id (если известны)
# - Атрибуты: style/material/theme + style_id/material_id/theme_id (если удалось сопоставить)
#
# КРИТИЧЕСКОЕ ТРЕБОВАНИЕ:
# - Если model_id отсутствует в models_root/model_info.json, объект ПОЛНОСТЬЮ ПРОПУСКАЕТСЯ:
#   не импортируется в Blender, не участвует в коллизиях и НЕ ПОПАДАЕТ в json-mini.
#

from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector


# ==========================
# Базовые утилиты
# ==========================

def norm_str(x: Any) -> str:
    return ("" if x is None else str(x)).strip()


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def to_int_or_none(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return int(x)
        s = norm_str(x)
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


# ==========================
# Координаты: вход <-> Blender
# ==========================

def to_blender_loc(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # (x,y,z) -> (x,z,y)
    return (float(x), float(z), float(y))


def from_blender_loc(bx: float, by: float, bz: float) -> Tuple[float, float, float]:
    # (x,z,y) <- (x,y,z)
    return (float(bx), float(bz), float(by))


def yaw_deg_to_rot_z(yaw_deg: float) -> float:
    return math.radians(float(yaw_deg))


def scale_to_blender(scale_xyz: List[float] | Tuple[float, float, float]) -> Tuple[float, float, float]:
    # input [sx, sy(height), sz] -> blender (sx, sz, sy)
    sx, sy, sz = float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])
    return (sx, sz, sy)


# ==========================
# Справочники: categories.py / model_info.json
# ==========================

def load_categories_py(categories_py: Path) -> Dict[str, Any]:
    if not categories_py.exists():
        return {}
    ns = runpy.run_path(str(categories_py))
    out: Dict[str, Any] = {}
    for k in (
        "_ATTR_STYLE",
        "_ATTR_MATERIAL",
        "_ATTR_THEME",
        "_SUPER_CATEGORIES_3D",
        "_CATEGORIES_3D",
        "_CATEGORIES_3D_TEXTURE",
    ):
        if k in ns:
            out[k] = ns[k]
    return out


def _build_attr_maps(items: List[Dict[str, Any]]) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        cid = to_int_or_none(it.get("id"))
        nm = norm_str(it.get("category"))
        if cid is None or not nm:
            continue
        id2name[cid] = nm
        if nm not in name2id:
            name2id[nm] = cid
    return id2name, name2id


def build_category_maps(cats_ns: Dict[str, Any]) -> Dict[str, Any]:
    super_name_to_id: Dict[str, int] = {}
    cat_name_to_id: Dict[str, int] = {}
    cat_name_to_super: Dict[str, str] = {}

    for it in (cats_ns.get("_SUPER_CATEGORIES_3D") or []):
        name = norm_str(it.get("category"))
        cid = to_int_or_none(it.get("id"))
        if name and cid is not None:
            super_name_to_id[name] = cid

    for it in (cats_ns.get("_CATEGORIES_3D") or []):
        name = norm_str(it.get("category"))
        sup = norm_str(it.get("super-category"))
        cid = to_int_or_none(it.get("id"))
        if name and sup:
            cat_name_to_super[name] = sup
        if name and cid is not None:
            cat_name_to_id[name] = cid

    if "Other" not in super_name_to_id:
        super_name_to_id["Other"] = 8

    style_id2name, style_name2id = _build_attr_maps(list(cats_ns.get("_ATTR_STYLE") or []))
    mat_id2name, mat_name2id = _build_attr_maps(list(cats_ns.get("_ATTR_MATERIAL") or []))
    theme_id2name, theme_name2id = _build_attr_maps(list(cats_ns.get("_ATTR_THEME") or []))

    return {
        "super_name_to_id": super_name_to_id,
        "cat_name_to_id": cat_name_to_id,
        "cat_name_to_super": cat_name_to_super,
        "style_id2name": style_id2name,
        "style_name2id": style_name2id,
        "material_id2name": mat_id2name,
        "material_name2id": mat_name2id,
        "theme_id2name": theme_id2name,
        "theme_name2id": theme_name2id,
    }


def load_model_info(model_info_json: Path) -> Dict[str, Dict[str, Any]]:
    if not model_info_json.exists():
        return {}
    data = json.loads(model_info_json.read_text(encoding="utf-8"))

    items: Any = data
    if isinstance(data, dict):
        items = data.get("items") or data.get("models") or data.get("data") or []
    if not isinstance(items, list):
        return {}

    mp: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = norm_str(it.get("model_id"))
        if not mid:
            continue
        mp[mid] = {
            "super-category": it.get("super-category"),
            "category": it.get("category"),
            "style": it.get("style"),
            "material": it.get("material"),
            "theme": it.get("theme"),
        }
    return mp


def norm_super_category(x: Any) -> str:
    s = norm_str(x)
    if not s:
        return "Other"
    if s.lower() == "others":
        return "Other"
    return s


def norm_category(x: Any) -> str:
    s = norm_str(x)
    if not s:
        return "Other"
    if s.lower() == "others":
        return "Other"
    return s


def extract_model_id(src_obj: Dict[str, Any]) -> str:
    jid = norm_str(src_obj.get("jid"))
    if jid:
        return jid
    ref = norm_str(src_obj.get("ref"))
    if ref:
        return ref.split("/")[0].strip()
    return ""


def _normalize_attr_value(
    raw_value: Any,
    id2name: Dict[int, str],
    name2id: Dict[str, int],
) -> Tuple[Optional[str], Optional[int]]:
    if raw_value is None:
        return (None, None)

    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        nm = id2name.get(int(raw_value))
        return (nm or None, int(raw_value) if nm is not None else None)

    s = norm_str(raw_value)
    if not s:
        return (None, None)

    maybe_id = to_int_or_none(s)
    if maybe_id is not None and maybe_id in id2name:
        return (id2name[maybe_id], maybe_id)

    if s in name2id:
        return (s, name2id[s])

    return (s, None)


def choose_labels_for_object(
    src_obj: Dict[str, Any],
    model_id: str,
    model_info_by_id: Dict[str, Dict[str, Any]],
    cat_maps: Dict[str, Any],
) -> Dict[str, Any]:
    mi = model_info_by_id.get(model_id, {}) if model_id else {}

    super_cat = norm_super_category(mi.get("super-category"))
    category = norm_category(mi.get("category"))

    cat_source = "model_info"
    if category == "Other":
        cand = norm_category(src_obj.get("category"))
        if cand != "Other":
            category = cand
            cat_source = "scene"

    cat_name_to_super: Dict[str, str] = cat_maps["cat_name_to_super"]
    if super_cat == "Other" and category != "Other":
        guess = norm_super_category(cat_name_to_super.get(category))
        if guess and guess != "Other":
            super_cat = guess

    super_name_to_id: Dict[str, int] = cat_maps["super_name_to_id"]
    if super_cat not in super_name_to_id:
        super_cat = "Other"
    super_id = int(super_name_to_id.get(super_cat, super_name_to_id["Other"]))

    cat_name_to_id: Dict[str, int] = cat_maps["cat_name_to_id"]
    cat_id = cat_name_to_id.get(category)

    if super_cat == "Other":
        label = "Other"
    elif category != "Other":
        label = f"{super_cat}/{category}"
    else:
        label = super_cat

    style_raw = mi.get("style", None)
    material_raw = mi.get("material", None)
    theme_raw = mi.get("theme", None)

    if style_raw is None:
        style_raw = src_obj.get("style", None)
    if material_raw is None:
        material_raw = src_obj.get("material", None)
    if theme_raw is None:
        theme_raw = src_obj.get("theme", None)

    style, style_id = _normalize_attr_value(style_raw, cat_maps["style_id2name"], cat_maps["style_name2id"])
    material, material_id = _normalize_attr_value(material_raw, cat_maps["material_id2name"], cat_maps["material_name2id"])
    theme, theme_id = _normalize_attr_value(theme_raw, cat_maps["theme_id2name"], cat_maps["theme_name2id"])

    return {
        "super-category": super_cat,
        "super_id": super_id,
        "category": category,
        "cat_id": int(cat_id) if isinstance(cat_id, int) else None,
        "label": label,
        "style": style,
        "style_id": style_id,
        "material": material,
        "material_id": material_id,
        "theme": theme,
        "theme_id": theme_id,
        "cat_source": cat_source,
    }


# ==========================
# Upright: 24 ориентации
# ==========================

def _rot_x90(k: int) -> Matrix:
    return Matrix.Rotation((math.pi / 2) * k, 4, "X")


def _rot_y90(k: int) -> Matrix:
    return Matrix.Rotation((math.pi / 2) * k, 4, "Y")


def _rot_z90(k: int) -> Matrix:
    return Matrix.Rotation((math.pi / 2) * k, 4, "Z")


def all_24_axis_rotations() -> List[Matrix]:
    rots: List[Matrix] = []
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


def apply_upright_rotation_world(obj: bpy.types.Object, world_rot: Matrix) -> None:
    pivot = obj.location.copy()
    T1 = Matrix.Translation(pivot)
    T2 = Matrix.Translation(-pivot)
    obj.matrix_world = (T1 @ world_rot @ T2) @ obj.matrix_world
    bpy.context.view_layer.update()


def apply_yaw_after_upright(obj: bpy.types.Object, yaw_deg: float) -> None:
    rz = yaw_deg_to_rot_z(-float(yaw_deg))
    Rz = Matrix.Rotation(rz, 4, "Z")
    pivot = obj.location.copy()
    T1 = Matrix.Translation(pivot)
    T2 = Matrix.Translation(-pivot)
    obj.matrix_world = (T1 @ Rz @ T2) @ obj.matrix_world
    bpy.context.view_layer.update()


# ==========================
# Blender: очистка и импорт
# ==========================

def reset_scene_empty() -> None:
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    except Exception:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)


def import_obj(obj_path: Path) -> List[bpy.types.Object]:
    if not obj_path.exists():
        return []
    before = set(bpy.data.objects)

    if hasattr(bpy.ops.wm, "obj_import"):
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="NEGATIVE_Z", up_axis="Y")
        except Exception:
            return []
    elif hasattr(bpy.ops.import_scene, "obj"):
        try:
            bpy.ops.import_scene.obj(filepath=str(obj_path), axis_forward="-Z", axis_up="Y", use_image_search=False)
        except Exception:
            return []
    else:
        return []

    after = set(bpy.data.objects)
    return [o for o in (after - before) if o.type in {"MESH", "EMPTY"}]


def join_meshes(objs: List[bpy.types.Object]) -> Optional[bpy.types.Object]:
    mesh_objs = [o for o in objs if o.type == "MESH"]
    if not mesh_objs:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = mesh_objs[0]
    for o in mesh_objs:
        o.select_set(True)

    if len(mesh_objs) > 1:
        try:
            bpy.ops.object.join()
        except Exception:
            pass

    merged = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action="DESELECT")
    return merged


def delete_objects(objs: List[bpy.types.Object]) -> None:
    if not objs:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        if o and o.name in bpy.data.objects:
            o.select_set(True)
    try:
        bpy.ops.object.delete(use_global=False)
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")


def apply_object_scale(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass
    bpy.ops.object.select_all(action="DESELECT")


def import_model_from_dir(model_dir: Path, prefer_raw: bool) -> Tuple[Optional[bpy.types.Object], Optional[Path], List[bpy.types.Object]]:
    candidates = ["raw_model.obj", "normalized_model.obj"] if prefer_raw else ["normalized_model.obj", "raw_model.obj"]

    obj_path: Optional[Path] = None
    for name in candidates:
        p = (model_dir / name).resolve()
        if p.exists():
            obj_path = p
            break
    if obj_path is None:
        return None, None, []

    new_objs = import_obj(obj_path)
    merged = join_meshes(new_objs)
    return merged, obj_path, new_objs


# ==========================
# Геометрия: bbox, пол, потолок
# ==========================

def object_world_bbox_xy(obj: bpy.types.Object) -> Tuple[float, float, float, float]:
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))


def object_world_center_xy(obj: bpy.types.Object) -> Tuple[float, float]:
    x0, x1, y0, y1 = object_world_bbox_xy(obj)
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def snap_object_to_floor(obj: bpy.types.Object, floor_z: float = 0.0) -> None:
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    zmin = min(p.z for p in pts)
    obj.location.z += (floor_z - zmin)
    bpy.context.view_layer.update()


def snap_object_to_ceiling(obj: bpy.types.Object, ceiling_z: float) -> None:
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    zmax = max(p.z for p in pts)
    obj.location.z += (ceiling_z - zmax)
    bpy.context.view_layer.update()


# ==========================
# Полигон комнаты и удержание внутри (Blender XY)
# ==========================

def room_polygon_xy(polygon_xz: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in polygon_xz or []:
        if "x" in p and "z" in p:
            out.append((float(p["x"]), float(p["z"])))
    return out


def poly_centroid(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not poly:
        return (0.0, 0.0)
    sx = sum(p[0] for p in poly)
    sy = sum(p[1] for p in poly)
    n = float(len(poly))
    return (sx / n, sy / n)


def point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
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


def closest_point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> Tuple[float, float]:
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return (ax, ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    return (ax + t * abx, ay + t * aby)


def closest_point_on_poly(px: float, py: float, poly: List[Tuple[float, float]]) -> Tuple[float, float]:
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


def keep_object_inside_room(obj: bpy.types.Object, poly_xy: List[Tuple[float, float]], margin: float) -> None:
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
        if vl > 1e-9:
            vx, vy = vx / vl, vy / vl
        else:
            vx, vy = 0.0, 0.0

        tx, ty = (qx + vx * margin, qy + vy * margin)
        dx, dy = (tx - bx, ty - by)

        obj.location.x += dx
        obj.location.y += dy
        bpy.context.view_layer.update()


# ==========================
# Размежевание (AABB 2D, только НЕ Lighting)
# ==========================

def aabb_overlap_2d(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], pad: float) -> Tuple[bool, float, float]:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    ok = (ox > -pad) and (oy > -pad)
    return ok, ox, oy


def resolve_overlaps_in_room(
    objs: List[bpy.types.Object],
    poly_xy: List[Tuple[float, float]],
    margin: float,
    max_iters: int = 40,
) -> None:
    movable = [o for o in objs if (o is not None) and (o.type == "MESH") and (not bool(o.get("is_lighting", False)))]
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
# Lighting правила
# ==========================

def is_lighting_category(super_cat: str, category: str) -> bool:
    s = (super_cat or "").strip().lower()
    c = (category or "").strip().lower()
    if s == "lighting":
        return True
    if "lighting" in c or "lamp" in c or "light" in c:
        return True
    return False


def lighting_should_attach_to_ceiling(src_obj: Dict[str, Any], category: str, wall_height: float) -> bool:
    for k in ("attach_to_ceiling", "ceiling", "is_ceiling", "ceiling_mounted", "hang_from_ceiling"):
        if bool(src_obj.get(k, False)):
            return True

    cat = (category or "").strip().lower()
    if any(t in cat for t in ("pendant", "ceiling", "hanging", "chandelier", "吊灯")):
        return True

    for k in ("anchor", "mount", "placement", "mount_type"):
        v = norm_str(src_obj.get(k, "")).lower()
        if v in {"ceiling", "hanging", "pendant", "top", "hang"}:
            return True

    try:
        py = float((src_obj.get("pos") or {}).get("y", 0.0))
        if py >= 1.2:
            return True
        if py >= 0.75 * float(wall_height):
            return True
    except Exception:
        pass

    return False


# ==========================
# CORE: build mini
# ==========================

def build_scene_mini(
    scene_json: Path,
    models_root: Path,
    out_json: Path,
    room_ids: Optional[List[str]],
    prefer_raw: int,
    resolve_collisions: int,
    collision_margin: float,
    wall_height: float,
    upright_idx: int,
    pretty: int,
) -> None:
    categories_py = (models_root / "categories.py").resolve()
    model_info_json = (models_root / "model_info.json").resolve()

    cats_ns = load_categories_py(categories_py)
    cat_maps = build_category_maps(cats_ns)
    model_info_by_id = load_model_info(model_info_json)

    scene_data = json.loads(scene_json.read_text(encoding="utf-8"))
    rooms_all = scene_data.get("rooms", []) or []
    if not isinstance(rooms_all, list) or not rooms_all:
        raise ValueError("scene_json: rooms пуст/неверный формат")

    if room_ids:
        want = set(room_ids)
        rooms = [r for r in rooms_all if norm_str(r.get("id")) in want]
    else:
        rooms = rooms_all[:1]

    if not rooms:
        raise ValueError("Не найдено ни одной комнаты по room_ids")

    rots24 = all_24_axis_rotations()
    k = int(upright_idx) % 24
    upright_rot = rots24[k]

    reset_scene_empty()

    out_rooms: List[Dict[str, Any]] = []

    for r in rooms:
        rid = norm_str(r.get("id")) or "room"
        polygon = r.get("polygon", []) or []
        poly_xy = room_polygon_xy(polygon)

        floor_polygon_xz = [{"x": float(p["x"]), "z": float(p["z"])} for p in polygon if ("x" in p and "z" in p)]

        src_objects_all: List[Dict[str, Any]] = list(r.get("objects", []) or [])

        # ВАЖНО: фильтруем объекты, которых НЕТ в model_info.json
        kept_src_objects: List[Dict[str, Any]] = []
        kept_model_ids: List[str] = []
        for src_o in src_objects_all:
            mid = extract_model_id(src_o)
            if not mid:
                continue
            if mid not in model_info_by_id:
                continue
            kept_src_objects.append(src_o)
            kept_model_ids.append(mid)

        imported_objs: List[bpy.types.Object] = []

        # --- импорт и постановка (ТОЛЬКО kept) ---
        for idx, (src_o, model_id) in enumerate(zip(kept_src_objects, kept_model_ids)):
            labels = choose_labels_for_object(src_o, model_id, model_info_by_id, cat_maps)

            model_dir = (models_root / str(model_id)).resolve()
            merged, _obj_path, new_imported = import_model_from_dir(model_dir, prefer_raw=bool(prefer_raw))
            obj = merged
            if obj is None:
                # Если модель отсутствует на диске/не импортируется, объект также пропускаем из json-mini
                delete_objects([x for x in new_imported if (x is not None) and (x.name in bpy.data.objects)])
                continue

            # удалить импортированные хвосты
            if new_imported:
                to_del = [x for x in new_imported if (x is not None) and (x.name in bpy.data.objects) and (x != obj)]
                delete_objects(to_del)

            imported_objs.append(obj)

            # lighting flag
            obj["is_lighting"] = bool(is_lighting_category(labels["super-category"], labels["category"]))

            # scale + apply
            sc = src_o.get("scale")
            if isinstance(sc, (list, tuple)) and len(sc) == 3:
                obj.scale = scale_to_blender(sc)
                bpy.context.view_layer.update()
                apply_object_scale(obj)

            # position
            pos = src_o.get("pos", {}) or {}
            bx, by, bz = to_blender_loc(pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0))
            obj.location = (bx, by, bz)
            bpy.context.view_layer.update()

            # upright fixed
            apply_upright_rotation_world(obj, upright_rot)

            # floor snap for non-light
            if not bool(obj.get("is_lighting", False)):
                snap_object_to_floor(obj, floor_z=0.0)

            # yaw after upright
            yaw = float(src_o.get("yaw_deg", 0.0))
            apply_yaw_after_upright(obj, yaw)

            if not bool(obj.get("is_lighting", False)):
                snap_object_to_floor(obj, floor_z=0.0)

            # lighting ceiling/height
            if bool(obj.get("is_lighting", False)):
                if lighting_should_attach_to_ceiling(src_o, labels["category"], wall_height=float(wall_height)):
                    snap_object_to_ceiling(obj, ceiling_z=float(wall_height))
                else:
                    desired_z = float(to_blender_loc(pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0))[2])
                    obj.location.z = desired_z
                    bpy.context.view_layer.update()

            # keep inside room
            keep_object_inside_room(obj, poly_xy, margin=float(collision_margin))

            # props
            ref = norm_str(src_o.get("ref"))
            if ref:
                obj["ref"] = ref
            obj["jid"] = model_id
            inst = norm_str(src_o.get("instanceid"))
            if inst:
                obj["instanceid"] = inst
            obj["valid"] = bool(src_o.get("valid", True))

        # Поскольку часть kept могла быть выкинута (не импортнулась модель), нужно синхронизировать списки.
        # Правило: сохраняем только те src_o, для которых реально есть imported obj.
        # imported_objs добавляется только при успешном импорте, значит надо собрать aligned списки заново.
        # Мы не можем просто zip(kept_src_objects, imported_objs), если где-то был continue.
        # Поэтому повторно строим aligned через второй проход: импортируемые добавлялись в том же порядке, где obj!=None.
        aligned_src: List[Dict[str, Any]] = []
        aligned_ids: List[str] = []
        # Повторим проход и добавим только те, чьи OBJ реально существуют в Blender-данных
        # (heuristic: по факту imported_objs уже в порядке; мы копим src/id параллельно в aligned внутри первого цикла нельзя без усложнения.
        # Здесь делаем корректно: пересобираем через второй импорт-цикл нельзя.
        # Поэтому делаем проще и надёжнее: в первом цикле при успешном импорте добавляли obj,
        # а здесь пересоберём aligned через атрибут obj["jid"].
        if imported_objs:
            keep_set = {o.get("jid", "") for o in imported_objs}
            for src_o, mid in zip(kept_src_objects, kept_model_ids):
                if mid in keep_set:
                    aligned_src.append(src_o)
                    aligned_ids.append(mid)

        # --- collisions ---
        if int(resolve_collisions) == 1 and imported_objs:
            resolve_overlaps_in_room(imported_objs, poly_xy=poly_xy, margin=float(collision_margin), max_iters=40)

            for idx, obj in enumerate(imported_objs):
                if obj is None or obj.type != "MESH":
                    continue
                mid = norm_str(obj.get("jid"))
                src_o = {}
                # найдём исходный объект по model_id (достаточно для корректной обработки светильников)
                # Если в комнате один и тот же model_id встречается несколько раз — это эвристика, но логика потолка/пола одинаковая.
                for so, mid2 in zip(aligned_src, aligned_ids):
                    if mid2 == mid:
                        src_o = so
                        break

                lbl = choose_labels_for_object(src_o, mid, model_info_by_id, cat_maps)

                if bool(obj.get("is_lighting", False)):
                    if lighting_should_attach_to_ceiling(src_o, lbl["category"], wall_height=float(wall_height)):
                        snap_object_to_ceiling(obj, ceiling_z=float(wall_height))
                    keep_object_inside_room(obj, poly_xy, margin=float(collision_margin))
                else:
                    snap_object_to_floor(obj, floor_z=0.0)
                    keep_object_inside_room(obj, poly_xy, margin=float(collision_margin))

        # --- export objects ---
        objs_out: List[Dict[str, Any]] = []
        for obj in imported_objs:
            if obj is None or obj.type != "MESH":
                continue

            model_id = norm_str(obj.get("jid"))
            if not model_id:
                continue
            if model_id not in model_info_by_id:
                # дополнительная защита: "если нет в model_info.json — не писать"
                continue

            # найдём src_o для yaw/scale/ref/instanceid (если не нашли — берём пустой dict)
            src_o: Dict[str, Any] = {}
            for so, mid2 in zip(aligned_src, aligned_ids):
                if mid2 == model_id:
                    src_o = so
                    break

            labels = choose_labels_for_object(src_o, model_id, model_info_by_id, cat_maps)

            x0, x1, y0, y1 = object_world_bbox_xy(obj)
            px, py, pz = from_blender_loc(obj.location.x, obj.location.y, obj.location.z)

            name = labels.get("label") or labels.get("category") or "Other"

            item: Dict[str, Any] = {
                "name": name,
                "model_id": model_id,

                "super-category": labels["super-category"],
                "super_id": labels["super_id"],
                "category": labels["category"],
                "cat_id": labels["cat_id"],
                "label": labels["label"],

                "style": labels["style"],
                "style_id": labels["style_id"],
                "material": labels["material"],
                "material_id": labels["material_id"],
                "theme": labels["theme"],
                "theme_id": labels["theme_id"],

                "cat_source": labels["cat_source"],

                "pos": {"x": float(px), "y": float(py), "z": float(pz)},
                "yaw_deg": float(src_o.get("yaw_deg", 0.0)),
                "upright_idx": int(upright_idx) % 24,
                "is_lighting": bool(obj.get("is_lighting", False)),
                "bbox_world_xy": [float(x0), float(x1), float(y0), float(y1)],
            }

            sc = src_o.get("scale")
            if isinstance(sc, (list, tuple)) and len(sc) == 3:
                item["scale"] = [float(sc[0]), float(sc[1]), float(sc[2])]

            ref = norm_str(src_o.get("ref"))
            if ref:
                item["ref"] = ref
            inst = norm_str(src_o.get("instanceid"))
            if inst:
                item["instanceid"] = inst

            item = {k: v for k, v in item.items() if v is not None}
            objs_out.append(item)

        out_rooms.append(
            {
                "id": rid,
                "floor_polygon_xz": floor_polygon_xz,
                "objects": objs_out,
            }
        )

    out = {
        "schema": "json-mini/v6-skip-unknown-models",
        "upright_idx": int(upright_idx) % 24,
        "rooms": out_rooms,
    }

    ensure_parent_dir(out_json)
    if int(pretty) == 1:
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        out_json.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


# ==========================
# CLI
# ==========================

def parse_args_blender() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_json", required=True)
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--room_id", default=None, help="deprecated; use --room_ids")
    ap.add_argument("--room_ids", default=None, help="room ids через запятую; если не задано — rooms[0]")

    ap.add_argument("--prefer_raw", type=int, default=1)
    ap.add_argument("--resolve_collisions", type=int, default=1)
    ap.add_argument("--collision_margin", type=float, default=0.02)
    ap.add_argument("--wall_height", type=float, default=2.7)

    ap.add_argument("--upright_idx", type=int, default=2)
    ap.add_argument("--pretty", type=int, default=1)

    argv: List[str] = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args_blender()

    room_ids: Optional[List[str]] = None
    if args.room_ids:
        room_ids = [s.strip() for s in str(args.room_ids).split(",") if s.strip()]
    elif args.room_id:
        rid = str(args.room_id).strip()
        room_ids = [rid] if rid else None

    scene_json = Path(args.scene_json).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    if not scene_json.exists():
        raise FileNotFoundError(f"scene_json not found: {scene_json}")
    if not models_root.exists():
        raise FileNotFoundError(f"models_root not found: {models_root}")

    build_scene_mini(
        scene_json=scene_json,
        models_root=models_root,
        out_json=out_json,
        room_ids=room_ids,
        prefer_raw=int(args.prefer_raw),
        resolve_collisions=int(args.resolve_collisions),
        collision_margin=float(args.collision_margin),
        wall_height=float(args.wall_height),
        upright_idx=int(args.upright_idx),
        pretty=int(args.pretty),
    )


if __name__ == "__main__":
    main()
