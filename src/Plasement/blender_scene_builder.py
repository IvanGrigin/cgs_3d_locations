# -*- coding: utf-8 -*-
# src/Plasement/blender_scene_builder.py
#
# Совместимый builder: GLB комнаты + placement JSON / scene.v1 JSON.
# Главная цель: корректные текстуры.
# Логика:
#   - если keep_existing_mats=True (по умолчанию), НЕ затираем авторские материалы OBJ/MTL
#   - пытаемся восстановить текстуры: relink TEX_IMAGE -> map_Kd из MTL -> largest-image fallback
#   - если материалов нет, создаём PBR (Principled) и назначаем
#
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import random
import traceback
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import bpy
import bmesh
import mathutils


# ============================================================
# CLI
# ============================================================

def _parse_argv(argv: List[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--json", required=True)

    ap.add_argument("--project-root", default=None)  # compat, unused
    ap.add_argument("--import-glb", action="store_true")
    ap.add_argument("--save-blend", default=None)
    ap.add_argument("--render", default=None)
    ap.add_argument("--draw-aabb", action="store_true")
    ap.add_argument("--background", action="store_true")  # compat
    ap.add_argument("--force-tint", action="store_true")

    # Texture policy
    ap.add_argument(
        "--rebuild-materials",
        action="store_true",
        help="Force rebuild PBR materials even if OBJ has materials. Default: keep existing.",
    )
    ap.add_argument(
        "--no-keep-existing-mats",
        action="store_true",
        help="Alias: disable keeping existing materials (same as --rebuild-materials).",
    )

    ap.add_argument("--no-pack-assets", action="store_true", help="Do not pack external files into .blend")
    ap.add_argument("--verbose", action="store_true")

    # Room environment textures
    ap.add_argument(
        "--env-textures-dir",
        default="data/sourse",
        help="Directory with environment textures (floor_*.jpg, wall_*.jpg, window_*.jpg, door_*.jpg)",
    )
    ap.add_argument(
        "--style-text",
        default=None,
        help="Optional text description for future texture selection (placeholder; currently random).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for env texture selection (0 = derive from json path).",
    )

    return ap.parse_args(argv)


# ============================================================
# Logging
# ============================================================

def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


# ============================================================
# Scene utils
# ============================================================

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    try:
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0
    except Exception:
        pass

    try:
        if "BLENDER_EEVEE_NEXT" in bpy.app.build_options:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        else:
            scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        pass

    for obj in list(bpy.data.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
    for me in list(bpy.data.meshes):
        try:
            bpy.data.meshes.remove(me, do_unlink=True)
        except Exception:
            pass


def ensure_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _unlink_from_all_collections(obj: bpy.types.Object) -> None:
    for c in list(obj.users_collection):
        try:
            c.objects.unlink(obj)
        except Exception:
            pass


def _world_bounds_mesh_objects(objs: List[bpy.types.Object]) -> Tuple[mathutils.Vector, mathutils.Vector]:
    bb_min = mathutils.Vector((+1e30, +1e30, +1e30))
    bb_max = mathutils.Vector((-1e30, -1e30, -1e30))

    deps = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        if o.type != "MESH":
            continue
        eo = o.evaluated_get(deps)
        me = eo.to_mesh()
        if not me:
            continue
        mat = eo.matrix_world
        for v in me.vertices:
            p = mat @ v.co
            bb_min.x = min(bb_min.x, p.x)
            bb_min.y = min(bb_min.y, p.y)
            bb_min.z = min(bb_min.z, p.z)
            bb_max.x = max(bb_max.x, p.x)
            bb_max.y = max(bb_max.y, p.y)
            bb_max.z = max(bb_max.z, p.z)
        eo.to_mesh_clear()

    if bb_min.x > bb_max.x:
        z = mathutils.Vector((0.0, 0.0, 0.0))
        return z, z
    return bb_min, bb_max


def _aabb_to_center_size(aabb: Dict[str, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    x1, x2 = float(aabb["x_min"]), float(aabb["x_max"])
    y1, y2 = float(aabb["y_min"]), float(aabb["y_max"])
    z1, z2 = float(aabb["z_min"]), float(aabb["z_max"])
    cx, cy, cz = 0.5 * (x1 + x2), 0.5 * (y1 + y2), 0.5 * (z1 + z2)
    sx, sy, sz = (x2 - x1), (y2 - y1), (z2 - z1)
    return (cx, cy, cz), (sx, sy, sz)


def _add_aabb_box(aabb: Dict[str, float], name: str, collection: bpy.types.Collection) -> bpy.types.Object:
    (cx, cy, cz), (sx, sy, sz) = _aabb_to_center_size(aabb)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = f"{name}_AABB"
    obj.scale = (sx * 0.5, sy * 0.5, sz * 0.5)

    _unlink_from_all_collections(obj)
    collection.objects.link(obj)
    return obj


def _ensure_world() -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    try:
        scene.world.use_nodes = True
        nt = scene.world.node_tree
        nodes = nt.nodes
        links = nt.links
        bg = next((n for n in nodes if n.type == "BACKGROUND"), None)
        out = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None)
        if not out:
            out = nodes.new("ShaderNodeOutputWorld")
        if not bg:
            bg = nodes.new("ShaderNodeBackground")
        try:
            links.new(bg.outputs["Background"], out.inputs["Surface"])
        except Exception:
            pass
        bg.inputs["Strength"].default_value = 1.0
    except Exception:
        pass


def _force_material_preview_if_ui() -> None:
    wm = bpy.context.window_manager
    if not wm or not getattr(wm, "windows", None):
        return
    for win in wm.windows:
        scr = win.screen
        if not scr:
            continue
        for area in scr.areas:
            if area.type != "VIEW_3D":
                continue
            sp = area.spaces.active
            if not hasattr(sp, "shading"):
                continue
            try:
                sp.shading.type = "MATERIAL"
                if hasattr(sp.shading, "use_scene_lights"):
                    sp.shading.use_scene_lights = True
                if hasattr(sp.shading, "use_scene_world"):
                    sp.shading.use_scene_world = False
            except Exception:
                pass


def _frame_camera_on_bounds(bb_min: mathutils.Vector, bb_max: mathutils.Vector) -> None:
    center = (bb_min + bb_max) * 0.5
    dims = bb_max - bb_min
    radius = max(dims.x, dims.y, dims.z, 0.1)

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    cam.location = (center.x - 1.6 * radius, center.y - 1.6 * radius, center.z + 1.2 * radius)
    cam.data.lens = 35
    direction = mathutils.Vector((center.x, center.y, center.z)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.context.scene.camera = cam


def _add_basic_lights(bb_min: mathutils.Vector, bb_max: mathutils.Vector) -> None:
    center = (bb_min + bb_max) * 0.5
    dims = bb_max - bb_min
    radius = max(dims.x, dims.y, dims.z, 0.1)

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = (center.x + 3.0 * radius, center.y + 2.5 * radius, center.z + 3.0 * radius)

    try:
        area_data = bpy.data.lights.new("Key", "AREA")
        area_data.energy = 500.0
        area_data.size = 2.0 * radius
        area = bpy.data.objects.new("Key", area_data)
        bpy.context.scene.collection.objects.link(area)
        area.location = (center.x - 1.5 * radius, center.y + 1.5 * radius, center.z + 1.8 * radius)
        direction = mathutils.Vector((center.x, center.y, center.z)) - area.location
        area.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    except Exception:
        pass


def _pack_assets_best_effort() -> None:
    try:
        bpy.ops.file.make_paths_relative()
    except Exception:
        pass
    try:
        bpy.ops.file.pack_all()
    except Exception:
        pass


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_item_aabb_to_room_glb(
    aabb: Dict[str, float],
    bb_min: mathutils.Vector,
    bb_max: mathutils.Vector,
    margin: float = 0.02,
) -> Dict[str, float]:
    x1, x2 = float(aabb["x_min"]), float(aabb["x_max"])
    y1, y2 = float(aabb["y_min"]), float(aabb["y_max"])
    z1, z2 = float(aabb["z_min"]), float(aabb["z_max"])

    sx = max(1e-6, x2 - x1)
    sy = max(1e-6, y2 - y1)
    sz = max(1e-6, z2 - z1)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    cz = 0.5 * (z1 + z2)

    cx = _clamp(cx, float(bb_min.x) + margin + sx * 0.5, float(bb_max.x) - margin - sx * 0.5)
    cy = _clamp(cy, float(bb_min.y) + margin + sy * 0.5, float(bb_max.y) - margin - sy * 0.5)
    cz = _clamp(cz, float(bb_min.z) + margin + sz * 0.5, float(bb_max.z) - margin - sz * 0.5)

    return {
        "x_min": cx - sx * 0.5, "x_max": cx + sx * 0.5,
        "y_min": cy - sy * 0.5, "y_max": cy + sy * 0.5,
        "z_min": cz - sz * 0.5, "z_max": cz + sz * 0.5,
    }


# ============================================================
# Path helpers
# ============================================================

def _resolve_path_maybe(base_dir: Path, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    s = str(p).strip().strip('"').strip("'")
    if not s:
        return None
    pp = Path(s).expanduser()
    if pp.is_absolute():
        return str(pp)
    return str((base_dir / pp).resolve())


def _item_asset_dict(it: dict) -> dict:
    asset = it.get("asset")
    return asset if isinstance(asset, dict) else {}


def _item_mesh_path_raw(it: dict) -> Optional[str]:
    asset = _item_asset_dict(it)
    return (
        it.get("mesh_path")
        or it.get("obj_path")
        or asset.get("mesh_path")
        or asset.get("obj_path")
    )


def _item_mesh_texture_dirs_raw(it: dict) -> List[str]:
    asset = _item_asset_dict(it)
    raw = it.get("mesh_texture_dirs")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]

    raw = asset.get("mesh_texture_dirs")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw]

    return []


def _item_mesh_fit_mode(it: dict) -> str:
    asset = _item_asset_dict(it)
    return str(it.get("mesh_fit_mode") or asset.get("mesh_fit_mode") or "stretch")


def _item_name(it: dict) -> str:
    return str(it.get("name") or it.get("category") or it.get("id") or "Item")


# ============================================================
# Import
# ============================================================

def import_glb_into_collection(glb_path: str, collection: bpy.types.Collection) -> List[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    after = [o for o in bpy.data.objects if o not in before]
    for o in after:
        _unlink_from_all_collections(o)
        collection.objects.link(o)
    return after


def _build_obj_import_override() -> Optional[dict]:
    wm = bpy.context.window_manager
    windows = getattr(wm, "windows", None) if wm else None
    if not windows:
        return None
    win = windows[0]
    scr = win.screen if win else None
    if not scr:
        return None

    area = None
    for a in scr.areas:
        if a.type == "VIEW_3D":
            area = a
            break
    if area is None and scr.areas:
        area = scr.areas[0]

    region = None
    if area:
        for r in area.regions:
            if r.type == "WINDOW":
                region = r
                break

    if not area or not region:
        return None

    space_data = area.spaces.active if getattr(area, "spaces", None) else None

    return {
        "window": win,
        "screen": scr,
        "area": area,
        "region": region,
        "space_data": space_data,
        "scene": bpy.context.scene,
        "view_layer": bpy.context.view_layer,
    }


def import_obj(mesh_path: str) -> List[bpy.types.Object]:
    before = set(bpy.data.objects)

    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    override = _build_obj_import_override()
    if hasattr(bpy.ops.wm, "obj_import"):
        try:
            if override:
                with bpy.context.temp_override(**override):
                    bpy.ops.wm.obj_import(filepath=mesh_path)
            else:
                with bpy.context.temp_override(scene=bpy.context.scene, view_layer=bpy.context.view_layer):
                    bpy.ops.wm.obj_import(filepath=mesh_path)
        except Exception:
            if hasattr(bpy.ops.import_scene, "obj"):
                try:
                    if override:
                        with bpy.context.temp_override(**override):
                            bpy.ops.import_scene.obj(filepath=mesh_path)
                    else:
                        bpy.ops.import_scene.obj(filepath=mesh_path)
                except Exception as e2:
                    raise RuntimeError(str(e2))
            else:
                raise RuntimeError("OBJ import failed (no suitable importer / context).")
    else:
        if hasattr(bpy.ops.import_scene, "obj"):
            try:
                if override:
                    with bpy.context.temp_override(**override):
                        bpy.ops.import_scene.obj(filepath=mesh_path)
                else:
                    bpy.ops.import_scene.obj(filepath=mesh_path)
            except Exception as e:
                raise RuntimeError(str(e))
        else:
            raise RuntimeError("No OBJ importer available in this Blender build.")

    after = [o for o in bpy.data.objects if o not in before]
    return after


# ============================================================
# Texture search / indexing
# ============================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".exr", ".webp"}
_IGNORE_DIRS = {".git", "__pycache__", ".venv", "node_modules", "textures_cache"}
_TRASH_TOKENS = {"preview", "render", "thumb", "thumbnail", "icon"}


def _is_hashlike_name(name: str) -> bool:
    if not name:
        return False
    if len(name) < 16:
        return False
    if any(ch.isspace() for ch in name):
        return False
    if re.search(r"[А-Яа-я]", name):
        return False
    alnum = sum(1 for ch in name.lower() if ("a" <= ch <= "z") or ("0" <= ch <= "9"))
    return alnum / max(len(name), 1) >= 0.85


def detect_asset_root(mesh_path: str) -> Path:
    model_dir = Path(mesh_path).resolve().parent
    if _is_hashlike_name(model_dir.name):
        return model_dir.parent
    return model_dir


def build_search_dirs(mesh_path: Optional[str], mesh_texture_dirs: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()

    def add_dir(p: Optional[Path]) -> None:
        if not p:
            return
        try:
            pp = str(p.resolve())
        except Exception:
            return
        if pp in seen:
            return
        if not p.exists() or not p.is_dir():
            return
        out.append(pp)
        seen.add(pp)

    if mesh_path:
        mp = Path(mesh_path).resolve()
        model_dir = mp.parent
        asset_root = detect_asset_root(str(mp))
        add_dir(model_dir)
        add_dir(asset_root)
        add_dir(asset_root.parent)

    for d in (mesh_texture_dirs or []):
        try:
            dd = Path(d).expanduser()
            if not dd.is_absolute() and mesh_path:
                dd = (Path(mesh_path).resolve().parent / dd).resolve()
            add_dir(dd)
        except Exception:
            pass

    return out


def _walk_images_limited(base_dir: str, max_depth: int, max_files: int) -> Iterable[Tuple[str, str]]:
    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        return

    emitted = 0
    base_parts = len(base.resolve().parts)

    for root, dirs, files in os.walk(str(base)):
        try:
            depth = len(Path(root).resolve().parts) - base_parts
        except Exception:
            depth = 0
        if depth > max_depth:
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]

        for fn in files:
            if emitted >= max_files:
                return
            ext = os.path.splitext(fn)[1].lower()
            if ext not in IMG_EXTS:
                continue
            fn_l = fn.lower()
            if any(tok in fn_l for tok in _TRASH_TOKENS):
                continue
            emitted += 1
            yield (fn_l, os.path.join(root, fn))


def build_image_index(search_dirs: List[str], max_depth: int = 4, max_files: int = 15000) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for d in search_dirs:
        for fn_l, full in _walk_images_limited(d, max_depth=max_depth, max_files=max_files):
            if fn_l not in idx:
                idx[fn_l] = full
    return idx


# ============================================================
# MTL parsing / resolving
# ============================================================

_MTL_MAP_RE = re.compile(r"^\s*(map_Kd|map_Bump|bump|map_d|map_Ks|map_Ka)\s+(.*)$", re.IGNORECASE)
_OBJ_MTLLIB_RE = re.compile(r"^\s*mtllib\s+(.*)$", re.IGNORECASE)


def _strip_mtl_opts(s: str) -> str:
    s = s.strip()
    if '"' in s:
        parts = re.findall(r'"([^"]+)"', s)
        if parts:
            return parts[-1].strip()
    toks = s.split()
    if not toks:
        return ""
    return toks[-1].strip()


def parse_obj_mtl_files(obj_path: str) -> List[str]:
    res: List[str] = []
    try:
        lines = Path(obj_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return res

    for line in lines:
        m = _OBJ_MTLLIB_RE.match(line)
        if not m:
            continue
        tail = m.group(1).strip()
        parts = tail.split()
        for p in parts:
            p = p.strip().strip('"').strip("'")
            if p:
                res.append(p)

    out: List[str] = []
    seen = set()
    for x in res:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_mtl_refs(mtl_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        lines = Path(mtl_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out

    for line in lines:
        m = _MTL_MAP_RE.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = _strip_mtl_opts(m.group(2))
        if not val:
            continue

        if key == "map_kd":
            out.setdefault("basecolor_ref", val)
        elif key in ("map_bump", "bump"):
            out.setdefault("bump_ref", val)
        elif key == "map_d":
            out.setdefault("opacity_ref", val)
        elif key == "map_ks":
            out.setdefault("specular_ref", val)
        elif key == "map_ka":
            out.setdefault("ambient_ref", val)

    return out


def parse_mtl_materials(mtl_path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None

    try:
        lines = Path(mtl_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        low = line.lower()
        if low.startswith("newmtl "):
            current = line[7:].strip()
            if current:
                out.setdefault(current, {})
            continue

        if current is None:
            continue

        m = _MTL_MAP_RE.match(line)
        if not m:
            continue

        key = m.group(1).lower()
        val = _strip_mtl_opts(m.group(2))
        if not val:
            continue

        slot = out[current]
        if key == "map_kd":
            slot["basecolor_ref"] = val
        elif key in ("map_bump", "bump"):
            slot["bump_ref"] = val
        elif key == "map_d":
            slot["opacity_ref"] = val
        elif key == "map_ks":
            slot["specular_ref"] = val
        elif key == "map_ka":
            slot["ambient_ref"] = val

    return out


def resolve_texture_ref(ref: str, model_dir: Path, idx: Dict[str, str]) -> Optional[str]:
    if not ref:
        return None
    s = ref.strip().strip('"').strip("'")
    if not s:
        return None

    p = Path(s).expanduser()
    if p.is_absolute() and p.is_file():
        return str(p)

    cand = (model_dir / p).resolve()
    if cand.is_file():
        return str(cand)

    base = os.path.basename(s).lower()
    if base in idx:
        return idx[base]
    return None


# ============================================================
# PBR material building
# ============================================================

def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)


def _auto_color_rgb(key: str) -> Tuple[float, float, float]:
    h = (hash(key) & 0xFFFFFFFF) / 0xFFFFFFFF
    return _hsv_to_rgb(h, 0.45, 0.75)


@dataclass
class PBRMaps:
    basecolor: Optional[str] = None
    normal: Optional[str] = None
    roughness: Optional[str] = None
    metallic: Optional[str] = None
    ao: Optional[str] = None
    height: Optional[str] = None
    emissive: Optional[str] = None
    opacity: Optional[str] = None


def _set_image_colorspace(img: bpy.types.Image, is_data: bool) -> None:
    try:
        img.colorspace_settings.name = "Non-Color" if is_data else "sRGB"
    except Exception:
        pass


def _make_pbr_material(name: str, maps: PBRMaps, tint_rgb: Optional[Tuple[float, float, float]], tex_scale: float) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (980, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (720, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (0, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (220, 0)
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    mapping.inputs["Scale"].default_value = (float(tex_scale), float(tex_scale), float(tex_scale))

    def add_tex_image(path: str, loc: Tuple[int, int], is_data: bool) -> bpy.types.Node:
        n = nodes.new("ShaderNodeTexImage")
        n.location = loc
        try:
            img = bpy.data.images.load(path, check_existing=True)
            n.image = img
            _set_image_colorspace(img, is_data=is_data)
        except Exception:
            pass
        links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    if maps.basecolor:
        base = add_tex_image(maps.basecolor, (440, 240), is_data=False)
        if maps.ao:
            ao = add_tex_image(maps.ao, (440, 460), is_data=True)
            mix = nodes.new("ShaderNodeMixRGB")
            mix.location = (600, 340)
            mix.blend_type = "MULTIPLY"
            mix.inputs["Fac"].default_value = 1.0
            links.new(base.outputs["Color"], mix.inputs["Color1"])
            links.new(ao.outputs["Color"], mix.inputs["Color2"])
            links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    elif tint_rgb:
        bsdf.inputs["Base Color"].default_value = (float(tint_rgb[0]), float(tint_rgb[1]), float(tint_rgb[2]), 1.0)

    if maps.roughness:
        r = add_tex_image(maps.roughness, (440, -40), is_data=True)
        links.new(r.outputs["Color"], bsdf.inputs["Roughness"])

    if maps.metallic:
        m = add_tex_image(maps.metallic, (440, -260), is_data=True)
        links.new(m.outputs["Color"], bsdf.inputs["Metallic"])

    if maps.normal:
        ntex = add_tex_image(maps.normal, (440, -480), is_data=True)
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (620, -480)
        links.new(ntex.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    if maps.height:
        h = add_tex_image(maps.height, (440, -700), is_data=True)
        disp = nodes.new("ShaderNodeDisplacement")
        disp.location = (720, -700)
        links.new(h.outputs["Color"], disp.inputs["Height"])
        links.new(disp.outputs["Displacement"], out.inputs["Displacement"])

    if maps.emissive:
        e = add_tex_image(maps.emissive, (440, 660), is_data=False)
        if "Emission" in bsdf.inputs:
            links.new(e.outputs["Color"], bsdf.inputs["Emission"])
        elif "Emission Color" in bsdf.inputs:
            links.new(e.outputs["Color"], bsdf.inputs["Emission Color"])
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 1.0

    if maps.opacity and "Alpha" in bsdf.inputs:
        o = add_tex_image(maps.opacity, (440, 880), is_data=True)
        links.new(o.outputs["Color"], bsdf.inputs["Alpha"])
        try:
            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
        except Exception:
            pass

    return mat


# ============================================================
# Heuristic guessing
# ============================================================

def _pick_best_match(stem: str, files_lower: Dict[str, str], keys: List[str]) -> Optional[str]:
    stem_l = (stem or "").lower()
    candidates: List[Tuple[int, str]] = []

    for fn_l, full in files_lower.items():
        if stem_l and stem_l not in fn_l:
            continue

        score = 0
        for k in keys:
            if k in fn_l:
                score += 12

        if any(tok in fn_l for tok in _TRASH_TOKENS):
            score -= 50

        score += max(0, 60 - len(fn_l))
        if score > 0:
            candidates.append((score, full))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _guess_maps_from_scan(mesh_path: Optional[str], search_dirs: List[str], explicit_base: Optional[str]) -> PBRMaps:
    idx = build_image_index(search_dirs, max_depth=4, max_files=15000)
    if not idx:
        return PBRMaps(basecolor=(explicit_base if (explicit_base and os.path.isfile(explicit_base)) else None))

    base_keys = ["basecolor", "base_color", "albedo", "diffuse", "dif", "color", "col", "base", "tex", "texture"]
    nor_keys = ["normal", "norm", "nrm", "nmap"]
    rou_keys = ["roughness", "rough", "rgh"]
    met_keys = ["metallic", "metalness", "metal", "mtl", "met"]
    ao_keys = ["ambientocclusion", "ambient_occlusion", "occlusion", "occ", "ao"]
    hgt_keys = ["height", "displacement", "disp"]
    emi_keys = ["emissive", "emission", "emit", "glow"]
    op_keys = ["opacity", "alpha", "transparency"]

    stems: List[str] = []
    if mesh_path:
        mp = Path(mesh_path)
        stems.append(mp.stem)
        ar = detect_asset_root(str(mp))
        if not _is_hashlike_name(ar.name):
            stems.append(ar.name)
    stems.append("")

    out = PBRMaps()

    if explicit_base and os.path.isfile(explicit_base):
        out.basecolor = explicit_base

    for st in stems:
        if not out.basecolor:
            out.basecolor = _pick_best_match(st, idx, base_keys)
        if not out.normal:
            out.normal = _pick_best_match(st, idx, nor_keys)
        if not out.roughness:
            out.roughness = _pick_best_match(st, idx, rou_keys)
        if not out.metallic:
            out.metallic = _pick_best_match(st, idx, met_keys)
        if not out.ao:
            out.ao = _pick_best_match(st, idx, ao_keys)
        if not out.height:
            out.height = _pick_best_match(st, idx, hgt_keys)
        if not out.emissive:
            out.emissive = _pick_best_match(st, idx, emi_keys)
        if not out.opacity:
            out.opacity = _pick_best_match(st, idx, op_keys)

    return out


# ============================================================
# Existing materials
# ============================================================

def _iter_mesh_children(root: bpy.types.Object) -> List[bpy.types.Object]:
    stack = [root]
    out: List[bpy.types.Object] = []
    while stack:
        o = stack.pop()
        if o.type == "MESH":
            out.append(o)
        stack.extend(list(o.children))
    return out


def _object_has_any_material_slots(parent: bpy.types.Object) -> bool:
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        if mats and len(mats) > 0 and any(m is not None for m in mats):
            return True
    return False


def _material_has_effective_basecolor_texture(mat: bpy.types.Material) -> bool:
    if not mat or not getattr(mat, "use_nodes", False) or not mat.node_tree:
        return False

    nt = mat.node_tree
    nodes = nt.nodes

    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False

    base_input = bsdf.inputs.get("Base Color")
    if base_input is None or not base_input.is_linked:
        return False

    visited = set()
    stack = [link.from_node for link in base_input.links]

    while stack:
        node = stack.pop()
        if node is None:
            continue
        ptr = node.as_pointer()
        if ptr in visited:
            continue
        visited.add(ptr)

        if node.type == "TEX_IMAGE":
            img = getattr(node, "image", None)
            if img is None:
                continue
            try:
                if getattr(img, "size", (0, 0))[0] > 0:
                    return True
            except Exception:
                pass
            fp = getattr(img, "filepath", "") or ""
            if fp:
                return True

        for inp in getattr(node, "inputs", []):
            if inp.is_linked:
                for link in inp.links:
                    stack.append(link.from_node)

    return False


def _has_loaded_textures(parent: bpy.types.Object) -> bool:
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        for mat in (mats or []):
            if _material_has_effective_basecolor_texture(mat):
                return True
    return False


def _apply_image_to_material_nodes(
    mat: bpy.types.Material,
    basecolor_img: Optional[bpy.types.Image] = None,
    normal_img: Optional[bpy.types.Image] = None,
    opacity_img: Optional[bpy.types.Image] = None,
) -> None:
    if not mat:
        return

    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        return

    nodes = nt.nodes
    links = nt.links

    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)

    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (500, 0)

    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (180, 0)
        try:
            links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        except Exception:
            pass

    def remove_links_to_socket(socket_name: str) -> None:
        for l in list(links):
            try:
                if l.to_node == bsdf and l.to_socket and l.to_socket.name == socket_name:
                    links.remove(l)
            except Exception:
                pass

    if basecolor_img is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = basecolor_img
        tex.location = (-180, 120)
        try:
            _set_image_colorspace(basecolor_img, is_data=False)
        except Exception:
            pass
        remove_links_to_socket("Base Color")
        try:
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        except Exception:
            pass

    if normal_img is not None:
        tex_n = nodes.new("ShaderNodeTexImage")
        tex_n.image = normal_img
        tex_n.location = (-180, -180)
        try:
            _set_image_colorspace(normal_img, is_data=True)
        except Exception:
            pass

        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (0, -180)

        remove_links_to_socket("Normal")
        try:
            links.new(tex_n.outputs["Color"], nmap.inputs["Color"])
            links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        except Exception:
            pass

    if opacity_img is not None and "Alpha" in bsdf.inputs:
        tex_a = nodes.new("ShaderNodeTexImage")
        tex_a.image = opacity_img
        tex_a.location = (-180, -420)
        try:
            _set_image_colorspace(opacity_img, is_data=True)
        except Exception:
            pass

        remove_links_to_socket("Alpha")
        try:
            links.new(tex_a.outputs["Color"], bsdf.inputs["Alpha"])
            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
        except Exception:
            pass


def _ensure_principled_basecolor_image(mat: bpy.types.Material, img: bpy.types.Image) -> None:
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)

    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (420, 0)

    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (120, 0)
        try:
            links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        except Exception:
            pass

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (-200, 0)

    try:
        for l in list(links):
            if l.to_node == bsdf and l.to_socket and l.to_socket.name == "Base Color":
                links.remove(l)
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    except Exception:
        pass


def _relink_missing_images(parent: bpy.types.Object, idx: Dict[str, str]) -> Tuple[int, List[str]]:
    used: List[str] = []
    fixed = 0
    if not idx:
        return 0, used

    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        for mat in (mats or []):
            if not mat or not getattr(mat, "use_nodes", False) or not mat.node_tree:
                continue
            for n in mat.node_tree.nodes:
                if n.type != "TEX_IMAGE":
                    continue

                img = getattr(n, "image", None)
                if img and getattr(img, "size", (0, 0))[0] > 0:
                    continue

                want = None
                if img and getattr(img, "filepath", ""):
                    want = os.path.basename(img.filepath)
                if not want:
                    want = (getattr(n, "label", "") or getattr(n, "name", "") or "").strip()
                if not want:
                    continue

                cand = idx.get(os.path.basename(want).lower())
                if not cand:
                    continue

                try:
                    new_img = bpy.data.images.load(cand, check_existing=True)
                    n.image = new_img
                    fixed += 1
                    used.append(cand)
                except Exception:
                    pass

    return fixed, used


def _apply_mtl_map_kd_to_existing_mats(parent: bpy.types.Object, mesh_path: str, idx: Dict[str, str], verbose: bool) -> bool:
    obj_path = Path(mesh_path).resolve()
    model_dir = obj_path.parent

    mtl_names = parse_obj_mtl_files(str(obj_path))
    if not mtl_names:
        cand = next(iter(model_dir.glob("*.mtl")), None)
        if cand:
            mtl_names = [cand.name]

    if not mtl_names:
        return False

    merged_mtl: Dict[str, Dict[str, str]] = {}
    for mtl_name in mtl_names:
        mtl_path = (model_dir / mtl_name).resolve()
        if not mtl_path.is_file():
            rp = _resolve_path_maybe(model_dir, mtl_name)
            if rp and Path(rp).is_file():
                mtl_path = Path(rp)
            else:
                continue

        parsed = parse_mtl_materials(str(mtl_path))
        for k, v in parsed.items():
            merged_mtl[k] = v

    if not merged_mtl:
        return False

    def load_img(tex_ref: Optional[str], is_data: bool) -> Optional[bpy.types.Image]:
        if not tex_ref:
            return None
        resolved = resolve_texture_ref(tex_ref, model_dir, idx)
        if not resolved or not os.path.isfile(resolved):
            return None
        try:
            img = bpy.data.images.load(resolved, check_existing=True)
            _set_image_colorspace(img, is_data=is_data)
            return img
        except Exception:
            return None

    applied_any = False

    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        if not mats:
            continue

        for mat in mats:
            if not mat:
                continue

            mtl_info = merged_mtl.get(mat.name)

            if mtl_info is None:
                mat_name_l = mat.name.lower().strip()
                for mk, mv in merged_mtl.items():
                    mk_l = mk.lower().strip()
                    if mk_l == mat_name_l or mk_l in mat_name_l or mat_name_l in mk_l:
                        mtl_info = mv
                        break

            if mtl_info is None:
                continue

            base_img = load_img(mtl_info.get("basecolor_ref"), is_data=False)
            if base_img is None:
                base_img = load_img(mtl_info.get("ambient_ref"), is_data=False)

            normal_img = load_img(mtl_info.get("bump_ref"), is_data=True)
            opacity_img = load_img(mtl_info.get("opacity_ref"), is_data=True)

            if base_img is None and normal_img is None and opacity_img is None:
                continue

            _apply_image_to_material_nodes(
                mat=mat,
                basecolor_img=base_img,
                normal_img=normal_img,
                opacity_img=opacity_img,
            )
            applied_any = True

            if verbose:
                print(
                    f"[MTL] {parent.name} | mat={mat.name} | "
                    f"base={getattr(base_img, 'filepath', None)} | "
                    f"normal={getattr(normal_img, 'filepath', None)} | "
                    f"alpha={getattr(opacity_img, 'filepath', None)}"
                )

    return applied_any


def _fallback_apply_largest_image_existing_mats(parent: bpy.types.Object, idx: Dict[str, str], verbose: bool) -> bool:
    if not idx:
        return False

    best_path = None
    best_size = -1
    for p in idx.values():
        try:
            sz = os.path.getsize(p)
        except Exception:
            continue
        if sz > best_size:
            best_size = sz
            best_path = p

    if not best_path:
        return False

    try:
        img = bpy.data.images.load(best_path, check_existing=True)
    except Exception:
        return False

    applied = False
    for o in _iter_mesh_children(parent):
        me = getattr(o, "data", None)
        mats = getattr(me, "materials", None) if me else None
        for mat in (mats or []):
            if mat:
                _ensure_principled_basecolor_image(mat, img)
                applied = True

    if applied:
        _log(verbose, f"[Textures] {parent.name}: fallback largest={best_path}")
    return applied


def _ensure_textures(
    parent: bpy.types.Object,
    mesh_path: Optional[str],
    mesh_texture_dirs: Optional[List[str]],
    texture_path: Optional[str],
    texture_files: Optional[List[str]],
    texture_scale: float,
    tint_rgb: Optional[Tuple[float, float, float]],
    keep_existing_mats: bool,
    verbose: bool,
) -> None:
    search_dirs = build_search_dirs(mesh_path, mesh_texture_dirs)
    idx = build_image_index(search_dirs, max_depth=4, max_files=15000)

    if keep_existing_mats:
        if _has_loaded_textures(parent):
            _log(verbose, f"[Textures] {parent.name}: effective basecolor texture already exists -> keep")
            return

        fixed, _ = _relink_missing_images(parent, idx)
        if fixed > 0 and _has_loaded_textures(parent):
            _log(verbose, f"[Textures] {parent.name}: relink fixed={fixed}")
            return

        if mesh_path and os.path.isfile(mesh_path):
            if _apply_mtl_map_kd_to_existing_mats(parent, mesh_path, idx, verbose):
                if _has_loaded_textures(parent):
                    _log(verbose, f"[Textures] {parent.name}: restored from MTL")
                    return

        if _object_has_any_material_slots(parent):
            if _fallback_apply_largest_image_existing_mats(parent, idx, verbose):
                if _has_loaded_textures(parent):
                    return
            _log(verbose, f"[Textures] {parent.name}: materials exist, but MTL restore failed")
            return

    maps = PBRMaps()
    explicit_base = texture_path if (texture_path and os.path.isfile(texture_path)) else None

    if mesh_path and os.path.isfile(mesh_path):
        model_dir = Path(mesh_path).resolve().parent
        mtl_names = parse_obj_mtl_files(mesh_path)
        if not mtl_names:
            cand = next(iter(model_dir.glob("*.mtl")), None)
            if cand:
                mtl_names = [cand.name]

        for mtl_name in mtl_names:
            mtl_path = (model_dir / mtl_name).resolve()
            if not mtl_path.is_file():
                rp = _resolve_path_maybe(model_dir, mtl_name)
                if rp and Path(rp).is_file():
                    mtl_path = Path(rp)
                else:
                    continue

            refs = parse_mtl_refs(str(mtl_path))
            if not maps.basecolor and "basecolor_ref" in refs:
                maps.basecolor = resolve_texture_ref(refs["basecolor_ref"], model_dir, idx)
            if not maps.normal and "bump_ref" in refs:
                maps.normal = resolve_texture_ref(refs["bump_ref"], model_dir, idx)
            if not maps.opacity and "opacity_ref" in refs:
                maps.opacity = resolve_texture_ref(refs["opacity_ref"], model_dir, idx)
            if not maps.basecolor and "ambient_ref" in refs:
                maps.basecolor = resolve_texture_ref(refs["ambient_ref"], model_dir, idx)

            if any([maps.basecolor, maps.normal, maps.opacity]):
                break

    base_dir = Path(mesh_path).resolve().parent if mesh_path else Path.cwd()
    resolved_files: List[str] = []
    for tf in (texture_files or []):
        rp = _resolve_path_maybe(base_dir, tf)
        if rp and os.path.isfile(rp):
            resolved_files.append(rp)
        else:
            bn = os.path.basename(str(tf)).lower()
            if bn in idx:
                resolved_files.append(idx[bn])

    for p in resolved_files:
        bn = os.path.basename(p).lower()
        if not maps.basecolor and any(k in bn for k in ["basecolor", "base_color", "albedo", "diffuse", "color", "col", "tex", "texture"]):
            maps.basecolor = p
        if not maps.normal and any(k in bn for k in ["normal", "norm", "nrm", "nmap"]):
            maps.normal = p
        if not maps.roughness and any(k in bn for k in ["roughness", "rough", "rgh"]):
            maps.roughness = p
        if not maps.metallic and any(k in bn for k in ["metallic", "metalness", "metal", "mtl", "met"]):
            maps.metallic = p
        if not maps.ao and any(k in bn for k in ["ao", "occlusion", "occ", "ambientocclusion"]):
            maps.ao = p

    if not maps.basecolor and explicit_base:
        maps.basecolor = explicit_base

    if not any([maps.basecolor, maps.normal, maps.roughness, maps.metallic, maps.ao, maps.height, maps.emissive, maps.opacity]):
        maps = _guess_maps_from_scan(mesh_path, search_dirs, explicit_base=explicit_base)
    else:
        guessed = _guess_maps_from_scan(mesh_path, search_dirs, explicit_base=explicit_base)
        maps.roughness = maps.roughness or guessed.roughness
        maps.metallic = maps.metallic or guessed.metallic
        maps.ao = maps.ao or guessed.ao
        maps.height = maps.height or guessed.height
        maps.emissive = maps.emissive or guessed.emissive
        maps.opacity = maps.opacity or guessed.opacity

    _log(verbose, f"[Textures] {parent.name}: build PBR maps={maps}")

    mat = _make_pbr_material(
        name=f"Mat_{parent.name}",
        maps=maps,
        tint_rgb=tint_rgb,
        tex_scale=float(texture_scale or 1.0),
    )

    for o in _iter_mesh_children(parent):
        me = o.data
        if not me.materials:
            me.materials.append(mat)
        else:
            for i in range(len(me.materials)):
                me.materials[i] = mat


# ============================================================
# Room environment textures
# ============================================================

def _list_env_textures(env_dir: str) -> Dict[str, List[str]]:
    d = str(Path(env_dir).expanduser().resolve())
    exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.tif", "*.tiff"]

    def collect(prefixes: List[str]) -> List[str]:
        out: List[str] = []
        for pref in prefixes:
            for e in exts:
                out.extend(glob(os.path.join(d, f"{pref}{e}")))
        out = sorted({str(Path(p).resolve()) for p in out})
        return out

    return {
        "floor": collect(["floor_", "floor"]),
        "wall": collect(["wall_", "wall"]),
        "window": collect(["window_", "window"]),
        "door": collect(["door_", "door"]),
    }


def _choose_texture(kind: str, candidates: List[str], style_text: Optional[str], rng: random.Random) -> Optional[str]:
    del kind
    del style_text
    if not candidates:
        return None
    return rng.choice(candidates)


def _make_image_material(name: str, image_path: str, uv_scale: float = 1.0, is_data: bool = False) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (450, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (0, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (170, 0)
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    mapping.inputs["Scale"].default_value = (float(uv_scale), float(uv_scale), float(uv_scale))

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (340, 0)
    try:
        img = bpy.data.images.load(image_path, check_existing=True)
        tex.image = img
        _set_image_colorspace(img, is_data=is_data)
    except Exception:
        pass
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.85
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.10

    return mat


def _assign_material_to_object(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.type != "MESH":
        return
    me = obj.data
    try:
        me.materials.clear()
    except Exception:
        pass
    if len(me.materials) == 0:
        me.materials.append(mat)
    else:
        for i in range(len(me.materials)):
            me.materials[i] = mat


def _make_glass_material(name: str, image_path: Optional[str]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (430, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 1.0

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.05
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.45

    if image_path and os.path.isfile(image_path):
        texcoord = nodes.new("ShaderNodeTexCoord")
        texcoord.location = (0, 0)

        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (170, 0)
        links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])

        tex = nodes.new("ShaderNodeTexImage")
        tex.location = (340, 0)
        img = bpy.data.images.load(image_path, check_existing=True)
        tex.image = img
        _set_image_colorspace(img, is_data=False)

        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

        if "Alpha" in bsdf.inputs:
            links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            try:
                mat.blend_method = "HASHED"
                mat.shadow_method = "HASHED"
            except Exception:
                pass
        else:
            try:
                mat.blend_method = "HASHED"
                mat.shadow_method = "HASHED"
            except Exception:
                pass

    return mat


# ============================================================
# Room spec mesh building
# ============================================================

def _poly_signed_area_xy(pts: List[Tuple[float, float]]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _normalize2(x: float, y: float) -> Tuple[float, float]:
    l = math.hypot(x, y)
    if l < 1e-12:
        return (1.0, 0.0)
    return (x / l, y / l)


def _make_mesh_object(name: str, verts: List[Tuple[float, float, float]], faces: List[Tuple[int, ...]], coll) -> bpy.types.Object:
    me = bpy.data.meshes.new(name + "_MESH")
    me.from_pydata(verts, [], faces)
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    _unlink_from_all_collections(obj)
    coll.objects.link(obj)
    return obj


def _ensure_uv_layer(me: bpy.types.Mesh, name: str = "UVMap") -> None:
    if not me.uv_layers:
        me.uv_layers.new(name=name)


def _set_uvs_floor_xy(obj: bpy.types.Object, uv_scale: float = 1.0) -> None:
    me = obj.data
    _ensure_uv_layer(me)
    uv = me.uv_layers.active.data

    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            x, y, _z = me.vertices[vi].co
            uv[li].uv = (x * uv_scale, y * uv_scale)


def _set_uvs_wall_sz(obj: bpy.types.Object, origin_xy: Tuple[float, float], dir_xy: Tuple[float, float], uv_scale: float = 1.0) -> None:
    me = obj.data
    _ensure_uv_layer(me)
    uv = me.uv_layers.active.data

    ox, oy = origin_xy
    dx, dy = dir_xy

    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            x, y, z = me.vertices[vi].co
            s = (x - ox) * dx + (y - oy) * dy
            uv[li].uv = (s * uv_scale, z * uv_scale)


def _make_floor_from_polygon(name: str, poly_xy: List[Tuple[float, float]], z: float, coll) -> bpy.types.Object:
    bm = bmesh.new()
    verts = [bm.verts.new((x, y, z)) for (x, y) in poly_xy]
    bm.faces.new(verts)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    me = bpy.data.meshes.new(name + "_MESH")
    bm.to_mesh(me)
    bm.free()
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    _unlink_from_all_collections(obj)
    coll.objects.link(obj)
    return obj


def _make_wall_quad(name: str, p0: Tuple[float, float], p1: Tuple[float, float], z0: float, z1: float, coll) -> bpy.types.Object:
    x0, y0 = p0
    x1, y1 = p1
    verts = [
        (x0, y0, z0),
        (x1, y1, z0),
        (x1, y1, z1),
        (x0, y0, z1),
    ]
    faces = [(0, 1, 2, 3)]
    return _make_mesh_object(name, verts, faces, coll)


def _make_decal_on_wall(
    name: str,
    wall_p0: Tuple[float, float],
    wall_p1: Tuple[float, float],
    inward_n: Tuple[float, float],
    s0: float,
    width: float,
    z0: float,
    height: float,
    coll,
    offset: float = 0.002,
) -> bpy.types.Object:
    x0, y0 = wall_p0
    x1, y1 = wall_p1
    ex, ey = (x1 - x0, y1 - y0)
    ex, ey = _normalize2(ex, ey)

    sc = s0 + 0.5 * width
    cx = x0 + ex * sc
    cy = y0 + ey * sc

    nx, ny = inward_n
    cx += nx * offset
    cy += ny * offset

    u0 = -0.5 * width
    u1 = +0.5 * width
    v0 = z0
    v1 = z0 + height

    verts = [
        (cx + ex * u0, cy + ey * u0, v0),
        (cx + ex * u1, cy + ey * u1, v0),
        (cx + ex * u1, cy + ey * u1, v1),
        (cx + ex * u0, cy + ey * u0, v1),
    ]
    faces = [(0, 1, 2, 3)]
    return _make_mesh_object(name, verts, faces, coll)


def build_room_from_spec(
    room_json: dict,
    coll_room: bpy.types.Collection,
    env_textures_dir: str,
    style_text: Optional[str],
    seed: int,
    verbose: bool,
) -> Tuple[List[bpy.types.Object], mathutils.Vector, mathutils.Vector]:
    r = room_json["room"]
    H = float(r.get("ceiling_height", 2.8))

    poly = r["floor_polygon"]
    poly_xy = [(float(p["x"]), float(p["y"])) for p in poly]

    area = _poly_signed_area_xy(poly_xy)
    ccw = (area > 0.0)

    floor_tex = wall_tex = door_tex = window_tex = None
    if env_textures_dir and os.path.isdir(env_textures_dir):
        tex = _list_env_textures(env_textures_dir)
        rng = random.Random(int(seed) if int(seed) != 0 else (hash(json.dumps(r, sort_keys=True)) & 0xFFFFFFFF))

        floor_tex = _choose_texture("floor", tex.get("floor", []), style_text, rng)
        wall_tex = _choose_texture("wall", tex.get("wall", []), style_text, rng)
        door_tex = _choose_texture("door", tex.get("door", []), style_text, rng)
        window_tex = _choose_texture("window", tex.get("window", []), style_text, rng)

    if verbose:
        print(f"[RoomSpec] ccw={ccw} H={H}")
        print(f"[RoomSpec] floor_tex={floor_tex}")
        print(f"[RoomSpec] wall_tex ={wall_tex}")
        print(f"[RoomSpec] door_tex ={door_tex}")
        print(f"[RoomSpec] win_tex  ={window_tex}")

    floor_mat = _make_image_material("MAT_ROOM_FLOOR", floor_tex, uv_scale=1.0) if floor_tex else None
    wall_mat = _make_image_material("MAT_ROOM_WALL", wall_tex, uv_scale=1.0) if wall_tex else None
    door_mat = _make_image_material("MAT_ROOM_DOOR", door_tex, uv_scale=1.0) if door_tex else None
    win_mat = _make_glass_material("MAT_ROOM_WINDOW", window_tex) if window_tex else _make_glass_material("MAT_ROOM_WINDOW", None)

    out_objs: List[bpy.types.Object] = []

    floor_obj = _make_floor_from_polygon("Room_Floor", poly_xy, z=0.0, coll=coll_room)
    _set_uvs_floor_xy(floor_obj, uv_scale=1.0)
    if floor_mat:
        _assign_material_to_object(floor_obj, floor_mat)
    out_objs.append(floor_obj)

    walls = r.get("walls", [])
    wall_by_id = {}

    for w in walls:
        wid = w["id"]
        i0 = int(w["from_vertex"])
        i1 = int(w["to_vertex"])
        p0 = poly_xy[i0]
        p1 = poly_xy[i1]

        ex, ey = (p1[0] - p0[0], p1[1] - p0[1])
        ex, ey = _normalize2(ex, ey)

        if ccw:
            nx, ny = (-ey, ex)
        else:
            nx, ny = (ey, -ex)

        wall_obj = _make_wall_quad(f"Room_Wall_{wid}", p0, p1, z0=0.0, z1=H, coll=coll_room)
        _set_uvs_wall_sz(wall_obj, origin_xy=p0, dir_xy=(ex, ey), uv_scale=1.0)
        if wall_mat:
            _assign_material_to_object(wall_obj, wall_mat)
        out_objs.append(wall_obj)

        wall_by_id[wid] = {
            "p0": p0, "p1": p1,
            "dir": (ex, ey),
            "in": (nx, ny),
        }

    for d in r.get("doors", []):
        wid = d["wall_id"]
        info = wall_by_id.get(wid)
        if not info:
            continue

        s0 = float(d["s"])
        width = float(d["width"])
        z0 = float(d.get("z0", 0.0))
        height = float(d.get("height", 2.0))

        door_obj = _make_decal_on_wall(
            name=f"Room_Door_{d['id']}",
            wall_p0=info["p0"],
            wall_p1=info["p1"],
            inward_n=info["in"],
            s0=s0,
            width=width,
            z0=z0,
            height=height,
            coll=coll_room,
            offset=0.003,
        )
        _set_uvs_wall_sz(door_obj, origin_xy=info["p0"], dir_xy=info["dir"], uv_scale=1.0)
        if door_mat:
            _assign_material_to_object(door_obj, door_mat)
        out_objs.append(door_obj)

    for w in r.get("windows", []):
        wid = w["wall_id"]
        info = wall_by_id.get(wid)
        if not info:
            continue

        s0 = float(w["s"])
        width = float(w["width"])
        z0 = float(w.get("z0", 0.9))
        height = float(w.get("height", 1.2))

        win_obj = _make_decal_on_wall(
            name=f"Room_Window_{w['id']}",
            wall_p0=info["p0"],
            wall_p1=info["p1"],
            inward_n=info["in"],
            s0=s0,
            width=width,
            z0=z0,
            height=height,
            coll=coll_room,
            offset=0.004,
        )
        _set_uvs_wall_sz(win_obj, origin_xy=info["p0"], dir_xy=info["dir"], uv_scale=1.0)
        if win_mat:
            _assign_material_to_object(win_obj, win_mat)
        out_objs.append(win_obj)

    bmin, bmax = _world_bounds_mesh_objects(out_objs)
    return out_objs, bmin, bmax


# ============================================================
# Placement
# ============================================================

def place_in_aabb(
    objs: List[bpy.types.Object],
    aabb: Dict[str, float],
    rotation_deg_engine: float,
    fit_mode: str,
    parent_name: str,
    collection: bpy.types.Collection,
    snap_to_floor: bool,
    floor_offset: float,
    snap_to_ceiling: bool = False,
    ceiling_offset: float = 0.0,
) -> Optional[bpy.types.Object]:
    mesh_objs = [o for o in objs if o.type == "MESH"]
    if not mesh_objs:
        return None

    parent = bpy.data.objects.new(parent_name, None)
    bpy.context.scene.collection.objects.link(parent)

    for o in objs:
        _unlink_from_all_collections(o)
        collection.objects.link(o)
        o.parent = parent

    bpy.context.view_layer.update()

    parent.rotation_euler = (0.0, 0.0, math.radians(float(rotation_deg_engine or 0.0)))
    bpy.context.view_layer.update()

    bmin, bmax = _world_bounds_mesh_objects(mesh_objs)
    cur = bmax - bmin

    tgt = mathutils.Vector(
        (
            float(aabb["x_max"]) - float(aabb["x_min"]),
            float(aabb["y_max"]) - float(aabb["y_min"]),
            float(aabb["z_max"]) - float(aabb["z_min"]),
        )
    )

    eps = 1e-9
    cur = mathutils.Vector((max(cur.x, eps), max(cur.y, eps), max(cur.z, eps)))
    sx, sy, sz = tgt.x / cur.x, tgt.y / cur.y, tgt.z / cur.z

    if (fit_mode or "stretch").lower() == "uniform":
        k = min(sx, sy, sz)
        parent.scale = (k, k, k)
    else:
        parent.scale = (sx, sy, sz)

    bpy.context.view_layer.update()

    bmin2, bmax2 = _world_bounds_mesh_objects(mesh_objs)
    cur_center = (bmin2 + bmax2) * 0.5
    tgt_center = mathutils.Vector(
        (
            0.5 * (float(aabb["x_min"]) + float(aabb["x_max"])),
            0.5 * (float(aabb["y_min"]) + float(aabb["y_max"])),
            0.5 * (float(aabb["z_min"]) + float(aabb["z_max"])),
        )
    )
    delta = tgt_center - cur_center

    if snap_to_floor and not snap_to_ceiling:
        bottom_after_center = bmin2.z + delta.z
        delta.z += (float(aabb["z_min"]) - bottom_after_center) + float(floor_offset)
    elif snap_to_ceiling and not snap_to_floor:
        top_after_center = bmax2.z + delta.z
        delta.z += (float(aabb["z_max"]) - top_after_center) - float(ceiling_offset)

    parent.location += delta
    bpy.context.view_layer.update()
    return parent


# ============================================================
# Main build
# ============================================================

def build_scene(
    glb_path: str,
    json_path: str,
    import_glb: bool,
    draw_aabb: bool,
    force_tint: bool,
    keep_existing_mats: bool,
    verbose: bool,
    env_textures_dir: str,
    style_text: Optional[str],
    seed: int,
) -> None:
    reset_scene()
    _ensure_world()

    json_file = Path(json_path).expanduser().resolve()
    json_dir = json_file.parent

    with open(str(json_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    is_room_spec = False
    try:
        is_room_spec = (
            isinstance(data, dict)
            and "room" in data
            and isinstance(data["room"], dict)
            and "floor_polygon" in data["room"]
            and "walls" in data["room"]
        )
    except Exception:
        is_room_spec = False

    coll_room = ensure_collection("Room")
    coll_items = ensure_collection("Items")

    room_ok = False
    room_objs: List[bpy.types.Object] = []
    glb_bb_min: Optional[mathutils.Vector] = None
    glb_bb_max: Optional[mathutils.Vector] = None
    room_meshes: List[bpy.types.Object] = []

    if "items" in data:
        room_engine = data.get("room") or {}
        items = data["items"] or []
    else:
        room_engine = data.get("room") or data.get("room_spec") or {}
        items = data.get("placements") or data.get("items") or []

    if is_room_spec:
        try:
            room_objs, bb_min, bb_max = build_room_from_spec(
                room_json=data,
                coll_room=coll_room,
                env_textures_dir=env_textures_dir,
                style_text=style_text,
                seed=seed,
                verbose=verbose,
            )
            room_ok = True
            glb_bb_min = bb_min.copy()
            glb_bb_max = bb_max.copy()
            z_floor = float(bb_min.z)
            z_ceil = float(bb_max.z)
        except Exception as e:
            room_ok = False
            room_objs = []
            bb_min = mathutils.Vector((0.0, 0.0, 0.0))
            bb_max = mathutils.Vector((5.0, 5.0, 3.0))
            z_floor = float(bb_min.z)
            z_ceil = float(bb_max.z)
            _log(verbose, f"[RoomSpec] build failed: {e}")
    else:
        if import_glb and os.path.isfile(glb_path):
            try:
                room_objs = import_glb_into_collection(glb_path, coll_room)
                room_ok = len(room_objs) > 0
                _log(verbose, f"[Room] Imported GLB objects: {len(room_objs)}")
            except Exception as e:
                print("Импорт GLB комнаты не удался:", e)
                room_ok = False
                room_objs = []

        if room_ok:
            room_meshes = [o for o in room_objs if o.type == "MESH"]
            glb_bb_min, glb_bb_max = _world_bounds_mesh_objects(room_meshes if room_meshes else room_objs)

        if room_ok and glb_bb_min is not None and glb_bb_max is not None:
            bb_min = glb_bb_min.copy()
            bb_max = glb_bb_max.copy()
        elif room_engine and all(k in room_engine for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]):
            x1, x2 = float(room_engine["x_min"]), float(room_engine["x_max"])
            y1, y2 = float(room_engine["y_min"]), float(room_engine["y_max"])
            z1, z2 = float(room_engine["z_min"]), float(room_engine["z_max"])
            bb_min = mathutils.Vector((x1, y1, z1))
            bb_max = mathutils.Vector((x2, y2, z2))
        else:
            bb_min = mathutils.Vector((0.0, 0.0, 0.0))
            bb_max = mathutils.Vector((5.0, 5.0, 3.0))

        z_floor = float(bb_min.z)
        z_ceil = float(bb_max.z)

        floor_mat = None
        wall_mat = None
        seed_eff = (hash(str(json_file)) & 0xFFFFFFFF) if int(seed) == 0 else int(seed)

        if env_textures_dir and os.path.isdir(env_textures_dir):
            try:
                tex = _list_env_textures(env_textures_dir)
                rng = random.Random(seed_eff)

                floor_path = _choose_texture("floor", tex.get("floor", []), style_text, rng)
                wall_path = _choose_texture("wall", tex.get("wall", []), style_text, rng)

                floor_mat = _make_image_material("MAT_ENV_FLOOR", floor_path, uv_scale=1.0) if floor_path else None
                wall_mat = _make_image_material("MAT_ENV_WALL", wall_path, uv_scale=1.0) if wall_path else None

                _log(verbose, f"[EnvTextures] floor={floor_path}")
                _log(verbose, f"[EnvTextures] wall ={wall_path}")
            except Exception as e:
                _log(verbose, f"[EnvTextures] selection/materials failed: {e}")

        preview_created = False
        try:
            dx = float(bb_max.x - bb_min.x)
            dy = float(bb_max.y - bb_min.y)
            dz = float(bb_max.z - bb_min.z)

            cx = float((bb_min.x + bb_max.x) * 0.5)
            cy = float((bb_min.y + bb_max.y) * 0.5)
            cz = float((bb_min.z + bb_max.z) * 0.5)

            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(cx, cy, float(bb_min.z) + 1e-4))
            floor = bpy.context.active_object
            floor.name = "Preview_Floor"
            floor.scale = (dx * 0.5, dy * 0.5, 1.0)
            _unlink_from_all_collections(floor)
            coll_room.objects.link(floor)
            if floor_mat:
                _assign_material_to_object(floor, floor_mat)

            bpy.ops.mesh.primitive_plane_add(
                size=1.0,
                location=(float(bb_min.x) + 1e-4, cy, cz),
                rotation=(0.0, math.radians(90.0), 0.0),
            )
            left = bpy.context.active_object
            left.name = "Preview_Wall_Left"
            left.scale = (dz * 0.5, dy * 0.5, 1.0)
            _unlink_from_all_collections(left)
            coll_room.objects.link(left)
            if wall_mat:
                _assign_material_to_object(left, wall_mat)

            bpy.ops.mesh.primitive_plane_add(
                size=1.0,
                location=(cx, float(bb_min.y) + 1e-4, cz),
                rotation=(math.radians(90.0), 0.0, 0.0),
            )
            front = bpy.context.active_object
            front.name = "Preview_Wall_Front"
            front.scale = (dx * 0.5, dz * 0.5, 1.0)
            _unlink_from_all_collections(front)
            coll_room.objects.link(front)
            if wall_mat:
                _assign_material_to_object(front, wall_mat)

            preview_created = True
        except Exception as e:
            _log(verbose, f"[PreviewRoom] failed: {e}")

        if room_ok and preview_created:
            for o in room_meshes:
                try:
                    o.hide_viewport = True
                    o.hide_render = True
                except Exception:
                    pass
        elif (not room_ok) and draw_aabb and room_engine:
            _add_aabb_box(room_engine, "Room", coll_room)

    def _quantize_rot_0_90_180_270(deg: float) -> float:
        a = float(deg or 0.0) % 360.0
        allowed = (0.0, 90.0, 180.0, 270.0)
        best = min(allowed, key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t))
        return best

    for it in items:
        aabb_eng = dict(it.get("aabb") or it.get("bbox") or {})
        if not aabb_eng:
            _log(verbose, f"⚠️ item without aabb: {it}")
            continue

        name = _item_name(it)
        constraints = it.get("constraints") or {}
        name_l = name.lower()

        is_ceiling_item = (
            constraints.get("mount_type") == "ceiling"
            or constraints.get("under_ceiling")
            or "lamp" in name_l
            or "light" in name_l
            or "люстр" in name_l
            or "светиль" in name_l
        )

        is_floor_item = (
            constraints.get("mount_type") == "floor"
            or (isinstance(constraints.get("touch_floor"), dict) and constraints["touch_floor"].get("side") == "bottom")
            or (not is_ceiling_item)
        )

        if glb_bb_min is not None and glb_bb_max is not None:
            try:
                aabb_eng = clamp_item_aabb_to_room_glb(aabb_eng, glb_bb_min, glb_bb_max, margin=0.05)
            except Exception as e:
                _log(verbose, f"[Clamp] failed for {name}: {e}")

        if is_floor_item:
            if float(aabb_eng.get("z_min", 0.0)) < float(z_floor):
                dz_fix = float(z_floor) - float(aabb_eng["z_min"])
                aabb_eng["z_min"] = float(z_floor)
                aabb_eng["z_max"] = float(aabb_eng["z_max"]) + dz_fix

            sz = float(aabb_eng["z_max"]) - float(aabb_eng["z_min"])
            aabb_eng["z_min"] = float(z_floor)
            aabb_eng["z_max"] = float(z_floor) + sz
        elif is_ceiling_item:
            sz = float(aabb_eng["z_max"]) - float(aabb_eng["z_min"])
            aabb_eng["z_max"] = float(z_ceil)
            aabb_eng["z_min"] = float(z_ceil) - sz

        rot_raw = float(it.get("rotation", it.get("yaw_deg", it.get("rotation_deg", 0.0))) or 0.0)
        rot_deg = _quantize_rot_0_90_180_270(rot_raw)

        fit_mode = _item_mesh_fit_mode(it)

        mesh_path_raw = _item_mesh_path_raw(it)
        mesh_path = _resolve_path_maybe(json_dir, mesh_path_raw)

        mesh_tex_dirs_raw = _item_mesh_texture_dirs_raw(it)
        mesh_tex_dirs: List[str] = []
        for d in mesh_tex_dirs_raw:
            rr = _resolve_path_maybe(json_dir, str(d))
            if rr:
                mesh_tex_dirs.append(rr)

        texture_path = _resolve_path_maybe(json_dir, it.get("texture_path"))
        texture_files = [str(x) for x in (it.get("texture_files") or [])]

        texture_scale = float(it.get("texture_scale", 1.0) or 1.0)
        color = it.get("color") or _auto_color_rgb(str(name))
        tint_rgb = (float(color[0]), float(color[1]), float(color[2]))

        placed_ok = False

        if mesh_path and os.path.isfile(mesh_path):
            print(f"[DBG] placement name={name}")
            print(f"[DBG] mesh_path={mesh_path}")
            ext = os.path.splitext(mesh_path)[1].lower()
            if ext != ".obj":
                print(f"⚠️ {name}: mesh_path не OBJ ({ext}). Сейчас поддерживается только .obj: {mesh_path}")
            else:
                try:
                    before_names = set(o.name for o in bpy.data.objects)
                    objs = import_obj(mesh_path)
                    after_names = set(o.name for o in bpy.data.objects)
                    created_names = sorted(after_names - before_names)
                    print(f"[DBG] import_obj returned {0 if objs is None else len(objs)} objects for {name}")
                    print(f"[DBG] bpy created {len(created_names)} objects for {name}: {created_names[:20]}")
                except Exception as e:
                    print(f"⚠️ {name}: import OBJ failed: {e}")
                    objs = []

                if objs:
                    parent = place_in_aabb(
                        objs=objs,
                        aabb=aabb_eng,
                        rotation_deg_engine=rot_deg,
                        fit_mode=fit_mode,
                        parent_name=name,
                        collection=coll_items,
                        snap_to_floor=is_floor_item,
                        floor_offset=(-0.001 if is_floor_item else 0.0),
                        snap_to_ceiling=is_ceiling_item,
                        ceiling_offset=0.0,
                    )
                    placed_ok = parent is not None

                    if parent is not None:
                        if force_tint:
                            mat = _make_pbr_material(
                                name=f"Mat_{name}_Tint",
                                maps=PBRMaps(),
                                tint_rgb=tint_rgb,
                                tex_scale=1.0,
                            )
                            for o in _iter_mesh_children(parent):
                                me = o.data
                                if not me.materials:
                                    me.materials.append(mat)
                                else:
                                    for i in range(len(me.materials)):
                                        me.materials[i] = mat
                        else:
                            _ensure_textures(
                                parent=parent,
                                mesh_path=mesh_path,
                                mesh_texture_dirs=mesh_tex_dirs,
                                texture_path=texture_path,
                                texture_files=texture_files,
                                texture_scale=texture_scale,
                                tint_rgb=tint_rgb,
                                keep_existing_mats=keep_existing_mats,
                                verbose=verbose,
                            )
                else:
                    print(f"⚠️ {name}: Не удалось импортировать модель: {mesh_path}")
        else:
            print(f"⚠️ {name}: mesh_path не найден в placement/asset или файл отсутствует: {mesh_path_raw}")

        if (not placed_ok) and draw_aabb:
            _add_aabb_box(aabb_eng, name, coll_items)

    _frame_camera_on_bounds(bb_min, bb_max)
    _add_basic_lights(bb_min, bb_max)
    _force_material_preview_if_ui()


# ============================================================
# RUN
# ============================================================

def main() -> None:
    args = _parse_argv(sys.argv)

    glb_path = str(Path(args.glb).expanduser().resolve())
    json_path = str(Path(args.json).expanduser().resolve())

    keep_existing = not (args.rebuild_materials or args.no_keep_existing_mats)

    build_scene(
        glb_path=glb_path,
        json_path=json_path,
        import_glb=bool(args.import_glb),
        draw_aabb=bool(args.draw_aabb),
        force_tint=bool(args.force_tint),
        keep_existing_mats=keep_existing,
        verbose=bool(args.verbose),
        env_textures_dir=str(Path(args.env_textures_dir).expanduser().resolve()),
        style_text=args.style_text,
        seed=int(args.seed or 0),
    )

    if not args.no_pack_assets:
        _pack_assets_best_effort()

    if args.save_blend:
        out_blend = str(Path(args.save_blend).expanduser().resolve())
        os.makedirs(os.path.dirname(out_blend), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)

    if args.render:
        out_png = str(Path(args.render).expanduser().resolve())
        os.makedirs(os.path.dirname(out_png), exist_ok=True)

        scene = bpy.context.scene
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = out_png

        try:
            scene.render.film_transparent = False
        except Exception:
            pass

        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)