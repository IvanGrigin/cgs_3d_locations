# cgs_blender_build_from_obj.py
# Full Blender scene build from OBJ path (import, textures, scale/center, cage, UI, frame view).
# Intended to be imported by BlenderOrientItem.py

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Vector

# ----------------------------- Tunables -----------------------------

CAGE_SIZE_M = 5.0
CAGE_HALF = CAGE_SIZE_M * 0.5

ITEM_TINT_RGBA: Tuple[float, float, float, float] = (1.0, 0.78, 0.82, 1.0)  # pink if no textures

WALL_ALPHA_SIDE = 0.25
WALL_ALPHA_TOPBOT = 0.20

DEFAULT_FRONT_CLEARANCE_M = 0.60

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".exr", ".webp"}

# ----------------------------- Small helpers -----------------------------

def _normalize_probs(p: Dict[str, float], fallback: Dict[str, float]) -> Dict[str, float]:
    keys = ["back", "left", "right", "front"]
    cleaned = {k: max(0.0, float(p.get(k, 0.0))) for k in keys}
    s = sum(cleaned.values())
    if s <= 0.0:
        return fallback.copy()
    return {k: cleaned[k] / s for k in keys}

def _find_override(prefer_view3d: bool = True) -> dict:
    wm = bpy.context.window_manager
    if not wm or not wm.windows:
        return {}
    win = wm.windows[0]
    scr = win.screen
    if not scr:
        return {"window": win}

    areas = list(scr.areas)
    if prefer_view3d:
        areas.sort(key=lambda a: 0 if a.type == "VIEW_3D" else 1)

    for area in areas:
        for region in area.regions:
            if region.type == "WINDOW":
                return {
                    "window": win,
                    "screen": scr,
                    "area": area,
                    "region": region,
                    "scene": bpy.context.scene,
                    "view_layer": bpy.context.view_layer,
                }
    return {"window": win, "screen": scr, "scene": bpy.context.scene, "view_layer": bpy.context.view_layer}

def _op_in_view3d(op_callable, **kwargs):
    ov = _find_override(prefer_view3d=True)
    if ov:
        with bpy.context.temp_override(**ov):
            return op_callable(**kwargs)
    return op_callable(**kwargs)

def _force_material_preview() -> None:
    wm = bpy.context.window_manager
    if not wm or not wm.windows:
        return
    scr = wm.windows[0].screen
    if not scr:
        return
    for area in scr.areas:
        if area.type != "VIEW_3D":
            continue
        sp = area.spaces.active
        if not hasattr(sp, "shading"):
            continue
        try:
            sp.shading.type = "MATERIAL"
            if hasattr(sp.shading, "color_type"):
                sp.shading.color_type = "MATERIAL"
            if hasattr(sp.shading, "use_scene_lights"):
                sp.shading.use_scene_lights = True
            if hasattr(sp.shading, "use_scene_world"):
                sp.shading.use_scene_world = False
        except Exception:
            pass

def _set_view_clipping(max_dim: float) -> None:
    wm = bpy.context.window_manager
    if not wm or not wm.windows:
        return
    scr = wm.windows[0].screen
    if not scr:
        return

    clip_end = max(1000.0, float(max_dim) * 20.0)
    for area in scr.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            try:
                if hasattr(space, "clip_start"):
                    space.clip_start = 0.001
                if hasattr(space, "clip_end"):
                    space.clip_end = clip_end
            except Exception:
                pass

# ----------------------------- Geometry helpers -----------------------------

def _world_bbox_from_eval_mesh(obj: bpy.types.Object) -> Tuple[Vector, Vector]:
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)

    mesh = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=deps)
    try:
        if not mesh or not mesh.vertices:
            z = Vector((0.0, 0.0, 0.0))
            return z, z

        mw = obj_eval.matrix_world
        xmin = ymin = zmin = float("inf")
        xmax = ymax = zmax = float("-inf")

        for v in mesh.vertices:
            p = mw @ v.co
            xmin = min(xmin, p.x); xmax = max(xmax, p.x)
            ymin = min(ymin, p.y); ymax = max(ymax, p.y)
            zmin = min(zmin, p.z); zmax = max(zmax, p.z)

        return Vector((xmin, ymin, zmin)), Vector((xmax, ymax, zmax))
    finally:
        try:
            obj_eval.to_mesh_clear()
        except Exception:
            pass

def _bbox_dims(min_v: Vector, max_v: Vector) -> Vector:
    return Vector((max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z))

def _guess_unit_scale_to_meters(max_dim_raw: float) -> Tuple[int, float, str]:
    if max_dim_raw <= 0.0:
        return 0, 1.0, "unknown"
    if max_dim_raw >= 100.0:
        return 3, 1e-3, "mm"
    if max_dim_raw >= 10.0:
        return 2, 1e-2, "cm"
    return 0, 1.0, "m"

# ----------------------------- Materials / textures -----------------------------

def _make_material(name: str, rgba: Tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    try:
        mat.use_nodes = True
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links

        for n in list(nodes):
            nodes.remove(n)

        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (320, 0)

        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (0, 0)

        principled.inputs["Base Color"].default_value = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
        if "Alpha" in principled.inputs:
            principled.inputs["Alpha"].default_value = float(rgba[3])

        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = 0.55
        if "Specular" in principled.inputs:
            principled.inputs["Specular"].default_value = 0.25

        links.new(principled.outputs["BSDF"], out.inputs["Surface"])

        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND" if rgba[3] < 0.999 else "OPAQUE"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "NONE" if rgba[3] < 0.999 else "OPAQUE"
    except Exception:
        pass

    return mat

def _force_tint_on_item(item: bpy.types.Object, rgba: Tuple[float, float, float, float]) -> None:
    tint = _make_material("MAT_ITEM_PINK", rgba)

    targets: List[bpy.types.Object] = []
    if item.type == "MESH":
        targets.append(item)
    targets.extend([ch for ch in item.children_recursive if ch.type == "MESH"])

    for o in targets:
        me = getattr(o, "data", None)
        if not me or not hasattr(me, "materials"):
            continue
        if len(me.materials) == 0:
            me.materials.append(tint)
        else:
            for i in range(len(me.materials)):
                me.materials[i] = tint

def _iter_mesh_objects(item: bpy.types.Object) -> List[bpy.types.Object]:
    out: List[bpy.types.Object] = []
    if item and item.type == "MESH":
        out.append(item)
    out.extend([ch for ch in item.children_recursive if ch.type == "MESH"])
    return out

def _collect_image_nodes(item: bpy.types.Object) -> List[bpy.types.Node]:
    nodes_out: List[bpy.types.Node] = []
    for o in _iter_mesh_objects(item):
        me = getattr(o, "data", None)
        if not me or not hasattr(me, "materials"):
            continue
        for mat in me.materials:
            if not mat or not getattr(mat, "use_nodes", False) or not mat.node_tree:
                continue
            for n in mat.node_tree.nodes:
                if n and n.type == "TEX_IMAGE":
                    nodes_out.append(n)
    return nodes_out

def _has_loaded_textures(item: bpy.types.Object) -> bool:
    for n in _collect_image_nodes(item):
        img = getattr(n, "image", None)
        if img and getattr(img, "size", (0, 0))[0] > 0:
            return True
    return False

def _build_image_index(search_dirs: List[Path]) -> Dict[str, Path]:
    name_to_path: Dict[str, Path] = {}
    for d in search_dirs:
        if not d or not d.exists():
            continue
        try:
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    name_to_path.setdefault(p.name.lower(), p)
        except Exception:
            pass
    return name_to_path

def _try_reload_missing_images(item: bpy.types.Object, search_dirs: List[Path]) -> Tuple[bool, List[str], Optional[str]]:
    loaded: List[str] = []
    try:
        nodes = _collect_image_nodes(item)
        if not nodes:
            return False, loaded, None

        idx = _build_image_index(search_dirs)

        any_loaded = False
        for n in nodes:
            img = getattr(n, "image", None)
            if img and getattr(img, "size", (0, 0))[0] > 0:
                any_loaded = True
                continue

            want_name = None
            if img and getattr(img, "filepath", ""):
                want_name = Path(img.filepath).name
            if not want_name and getattr(n, "label", ""):
                want_name = n.label
            if not want_name and getattr(n, "name", ""):
                want_name = n.name

            if not want_name:
                continue

            cand = idx.get(Path(want_name).name.lower())
            if not cand:
                continue

            try:
                new_img = bpy.data.images.load(str(cand), check_existing=True)
                n.image = new_img
                any_loaded = True
                loaded.append(str(cand))
            except Exception:
                continue

        return any_loaded, loaded, None
    except Exception as e:
        return False, loaded, repr(e)

def _parse_mtllib_from_obj(obj_path: Path) -> Optional[str]:
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(600):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.lower().startswith("mtllib "):
                    return s.split(None, 1)[1].strip()
    except Exception:
        return None
    return None

def _parse_map_kd_from_mtl(mtl_path: Path) -> List[str]:
    tex = []
    try:
        with mtl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                low = s.lower()
                if low.startswith("map_kd "):
                    parts = s.split()
                    for token in reversed(parts[1:]):
                        if "." in token:
                            tex.append(token.strip('"'))
                            break
    except Exception:
        pass
    return tex

def _ensure_principled_basecolor_image(mat: bpy.types.Material, img: bpy.types.Image) -> None:
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    out = None
    principled = None
    for n in nodes:
        if n.type == "OUTPUT_MATERIAL":
            out = n
        elif n.type == "BSDF_PRINCIPLED":
            principled = n
    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (420, 0)
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (120, 0)
        try:
            links.new(principled.outputs["BSDF"], out.inputs["Surface"])
        except Exception:
            pass

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    tex_node.location = (-200, 0)

    try:
        for l in list(links):
            if l.to_node == principled and l.to_socket and l.to_socket.name == "Base Color":
                links.remove(l)
        links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])
    except Exception:
        pass

def _try_apply_textures_from_mtl(obj_path: Path, item: bpy.types.Object, search_dirs: List[Path]) -> Tuple[bool, List[str], Optional[str]]:
    applied: List[str] = []
    try:
        mtllib = _parse_mtllib_from_obj(obj_path)
        if not mtllib:
            return False, applied, None

        mtl_path = obj_path.parent / mtllib
        if not mtl_path.exists():
            found = []
            for d in search_dirs:
                if d and d.exists():
                    found.extend(list(d.rglob(Path(mtllib).name)))
            if found:
                mtl_path = found[0]
        if not mtl_path.exists():
            return False, applied, None

        tex_names = _parse_map_kd_from_mtl(mtl_path)
        if not tex_names:
            return False, applied, None

        img_idx = _build_image_index(search_dirs)

        chosen_img = None
        chosen_path = None
        for tn in tex_names:
            cand = img_idx.get(Path(tn).name.lower())
            if cand and cand.exists():
                chosen_path = cand
                try:
                    chosen_img = bpy.data.images.load(str(cand), check_existing=True)
                except Exception:
                    chosen_img = None
                if chosen_img:
                    break

        if not chosen_img:
            return False, applied, None

        for o in _iter_mesh_objects(item):
            me = getattr(o, "data", None)
            if not me or not hasattr(me, "materials"):
                continue
            for mat in me.materials:
                if not mat:
                    continue
                _ensure_principled_basecolor_image(mat, chosen_img)

        applied.append(str(chosen_path))
        return True, applied, None
    except Exception as e:
        return False, applied, repr(e)

def _fallback_apply_largest_image(item: bpy.types.Object, search_dirs: List[Path]) -> Tuple[bool, List[str], Optional[str]]:
    picked: List[str] = []
    try:
        imgs: List[Path] = []
        for d in search_dirs:
            if not d or not d.exists():
                continue
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    imgs.append(p)
        if not imgs:
            return False, picked, None

        imgs.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        best = imgs[0]

        try:
            img = bpy.data.images.load(str(best), check_existing=True)
        except Exception:
            return False, picked, None

        for o in _iter_mesh_objects(item):
            me = getattr(o, "data", None)
            if not me or not hasattr(me, "materials"):
                continue
            if len(me.materials) == 0:
                me.materials.append(bpy.data.materials.new(name="MAT_AUTO"))
            for mat in me.materials:
                if not mat:
                    continue
                _ensure_principled_basecolor_image(mat, img)

        picked.append(str(best))
        return True, picked, None
    except Exception as e:
        return False, picked, repr(e)

def try_apply_textures(obj_path: Path, item: bpy.types.Object) -> Tuple[bool, List[str], Optional[str]]:
    search_dirs = [obj_path.parent]
    if obj_path.parent.parent:
        search_dirs.append(obj_path.parent.parent)

    if _has_loaded_textures(item):
        return True, [], None

    _, files1, err1 = _try_reload_missing_images(item, search_dirs)
    if _has_loaded_textures(item):
        return True, files1, err1

    _, files2, err2 = _try_apply_textures_from_mtl(obj_path, item, search_dirs)
    if _has_loaded_textures(item):
        return True, files1 + files2, err2 or err1

    _, files3, err3 = _fallback_apply_largest_image(item, search_dirs)
    if _has_loaded_textures(item):
        return True, files1 + files2 + files3, err3 or err2 or err1

    return False, files1 + files2 + files3, err3 or err2 or err1

# ----------------------------- Scene ops -----------------------------

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

def import_obj(obj_path: Path) -> List[bpy.types.Object]:
    obj_path = obj_path.resolve()
    if not obj_path.exists():
        raise FileNotFoundError(str(obj_path))

    bpy.ops.object.select_all(action="DESELECT")
    before = set(bpy.data.objects)

    if not hasattr(bpy.ops.wm, "obj_import"):
        raise RuntimeError("Ожидался Blender 5.x с оператором bpy.ops.wm.obj_import")

    override = _find_override(prefer_view3d=True)
    with bpy.context.temp_override(**override):
        bpy.ops.wm.obj_import(filepath=str(obj_path))

    after = set(bpy.data.objects)
    imported = list(after - before)
    view_objs = set(bpy.context.view_layer.objects)
    return [o for o in imported if o in view_objs]

def join_meshes(objs: List[bpy.types.Object], name: str = "ITEM") -> bpy.types.Object:
    meshes = [o for o in objs if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("После импорта OBJ не найдено MESH объектов.")

    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    if len(meshes) > 1:
        ov = _find_override(prefer_view3d=True)
        with bpy.context.temp_override(**ov):
            bpy.ops.object.join()

    item = bpy.context.view_layer.objects.active
    item.name = name
    return item

def apply_scale_center_apply(item: bpy.types.Object) -> Tuple[int, float, str, Vector, Vector]:
    bpy.context.view_layer.update()

    min0, max0 = _world_bbox_from_eval_mesh(item)
    dims0 = _bbox_dims(min0, max0)
    max_dim0 = float(max(dims0.x, dims0.y, dims0.z))

    k, s, unit = _guess_unit_scale_to_meters(max_dim0)

    if s != 1.0:
        item.scale = item.scale * float(s)

    bpy.context.view_layer.objects.active = item
    ov = _find_override(prefer_view3d=True)
    with bpy.context.temp_override(**ov):
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.context.view_layer.update()
    min1, max1 = _world_bbox_from_eval_mesh(item)
    center = (min1 + max1) * 0.5
    item.location = item.location - center

    with bpy.context.temp_override(**ov):
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    bpy.context.view_layer.update()
    min2, max2 = _world_bbox_from_eval_mesh(item)
    dims2 = _bbox_dims(min2, max2)

    return k, float(s), unit, dims0, dims2

def _ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def _relink_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    for c in list(obj.users_collection):
        try:
            c.objects.unlink(obj)
        except Exception:
            pass
    try:
        col.objects.link(obj)
    except Exception:
        pass

# ----------------------------- Cage -----------------------------

def _add_wall_plane(
    name: str,
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    _op_in_view3d(bpy.ops.mesh.primitive_plane_add, size=CAGE_SIZE_M, location=location, rotation=rotation)
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    if obj.data and hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    _relink_to_collection(obj, collection)
    return obj

def _add_text_label(
    text: str,
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    _op_in_view3d(bpy.ops.object.text_add, location=location, rotation=rotation)
    t = bpy.context.view_layer.objects.active
    t.data.body = text
    t.data.size = 0.18
    t.name = f"LBL_{text}"
    _relink_to_collection(t, collection)
    return t

def build_cage() -> None:
    cage = _ensure_collection("ORIENT_CAGE")

    mats = {
        "BACK":  _make_material("MAT_BACK",  (1.0, 0.2, 0.2, WALL_ALPHA_SIDE)),
        "FRONT": _make_material("MAT_FRONT", (0.2, 1.0, 0.2, WALL_ALPHA_SIDE)),
        "LEFT":  _make_material("MAT_LEFT",  (0.2, 0.4, 1.0, WALL_ALPHA_SIDE)),
        "RIGHT": _make_material("MAT_RIGHT", (1.0, 1.0, 0.2, WALL_ALPHA_SIDE)),
        "TOP":   _make_material("MAT_TOP",   (1.0, 0.2, 1.0, WALL_ALPHA_TOPBOT)),
        "BOT":   _make_material("MAT_BOT",   (0.2, 1.0, 1.0, WALL_ALPHA_TOPBOT)),
    }

    _add_wall_plane("WALL_TOP",    (0, 0,  CAGE_HALF), (0, 0, 0), mats["TOP"], cage)
    _add_wall_plane("WALL_BOTTOM", (0, 0, -CAGE_HALF), (math.pi, 0, 0), mats["BOT"], cage)

    _add_wall_plane("WALL_FRONT", (0,  CAGE_HALF, 0), (-math.pi / 2, 0, 0), mats["FRONT"], cage)
    _add_wall_plane("WALL_BACK",  (0, -CAGE_HALF, 0), ( math.pi / 2, 0, 0), mats["BACK"], cage)

    _add_wall_plane("WALL_RIGHT", ( CAGE_HALF, 0, 0), (0, -math.pi / 2, 0), mats["RIGHT"], cage)
    _add_wall_plane("WALL_LEFT",  (-CAGE_HALF, 0, 0), (0,  math.pi / 2, 0), mats["LEFT"], cage)

    _add_text_label("FRONT (+Y)", (0,  CAGE_HALF * 0.92, 0.0), (0, 0, 0), cage)
    _add_text_label("BACK (-Y)",  (0, -CAGE_HALF * 0.92, 0.0), (0, math.pi, 0), cage)
    _add_text_label("RIGHT (+X)", (CAGE_HALF * 0.92, 0, 0.0), (0, -math.pi/2, 0), cage)
    _add_text_label("LEFT (-X)",  (-CAGE_HALF * 0.92, 0, 0.0), (0, math.pi/2, 0), cage)
    _add_text_label("TOP (+Z)",   (0, 0, CAGE_HALF * 0.92), (math.pi/2, 0, 0), cage)
    _add_text_label("BOTTOM (-Z)",(0, 0, -CAGE_HALF * 0.92), (-math.pi/2, 0, 0), cage)

def frame_view_on_object(obj: bpy.types.Object) -> None:
    ov = _find_override(prefer_view3d=True)
    try:
        with bpy.context.temp_override(**ov):
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            if hasattr(bpy.ops.view3d, "view_selected"):
                bpy.ops.view3d.view_selected()
    except Exception:
        pass

# ----------------------------- UI -----------------------------

class CGSOrientSettings(bpy.types.PropertyGroup):
    wall_required: bpy.props.BoolProperty(name="Wall contact required", default=True)
    floor_required: bpy.props.BoolProperty(name="Floor contact required", default=True)
    ceiling_required: bpy.props.BoolProperty(name="Ceiling contact required", default=False)
    front_clearance: bpy.props.FloatProperty(name="Front clearance (m)", default=DEFAULT_FRONT_CLEARANCE_M, min=0.0)
    p_back: bpy.props.FloatProperty(name="P(back)", default=0.8, min=0.0)
    p_left: bpy.props.FloatProperty(name="P(left)", default=0.1, min=0.0)
    p_right: bpy.props.FloatProperty(name="P(right)", default=0.1, min=0.0)
    p_front: bpy.props.FloatProperty(name="P(front)", default=0.0, min=0.0)

class CGS_PT_OrientPanel(bpy.types.Panel):
    bl_label = "CGS Orient"
    bl_idname = "CGS_PT_orient_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CGS Orient"

    def draw(self, context):
        s = context.scene.cgs_orient_settings
        layout = self.layout
        layout.label(text="Axis / walls mapping (world):")
        layout.label(text="FRONT=+Y (green), BACK=-Y (red)")
        layout.label(text="RIGHT=+X (yellow), LEFT=-X (blue)")
        layout.label(text="TOP=+Z (magenta), BOTTOM=-Z (cyan)")
        layout.separator()
        layout.label(text="Required contacts (wizard will ask too)")
        layout.prop(s, "wall_required")
        layout.prop(s, "floor_required")
        layout.prop(s, "ceiling_required")
        layout.separator()
        layout.label(text="Wall probabilities (wizard will ask too)")
        row = layout.row(align=True)
        row.prop(s, "p_back")
        row.prop(s, "p_left")
        row = layout.row(align=True)
        row.prop(s, "p_right")
        row.prop(s, "p_front")
        layout.separator()
        layout.prop(s, "front_clearance")

_CLASSES = [CGSOrientSettings, CGS_PT_OrientPanel]

def register_ui() -> None:
    for c in _CLASSES[::-1]:
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
    for c in _CLASSES:
        try:
            bpy.utils.register_class(c)
        except Exception:
            pass
    try:
        bpy.types.Scene.cgs_orient_settings
    except Exception:
        bpy.types.Scene.cgs_orient_settings = bpy.props.PointerProperty(type=CGSOrientSettings)

# ----------------------------- Public API -----------------------------

def build_scene_from_obj(obj_path: Path) -> Dict[str, object]:
    """
    Полное построение сцены в Blender по OBJ.
    Возвращает словарь init_meta, который можно сохранить/использовать дальше.
    """
    reset_scene()

    imported = import_obj(obj_path)
    item = join_meshes(imported, name="ITEM")

    has_tex, tex_files, tex_err = try_apply_textures(obj_path, item)
    if not has_tex:
        _force_tint_on_item(item, ITEM_TINT_RGBA)
        print("[CGS] No textures applied -> tinted pink.")
    else:
        print("[CGS] Textures detected/applied -> keep materials.")

    k, s, unit, dims0, dims2 = apply_scale_center_apply(item)

    max_dim_after = float(max(dims2.x, dims2.y, dims2.z))
    _set_view_clipping(max_dim_after)

    build_cage()
    register_ui()

    ui = bpy.context.scene.cgs_orient_settings
    ui.wall_required = True
    ui.floor_required = True
    ui.ceiling_required = False
    ui.front_clearance = DEFAULT_FRONT_CLEARANCE_M
    ui.p_back = 0.8
    ui.p_left = 0.1
    ui.p_right = 0.1
    ui.p_front = 0.0

    _force_material_preview()
    frame_view_on_object(item)

    init_meta = {
        "item_source": str(obj_path),
        "item_name": obj_path.stem,
        "scale_power10_k": int(k),
        "scale_factor": float(s),
        "unit_guess": str(unit),
        "bbox_before_raw": (float(dims0.x), float(dims0.y), float(dims0.z)),
        "bbox_after_m": (float(dims2.x), float(dims2.y), float(dims2.z)),
        "has_base_textures": bool(has_tex),
        "texture_files": list(tex_files or []),
        "texture_apply_error": tex_err,
    }

    print("\n[CGS] Scene ready.")
    print("[CGS] Axis mapping (world):")
    print("  FRONT  = +Y (green wall)")
    print("  BACK   = -Y (red wall)")
    print("  RIGHT  = +X (yellow wall)")
    print("  LEFT   = -X (blue wall)")
    print("  TOP    = +Z (magenta wall)")
    print("  BOTTOM = -Z (cyan wall)\n")
    print("[CGS] Import OK")
    print(f"[CGS] BBox before (raw): {dims0.x:.3f} {dims0.y:.3f} {dims0.z:.3f}")
    print(f"[CGS] Unit guess: {unit} -> meters via factor={s} (10^-{k})")
    print(f"[CGS] BBox after (m): {dims2.x:.3f} {dims2.y:.3f} {dims2.z:.3f}\n")

    return init_meta
