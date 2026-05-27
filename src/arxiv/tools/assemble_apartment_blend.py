#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import bpy
import mathutils


_BLENDER_BUILDER_MODULE = None

ROOM_BLEND_CANDIDATES = (
    "pipeline/optimal/scene_infinigen_clean_supplier.requirements.blend",
    "pipeline/optimal/scene_kitchen_requirements.blend",
    "pipeline/optimal/scene_infinigen_clean_supplier.optimal.memfix.blend",
    "pipeline/optimal/scene_infinigen_clean_supplier.optimal.blend",
    "kitchen/{room_id}.blend",
)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.materials,
        bpy.data.images,
    ):
        for item in list(block):
            if getattr(item, "users", 0) == 0:
                block.remove(item)


def room_scene_path(room_dir: Path, mode: str) -> Path:
    return room_dir / "pipeline" / mode / "scene_requirements.v1.json"


def find_room_blend(room_dir: Path, room_id: str, mode: str) -> Path | None:
    for rel in ROOM_BLEND_CANDIDATES:
        path = room_dir / rel.format(room_id=room_id)
        if path.is_file():
            return path
    return None


def apt_min_from_scene(apartment_scene: Path | None, apartment_json: Path) -> tuple[float, float]:
    if apartment_scene and apartment_scene.is_file():
        data = read_json(apartment_scene)
        raw = ((data.get("meta") or {}).get("apartment_global_min_xy") or [])
        if isinstance(raw, list) and len(raw) >= 2:
            return float(raw[0]), float(raw[1])
    apt = read_json(apartment_json)
    poly = (apt.get("room") or {}).get("floor_polygon") or []
    if not poly:
        return 0.0, 0.0
    return (
        min(float(p.get("x", 0.0)) for p in poly if isinstance(p, dict)),
        min(float(p.get("y", 0.0)) for p in poly if isinstance(p, dict)),
    )


def transform_from_frame(frame: dict, apt_min: tuple[float, float]) -> mathutils.Matrix:
    off = frame.get("offset_xy") or [0.0, 0.0]
    origin = frame.get("origin_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or math.radians(float(frame.get("rotation_deg") or 0.0)))
    return (
        mathutils.Matrix.Translation((float(origin[0]) - apt_min[0], float(origin[1]) - apt_min[1], 0.0))
        @ mathutils.Matrix.Rotation(angle, 4, "Z")
        @ mathutils.Matrix.Translation((-float(off[0]), -float(off[1]), 0.0))
    )


def link_loaded_objects(objects: list[bpy.types.Object], collection: bpy.types.Collection) -> None:
    for obj in objects:
        if obj is None:
            continue
        if not any(existing == obj for existing in collection.objects):
            collection.objects.link(obj)


def rename_room_objects(objects: list[bpy.types.Object], room_id: str) -> None:
    for obj in objects:
        if obj is None:
            continue
        obj.name = f"{room_id}__{obj.name}"
        if obj.data is not None and hasattr(obj.data, "name"):
            obj.data.name = f"{room_id}__{obj.data.name}"


def looks_like_architectural_door_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    if any(
        token in low
        for token in (
            "base_",
            "upper_",
            "cabinet",
            "wardrobe",
            "cupboard",
            "drawer",
            "fridge",
            "dishwasher",
            "washing",
            "appliance",
        )
    ) and any(token in low for token in ("_door_", "door_line", "door_recess")):
        return False
    return (
        low in {"door", "doors"}
        or low.endswith("_door")
        or low.endswith("__door")
        or low.endswith(".door")
        or low.endswith("/door")
        or low.endswith("_doors")
        or low.endswith("__doors")
        or low.endswith(".doors")
        or low.endswith("/doors")
        or low.startswith("door_")
        or low.startswith("doors_")
        or "_door_" in low
        or "_doors_" in low
        or "__door." in low
        or "__doors." in low
        or ".door." in low
        or ".doors." in low
        or "door_internal" in low
        or "interior_door" in low
        or "entry_door" in low
        or "door_frame" in low
        or "doorframe" in low
        or "door_leaf" in low
        or "doorleaf" in low
        or "door_panel" in low
        or "slidingdoor" in low
        or "sliding_door" in low
        or "doorfactory" in low
        or "paneldoorfactory" in low
        or "louverdoorfactory" in low
        or "glasspaneldoorfactory" in low
    )


def strip_blender_numeric_suffix(name: str) -> str:
    return re.sub(r"\.\d{3}$", "", str(name or "").strip())


def generic_object_basename(name: str) -> str:
    low = str(name or "").strip().lower()
    low = re.split(r"[\\/]", low)[-1]
    if "__" in low:
        low = low.rsplit("__", 1)[-1]
    return strip_blender_numeric_suffix(low)


def looks_like_orphan_architectural_door_panel(obj: bpy.types.Object) -> bool:
    if getattr(obj, "type", None) != "MESH":
        return False
    text = " ".join(
        str(part or "").lower()
        for part in (
            getattr(obj, "name", ""),
            getattr(getattr(obj, "data", None), "name", ""),
            getattr(getattr(obj, "parent", None), "name", ""),
        )
    )
    if any(
        token in text
        for token in (
            "room_wall",
            "room_floor",
            "room_window",
            "supplieroverlay",
            "skirting",
            "base_",
            "upper_",
            "cabinet",
            "wardrobe",
            "cupboard",
            "drawer",
            "shelf",
            "bookstack",
            "fridge",
            "dishwasher",
            "appliance",
            "countertop",
        )
    ):
        return False
    base = generic_object_basename(getattr(obj, "name", ""))
    data_base = generic_object_basename(getattr(getattr(obj, "data", None), "name", ""))
    if base not in {"cube", "plane"} and data_base not in {"cube", "plane"}:
        return False
    try:
        dims = obj.dimensions
        width = max(float(dims.x), float(dims.y))
        thickness = min(float(dims.x), float(dims.y))
        height = float(dims.z)
    except Exception:
        return False
    return 1.65 <= height <= 3.15 and 0.55 <= width <= 1.35 and thickness <= 0.22


def looks_like_bbox_helper_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    return (
        "bbox_placeholder" in low
        or "spawn_placeholder" in low
        or low.endswith("_aabb")
        or "_aabb." in low
        or low.startswith("invalid_")
        or "__invalid_" in low
    )


def looks_like_room_wrapper_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    if any(token in low for token in ("ceilinglight", "ceiling_light", "lamp_ceiling", "flat_ceiling_light")):
        return False
    if any(token in low for token in ("room_floor_supplier", "room_wall_", "room_window_", "room_door_")):
        return False
    patterns = (
        r"(^|__)[a-z_]+_0/0(\.\d+)?$",
        r"(^|[./_])ceiling($|[./_0-9])",
        r"(^|[./_])exterior($|[./_0-9])",
        r"(^|[./_])meshed($|[./_0-9])",
        r"(^|[./_])room_shell($|[./_0-9])",
        r"(^|[./_])shell($|[./_0-9])",
    )
    return any(re.search(pattern, low) for pattern in patterns)


def looks_like_kitchen_wallpaper_overlay(name: str) -> bool:
    low = str(name or "").strip().lower()
    return "kitchen" in low and "room_wallpaper_supplieroverlay" in low


def looks_like_functional_light_helper(obj: bpy.types.Object) -> bool:
    low = str(getattr(obj, "name", "") or "").lower()
    return bool(obj.get("cgs_functional_light")) or "cgs_functionallight_" in low


def looks_like_window_cover_block(obj: bpy.types.Object) -> bool:
    if getattr(obj, "type", None) != "MESH":
        return False
    text = " ".join(
        str(part or "").lower()
        for part in (
            getattr(obj, "name", ""),
            getattr(getattr(obj, "data", None), "name", ""),
            getattr(getattr(obj, "parent", None), "name", ""),
        )
    )
    if "curtain" in text or "preview_window" in text:
        return False
    if not any(token in text for token in ("room_window", "__window", ".window", "_window_")):
        return False
    try:
        dims = obj.dimensions
        return max(float(dims.x), float(dims.y), float(dims.z)) >= 0.45
    except Exception:
        return True


def remove_object_family(root: bpy.types.Object) -> int:
    removed = 0
    stack = [root]
    seen: set[bpy.types.Object] = set()
    while stack:
        obj = stack.pop()
        if obj is None or obj in seen:
            continue
        seen.add(obj)
        stack.extend(list(obj.children))
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        except Exception:
            hide_object_family(obj)
    return removed


def hide_single_object(obj: bpy.types.Object) -> int:
    try:
        obj.hide_set(True)
    except Exception:
        pass
    try:
        obj.hide_viewport = True
    except Exception:
        pass
    try:
        obj.hide_render = True
    except Exception:
        pass
    return 1


def image_has_real_pixels(img: bpy.types.Image | None) -> bool:
    if img is None:
        return False
    try:
        name = str(getattr(img, "name", "") or "").strip().lower()
        filepath = str(getattr(img, "filepath", "") or "").strip()
        filepath_raw = str(getattr(img, "filepath_raw", "") or "").strip()
    except Exception:
        return False
    if name.startswith("map #") and not filepath and not filepath_raw:
        return False
    if filepath.startswith("/Map #") or filepath_raw.startswith("/Map #"):
        return False
    try:
        if getattr(img, "packed_file", None) is not None:
            return True
    except Exception:
        pass
    try:
        if str(getattr(img, "source", "") or "").upper() in {"GENERATED", "VIEWER"} and getattr(img, "size", (0, 0))[0] > 0:
            return True
    except Exception:
        pass
    fp = filepath or filepath_raw
    if fp:
        try:
            return os.path.isfile(bpy.path.abspath(fp))
        except Exception:
            return os.path.isfile(fp)
    try:
        return bool(getattr(img, "has_data", False)) and getattr(img, "size", (0, 0))[0] > 0
    except Exception:
        return False


def image_looks_non_color_texture(img: bpy.types.Image | None) -> bool:
    if img is None:
        return False
    text = " ".join(
        str(part or "").lower()
        for part in (
            getattr(img, "name", ""),
            getattr(img, "filepath", ""),
            getattr(img, "filepath_raw", ""),
        )
    )
    if any(
        token in text
        for token in (
            "normal",
            "roughness",
            "metallic",
            "metalness",
            "ambientocclusion",
            "ambient_occlusion",
            "occlusion",
            "displacement",
            "height",
            "bump",
            "opacity",
            "alpha",
            "specular",
            "gloss",
        )
    ):
        return True
    stem = Path(str(getattr(img, "filepath", "") or getattr(img, "name", "") or "")).stem.lower()
    return bool(re.search(r"(^|[_\-.])(n|nor|norm|normal|r|rough|roughness|m|metal|metallic|ao|h|height|bump|disp|opacity|alpha)([_\-.]|$)", stem))


def socket_chain_image_state(socket) -> tuple[bool, bool]:
    if socket is None or not getattr(socket, "is_linked", False):
        return False, False
    has_real = False
    has_missing = False
    visited: set[int] = set()
    stack = [link.from_node for link in socket.links]
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
            if image_has_real_pixels(img) and not image_looks_non_color_texture(img):
                has_real = True
            elif img is not None:
                has_missing = True
        for inp in getattr(node, "inputs", []):
            if inp.is_linked:
                for link in inp.links:
                    stack.append(link.from_node)
    return has_real, has_missing


def material_basecolor_state(mat: bpy.types.Material) -> tuple[bool, bool]:
    if not mat or not getattr(mat, "use_nodes", False) or not mat.node_tree:
        return False, False
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False, False
    base_input = bsdf.inputs.get("Base Color")
    return socket_chain_image_state(base_input)


def material_looks_magenta_missing(mat: bpy.types.Material) -> bool:
    try:
        r, g, b, _a = [float(x) for x in mat.diffuse_color[:4]]
    except Exception:
        return False
    return r > 0.62 and b > 0.62 and g < 0.38


def neutral_missing_texture_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("MAT_CGS_MISSING_TEXTURE_NEUTRAL_GRAY")
    if mat is None:
        mat = bpy.data.materials.new("MAT_CGS_MISSING_TEXTURE_NEUTRAL_GRAY")
    mat.diffuse_color = (0.62, 0.64, 0.65, 1.0)
    mat.use_nodes = True
    if mat.node_tree:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        try:
            bsdf.inputs["Base Color"].default_value = (0.62, 0.64, 0.65, 1.0)
            bsdf.inputs["Metallic"].default_value = 0.12
            bsdf.inputs["Roughness"].default_value = 0.48
        except Exception:
            pass
        try:
            links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        except Exception:
            pass
    return mat


def replace_missing_texture_materials(objects: list[bpy.types.Object]) -> int:
    neutral = neutral_missing_texture_material()
    changed = 0
    seen_meshes: set[int] = set()
    for obj in objects:
        if obj is None:
            continue
        stack = [obj]
        while stack:
            cur = stack.pop()
            stack.extend(list(cur.children))
            if cur.type != "MESH":
                continue
            ptr = cur.as_pointer()
            if ptr in seen_meshes:
                continue
            seen_meshes.add(ptr)
            mats = getattr(getattr(cur, "data", None), "materials", None)
            if not mats:
                continue
            for idx, mat in enumerate(list(mats)):
                if mat is None:
                    continue
                has_real, has_missing = material_basecolor_state(mat)
                if has_real:
                    continue
                if not (has_missing or material_looks_magenta_missing(mat)):
                    continue
                mats[idx] = neutral
                changed += 1
    return changed


def cleanup_room_visual_helpers(objects: list[bpy.types.Object]) -> dict[str, int]:
    report = {
        "hidden_architectural_door_objects": 0,
        "hidden_window_cover_objects": 0,
        "removed_bbox_helper_objects": 0,
        "removed_functional_light_helpers": 0,
        "preserved_functional_light_objects": 0,
        "hidden_room_wrapper_objects": 0,
        "replaced_missing_texture_materials": 0,
    }
    report["replaced_missing_texture_materials"] = replace_missing_texture_materials(objects)
    for obj in list(objects):
        if obj.name not in bpy.data.objects:
            continue
        if looks_like_functional_light_helper(obj):
            if getattr(obj, "type", None) == "LIGHT":
                try:
                    obj.hide_viewport = True
                    obj.hide_render = False
                except Exception:
                    pass
                report["preserved_functional_light_objects"] += 1
            else:
                report["removed_functional_light_helpers"] += remove_object_family(obj)
            continue
        if looks_like_window_cover_block(obj):
            report["hidden_window_cover_objects"] += hide_object_family(obj)
            continue
        if looks_like_kitchen_wallpaper_overlay(obj.name):
            report["hidden_room_wrapper_objects"] += hide_single_object(obj)
            continue
        if looks_like_room_wrapper_name(obj.name):
            report["hidden_room_wrapper_objects"] += hide_single_object(obj)
            continue
        if looks_like_architectural_door_name(obj.name) or looks_like_orphan_architectural_door_panel(obj):
            report["hidden_architectural_door_objects"] += hide_object_family(obj)
            continue
        if looks_like_bbox_helper_name(obj.name):
            report["removed_bbox_helper_objects"] += remove_object_family(obj)
            continue
    return report


def make_simple_material(name: str, rgba: tuple[float, float, float, float], *, emissive: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    if emissive <= 0.0:
        return mat
    mat.use_nodes = True
    if mat.node_tree:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = rgba
        emission.inputs["Strength"].default_value = emissive
        links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def point_xy(point: dict) -> tuple[float, float] | None:
    if not isinstance(point, dict):
        return None
    try:
        return float(point.get("x")), float(point.get("y", point.get("z")))
    except Exception:
        return None


def room_polygon_xy(room: dict) -> list[tuple[float, float]]:
    raw = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    poly = [xy for point in raw if (xy := point_xy(point)) is not None]
    if len(poly) >= 3:
        return poly
    width = float(room.get("width_m") or room.get("width") or 3.2)
    depth = float(room.get("depth_m") or room.get("depth") or 3.0)
    return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]


def polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    if not poly:
        return 0.0, 0.0
    return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)


def room_walls(room: dict) -> list[dict]:
    poly = room_polygon_xy(room)
    centroid = mathutils.Vector((polygon_centroid(poly)[0], polygon_centroid(poly)[1], 0.0))
    raw_walls = room.get("walls") if isinstance(room.get("walls"), list) else []
    if not raw_walls:
        raw_walls = [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % len(poly)} for i in range(len(poly))]
    walls: list[dict] = []
    for idx, wall in enumerate(raw_walls):
        if not isinstance(wall, dict):
            continue
        try:
            ai = int(wall.get("from_vertex", idx))
            bi = int(wall.get("to_vertex", (idx + 1) % len(poly)))
            ax, ay = poly[ai]
            bx, by = poly[bi]
        except Exception:
            continue
        vec = mathutils.Vector((bx - ax, by - ay, 0.0))
        length = float(vec.length)
        if length <= 1e-5:
            continue
        tangent = vec.normalized()
        normal_a = mathutils.Vector((-tangent.y, tangent.x, 0.0))
        midpoint = mathutils.Vector(((ax + bx) * 0.5, (ay + by) * 0.5, 0.0))
        inward = normal_a if (midpoint + normal_a * 0.1 - centroid).length < (midpoint - normal_a * 0.1 - centroid).length else -normal_a
        walls.append(
            {
                "id": str(wall.get("id") or f"w{idx}"),
                "a": mathutils.Vector((ax, ay, 0.0)),
                "b": mathutils.Vector((bx, by, 0.0)),
                "length": length,
                "tangent": tangent,
                "normal": inward.normalized(),
            }
        )
    return walls


def opening_interval(opening: dict, wall: dict) -> tuple[float, float, float, float] | None:
    if not isinstance(opening, dict) or str(opening.get("wall_id") or "") != wall["id"]:
        return None
    try:
        if opening.get("s") is not None:
            width = max(0.05, float(opening.get("width") or 0.8))
            center = float(opening.get("s"))
            s0 = center - width * 0.5
            s1 = center + width * 0.5
        elif isinstance(opening.get("segment"), dict):
            segment = opening["segment"]
            p1 = mathutils.Vector((float(segment.get("x1")), float(segment.get("y1")), 0.0))
            p2 = mathutils.Vector((float(segment.get("x2")), float(segment.get("y2")), 0.0))
            s0 = (p1 - wall["a"]).dot(wall["tangent"])
            s1 = (p2 - wall["a"]).dot(wall["tangent"])
            if s1 < s0:
                s0, s1 = s1, s0
        else:
            return None
        z0 = float(opening.get("z0") or 0.0)
        height = float(opening.get("height") or (2.05 if str(opening.get("type") or "").lower() == "door" else 1.1))
    except Exception:
        return None
    s0 = max(0.0, min(float(wall["length"]), s0))
    s1 = max(0.0, min(float(wall["length"]), s1))
    if s1 - s0 <= 0.03:
        return None
    return s0, s1, z0, z0 + height


def add_oriented_box(
    *,
    name: str,
    collection: bpy.types.Collection,
    center: mathutils.Vector,
    tangent: mathutils.Vector,
    normal: mathutils.Vector,
    size_t: float,
    size_n: float,
    size_z: float,
    matrix_world: mathutils.Matrix,
    material: bpy.types.Material,
) -> bpy.types.Object:
    t = tangent.normalized()
    n = normal.normalized()
    z = mathutils.Vector((0.0, 0.0, 1.0))
    ht, hn, hz = size_t * 0.5, size_n * 0.5, size_z * 0.5
    verts = []
    for dz in (-hz, hz):
        for dn in (-hn, hn):
            for dt in (-ht, ht):
                p = center + t * dt + n * dn + z * dz
                verts.append((p.x, p.y, p.z))
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = matrix_world
    obj.data.materials.append(material)
    collection.objects.link(obj)
    return obj


def _blender_builder_module():
    global _BLENDER_BUILDER_MODULE
    if _BLENDER_BUILDER_MODULE is not None:
        return _BLENDER_BUILDER_MODULE
    path = Path(__file__).resolve().parents[2] / "src" / "Plasement" / "blender_scene_builder.py"
    spec = importlib.util.spec_from_file_location("cgs_apartment_curtain_builder", str(path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _BLENDER_BUILDER_MODULE = module
    return module


def _scene_curtain_items(scene: dict) -> list[dict]:
    items = scene.get("placements") if isinstance(scene.get("placements"), list) else scene.get("items")
    out: list[dict] = []
    for item in (items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        text = placement_text(item)
        asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
        if (
            "curtain" in text
            or "shtor" in text
            or "штор" in text
            or "занавес" in text
            or str(asset.get("kind") or "").startswith("curtain")
        ):
            out.append(item)
    return out


def _mark_object_family(root: bpy.types.Object, room_id: str) -> int:
    count = 0
    stack = [root]
    seen: set[bpy.types.Object] = set()
    while stack:
        obj = stack.pop()
        if obj is None or obj in seen:
            continue
        seen.add(obj)
        obj["source_room_id"] = room_id
        obj["apartment_auto_curtain"] = "shtorystore_textured"
        count += 1
        stack.extend(list(obj.children))
    return count


def _object_family(root: bpy.types.Object) -> list[bpy.types.Object]:
    out: list[bpy.types.Object] = []
    stack = [root]
    seen: set[bpy.types.Object] = set()
    while stack:
        obj = stack.pop()
        if obj is None or obj in seen:
            continue
        seen.add(obj)
        out.append(obj)
        stack.extend(list(obj.children))
    return out


def room_has_visible_curtain_objects(room_id: str) -> bool:
    for obj in bpy.data.objects:
        if str(obj.get("source_room_id") or "") != room_id:
            continue
        if bool(getattr(obj, "hide_render", False)):
            continue
        text = " ".join(
            str(part or "").lower()
            for part in (
                getattr(obj, "name", ""),
                getattr(getattr(obj, "data", None), "name", ""),
                obj.get("cgs_procedural_proxy"),
                obj.get("apartment_auto_curtain"),
            )
        )
        if any(token in text for token in ("curtain", "shtor", "штор", "занавес")):
            return True
    return False


def add_room_curtains(scene: dict, room_id: str, xform: mathutils.Matrix, collection: bpy.types.Collection) -> int:
    if room_has_visible_curtain_objects(room_id):
        return 0
    items = _scene_curtain_items(scene)
    if not items:
        return 0
    builder = _blender_builder_module()
    make_proxy = getattr(builder, "_make_curtain_proxy_mesh", None) if builder is not None else None
    if make_proxy is None:
        return 0
    added = 0
    for idx, item in enumerate(items):
        aabb = item.get("aabb") if isinstance(item.get("aabb"), dict) else None
        if not aabb:
            continue
        asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
        texture_path = str(asset.get("texture_path") or item.get("texture_path") or "").strip()
        if not texture_path or not Path(texture_path).expanduser().is_file():
            continue
        try:
            yaw = float(item.get("yaw_deg") if item.get("yaw_deg") is not None else item.get("rotation_deg") or 0.0)
            parent = make_proxy(
                item=item,
                aabb=aabb,
                rotation_deg_engine=yaw,
                texture_path=texture_path,
                collection=collection,
                name=f"{room_id}__ShtorystoreCurtain_{idx:02d}",
            )
        except Exception as exc:
            print(f"[assemble_apartment] curtain proxy failed for {room_id}: {exc!r}")
            parent = None
        if parent is None:
            continue
        family = _object_family(parent)
        transformed_world = {obj: xform @ obj.matrix_world.copy() for obj in family}
        for obj in family:
            obj.parent = None
        for obj, matrix_world in transformed_world.items():
            obj.matrix_world = matrix_world
        added += _mark_object_family(parent, room_id)
    return added


def placement_text(item: dict) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    return " ".join(str(x or "").lower() for x in (item.get("id"), item.get("name"), item.get("category"), item.get("semantic_group"), candidate.get("title")))


def item_aabb(item: dict) -> dict[str, float] | None:
    raw = item.get("aabb") if isinstance(item.get("aabb"), dict) else None
    if raw:
        try:
            return {key: float(raw[key]) for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
        except Exception:
            pass
    pos = item.get("position_m") if isinstance(item.get("position_m"), list) else None
    size = item.get("size_m") if isinstance(item.get("size_m"), list) else None
    if not pos or not size or len(pos) < 3 or len(size) < 3:
        return None
    try:
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    except Exception:
        return None
    return {"x_min": cx - sx / 2, "x_max": cx + sx / 2, "y_min": cy - sy / 2, "y_max": cy + sy / 2, "z_min": cz - sz / 2, "z_max": cz + sz / 2}


def subtract_interval(ranges: list[tuple[float, float]], cut: tuple[float, float]) -> list[tuple[float, float]]:
    c0, c1 = cut
    out: list[tuple[float, float]] = []
    for r0, r1 in ranges:
        if c1 <= r0 or c0 >= r1:
            out.append((r0, r1))
            continue
        if c0 - r0 > 0.05:
            out.append((r0, max(r0, c0)))
        if r1 - c1 > 0.05:
            out.append((min(r1, c1), r1))
    return out


def wall_free_intervals_for_tv(room: dict, wall: dict, placements: list[dict], pad: float = 0.22) -> list[tuple[float, float]]:
    free = [(0.22, max(0.22, float(wall["length"]) - 0.22))]
    for key in ("doors", "windows"):
        for opening in room.get(key) or []:
            interval = opening_interval(opening, wall)
            if interval is None:
                continue
            s0, s1, _z0, _z1 = interval
            free = subtract_interval(free, (s0 - pad, s1 + pad))
    for item in placements:
        aabb = item_aabb(item)
        if not aabb or aabb["z_max"] < 0.75:
            continue
        corners = (
            mathutils.Vector((aabb["x_min"], aabb["y_min"], 0.0)),
            mathutils.Vector((aabb["x_min"], aabb["y_max"], 0.0)),
            mathutils.Vector((aabb["x_max"], aabb["y_min"], 0.0)),
            mathutils.Vector((aabb["x_max"], aabb["y_max"], 0.0)),
        )
        rel = [corner - wall["a"] for corner in corners]
        distances = [abs(v.dot(wall["normal"])) for v in rel]
        if min(distances) > 0.55:
            continue
        ss = [v.dot(wall["tangent"]) for v in rel]
        free = subtract_interval(free, (min(ss) - pad, max(ss) + pad))
    return [(max(0.0, a), min(float(wall["length"]), b)) for a, b in free if b - a >= 0.86]


def add_bed_opposite_tv(scene: dict, room_id: str, xform: mathutils.Matrix, collection: bpy.types.Collection) -> int:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    room_text = " ".join(str(x or "").lower() for x in (room_id, room.get("type"), room.get("name"), room.get("prompt_room_type")))
    placements = [x for x in scene.get("placements") or scene.get("items") or [] if isinstance(x, dict)]
    if "bedroom" not in room_text and "спаль" not in room_text:
        if not any("bed" in placement_text(item) or "кровать" in placement_text(item) for item in placements):
            return 0
    if any(token in placement_text(item) for item in placements for token in ("television", " tv", "tv_", "телевизор")):
        return 0
    beds = []
    for item in placements:
        text = placement_text(item)
        if "bed" not in text and "кровать" not in text:
            continue
        aabb = item_aabb(item)
        if aabb:
            area = max(0.0, aabb["x_max"] - aabb["x_min"]) * max(0.0, aabb["y_max"] - aabb["y_min"])
            beds.append((area, aabb))
    if not beds:
        return 0
    bed_aabb = sorted(beds, key=lambda x: x[0], reverse=True)[0][1]
    bed_center = mathutils.Vector(((bed_aabb["x_min"] + bed_aabb["x_max"]) * 0.5, (bed_aabb["y_min"] + bed_aabb["y_max"]) * 0.5, 0.0))
    candidates = []
    for wall in room_walls(room):
        free = wall_free_intervals_for_tv(room, wall, placements)
        if not free:
            continue
        best_interval = max(free, key=lambda x: x[1] - x[0])
        available = best_interval[1] - best_interval[0]
        tv_width = min(1.45, max(0.86, available * 0.82))
        if available < tv_width:
            continue
        bed_s = (bed_center - wall["a"]).dot(wall["tangent"])
        s = min(max(bed_s, best_interval[0] + tv_width * 0.5), best_interval[1] - tv_width * 0.5)
        wall_point = wall["a"] + wall["tangent"] * s
        faces_bed = max(0.0, (bed_center - wall_point).normalized().dot(wall["normal"])) if (bed_center - wall_point).length > 1e-5 else 0.0
        score = (wall_point - bed_center).length + faces_bed * 2.0 + available * 0.1
        candidates.append((score, wall, s, tv_width))
    if not candidates:
        return 0
    _score, wall, s, tv_width = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    tv_height = tv_width * 9.0 / 16.0
    ceiling = float(room.get("ceiling_height") or room.get("height_m") or 2.7)
    center_z = min(ceiling - tv_height * 0.5 - 0.18, max(1.28, ceiling * 0.52))
    center = wall["a"] + wall["tangent"] * s + wall["normal"] * 0.045 + mathutils.Vector((0, 0, center_z))
    screen_mat = make_simple_material("MAT_CGS_WALL_TV_DARK_SCREEN", (0.005, 0.005, 0.006, 1.0), emissive=0.05)
    obj = add_oriented_box(
        name=f"{room_id}__AutoOppositeBedTV",
        collection=collection,
        center=center,
        tangent=wall["tangent"],
        normal=wall["normal"],
        size_t=tv_width,
        size_n=0.055,
        size_z=tv_height,
        matrix_world=xform,
        material=screen_mat,
    )
    obj["source_room_id"] = room_id
    obj["apartment_auto_tv"] = True
    return 1


def add_room_apartment_overlays(scene: dict, room_id: str, xform: mathutils.Matrix, collection: bpy.types.Collection) -> dict[str, int]:
    return {
        "added_curtain_objects": add_room_curtains(scene, room_id, xform, collection),
        "added_bed_opposite_tvs": add_bed_opposite_tv(scene, room_id, xform, collection),
    }


def append_room_blend(blend_path: Path, room_id: str, xform: mathutils.Matrix, root_collection: bpy.types.Collection) -> dict:
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
    loaded = [obj for obj in bpy.data.objects if obj not in before]
    removed_cameras = 0
    for obj in list(loaded):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
            loaded.remove(obj)
            removed_cameras += 1

    room_coll = bpy.data.collections.new(room_id)
    root_collection.children.link(room_coll)
    link_loaded_objects(loaded, room_coll)
    rename_room_objects(loaded, room_id)
    bpy.context.view_layer.update()

    loaded_set = set(loaded)
    roots = [obj for obj in loaded if obj.parent not in loaded_set]
    for obj in roots:
        obj.matrix_world = xform @ obj.matrix_world
    for obj in loaded:
        obj["source_room_id"] = room_id
        obj["source_room_blend"] = str(blend_path)
    bpy.context.view_layer.update()
    cleanup_report = cleanup_room_visual_helpers(loaded)
    return {
        "room_id": room_id,
        "blend": str(blend_path.resolve()),
        "objects_loaded": len(loaded),
        "root_objects_transformed": len(roots),
        "removed_cameras": removed_cameras,
        **cleanup_report,
    }


def world_bbox(objects: list[bpy.types.Object]) -> tuple[mathutils.Vector, mathutils.Vector]:
    pts: list[mathutils.Vector] = []
    for obj in objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        pts.extend(obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box)
    if not pts:
        return mathutils.Vector((0, 0, 0)), mathutils.Vector((5, 5, 3))
    return (
        mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
        mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))),
    )


def object_world_bbox(obj: bpy.types.Object) -> tuple[mathutils.Vector, mathutils.Vector] | None:
    if obj.type != "MESH" or obj.hide_render:
        return None
    pts = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    if not pts:
        return None
    return (
        mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))),
        mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))),
    )


def hide_object_family(root: bpy.types.Object) -> int:
    hidden = 0
    stack = [root]
    while stack:
        obj = stack.pop()
        if not obj.hide_render or not obj.hide_viewport:
            hidden += 1
        obj.hide_render = True
        obj.hide_viewport = True
        stack.extend(list(obj.children))
    return hidden


def hide_outlier_meshes(max_abs: float = 100.0, max_size: float = 30.0) -> list[dict]:
    """Hide imported supplier geometry that has broken transforms and ruins camera bounds."""
    bpy.context.view_layer.update()
    hidden: list[dict] = []
    for obj in list(bpy.data.objects):
        bounds = object_world_bbox(obj)
        if bounds is None:
            continue
        bmin, bmax = bounds
        size = bmax - bmin
        max_dim = max(float(abs(size.x)), float(abs(size.y)), float(abs(size.z)))
        max_coord = max(abs(float(v)) for v in [bmin.x, bmax.x, bmin.y, bmax.y, bmin.z, bmax.z])
        if max_coord <= max_abs and max_dim <= max_size:
            continue
        hide_object_family(obj)
        hidden.append(
            {
                "name": obj.name,
                "source_room_id": str(obj.get("source_room_id") or ""),
                "max_abs": round(max_coord, 4),
                "max_size": round(max_dim, 4),
                "min": [round(float(bmin.x), 4), round(float(bmin.y), 4), round(float(bmin.z), 4)],
                "max": [round(float(bmax.x), 4), round(float(bmax.y), 4), round(float(bmax.z), 4)],
            }
        )
    if hidden:
        print(f"[assemble_apartment] hidden_outlier_meshes={len(hidden)}")
    return hidden


def setup_camera_and_light() -> None:
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    bmin, bmax = world_bbox(mesh_objects)
    center = (bmin + bmax) * 0.5
    size = bmax - bmin
    radius = max(size.x, size.y, size.z, 1.0)

    cam_data = bpy.data.cameras.new("ApartmentCamera")
    cam = bpy.data.objects.new("ApartmentCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (
        center.x + radius * 0.08,
        center.y - radius * 0.12,
        bmax.z + max(radius * 1.55, 5.5),
    )
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(float(size.x), float(size.y), 1.0) * 1.16
    cam.data.clip_end = max(250.0, float(radius) * 8.0)

    target = bpy.data.objects.new("ApartmentCameraTarget", None)
    bpy.context.scene.collection.objects.link(target)
    target.location = (center.x, center.y, min(float(center.z), 0.8))
    con = cam.constraints.new(type="TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    cam.data.lens = 35
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("ApartmentKeyArea", type="AREA")
    light = bpy.data.objects.new("ApartmentKeyArea", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (center.x, center.y, bmax.z + max(radius * 0.35, 2.0))
    light.data.energy = 500
    light.data.size = max(radius, 3.0)


def setup_render(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except Exception:
            scene.render.engine = "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        for attr in ("taa_render_samples", "taa_samples"):
            if hasattr(scene.eevee, attr):
                try:
                    setattr(scene.eevee, attr, int(samples))
                except Exception:
                    pass
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Assemble ready room .blend files into one apartment .blend.")
    p.add_argument("--apt-dir", required=True)
    p.add_argument("--mode", default="optimal")
    p.add_argument("--apartment-scene", default=None)
    p.add_argument("--save-blend", required=True)
    p.add_argument("--render", default=None)
    p.add_argument("--build-report", default=None)
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--samples", type=int, default=64)
    return p


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    args = build_cli().parse_args(argv)
    apt_dir = Path(args.apt_dir).expanduser().resolve()
    manifest = read_json(apt_dir / "manifest.json")
    apartment_json = apt_dir / "apartment.json"
    apartment_scene = Path(args.apartment_scene).expanduser().resolve() if args.apartment_scene else None
    apt_min = apt_min_from_scene(apartment_scene, apartment_json)

    clear_scene()
    root_collection = bpy.data.collections.new("ApartmentRooms")
    bpy.context.scene.collection.children.link(root_collection)

    reports = []
    for room_meta in manifest.get("rooms") or []:
        room_id = str(room_meta.get("room_id") or "")
        if not room_id:
            continue
        room_dir = apt_dir / "rooms" / room_id
        scene_path = room_scene_path(room_dir, args.mode)
        if not scene_path.is_file():
            reports.append({"room_id": room_id, "status": "missing_scene_requirements", "path": str(scene_path)})
            continue
        blend_path = find_room_blend(room_dir, room_id, args.mode)
        if blend_path is None:
            reports.append({"room_id": room_id, "status": "missing_room_blend"})
            continue
        scene = read_json(scene_path)
        frame = (((scene.get("room") or {}).get("meta") or {}).get("coordinate_frame") or {})
        if not frame:
            reports.append({"room_id": room_id, "status": "missing_coordinate_frame", "blend": str(blend_path)})
            continue
        xform = transform_from_frame(frame, apt_min)
        report = append_room_blend(blend_path, room_id, xform, root_collection)
        report.update(add_room_apartment_overlays(scene, room_id, xform, root_collection))
        report["status"] = "ok"
        reports.append(report)

    hidden_outlier_objects = hide_outlier_meshes()
    setup_camera_and_light()
    setup_render(args.width, args.height, args.samples)

    save_path = Path(args.save_blend).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path))

    render_path = None
    if args.render:
        render_path = Path(args.render).expanduser().resolve()
        render_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)

    if args.build_report:
        bmin, bmax = world_bbox([obj for obj in bpy.data.objects if obj.type == "MESH"])
        report_path = Path(args.build_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "apartment_dir": str(apt_dir),
                    "save_blend": str(save_path),
                    "render": str(render_path) if render_path else None,
                    "apartment_global_min_xy": list(apt_min),
                    "room_reports": reports,
                    "hidden_outlier_object_count": len(hidden_outlier_objects),
                    "hidden_outlier_objects": hidden_outlier_objects[:200],
                    "object_count": len(bpy.data.objects),
                    "mesh_count": sum(1 for obj in bpy.data.objects if obj.type == "MESH"),
                    "bounds": {
                        "x_min": round(float(bmin.x), 4),
                        "x_max": round(float(bmax.x), 4),
                        "y_min": round(float(bmin.y), 4),
                        "y_max": round(float(bmax.y), 4),
                        "z_min": round(float(bmin.z), 4),
                        "z_max": round(float(bmax.z), 4),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
