from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _require_bpy():
    try:
        import bpy  # type: ignore
    except Exception as exc:
        raise RuntimeError("kitchen_blender_builder must be executed inside Blender Python") from exc
    return bpy


def _visual_color(material_binding: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not material_binding:
        return (0.8, 0.8, 0.8, 1.0)
    chosen = material_binding.get("chosen_material") or material_binding
    visual = chosen.get("visual") or {}
    colors = set(visual.get("base_colors") or [])
    pattern = visual.get("pattern")
    if "black" in colors:
        return (0.03, 0.03, 0.03, 1.0)
    if "gray" in colors:
        return (0.45, 0.45, 0.45, 1.0)
    if "white" in colors:
        return (0.92, 0.90, 0.86, 1.0)
    if "beige" in colors:
        return (0.72, 0.63, 0.50, 1.0)
    if "light_wood" in colors or pattern == "wood":
        return (0.68, 0.52, 0.34, 1.0)
    if "dark_wood" in colors:
        return (0.25, 0.14, 0.08, 1.0)
    if pattern in {"marble", "stone", "concrete"}:
        return (0.68, 0.68, 0.66, 1.0)
    return (0.75, 0.75, 0.72, 1.0)


def _resolve_texture_path(material_binding: dict[str, Any] | None) -> Path | None:
    if not material_binding:
        return None

    chosen = material_binding.get("chosen_material") or material_binding
    raw_path = chosen.get("local_image")
    if not raw_path:
        return None

    path = Path(str(raw_path))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                Path.cwd() / "data/floor_materials/basisrf" / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def _get_or_create_material(
    name: str,
    color: tuple[float, float, float, float],
    texture_path: Path | None = None,
):
    bpy = _require_bpy()
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color

    if texture_path is not None and texture_path.exists():
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        image_node = nodes.new(type="ShaderNodeTexImage")
        image_node.name = f"{name}_image"
        image_node.image = bpy.data.images.load(str(texture_path), check_existing=True)
        image_node.extension = "REPEAT"
        mat.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55
        mat["basisrf_texture_path"] = str(texture_path)

    return mat


def _create_box(name: str, center: tuple[float, float, float], size: tuple[float, float, float], material=None, collection=None):
    bpy = _require_bpy()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = getattr(bpy.context, "object", None) or bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError(f"Blender failed to create cube object: {name}")
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if material is not None:
        obj.data.materials.append(material)
    if collection is not None:
        for col in obj.users_collection:
            col.objects.unlink(obj)
        collection.objects.link(obj)
    return obj


def _apply_rectangular_cutout(
    target_obj,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    collection=None,
) -> bool:
    bpy = _require_bpy()
    cutter = _create_box(f"{name}_boolean_cutter", center, size, None, collection)
    try:
        bpy.context.view_layer.objects.active = target_obj
        target_obj.select_set(True)
        modifier = target_obj.modifiers.new(f"{name}_cutout", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception as exc:
        print(f"[kitchen] failed to apply cutout {name}: {exc}")
        cutter.hide_viewport = True
        cutter.hide_render = True
        return False

    bpy.data.objects.remove(cutter, do_unlink=True)
    return True


def _apply_mesh_objects_cutout(
    target_obj,
    name: str,
    source_objects: list[Any],
    collection=None,
) -> bool:
    bpy = _require_bpy()
    mesh_sources = [obj for obj in source_objects if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None]
    if not mesh_sources:
        return False

    ok = False
    cutters: list[Any] = []
    for idx, source in enumerate(mesh_sources, start=1):
        cutter = source.copy()
        cutter.data = source.data.copy()
        cutter.name = f"{name}_mesh_boolean_cutter_{idx:02d}"
        if collection is not None:
            collection.objects.link(cutter)
        else:
            bpy.context.scene.collection.objects.link(cutter)
        cutter.hide_viewport = True
        cutter.hide_render = True
        cutters.append(cutter)

        try:
            bpy.context.view_layer.objects.active = target_obj
            target_obj.select_set(True)
            modifier = target_obj.modifiers.new(f"{name}_mesh_cutout_{idx:02d}", "BOOLEAN")
            modifier.operation = "DIFFERENCE"
            modifier.object = cutter
            if hasattr(modifier, "solver"):
                modifier.solver = "EXACT"
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            ok = True
        except Exception as exc:
            print(f"[kitchen] failed to apply mesh cutout {name}: {exc}")
            try:
                target_obj.modifiers.remove(modifier)
            except Exception:
                pass

    for cutter in cutters:
        try:
            bpy.data.objects.remove(cutter, do_unlink=True)
        except Exception:
            pass
    return ok


def _convex_hull_xy(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set((round(x, 5), round(y, 5)) for x, y in points))
    if len(unique) <= 2:
        return unique

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _simplify_polygon_xy(
    polygon_xy: list[tuple[float, float]],
    *,
    min_edge_m: float = 0.006,
    collinear_epsilon: float = 0.00008,
) -> list[tuple[float, float]]:
    if len(polygon_xy) <= 3:
        return polygon_xy

    deduped: list[tuple[float, float]] = []
    for point in polygon_xy:
        if not deduped:
            deduped.append(point)
            continue
        dx = point[0] - deduped[-1][0]
        dy = point[1] - deduped[-1][1]
        if (dx * dx + dy * dy) ** 0.5 >= min_edge_m:
            deduped.append(point)

    if len(deduped) > 2:
        dx = deduped[0][0] - deduped[-1][0]
        dy = deduped[0][1] - deduped[-1][1]
        if (dx * dx + dy * dy) ** 0.5 < min_edge_m:
            deduped.pop()

    if len(deduped) <= 3:
        return deduped

    simplified: list[tuple[float, float]] = []
    count = len(deduped)
    for idx, point in enumerate(deduped):
        prev_point = deduped[(idx - 1) % count]
        next_point = deduped[(idx + 1) % count]
        ax = point[0] - prev_point[0]
        ay = point[1] - prev_point[1]
        bx = next_point[0] - point[0]
        by = next_point[1] - point[1]
        cross = abs(ax * by - ay * bx)
        if cross >= collinear_epsilon:
            simplified.append(point)

    return simplified if len(simplified) >= 3 else deduped


def _mesh_outer_polygon_xy_from_objects(
    source_objects: list[Any],
    *,
    sample_z_min: float,
    sample_z_max: float,
    inset_m: float = 0.0,
    radial_bins: int = 96,
) -> list[tuple[float, float]]:
    """Approximate the visible top footprint as an ordered polygon.

    The sink cutout must follow the external rim silhouette, not a hardcoded
    rectangle. Supplier meshes differ a lot, so this samples vertices around
    the countertop-height band and keeps the farthest point in each radial
    direction. This covers rectangular, round and most star-shaped top rims.
    Sparse models fall back to a convex hull.
    """

    import math

    mesh_sources = [obj for obj in source_objects if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None]
    if not mesh_sources:
        return []

    points: list[tuple[float, float]] = []
    for obj in mesh_sources:
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            z = float(world.z)
            if sample_z_min <= z <= sample_z_max:
                points.append((float(world.x), float(world.y)))

    if len(points) < 12:
        return _mesh_outer_hull_xy_from_objects(
            source_objects,
            sample_z_min=sample_z_min,
            sample_z_max=sample_z_max,
            inset_m=inset_m,
        )

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

    bins: list[tuple[float, tuple[float, float]] | None] = [None] * max(16, radial_bins)
    for x, y in points:
        dx = x - center[0]
        dy = y - center[1]
        radius = (dx * dx + dy * dy) ** 0.5
        if radius < 1e-5:
            continue
        angle = math.atan2(dy, dx)
        bucket = int(((angle + math.pi) / (2.0 * math.pi)) * len(bins)) % len(bins)
        current = bins[bucket]
        if current is None or radius > current[0]:
            bins[bucket] = (radius, (x, y))

    polygon = [entry[1] for entry in bins if entry is not None]
    if len(polygon) < 8:
        return _mesh_outer_hull_xy_from_objects(
            source_objects,
            sample_z_min=sample_z_min,
            sample_z_max=sample_z_max,
            inset_m=inset_m,
        )

    if inset_m > 0.0:
        polygon = _scale_polygon_xy(polygon, inset_m)

    return _simplify_polygon_xy(polygon)


def _mesh_outer_hull_xy_from_objects(
    source_objects: list[Any],
    *,
    sample_z_min: float,
    sample_z_max: float,
    inset_m: float = 0.014,
) -> list[tuple[float, float]]:
    mesh_sources = [obj for obj in source_objects if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None]
    if not mesh_sources:
        return []

    points: list[tuple[float, float]] = []
    for obj in mesh_sources:
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            z = float(world.z)
            if sample_z_min <= z <= sample_z_max:
                points.append((float(world.x), float(world.y)))

    if len(points) < 4:
        bbox = _bbox_world(mesh_sources)
        if bbox is None:
            return []
        mins, maxs = bbox
        points = [
            (mins[0], mins[1]),
            (maxs[0], mins[1]),
            (maxs[0], maxs[1]),
            (mins[0], maxs[1]),
        ]

    hull = _convex_hull_xy(points)
    if len(hull) < 3:
        return []

    bbox = _bbox_world(mesh_sources)
    target_center = None
    if bbox is not None:
        mins, maxs = bbox
        target_center = ((mins[0] + maxs[0]) / 2.0, (mins[1] + maxs[1]) / 2.0)

    cx = sum(point[0] for point in hull) / len(hull)
    cy = sum(point[1] for point in hull) / len(hull)
    if inset_m <= 0:
        fitted = hull
    else:
        fitted = []
        for x, y in hull:
            dx = x - cx
            dy = y - cy
            scale_x = max(0.0, (abs(dx) - inset_m) / max(abs(dx), 1e-6))
            scale_y = max(0.0, (abs(dy) - inset_m) / max(abs(dy), 1e-6))
            fitted.append((cx + dx * scale_x, cy + dy * scale_y))

    if target_center is None:
        return fitted

    fitted_center = (
        sum(point[0] for point in fitted) / len(fitted),
        sum(point[1] for point in fitted) / len(fitted),
    )
    dx = target_center[0] - fitted_center[0]
    dy = target_center[1] - fitted_center[1]
    return [(x + dx, y + dy) for x, y in fitted]


def _apply_polygon_cutout(
    target_obj,
    name: str,
    polygon_xy: list[tuple[float, float]],
    *,
    cutter_z_min: float,
    cutter_z_max: float,
    collection=None,
) -> bool:
    bpy = _require_bpy()
    if len(polygon_xy) < 3:
        return False

    cx = sum(point[0] for point in polygon_xy) / len(polygon_xy)
    cy = sum(point[1] for point in polygon_xy) / len(polygon_xy)
    vertices = (
        [(x, y, cutter_z_min) for x, y in polygon_xy]
        + [(x, y, cutter_z_max) for x, y in polygon_xy]
        + [(cx, cy, cutter_z_min), (cx, cy, cutter_z_max)]
    )
    count = len(polygon_xy)
    bottom_center = count * 2
    top_center = count * 2 + 1
    faces: list[tuple[int, ...]] = []
    for idx in range(count):
        nxt = (idx + 1) % count
        faces.append((bottom_center, nxt, idx))
        faces.append((top_center, idx + count, nxt + count))
        faces.append((idx, nxt, nxt + count, idx + count))

    mesh = bpy.data.meshes.new(f"{name}_footprint_cutout_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    cutter = bpy.data.objects.new(f"{name}_footprint_boolean_cutter", mesh)
    if collection is not None:
        collection.objects.link(cutter)
    else:
        bpy.context.scene.collection.objects.link(cutter)
    cutter.hide_viewport = True
    cutter.hide_render = True

    try:
        bpy.context.view_layer.objects.active = target_obj
        target_obj.select_set(True)
        modifier = target_obj.modifiers.new(f"{name}_footprint_cutout", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        if hasattr(modifier, "solver"):
            modifier.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception as exc:
        print(f"[kitchen] failed to apply footprint cutout {name}: {exc}")
        try:
            target_obj.modifiers.remove(modifier)
        except Exception:
            pass
        bpy.data.objects.remove(cutter, do_unlink=True)
        return False

    bpy.data.objects.remove(cutter, do_unlink=True)
    return True


def _apply_mesh_footprint_cutout(
    target_obj,
    name: str,
    source_objects: list[Any],
    *,
    sample_z_min: float,
    sample_z_max: float,
    cutter_z_min: float,
    cutter_z_max: float,
    inset_m: float = 0.014,
    collection=None,
) -> bool:
    hull = _mesh_outer_hull_xy_from_objects(
        source_objects,
        sample_z_min=sample_z_min,
        sample_z_max=sample_z_max,
        inset_m=inset_m,
    )
    return _apply_polygon_cutout(
        target_obj,
        name,
        hull,
        cutter_z_min=cutter_z_min,
        cutter_z_max=cutter_z_max,
        collection=collection,
    )


def _real_bbox_opening_from_objects(
    objects: list[Any],
    *,
    inset_x: float = 0.0,
    inset_y: float = 0.0,
    min_size: float = 0.18,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    bbox = _bbox_world([obj for obj in objects if getattr(obj, "type", None) == "MESH"])
    if bbox is None:
        return None
    mins, maxs = bbox
    center = ((mins[0] + maxs[0]) / 2.0, (mins[1] + maxs[1]) / 2.0)
    size = (
        max(min_size, (maxs[0] - mins[0]) - inset_x * 2.0),
        max(min_size, (maxs[1] - mins[1]) - inset_y * 2.0),
    )
    return center, size


def _create_oriented_box(
    name: str,
    origin_x: float,
    origin_y: float,
    along: float,
    depth: float,
    height: float,
    z: float,
    orientation: str,
    material=None,
    collection=None,
):
    if orientation == "y":
        return _create_box(name, (origin_x + depth / 2, origin_y + along / 2, z + height / 2), (depth, along, height), material, collection)
    return _create_box(name, (origin_x + along / 2, origin_y + depth / 2, z + height / 2), (along, depth, height), material, collection)


def _surface_point(
    px: float,
    py: float,
    segment_x: float,
    segment_y: float,
    along_x: float,
    local_y: float,
    orientation: str,
) -> tuple[float, float]:
    if orientation == "y":
        return px + segment_x + local_y, py + along_x
    return px + along_x, py + segment_y + local_y


def _surface_point_local(
    px: float,
    py: float,
    segment_x: float,
    segment_y: float,
    along_x: float,
    local_y: float,
    orientation: str,
) -> tuple[float, float]:
    if orientation == "y":
        return px + segment_x + local_y, py + segment_y + along_x
    return px + segment_x + along_x, py + segment_y + local_y


def _create_cylinder(
    name: str,
    center: tuple[float, float, float],
    radius: float,
    depth: float,
    material=None,
    collection=None,
    vertices: int = 48,
):
    bpy = _require_bpy()
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=center)
    obj = getattr(bpy.context, "object", None) or bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError(f"Blender failed to create cylinder object: {name}")
    obj.name = name
    if material is not None:
        obj.data.materials.append(material)
    if collection is not None:
        for col in obj.users_collection:
            col.objects.unlink(obj)
        collection.objects.link(obj)
    return obj


def _create_torus(
    name: str,
    center: tuple[float, float, float],
    major_radius: float,
    minor_radius: float,
    material=None,
    collection=None,
):
    bpy = _require_bpy()
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=72,
        minor_segments=8,
        location=center,
    )
    obj = getattr(bpy.context, "object", None) or bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError(f"Blender failed to create torus object: {name}")
    obj.name = name
    if material is not None:
        obj.data.materials.append(material)
    if collection is not None:
        for col in obj.users_collection:
            col.objects.unlink(obj)
        collection.objects.link(obj)
    return obj


def _collection(name: str):
    bpy = _require_bpy()
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def _appliance_asset(assembly: dict[str, Any], role: str) -> dict[str, Any] | None:
    appliances = (assembly.get("appliance_bindings") or {}).get("appliances") or {}
    entry = appliances.get(role) or {}
    asset = entry.get("chosen_asset")
    return asset if isinstance(asset, dict) and asset.get("asset_local_path") else None


def _appliance_asset_candidates(assembly: dict[str, Any], role: str) -> list[dict[str, Any]]:
    appliances = (assembly.get("appliance_bindings") or {}).get("appliances") or {}
    entry = appliances.get(role) or {}
    raw_candidates = [entry.get("chosen_asset")] + list(entry.get("top_candidates") or [])

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for asset in raw_candidates:
        if not isinstance(asset, dict) or not asset.get("asset_local_path"):
            continue
        key = str(asset.get("unique_key") or asset.get("asset_local_path") or "")
        if key in seen:
            continue
        seen.add(key)
        candidates.append(asset)
    return candidates


def _asset_title(asset: dict[str, Any] | None) -> str:
    return str((asset or {}).get("title") or (asset or {}).get("name") or "").lower()


def _sink_asset_includes_faucet(asset: dict[str, Any] | None) -> bool:
    title = _asset_title(asset)
    return any(term in title for term in ("смеситель", "mixer", "faucet"))


def _import_asset_objects(asset_path: str, collection=None) -> list[Any]:
    bpy = _require_bpy()
    path = Path(asset_path)
    if not path.exists():
        fixed = Path(str(asset_path).replace("\\", "/"))
        path = fixed if fixed.exists() else path
    if not path.exists():
        return []

    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    try:
        if suffix == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(path))
        elif suffix == ".obj":
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=str(path))
            else:
                bpy.ops.import_scene.obj(filepath=str(path))
        elif suffix in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=str(path))
        else:
            return []
    except Exception as exc:
        print(f"[kitchen] failed to import appliance asset {path}: {exc}")
        return []

    imported = [obj for obj in bpy.data.objects if obj not in before]
    if collection is not None:
        for obj in imported:
            for col in list(obj.users_collection):
                col.objects.unlink(obj)
            collection.objects.link(obj)
    return imported


def _delete_objects(objects: list[Any]) -> None:
    bpy = _require_bpy()
    to_delete: list[Any] = []
    seen: set[Any] = set()
    for obj in objects:
        parent = getattr(obj, "parent", None)
        if parent is not None and str(getattr(parent, "name", "")).startswith("kitchen_appliance_asset_root"):
            if parent not in seen:
                to_delete.append(parent)
                seen.add(parent)
        if obj not in seen:
            to_delete.append(obj)
            seen.add(obj)

    # Remove children before wrapper empties so failed fitted assets do not leave
    # orphaned roots that still affect subsequent bbox/camera logic.
    to_delete.sort(key=lambda obj: 1 if getattr(obj, "children", None) else 0)
    for obj in to_delete:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def _filter_imported_appliance_objects(
    role: str,
    objects: list[Any],
    asset: dict[str, Any] | None = None,
) -> list[Any]:
    if role == "sink":
        mesh_objects = [obj for obj in objects if getattr(obj, "type", None) == "MESH"]
        candidates = [
            obj
            for obj in mesh_objects
            if "cube" not in obj.name.lower()
            and max(float(v) for v in obj.dimensions) < 1.2
        ]
        if candidates and str((asset or {}).get("asset_format") or "").lower() == "fbx":
            chosen = max(
                candidates,
                key=lambda obj: float(obj.dimensions.x) * float(obj.dimensions.y) * float(obj.dimensions.z),
            )
            _delete_objects([obj for obj in objects if obj is not chosen])
            return [chosen]
        if candidates:
            keep_set = set(candidates)
            _delete_objects([obj for obj in objects if obj not in keep_set])
            return candidates
    if role == "faucet":
        mesh_objects = [obj for obj in objects if getattr(obj, "type", None) == "MESH"]
        candidates = [
            obj
            for obj in mesh_objects
            if "cube" not in obj.name.lower()
            and max(float(v) for v in obj.dimensions) < 4.0
            and min(float(v) for v in obj.dimensions) >= 0.0
        ]
        if candidates:
            preferred_parts = []
            for obj in candidates:
                try:
                    part_index = int(obj.name.rsplit("_", 1)[-1])
                except Exception:
                    part_index = -1
                if 13 <= part_index <= 15 or 18 <= part_index <= 23:
                    preferred_parts.append(obj)
            if preferred_parts:
                candidates = preferred_parts
        if candidates:
            centered_candidates = []
            vector_cls = __import__("mathutils").Vector
            for obj in candidates:
                corners = [obj.matrix_world @ vector_cls(corner) for corner in obj.bound_box]
                center_z = sum(float(corner.z) for corner in corners) / max(1, len(corners))
                if center_z > 8.0:
                    centered_candidates.append(obj)
            if centered_candidates:
                candidates = centered_candidates
        if candidates:
            keep_set = set(candidates)
            _delete_objects([obj for obj in objects if obj not in keep_set])
            return candidates
    return objects


def _bbox_world(objects: list[Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    corners: list[tuple[float, float, float]] = []
    for obj in objects:
        if not hasattr(obj, "bound_box"):
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ __import__("mathutils").Vector(corner)
            corners.append((float(world.x), float(world.y), float(world.z)))
    if not corners:
        return None
    mins = tuple(min(c[i] for c in corners) for i in range(3))
    maxs = tuple(max(c[i] for c in corners) for i in range(3))
    return mins, maxs


def _objects_fit_within_size(
    objects: list[Any],
    size: tuple[float, float, float],
    *,
    tolerance: float = 1.18,
) -> bool:
    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, maxs = bbox
    actual = tuple(maxs[i] - mins[i] for i in range(3))
    for actual_value, target_value in zip(actual, size):
        if target_value <= 0:
            continue
        if actual_value > target_value * tolerance:
            return False
    return True


def _translation_roots(objects: list[Any]) -> list[Any]:
    object_set = set(objects)
    roots: list[Any] = []
    seen: set[Any] = set()
    for obj in objects:
        parent = getattr(obj, "parent", None)
        if parent is not None and str(getattr(parent, "name", "")).startswith("kitchen_appliance_asset_root"):
            root = parent
        else:
            root = obj
            while getattr(root, "parent", None) is not None and root.parent in object_set:
                root = root.parent
        if root not in seen:
            roots.append(root)
            seen.add(root)
    return roots


def _translate_object_group(objects: list[Any], delta: tuple[float, float, float]) -> None:
    vector_cls = __import__("mathutils").Vector
    move = vector_cls(delta)
    for root in _translation_roots(objects):
        root.location += move
    _require_bpy().context.view_layer.update()


def _snap_objects_bottom_to_z(objects: list[Any], target_z: float) -> bool:
    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, _ = bbox
    _translate_object_group(objects, (0.0, 0.0, float(target_z) - float(mins[2])))
    return True


def _mesh_xy_bbox_below_z(
    objects: list[Any],
    z_limit: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    vector_cls = __import__("mathutils").Vector
    points: list[tuple[float, float]] = []
    for obj in objects:
        if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
            continue
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            if float(world.z) <= z_limit:
                points.append((float(world.x), float(world.y)))
        if points:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ vector_cls(corner)
            if float(world.z) <= z_limit:
                points.append((float(world.x), float(world.y)))
    if not points:
        return None
    mins = (min(point[0] for point in points), min(point[1] for point in points))
    maxs = (max(point[0] for point in points), max(point[1] for point in points))
    return mins, maxs


def _mesh_xy_bbox_between_z(
    objects: list[Any],
    z_min: float,
    z_max: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    vector_cls = __import__("mathutils").Vector
    points: list[tuple[float, float]] = []
    for obj in objects:
        if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
            continue
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            z = float(world.z)
            if z_min <= z <= z_max:
                points.append((float(world.x), float(world.y)))
        if points:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ vector_cls(corner)
            z = float(world.z)
            if z_min <= z <= z_max:
                points.append((float(world.x), float(world.y)))
    if not points:
        return None
    mins = (min(point[0] for point in points), min(point[1] for point in points))
    maxs = (max(point[0] for point in points), max(point[1] for point in points))
    return mins, maxs


def _mesh_xy_inner_bbox_between_z(
    objects: list[Any],
    z_min: float,
    z_max: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    vector_cls = __import__("mathutils").Vector
    points: list[tuple[float, float]] = []
    for obj in objects:
        if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
            continue
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            z = float(world.z)
            if z_min <= z <= z_max:
                points.append((float(world.x), float(world.y)))
        if points:
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ vector_cls(corner)
            z = float(world.z)
            if z_min <= z <= z_max:
                points.append((float(world.x), float(world.y)))
    if len(points) < 8:
        return _mesh_xy_bbox_between_z(objects, z_min, z_max)

    xs = sorted(point[0] for point in points)
    ys = sorted(point[1] for point in points)

    def quantile(values: list[float], q: float) -> float:
        return values[int((len(values) - 1) * q)]

    return (
        (quantile(xs, 0.20), quantile(ys, 0.20)),
        (quantile(xs, 0.80), quantile(ys, 0.80)),
    )


def _fit_objects_to_box(
    objects: list[Any],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    margin: float = 0.96,
) -> bool:
    bpy = _require_bpy()
    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, maxs = bbox
    current_size = tuple(max(1e-6, maxs[i] - mins[i]) for i in range(3))
    scale = min((size[i] * margin) / current_size[i] for i in range(3))
    current_center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))

    vector_cls = __import__("mathutils").Vector
    matrix_cls = __import__("mathutils").Matrix
    object_set = set(objects)
    top_level = [obj for obj in objects if obj.parent not in object_set]
    if not top_level:
        top_level = objects

    wrapper = bpy.data.objects.new("kitchen_appliance_fit_wrapper", None)
    bpy.context.scene.collection.objects.link(wrapper)
    wrapper.location = vector_cls(current_center)
    wrapper.matrix_world = matrix_cls.Translation(vector_cls(current_center))

    # Parent only top-level imported roots and preserve each root's world transform.
    # This keeps nested FBX/OBJ hierarchies intact and prevents child meshes from
    # being scaled/translated twice.
    for obj in top_level:
        obj.parent = wrapper
        obj.matrix_parent_inverse = wrapper.matrix_world.inverted()

    wrapper.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, maxs = bbox
    scaled_center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))
    wrapper.location += vector_cls((center[0] - scaled_center[0], center[1] - scaled_center[1], center[2] - scaled_center[2]))
    wrapper.name = "kitchen_appliance_asset_root"

    return True


def _fit_mesh_objects_to_box_baked(
    objects: list[Any],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    margin: float = 0.92,
    compact_disconnected: bool = False,
) -> bool:
    bpy = _require_bpy()
    vector_cls = __import__("mathutils").Vector
    matrix_cls = __import__("mathutils").Matrix
    mesh_objects = [obj for obj in objects if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None)]
    if not mesh_objects:
        return False

    for obj in mesh_objects:
        world = obj.matrix_world.copy()
        obj.data = obj.data.copy()
        obj.parent = None
        obj.matrix_parent_inverse = matrix_cls.Identity(4)
        obj.matrix_world = matrix_cls.Identity(4)
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)
        obj.data.transform(world)

    bpy.context.view_layer.update()
    bbox = _bbox_world(mesh_objects)
    if bbox is None:
        return False

    mins, maxs = bbox
    current_size = tuple(max(1e-6, maxs[i] - mins[i]) for i in range(3))
    scale = min((size[i] * margin) / current_size[i] for i in range(3))
    current_center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))
    transform = (
        matrix_cls.Translation(vector_cls(center))
        @ matrix_cls.Diagonal((scale, scale, scale, 1.0))
        @ matrix_cls.Translation(-vector_cls(current_center))
    )

    for obj in mesh_objects:
        obj.data.transform(transform)
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)

    bpy.context.view_layer.update()
    if compact_disconnected:
        bbox = _bbox_world(mesh_objects)
        if bbox is not None:
            mins, maxs = bbox
            if (maxs[0] - mins[0]) > size[0] * 2.5 or (maxs[1] - mins[1]) > size[1] * 2.5:
                centers: dict[Any, tuple[float, float, float]] = {}
                for obj in mesh_objects:
                    corners = [obj.matrix_world @ vector_cls(corner) for corner in obj.bound_box]
                    centers[obj] = tuple(sum(float(corner[i]) for corner in corners) / max(1, len(corners)) for i in range(3))
                split_x = (mins[0] + maxs[0]) / 2.0
                left = [obj for obj, obj_center in centers.items() if obj_center[0] < split_x]
                right = [obj for obj, obj_center in centers.items() if obj_center[0] >= split_x]
                if left and right:
                    main = left if len(left) >= len(right) else right
                    loose = right if main is left else left
                    main_center = tuple(sum(centers[obj][i] for obj in main) / len(main) for i in range(3))
                    loose_center = tuple(sum(centers[obj][i] for obj in loose) / len(loose) for i in range(3))
                    delta = vector_cls(
                        (
                            main_center[0] - loose_center[0],
                            main_center[1] - loose_center[1],
                            0.0,
                        )
                    )
                    for obj in loose:
                        obj.data.transform(matrix_cls.Translation(delta))
                    bpy.context.view_layer.update()
    return True


def _fit_objects_to_footprint(
    objects: list[Any],
    center_xy: tuple[float, float],
    footprint: tuple[float, float],
    bottom_z: float,
    margin: float = 0.92,
) -> bool:
    bpy = _require_bpy()
    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, maxs = bbox
    current_size = tuple(max(1e-6, maxs[i] - mins[i]) for i in range(3))
    scale = min((footprint[0] * margin) / current_size[0], (footprint[1] * margin) / current_size[1])
    current_center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))

    vector_cls = __import__("mathutils").Vector
    matrix_cls = __import__("mathutils").Matrix
    object_set = set(objects)
    top_level = [obj for obj in objects if obj.parent not in object_set] or objects

    wrapper = bpy.data.objects.new("kitchen_appliance_fit_wrapper", None)
    bpy.context.scene.collection.objects.link(wrapper)
    wrapper.location = vector_cls(current_center)
    wrapper.matrix_world = matrix_cls.Translation(vector_cls(current_center))

    for obj in top_level:
        obj.parent = wrapper
        obj.matrix_parent_inverse = wrapper.matrix_world.inverted()

    wrapper.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, maxs = bbox
    scaled_center_xy = ((mins[0] + maxs[0]) / 2.0, (mins[1] + maxs[1]) / 2.0)
    wrapper.location += vector_cls(
        (
            center_xy[0] - scaled_center_xy[0],
            center_xy[1] - scaled_center_xy[1],
            bottom_z - mins[2],
        )
    )
    wrapper.name = "kitchen_appliance_asset_root"
    return True


def _fit_objects_to_footprint_top(
    objects: list[Any],
    center_xy: tuple[float, float],
    footprint: tuple[float, float],
    top_z: float,
    margin: float = 0.92,
) -> bool:
    bpy = _require_bpy()
    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, maxs = bbox
    current_size = tuple(max(1e-6, maxs[i] - mins[i]) for i in range(3))
    scale = min((footprint[0] * margin) / current_size[0], (footprint[1] * margin) / current_size[1])
    current_center = tuple((mins[i] + maxs[i]) / 2.0 for i in range(3))

    vector_cls = __import__("mathutils").Vector
    matrix_cls = __import__("mathutils").Matrix
    object_set = set(objects)
    top_level = [obj for obj in objects if obj.parent not in object_set] or objects

    wrapper = bpy.data.objects.new("kitchen_appliance_fit_wrapper", None)
    bpy.context.scene.collection.objects.link(wrapper)
    wrapper.location = vector_cls(current_center)
    wrapper.matrix_world = matrix_cls.Translation(vector_cls(current_center))

    for obj in top_level:
        obj.parent = wrapper
        obj.matrix_parent_inverse = wrapper.matrix_world.inverted()

    wrapper.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    bbox = _bbox_world(objects)
    if bbox is None:
        return False
    mins, maxs = bbox
    scaled_center_xy = ((mins[0] + maxs[0]) / 2.0, (mins[1] + maxs[1]) / 2.0)
    wrapper.location += vector_cls(
        (
            center_xy[0] - scaled_center_xy[0],
            center_xy[1] - scaled_center_xy[1],
            top_z - maxs[2],
        )
    )
    wrapper.name = "kitchen_appliance_asset_root"
    return True


def _rotate_imported_roots_z(objects: list[Any], angle_rad: float) -> None:
    roots = {obj.parent for obj in objects if getattr(obj, "parent", None) is not None}
    roots = {root for root in roots if root is not None and str(root.name).startswith("kitchen_appliance_asset_root")}
    targets = list(roots) or [obj for obj in objects if getattr(obj, "parent", None) is None]
    for obj in targets:
        obj.rotation_euler[2] += angle_rad


def _orient_countertop_appliance_front(objects: list[Any], orientation: str) -> None:
    import math

    if orientation == "y":
        _rotate_imported_roots_z(objects, -math.pi / 2.0)


def _orient_wall_appliance_front(objects: list[Any], orientation: str) -> None:
    import math

    if orientation == "y":
        _rotate_imported_roots_z(objects, -math.pi / 2.0)


def _apply_asset_import_orientation(asset: dict[str, Any], objects: list[Any], orientation: str) -> None:
    import math

    blender_import = asset.get("blender_import") if isinstance(asset.get("blender_import"), dict) else {}
    rotations = blender_import.get("rotation_z_deg_by_layout") if isinstance(blender_import.get("rotation_z_deg_by_layout"), dict) else {}
    angle = rotations.get(orientation)
    if not isinstance(angle, (int, float)):
        return
    _rotate_imported_roots_z(objects, math.radians(float(angle)))


def _asset_rotation_z_deg(asset: dict[str, Any], orientation: str) -> float:
    blender_import = asset.get("blender_import") if isinstance(asset.get("blender_import"), dict) else {}
    rotations = blender_import.get("rotation_z_deg_by_layout") if isinstance(blender_import.get("rotation_z_deg_by_layout"), dict) else {}
    angle = rotations.get(orientation)
    return float(angle) if isinstance(angle, (int, float)) else 0.0


def _rotate_baked_mesh_objects_around_point_z(
    objects: list[Any],
    pivot_xy: tuple[float, float],
    angle_deg: float,
) -> None:
    if not angle_deg:
        return

    import math

    bpy = _require_bpy()
    matrix_cls = __import__("mathutils").Matrix
    vector_cls = __import__("mathutils").Vector
    pivot = vector_cls((pivot_xy[0], pivot_xy[1], 0.0))
    transform = (
        matrix_cls.Translation(pivot)
        @ matrix_cls.Rotation(math.radians(float(angle_deg)), 4, "Z")
        @ matrix_cls.Translation(-pivot)
    )

    for obj in objects:
        if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None:
            obj.data.transform(transform)
    bpy.context.view_layer.update()


def _translate_baked_mesh_objects_xy(objects: list[Any], dx: float, dy: float) -> None:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return

    bpy = _require_bpy()
    matrix_cls = __import__("mathutils").Matrix
    transform = matrix_cls.Translation((dx, dy, 0.0))
    for obj in objects:
        if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None:
            obj.data.transform(transform)
    bpy.context.view_layer.update()


def _translate_baked_mesh_objects_z(objects: list[Any], dz: float) -> None:
    if abs(dz) < 1e-6:
        return

    bpy = _require_bpy()
    matrix_cls = __import__("mathutils").Matrix
    transform = matrix_cls.Translation((0.0, 0.0, dz))
    for obj in objects:
        if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None:
            obj.data.transform(transform)
    bpy.context.view_layer.update()


def _snap_baked_mesh_objects_bottom_to_z(objects: list[Any], target_z: float) -> bool:
    mesh_objects = [obj for obj in objects if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None) is not None]
    bbox = _bbox_world(mesh_objects)
    if bbox is None:
        return False
    mins, _ = bbox
    _translate_baked_mesh_objects_z(mesh_objects, float(target_z) - float(mins[2]))
    return True


def _faucet_base_anchor_xy(
    objects: list[Any],
    target_xy: tuple[float, float],
) -> tuple[float, float] | None:
    vector_cls = __import__("mathutils").Vector
    mesh_objects = [obj for obj in objects if getattr(obj, "type", None) == "MESH" and getattr(obj, "data", None)]
    if not mesh_objects:
        return None

    object_boxes: list[tuple[Any, list[Any], tuple[float, float, float], tuple[float, float, float]]] = []
    global_min_z: float | None = None
    global_max_z: float | None = None
    for obj in mesh_objects:
        corners = [obj.matrix_world @ vector_cls(corner) for corner in obj.bound_box]
        if not corners:
            continue
        mins = tuple(min(float(corner[i]) for corner in corners) for i in range(3))
        maxs = tuple(max(float(corner[i]) for corner in corners) for i in range(3))
        object_boxes.append((obj, corners, mins, maxs))
        global_min_z = mins[2] if global_min_z is None else min(global_min_z, mins[2])
        global_max_z = maxs[2] if global_max_z is None else max(global_max_z, maxs[2])

    if global_min_z is None or global_max_z is None:
        return None

    height = max(1e-6, global_max_z - global_min_z)
    base_z_limit = global_min_z + min(0.055, height * 0.18)
    base_candidates: list[tuple[float, float, float]] = []

    for _obj, corners, mins, maxs in object_boxes:
        if mins[2] > base_z_limit:
            continue
        width_x = maxs[0] - mins[0]
        width_y = maxs[1] - mins[1]
        # Prefer compact low parts: those are usually circular mounting feet.
        compact_penalty = max(0.0, width_x - 0.22) + max(0.0, width_y - 0.22)
        center_x = sum(float(corner.x) for corner in corners) / len(corners)
        center_y = sum(float(corner.y) for corner in corners) / len(corners)
        dist = ((center_x - target_xy[0]) ** 2 + (center_y - target_xy[1]) ** 2) ** 0.5
        base_candidates.append((dist + compact_penalty * 2.0, center_x, center_y))

    if base_candidates:
        _score, anchor_x, anchor_y = min(base_candidates, key=lambda item: item[0])
        return anchor_x, anchor_y

    bottom_points: list[tuple[float, float, float]] = []
    for obj in mesh_objects:
        for vertex in obj.data.vertices:
            point = obj.matrix_world @ vertex.co
            if float(point.z) <= base_z_limit:
                dist = ((float(point.x) - target_xy[0]) ** 2 + (float(point.y) - target_xy[1]) ** 2) ** 0.5
                bottom_points.append((dist, float(point.x), float(point.y)))

    if not bottom_points:
        return None

    bottom_points.sort(key=lambda item: item[0])
    closest = bottom_points[: max(1, min(24, len(bottom_points) // 4))]
    return (
        sum(point[1] for point in closest) / len(closest),
        sum(point[2] for point in closest) / len(closest),
    )


def _faucet_lowest_mount_xy(objects: list[Any]) -> tuple[float, float] | None:
    vector_cls = __import__("mathutils").Vector
    points: list[tuple[float, float, float]] = []
    for obj in objects:
        if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
            continue
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            points.append((float(world.x), float(world.y), float(world.z)))

    if not points:
        return _faucet_base_anchor_xy(objects, (0.0, 0.0))

    min_z = min(point[2] for point in points)
    max_z = max(point[2] for point in points)
    height = max(1e-6, max_z - min_z)
    threshold = min_z + min(0.012, height * 0.035)
    low_points = [point for point in points if point[2] <= threshold]
    if len(low_points) < 4:
        low_points = sorted(points, key=lambda point: point[2])[: max(4, min(32, len(points)))]

    return (
        sum(point[0] for point in low_points) / len(low_points),
        sum(point[1] for point in low_points) / len(low_points),
    )


def _faucet_direction_xy(
    objects: list[Any],
    anchor_xy: tuple[float, float],
) -> tuple[float, float] | None:
    points: list[tuple[float, float, float, float]] = []
    for obj in objects:
        if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
            continue
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            x = float(world.x)
            y = float(world.y)
            z = float(world.z)
            dist = ((x - anchor_xy[0]) ** 2 + (y - anchor_xy[1]) ** 2) ** 0.5
            points.append((x, y, z, dist))

    if not points:
        return None

    min_z = min(point[2] for point in points)
    max_z = max(point[2] for point in points)
    height = max(1e-6, max_z - min_z)

    # The faucet should aim from the mounting foot toward the visible spout/neck.
    # Use high and horizontally distant vertices so the base flange does not
    # dominate the direction estimate.
    candidates = [
        point
        for point in points
        if point[2] >= min_z + height * 0.45 and point[3] >= 0.045
    ]
    if not candidates:
        candidates = sorted(points, key=lambda point: (point[3], point[2]), reverse=True)[: max(4, min(64, len(points)))]
    else:
        candidates = sorted(candidates, key=lambda point: point[3], reverse=True)[: max(4, min(64, len(candidates)))]

    x = sum(point[0] for point in candidates) / len(candidates)
    y = sum(point[1] for point in candidates) / len(candidates)
    dx = x - anchor_xy[0]
    dy = y - anchor_xy[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return None
    return dx / length, dy / length


def _signed_angle_between_xy(
    source: tuple[float, float],
    target: tuple[float, float],
) -> float:
    import math

    sx, sy = source
    tx, ty = target
    source_len = max(1e-6, (sx * sx + sy * sy) ** 0.5)
    target_len = max(1e-6, (tx * tx + ty * ty) ** 0.5)
    sx /= source_len
    sy /= source_len
    tx /= target_len
    ty /= target_len
    cross = sx * ty - sy * tx
    dot = max(-1.0, min(1.0, sx * tx + sy * ty))
    return math.degrees(math.atan2(cross, dot))


def _sanitize_imported_appliance_materials(role: str, objects: list[Any]) -> None:
    if role == "sink":
        sink_mat = _get_or_create_material("kitchen_sink_asset_dark_pvd", (0.025, 0.026, 0.026, 1.0))
        for obj in objects:
            if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
                continue
            obj.data.materials.clear()
            obj.data.materials.append(sink_mat)
        return

    if role == "cooktop":
        glass_mat = _get_or_create_material("kitchen_cooktop_asset_black_glass", (0.005, 0.006, 0.007, 1.0))
        trim_mat = _get_or_create_material("kitchen_cooktop_asset_dark_trim", (0.025, 0.026, 0.028, 1.0))
        for obj in objects:
            if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
                continue
            name = obj.name.lower()
            mat = trim_mat if "box" in name else glass_mat
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        return

    if role == "fridge":
        body_mat = _get_or_create_material("kitchen_fridge_asset_satin_white", (0.86, 0.85, 0.82, 1.0))
        trim_mat = _get_or_create_material("kitchen_fridge_asset_warm_gray_trim", (0.58, 0.57, 0.54, 1.0))
        dark_mat = _get_or_create_material("kitchen_fridge_asset_dark_display", (0.08, 0.08, 0.075, 1.0))

        for obj in objects:
            if getattr(obj, "type", None) != "MESH" or getattr(obj, "data", None) is None:
                continue

            name = obj.name.lower()
            if "shape" in name or "display" in name:
                mat = dark_mat
            elif "cylinder" in name or "handle" in name:
                mat = trim_mat
            else:
                mat = body_mat

            # Several supplier FBX files import with missing texture links, which
            # Blender renders as magenta. Keep the geometry, but force usable
            # refrigerator colors instead of broken materials.
            obj.data.materials.clear()
            obj.data.materials.append(mat)


def _create_or_import_appliance(
    assembly: dict[str, Any],
    role: str,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    fallback_mat,
    collection,
    layout_orientation: str = "x",
    aim_xy: tuple[float, float] | None = None,
) -> list[Any]:
    for asset_index, asset in enumerate(_appliance_asset_candidates(assembly, role), start=1):
        objects = _import_asset_objects(asset["asset_local_path"], collection)
        objects = _filter_imported_appliance_objects(role, objects, asset)
        orientation_applied = False
        if role != "faucet" and objects:
            _apply_asset_import_orientation(asset, objects, layout_orientation)
            orientation_applied = True
        if role == "sink" and objects and _fit_objects_to_footprint_top(
            objects,
            (center[0], center[1]),
            (size[0], size[1]),
            center[2] + size[2] / 2.0,
        ):
            fit_ok = True
        elif role == "cooktop" and objects and _fit_objects_to_footprint_top(
            objects,
            (center[0], center[1]),
            (size[0], size[1]),
            center[2] + size[2] / 2.0,
            margin=0.98,
        ):
            fit_ok = True
        elif role == "faucet" and objects and _fit_mesh_objects_to_box_baked(
            objects,
            center,
            size,
            margin=0.88,
            compact_disconnected=True,
        ):
            anchor_xy = _faucet_lowest_mount_xy(objects) or _faucet_base_anchor_xy(objects, (center[0], center[1]))
            if anchor_xy is not None:
                _translate_baked_mesh_objects_xy(
                    objects,
                    center[0] - anchor_xy[0],
                    center[1] - anchor_xy[1],
                )
            _rotate_baked_mesh_objects_around_point_z(
                objects,
                (center[0], center[1]),
                _asset_rotation_z_deg(asset, layout_orientation),
            )
            if aim_xy is not None:
                current_anchor = _faucet_lowest_mount_xy(objects) or (center[0], center[1])
                current_direction = _faucet_direction_xy(objects, current_anchor)
                target_direction = (aim_xy[0] - center[0], aim_xy[1] - center[1])
                if current_direction is not None and (target_direction[0] ** 2 + target_direction[1] ** 2) > 1e-6:
                    _rotate_baked_mesh_objects_around_point_z(
                        objects,
                        current_anchor,
                        _signed_angle_between_xy(current_direction, target_direction),
                    )
                    current_anchor = _faucet_lowest_mount_xy(objects)
                    if current_anchor is not None:
                        _translate_baked_mesh_objects_xy(
                            objects,
                            center[0] - current_anchor[0],
                            center[1] - current_anchor[1],
                        )
            orientation_applied = True
            fit_ok = True
        elif role == "microwave" and objects and _fit_objects_to_box(objects, center, size, margin=0.94):
            fit_ok = True
        else:
            fit_ok = bool(objects and _fit_objects_to_box(objects, center, size))
        if objects and fit_ok:
            tolerance = 1.35 if role in {"faucet", "hood"} else 1.12
            if not _objects_fit_within_size(objects, size, tolerance=tolerance):
                print(
                    "[kitchen] appliance asset rejected after fit: "
                    f"role={role} candidate={asset_index} title={asset.get('title')!r}"
                )
                _delete_objects(objects)
                continue
            if not orientation_applied:
                _apply_asset_import_orientation(asset, objects, layout_orientation)
            _sanitize_imported_appliance_materials(role, objects)
            for obj in objects:
                obj.name = f"{name}_{obj.name}"
                obj["kitchen_appliance_role"] = role
                obj["supplier_unique_key"] = asset.get("unique_key")
                obj["supplier_title"] = asset.get("title")
            return objects
        if objects:
            print(
                "[kitchen] appliance asset rejected: "
                f"role={role} candidate={asset_index} title={asset.get('title')!r}"
            )
            _delete_objects(objects)
    return [_create_box(name, center, size, fallback_mat, collection)]


def _create_or_import_countertop_decor_asset(
    assembly: dict[str, Any],
    role: str,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    bottom_z: float,
    fallback_mat,
    collection,
    layout_orientation: str = "x",
) -> list[Any]:
    asset = _appliance_asset(assembly, role)
    if asset:
        objects = _import_asset_objects(asset["asset_local_path"], collection)
        objects = _filter_imported_appliance_objects(role, objects, asset)
        if objects:
            _apply_asset_import_orientation(asset, objects, layout_orientation)
            if _fit_mesh_objects_to_box_baked(objects, center, size, margin=0.90, compact_disconnected=True):
                _snap_baked_mesh_objects_bottom_to_z(objects, bottom_z)
                for obj in objects:
                    obj.name = f"{name}_{obj.name}"
                    obj["kitchen_appliance_role"] = role
                    obj["supplier_unique_key"] = asset.get("unique_key")
                    obj["supplier_title"] = asset.get("title")
                return objects
            _delete_objects(objects)

    fallback = _create_box(name, center, size, fallback_mat, collection)
    _snap_objects_bottom_to_z([fallback], bottom_z)
    return [fallback]


def _create_cooktop_placeholder(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    panel_mat,
    burner_mat,
    collection,
) -> list[Any]:
    x, y, z = center
    w, d, h = size
    objects = [
        _create_box(name, (x, y, z), (w, d, max(0.008, h)), panel_mat, collection),
    ]
    burner_radius = min(w, d) * 0.13
    for idx, (dx, dy, radius_scale) in enumerate(
        [(-0.22, -0.22, 0.92), (0.22, -0.20, 0.72), (-0.22, 0.22, 0.72), (0.22, 0.22, 0.92)],
        start=1,
    ):
        objects.append(
            _create_cylinder(
                f"{name}_burner_{idx}",
                (x + dx * w, y + dy * d, z + h / 2 + 0.003),
                burner_radius * radius_scale,
                0.004,
                burner_mat,
                collection,
                vertices=64,
            )
        )
    return objects


def _create_handwash_placeholder(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    basin_mat,
    cutout_mat,
    collection,
) -> list[Any]:
    x, y, z = center
    w, d, h = size
    rim_height = 0.010
    bowl_height = max(0.07, h * 0.75)
    rail = min(w, d) * 0.08
    objects = [
        _create_box(f"{name}_rim_back", (x, y + d / 2 - rail / 2, z - rim_height / 2), (w, rail, rim_height), basin_mat, collection),
        _create_box(f"{name}_rim_front", (x, y - d / 2 + rail / 2, z - rim_height / 2), (w, rail, rim_height), basin_mat, collection),
        _create_box(f"{name}_rim_left", (x - w / 2 + rail / 2, y, z - rim_height / 2), (rail, d, rim_height), basin_mat, collection),
        _create_box(f"{name}_rim_right", (x + w / 2 - rail / 2, y, z - rim_height / 2), (rail, d, rim_height), basin_mat, collection),
        _create_box(f"{name}_bowl_floor", (x, y, z - rim_height - bowl_height + 0.004), (w * 0.62, d * 0.54, 0.008), cutout_mat, collection),
        _create_box(f"{name}_bowl_back", (x, y + d * 0.28, z - rim_height - bowl_height / 2), (w * 0.62, 0.008, bowl_height), cutout_mat, collection),
        _create_box(f"{name}_bowl_front", (x, y - d * 0.28, z - rim_height - bowl_height / 2), (w * 0.62, 0.008, bowl_height), cutout_mat, collection),
        _create_box(f"{name}_bowl_left", (x - w * 0.31, y, z - rim_height - bowl_height / 2), (0.008, d * 0.54, bowl_height), cutout_mat, collection),
        _create_box(f"{name}_bowl_right", (x + w * 0.31, y, z - rim_height - bowl_height / 2), (0.008, d * 0.54, bowl_height), cutout_mat, collection),
        _create_cylinder(f"{name}_drain", (x, y, z - rim_height - bowl_height + 0.004), min(w, d) * 0.055, 0.004, cutout_mat, collection),
    ]
    return objects


def _create_sink_placeholder(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    rim_mat,
    basin_mat,
    collection,
) -> list[Any]:
    x, y, z = center
    w, d, h = size
    rim_height = 0.012
    bowl_height = max(0.09, h * 0.85)
    rail = min(w, d) * 0.08
    objects = [
        _create_box(f"{name}_rim_back", (x, y + d / 2 - rail / 2, z - rim_height / 2), (w, rail, rim_height), rim_mat, collection),
        _create_box(f"{name}_rim_front", (x, y - d / 2 + rail / 2, z - rim_height / 2), (w, rail, rim_height), rim_mat, collection),
        _create_box(f"{name}_rim_left", (x - w / 2 + rail / 2, y, z - rim_height / 2), (rail, d, rim_height), rim_mat, collection),
        _create_box(f"{name}_rim_right", (x + w / 2 - rail / 2, y, z - rim_height / 2), (rail, d, rim_height), rim_mat, collection),
        _create_box(f"{name}_bowl_floor", (x, y, z - rim_height - bowl_height + 0.004), (w * 0.68, d * 0.62, 0.008), basin_mat, collection),
        _create_box(f"{name}_bowl_back", (x, y + d * 0.32, z - rim_height - bowl_height / 2), (w * 0.68, 0.008, bowl_height), basin_mat, collection),
        _create_box(f"{name}_bowl_front", (x, y - d * 0.32, z - rim_height - bowl_height / 2), (w * 0.68, 0.008, bowl_height), basin_mat, collection),
        _create_box(f"{name}_bowl_left", (x - w * 0.34, y, z - rim_height - bowl_height / 2), (0.008, d * 0.62, bowl_height), basin_mat, collection),
        _create_box(f"{name}_bowl_right", (x + w * 0.34, y, z - rim_height - bowl_height / 2), (0.008, d * 0.62, bowl_height), basin_mat, collection),
        _create_cylinder(f"{name}_drain", (x, y, z - rim_height - bowl_height + 0.004), min(w, d) * 0.055, 0.004, basin_mat, collection),
    ]
    return objects


def _create_sink_backing_basin(
    name: str,
    center_xy: tuple[float, float],
    opening_size: tuple[float, float],
    countertop_top_z: float,
    material,
    collection,
) -> list[Any]:
    x, y = center_xy
    w, d = opening_size
    wall = 0.014
    depth = 0.145
    top_z = countertop_top_z - 0.006
    bottom_z = top_z - depth
    objects = [
        _create_box(f"{name}_bottom", (x, y, bottom_z + wall / 2.0), (max(0.02, w), max(0.02, d), wall), material, collection),
        _create_box(f"{name}_left_wall", (x - w / 2.0 + wall / 2.0, y, bottom_z + depth / 2.0), (wall, max(0.02, d), depth), material, collection),
        _create_box(f"{name}_right_wall", (x + w / 2.0 - wall / 2.0, y, bottom_z + depth / 2.0), (wall, max(0.02, d), depth), material, collection),
        _create_box(f"{name}_back_wall", (x, y - d / 2.0 + wall / 2.0, bottom_z + depth / 2.0), (max(0.02, w), wall, depth), material, collection),
        _create_box(f"{name}_front_wall", (x, y + d / 2.0 - wall / 2.0, bottom_z + depth / 2.0), (max(0.02, w), wall, depth), material, collection),
    ]
    for obj in objects:
        obj["kitchen_appliance_role"] = "sink"
    return objects


def _scale_polygon_xy(
    polygon_xy: list[tuple[float, float]],
    inset_m: float,
) -> list[tuple[float, float]]:
    if not polygon_xy:
        return []
    cx = sum(point[0] for point in polygon_xy) / len(polygon_xy)
    cy = sum(point[1] for point in polygon_xy) / len(polygon_xy)
    scaled: list[tuple[float, float]] = []
    for x, y in polygon_xy:
        dx = x - cx
        dy = y - cy
        scale_x = max(0.0, (abs(dx) - inset_m) / max(abs(dx), 1e-6))
        scale_y = max(0.0, (abs(dy) - inset_m) / max(abs(dy), 1e-6))
        scaled.append((cx + dx * scale_x, cy + dy * scale_y))
    return scaled


def _fit_polygon_xy_to_opening(
    polygon_xy: list[tuple[float, float]],
    opening_center: tuple[float, float],
    opening_size: tuple[float, float],
) -> list[tuple[float, float]]:
    if len(polygon_xy) < 3:
        return polygon_xy

    min_x = min(point[0] for point in polygon_xy)
    max_x = max(point[0] for point in polygon_xy)
    min_y = min(point[1] for point in polygon_xy)
    max_y = max(point[1] for point in polygon_xy)
    current_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    current_size = (max(1e-6, max_x - min_x), max(1e-6, max_y - min_y))
    # The imported sink object has already been fitted to the requested slot.
    # The cutout must follow the actual visible rim contour, not the whole
    # procedural slot. Only clamp oversized contours caused by supplier helper
    # geometry or noisy vertices.
    scale_x = 1.0
    scale_y = 1.0
    max_x_size = max(0.08, opening_size[0] * 1.02)
    max_y_size = max(0.08, opening_size[1] * 1.02)
    if current_size[0] > max_x_size:
        scale_x = max_x_size / current_size[0]
    if current_size[1] > max_y_size:
        scale_y = max_y_size / current_size[1]

    return [
        (
            opening_center[0] + (x - current_center[0]) * scale_x,
            opening_center[1] + (y - current_center[1]) * scale_y,
        )
        for x, y in polygon_xy
    ]


def _create_polygon_sink_backing_basin(
    name: str,
    polygon_xy: list[tuple[float, float]],
    countertop_top_z: float,
    material,
    collection,
) -> list[Any]:
    bpy = _require_bpy()
    if len(polygon_xy) < 3:
        return []

    inner = _scale_polygon_xy(polygon_xy, 0.018)
    if len(inner) < 3:
        inner = polygon_xy
    depth = 0.145
    wall_top_z = countertop_top_z - 0.006
    bottom_z = wall_top_z - depth
    count = len(inner)
    vertices = [(x, y, bottom_z) for x, y in inner] + [(x, y, wall_top_z) for x, y in inner]
    faces: list[tuple[int, ...]] = [tuple(reversed(range(count)))]
    for idx in range(count):
        nxt = (idx + 1) % count
        faces.append((idx, nxt, nxt + count, idx + count))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    obj["kitchen_appliance_role"] = "sink"
    if collection is not None:
        collection.objects.link(obj)
    else:
        bpy.context.scene.collection.objects.link(obj)

    return [obj]


def _create_faucet_placeholder(
    name: str,
    base: tuple[float, float, float],
    orientation: str,
    metal_mat,
    collection,
) -> list[Any]:
    x, y, z = base
    objects = [
        _create_cylinder(f"{name}_stem", (x, y, z + 0.17), 0.024, 0.34, metal_mat, collection, vertices=32),
    ]
    if orientation == "y":
        objects.append(_create_box(f"{name}_spout", (x + 0.095, y, z + 0.335), (0.19, 0.030, 0.030), metal_mat, collection))
        objects.append(_create_cylinder(f"{name}_nozzle", (x + 0.180, y, z + 0.270), 0.015, 0.10, metal_mat, collection, vertices=24))
    else:
        objects.append(_create_box(f"{name}_spout", (x, y + 0.095, z + 0.335), (0.030, 0.19, 0.030), metal_mat, collection))
        objects.append(_create_cylinder(f"{name}_nozzle", (x, y + 0.180, z + 0.270), 0.015, 0.10, metal_mat, collection, vertices=24))
    return objects


def _create_dishwasher_placeholder(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    body_mat,
    detail_mat,
    collection,
) -> list[Any]:
    x, y, z = center
    w, d, h = size
    front_y = y + d / 2 + 0.004
    objects = [_create_box(name, center, size, body_mat, collection)]
    objects.append(_create_box(f"{name}_front_panel", (x, front_y, z + h * 0.02), (w * 0.90, 0.012, h * 0.82), body_mat, collection))
    objects.append(_create_box(f"{name}_control_strip", (x, front_y + 0.004, z + h * 0.36), (w * 0.78, 0.008, h * 0.035), detail_mat, collection))
    objects.append(_create_box(f"{name}_handle", (x, front_y + 0.008, z + h * 0.25), (w * 0.46, 0.010, h * 0.025), detail_mat, collection))
    return objects


def _create_integrated_appliance_front(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    orientation: str,
    facade_mat,
    handle_mat,
    collection,
) -> list[Any]:
    x, y, z = center
    w, d, h = size
    objects = [_create_box(f"{name}_hidden_body", center, size, facade_mat, collection)]
    if orientation == "y":
        front_x = x + w / 2 + 0.006
        objects.append(_create_box(f"{name}_integrated_front", (front_x, y, z + h * 0.02), (0.012, d * 0.92, h * 0.82), facade_mat, collection))
        objects.append(_create_box(f"{name}_handle", (front_x + 0.006, y, z + h * 0.28), (0.010, d * 0.42, h * 0.025), handle_mat, collection))
    else:
        front_y = y + d / 2 + 0.006
        objects.append(_create_box(f"{name}_integrated_front", (x, front_y, z + h * 0.02), (w * 0.92, 0.012, h * 0.82), facade_mat, collection))
        objects.append(_create_box(f"{name}_handle", (x, front_y + 0.006, z + h * 0.28), (w * 0.42, 0.010, h * 0.025), handle_mat, collection))
    return objects


def _create_fridge_placeholder(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    orientation: str,
    body_mat,
    handle_mat,
    collection,
) -> list[Any]:
    x, y, z = center
    w, d, h = size
    objects = [_create_box(name, center, size, body_mat, collection)]
    if orientation == "y":
        front_x = x + w / 2 + 0.006
        objects.append(_create_box(f"{name}_door_line", (front_x + 0.002, y, z + h * 0.12), (0.004, d * 0.90, 0.012), handle_mat, collection))
        objects.append(_create_box(f"{name}_upper_handle_recess", (front_x + 0.014, y - d * 0.18, z - h * 0.06), (0.022, d * 0.14, 0.018), handle_mat, collection))
        objects.append(_create_box(f"{name}_lower_handle_recess", (front_x + 0.014, y + d * 0.18, z - h * 0.06), (0.022, d * 0.14, 0.018), handle_mat, collection))
        objects.append(_create_box(f"{name}_display", (front_x + 0.016, y, z + h * 0.30), (0.026, d * 0.12, h * 0.085), handle_mat, collection))
    else:
        front_y = y + d / 2 + 0.006
        objects.append(_create_box(f"{name}_door_line", (x, front_y + 0.002, z + h * 0.12), (w * 0.90, 0.004, 0.012), handle_mat, collection))
        objects.append(_create_box(f"{name}_upper_handle_recess", (x - w * 0.18, front_y + 0.014, z - h * 0.06), (w * 0.14, 0.022, 0.018), handle_mat, collection))
        objects.append(_create_box(f"{name}_lower_handle_recess", (x + w * 0.18, front_y + 0.014, z - h * 0.06), (w * 0.14, 0.022, 0.018), handle_mat, collection))
        objects.append(_create_box(f"{name}_display", (x, front_y + 0.016, z + h * 0.30), (w * 0.10, 0.026, h * 0.085), handle_mat, collection))
    return objects


def _create_hood_placeholder(
    name: str,
    origin_x: float,
    origin_y: float,
    width: float,
    depth: float,
    z: float,
    orientation: str,
    body_mat,
    dark_mat,
    collection,
) -> list[Any]:
    objects: list[Any] = []
    if orientation == "y":
        objects.append(_create_box(f"{name}_hood_body", (origin_x + depth / 2, origin_y + width / 2, z + 0.10), (depth, width * 0.92, 0.12), body_mat, collection))
        objects.append(_create_box(f"{name}_hood_filter", (origin_x + depth + 0.006, origin_y + width / 2, z + 0.03), (0.012, width * 0.78, 0.18), dark_mat, collection))
        objects.append(_create_box(f"{name}_chimney", (origin_x + depth * 0.30, origin_y + width / 2, z + 0.33), (depth * 0.42, width * 0.32, 0.46), body_mat, collection))
    else:
        objects.append(_create_box(f"{name}_hood_body", (origin_x + width / 2, origin_y + depth / 2, z + 0.10), (width * 0.92, depth, 0.12), body_mat, collection))
        objects.append(_create_box(f"{name}_hood_filter", (origin_x + width / 2, origin_y + depth + 0.006, z + 0.03), (width * 0.78, 0.012, 0.18), dark_mat, collection))
        objects.append(_create_box(f"{name}_chimney", (origin_x + width / 2, origin_y + depth * 0.30, z + 0.33), (width * 0.32, depth * 0.42, 0.46), body_mat, collection))
    return objects


def _find_cooktop_cutout_center(
    assembly: dict[str, Any],
    module_id: str | None,
    px: float,
    py: float,
) -> tuple[float, float, float, float, str] | None:
    if not module_id:
        return None
    for segment in assembly.get("countertop_segments") or []:
        orientation = segment.get("orientation") or "x"
        segment_x = float(segment.get("x_m", 0.0))
        segment_y = float(segment.get("y_m", 0.0))
        for cutout in segment.get("cutouts") or []:
            if cutout.get("type") != "cooktop" or cutout.get("module_id") != module_id:
                continue
            cx = float(cutout.get("x_m", 0.0))
            cy = float(cutout.get("y_m", 0.0))
            cw = float(cutout.get("width_m", 0.56))
            cd = float(cutout.get("depth_m", 0.49))
            sx, sy = _surface_point_local(px, py, segment_x, segment_y, cx + cw / 2.0, cy + cd / 2.0, orientation)
            return sx, sy, cw, cd, orientation
    return None


def _create_kitchen_decor_item(
    assembly: dict[str, Any],
    item: dict[str, Any],
    px: float,
    py: float,
    pz: float,
    mat,
    accent_mat,
    collection,
) -> list[Any]:
    orientation = item.get("orientation") or "x"
    x = float(item.get("x_m", 0.0))
    y = float(item.get("y_m", 0.0))
    z = float(item.get("z_m", 0.86))
    item_type = item.get("type")
    sx, sy = _surface_point(px, py, 0.0, 0.0, x, y, orientation)
    if item_type == "cutting_board":
        size = (0.28, 0.16, 0.018)
        if orientation == "y":
            size = (size[1], size[0], size[2])
        return [_create_box(item.get("id", "decor_cutting_board"), (sx, sy, pz + z + 0.012), size, mat, collection)]
    if item_type == "microwave":
        if item.get("placement") == "upper_open_shelf":
            shelf_width = float(item.get("shelf_width_m") or 0.50)
            shelf_depth = float(item.get("shelf_depth_m") or 0.32)
            size = (max(0.28, min(0.42, shelf_width - 0.06)), max(0.22, min(0.30, shelf_depth - 0.04)), 0.24)
        else:
            size = (0.45, 0.34, 0.26)
        if orientation == "y":
            size = (size[1], size[0], size[2])
        objects = _create_or_import_appliance(
            assembly,
            "microwave",
            item.get("id", "decor_microwave"),
            (sx, sy, pz + z + size[2] / 2.0),
            size,
            accent_mat,
            collection,
            layout_orientation=orientation,
        )
        if item.get("placement") == "countertop":
            _snap_objects_bottom_to_z(objects, pz + z)
        return objects
    if item_type == "small_kitchen_appliance":
        size = (0.24, 0.24, 0.30)
        if orientation == "y":
            size = (size[1], size[0], size[2])
        objects = _create_or_import_countertop_decor_asset(
            assembly,
            "small_kitchen_appliance",
            item.get("id", "decor_countertop_appliance"),
            (sx, sy, pz + z + size[2] / 2.0),
            size,
            pz + z,
            mat,
            collection,
            layout_orientation=orientation,
        )
        return objects
    if item_type in {"flowers_vase", "oil_bottles_decor", "decorative_kitchen_set"}:
        sizes = {
            "flowers_vase": (0.28, 0.28, 0.46),
            "oil_bottles_decor": (0.26, 0.16, 0.32),
            "decorative_kitchen_set": (0.34, 0.24, 0.18),
        }
        size = sizes[item_type]
        if orientation == "y":
            size = (size[1], size[0], size[2])
        objects = _create_or_import_countertop_decor_asset(
            assembly,
            item_type,
            item.get("id", f"decor_{item_type}"),
            (sx, sy, pz + z + size[2] / 2.0),
            size,
            pz + z,
            mat,
            collection,
            layout_orientation=orientation,
        )
        return objects
    return []


def build_kitchen_assembly_in_blender(
    assembly: dict[str, Any],
    parent_collection_name: str = "ProceduralKitchen",
    use_boolean_cutouts: bool = False,
) -> list[Any]:
    del use_boolean_cutouts
    collection = _collection(parent_collection_name)
    created: list[Any] = []
    bindings = assembly.get("material_bindings") or {}
    body_mat = _get_or_create_material(
        "kitchen_body_basisrf",
        _visual_color(bindings.get("body")),
        _resolve_texture_path(bindings.get("body")),
    )
    facade_mat = _get_or_create_material(
        "kitchen_facade_basisrf",
        _visual_color(bindings.get("facade")),
        _resolve_texture_path(bindings.get("facade")),
    )
    countertop_mat = _get_or_create_material(
        "kitchen_countertop_basisrf",
        _visual_color(bindings.get("countertop")),
        _resolve_texture_path(bindings.get("countertop")),
    )
    backsplash_mat = _get_or_create_material(
        "kitchen_backsplash_basisrf",
        _visual_color(bindings.get("backsplash")),
        _resolve_texture_path(bindings.get("backsplash")),
    )
    appliance_mat = _get_or_create_material("kitchen_appliance", (0.12, 0.12, 0.12, 1.0))
    fridge_mat = _get_or_create_material("kitchen_fridge_light_gray", (0.62, 0.62, 0.60, 1.0))
    cutout_mat = _get_or_create_material("kitchen_cutout_dark", (0.01, 0.01, 0.01, 1.0))
    cooktop_mat = _get_or_create_material("kitchen_cooktop_glass", (0.005, 0.005, 0.005, 1.0))
    burner_mat = _get_or_create_material("kitchen_cooktop_burners", (0.10, 0.10, 0.10, 1.0))
    basin_mat = _get_or_create_material("kitchen_basin_ceramic", (0.92, 0.92, 0.88, 1.0))
    sink_rim_mat = _get_or_create_material("kitchen_sink_stainless", (0.62, 0.64, 0.63, 1.0))
    sink_bowl_mat = _get_or_create_material("kitchen_sink_bowl_dark", (0.08, 0.08, 0.075, 1.0))
    dishwasher_mat = _get_or_create_material("kitchen_dishwasher_white", (0.86, 0.86, 0.84, 1.0))
    faucet_mat = _get_or_create_material("kitchen_faucet_chrome", (0.58, 0.60, 0.60, 1.0))
    decor_mat = _get_or_create_material("kitchen_decor_warm_neutral", (0.72, 0.62, 0.48, 1.0))
    px, py, pz = assembly.get("position") or [0.0, 0.0, 0.0]

    def world(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (px + x, py + y, pz + z)

    base_carcasses: dict[str, tuple[Any, dict[str, Any]]] = {}

    for module in assembly.get("base_modules") or []:
        x = float(module.get("x_m", 0.0))
        y = float(module.get("y_m", 0.0))
        w = float(module.get("width_m", 0.6))
        d = float(module.get("depth_m", 0.56))
        h = float(module.get("height_m", 0.72))
        z = float(module.get("z_m", 0.1))
        module_type = module.get("type") or "base"
        orientation = module.get("orientation") or "x"
        if module_type == "fridge_slot":
            cx, cy = _surface_point(px, py, x, y, x + w / 2 if orientation == "x" else y + w / 2, d / 2, orientation)
            fridge_size = (d, w, h) if orientation == "y" else (w, d, h)
            if _appliance_asset(assembly, "fridge"):
                created.extend(
                    _create_or_import_appliance(
                        assembly,
                        "fridge",
                        module.get("id", "fridge"),
                        (cx, cy, pz + z + h / 2),
                        fridge_size,
                        fridge_mat,
                        collection,
                        layout_orientation=orientation,
                    )
                )
            else:
                created.extend(_create_fridge_placeholder(module.get("id", "fridge"), (cx, cy, pz + z + h / 2), fridge_size, orientation, fridge_mat, appliance_mat, collection))
            continue
        if module_type in {"washing_machine_slot", "dishwasher_slot"}:
            role = "washing_machine" if module_type == "washing_machine_slot" else "dishwasher"
            cx, cy = _surface_point(px, py, x, y, x + w / 2 if orientation == "x" else y + w / 2, d / 2, orientation)
            if role == "dishwasher" and _appliance_asset(assembly, role) is None:
                integrated_h = min(0.70, h * 0.82)
                created.extend(
                    _create_integrated_appliance_front(
                        module.get("id", module_type),
                        (cx, cy, pz + z + integrated_h / 2),
                        ((d * 0.94, w * 0.96, integrated_h) if orientation == "y" else (w * 0.96, d * 0.94, integrated_h)),
                        orientation,
                        facade_mat,
                        appliance_mat,
                        collection,
                    )
                )
            else:
                created.extend(
                    _create_or_import_appliance(
                        assembly,
                        role,
                        module.get("id", module_type),
                        (cx, cy, pz + z + h / 2),
                        ((d * 0.94, w * 0.96, h * 0.96) if orientation == "y" else (w * 0.96, d * 0.94, h * 0.96)),
                        appliance_mat,
                        collection,
                        layout_orientation=orientation,
                    )
                )
            continue
        carcass = _create_oriented_box(module.get("id", module_type), px + x, py + y, w, d, h, pz + z, orientation, body_mat, collection)
        created.append(carcass)
        if module.get("id"):
            base_carcasses[str(module["id"])] = (carcass, module)
        if not module.get("has_facade", True):
            continue
        facade_layout = module.get("facade_layout")
        if facade_layout == "three_drawers":
            for i in range(3):
                fh = h / 3.0 - 0.006
                if orientation == "y":
                    created.append(_create_box(f"{module.get('id')}_drawer_{i + 1}", world(x + d + 0.012, y + w / 2, z + fh / 2.0 + i * h / 3.0), (0.024, w * 0.96, fh), facade_mat, collection))
                else:
                    created.append(_create_box(f"{module.get('id')}_drawer_{i + 1}", world(x + w / 2, y + d + 0.012, z + fh / 2.0 + i * h / 3.0), (w * 0.96, 0.024, fh), facade_mat, collection))
        elif facade_layout == "two_doors":
            for i in range(2):
                fw = w / 2.0 - 0.006
                if orientation == "y":
                    created.append(_create_box(f"{module.get('id')}_door_{i + 1}", world(x + d + 0.012, y + fw / 2.0 + i * w / 2.0, z + h / 2), (0.024, fw, h * 0.96), facade_mat, collection))
                else:
                    created.append(_create_box(f"{module.get('id')}_door_{i + 1}", world(x + fw / 2.0 + i * w / 2.0, y + d + 0.012, z + h / 2), (fw, 0.024, h * 0.96), facade_mat, collection))
        elif facade_layout == "oven_front":
            cx, cy = _surface_point(px, py, x, y, x + w / 2 if orientation == "x" else y + w / 2, max(0.08, d - 0.08), orientation)
            created.extend(
                _create_or_import_appliance(
                    assembly,
                    "oven",
                    f"{module.get('id')}_oven",
                    (cx, cy, pz + z + h / 2),
                    ((0.16, w * 0.92, h * 0.78) if orientation == "y" else (w * 0.92, 0.16, h * 0.78)),
                    appliance_mat,
                    collection,
                    layout_orientation=orientation,
                )
            )
        else:
            if orientation == "y":
                created.append(_create_box(f"{module.get('id')}_facade", world(x + d + 0.012, y + w / 2, z + h / 2), (0.024, w * 0.96, h * 0.96), facade_mat, collection))
            else:
                created.append(_create_box(f"{module.get('id')}_facade", world(x + w / 2, y + d + 0.012, z + h / 2), (w * 0.96, 0.024, h * 0.96), facade_mat, collection))

    for segment in assembly.get("countertop_segments") or []:
        x = float(segment.get("x_m", 0.0))
        y = float(segment.get("y_m", 0.0))
        w = float(segment.get("width_m", 0.0))
        d = float(segment.get("depth_m", 0.6))
        h = float(segment.get("thickness_m", 0.038))
        z = float(segment.get("z_m", 0.82))
        orientation = segment.get("orientation") or "x"
        countertop_obj = _create_oriented_box(
            segment.get("id", "countertop"),
            px + x,
            py + y,
            w,
            d,
            h,
            pz + z,
            orientation,
            countertop_mat,
            collection,
        )
        created.append(countertop_obj)
        for cutout in segment.get("cutouts") or []:
            cx = float(cutout.get("x_m", 0.0))
            cy = float(cutout.get("y_m", 0.0))
            cw = float(cutout.get("width_m", 0.3))
            cd = float(cutout.get("depth_m", 0.25))
            sx, sy = _surface_point_local(px, py, x, y, cx + cw / 2, cy + cd / 2, orientation)
            cutout_size = (cd, cw, 0.004) if orientation == "y" else (cw, cd, 0.004)
            if cutout.get("type") == "sink":
                sink_height = 0.16
                sink_size = (cutout_size[0] * 0.92, cutout_size[1] * 0.92, sink_height)
                sink_hole_size = (sink_size[0] * 0.86, sink_size[1] * 0.84, h * 3.0)
                sink_asset = _appliance_asset(assembly, "sink")
                sink_imported = False
                sink_hole_center = (sx, sy)
                countertop_cut_ok = False
                imported_sink_objects: list[Any] = []
                if sink_asset:
                    sink_asset_title = _asset_title(sink_asset)
                    sink_asset_mat = (
                        cutout_mat
                        if any(term in sink_asset_title for term in ("черн", "black", "pvd"))
                        else sink_rim_mat
                    )
                    imported_sink_objects = _create_or_import_appliance(
                        assembly,
                        "sink",
                        f"{segment.get('id')}_sink_asset",
                        (sx, sy, pz + z + h + 0.004 - sink_height / 2),
                        sink_size,
                        sink_asset_mat,
                        collection,
                        layout_orientation=orientation,
                    )
                    created.extend(imported_sink_objects)
                    sink_imported = any(obj.get("kitchen_appliance_role") == "sink" for obj in imported_sink_objects)
                    if sink_imported:
                        _require_bpy().context.view_layer.update()
                        countertop_top = pz + z + h
                        sink_outer_polygon = _mesh_outer_hull_xy_from_objects(
                            imported_sink_objects,
                            sample_z_min=countertop_top - 0.016,
                            sample_z_max=countertop_top + 0.018,
                            inset_m=0.0,
                        )
                        real_opening = _real_bbox_opening_from_objects(imported_sink_objects)
                        if sink_outer_polygon:
                            sink_outer_polygon = _fit_polygon_xy_to_opening(
                                sink_outer_polygon,
                                sink_hole_center,
                                (sink_size[0], sink_size[1]),
                            )
                        if sink_outer_polygon:
                            countertop_cut_ok = _apply_polygon_cutout(
                                countertop_obj,
                                f"{segment.get('id')}_{cutout.get('type')}",
                                sink_outer_polygon,
                                cutter_z_min=pz + z - h * 0.70,
                                cutter_z_max=countertop_top + h * 0.45,
                                collection=collection,
                            )
                            created.extend(
                                _create_polygon_sink_backing_basin(
                                    f"{segment.get('id')}_sink_fbx_polygon_backing_basin",
                                    sink_outer_polygon,
                                    countertop_top,
                                    cutout_mat,
                                    collection,
                                )
                            )
                            min_x = min(point[0] for point in sink_outer_polygon)
                            max_x = max(point[0] for point in sink_outer_polygon)
                            min_y = min(point[1] for point in sink_outer_polygon)
                            max_y = max(point[1] for point in sink_outer_polygon)
                            sink_hole_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
                            sink_hole_size = (max_x - min_x, max_y - min_y, h * 3.0)
                        elif real_opening is not None:
                            real_center, real_size = real_opening
                            sink_hole_center = real_center
                            sink_hole_size = (real_size[0], real_size[1], h * 3.0)
                            countertop_cut_ok = _apply_rectangular_cutout(
                                countertop_obj,
                                f"{segment.get('id')}_{cutout.get('type')}",
                                (sink_hole_center[0], sink_hole_center[1], pz + z + h / 2.0),
                                sink_hole_size,
                                collection,
                            )
                            created.extend(
                                _create_sink_backing_basin(
                                    f"{segment.get('id')}_sink_fbx_backing_basin",
                                    sink_hole_center,
                                    (real_size[0], real_size[1]),
                                    countertop_top,
                                    cutout_mat,
                                    collection,
                                )
                            )
                else:
                    created.append(
                        _create_box(
                            f"{segment.get('id')}_{cutout.get('type')}_recess",
                            (sx, sy, pz + z + h + 0.001),
                            cutout_size,
                            cutout_mat,
                            collection,
                        )
                    )
                    created.extend(
                        _create_sink_placeholder(
                            f"{segment.get('id')}_sink",
                            (sx, sy, pz + z + h),
                            sink_size,
                            sink_rim_mat,
                            sink_bowl_mat,
                            collection,
                        )
                    )
                if not countertop_cut_ok:
                    _apply_rectangular_cutout(
                        countertop_obj,
                        f"{segment.get('id')}_{cutout.get('type')}",
                        (sink_hole_center[0], sink_hole_center[1], pz + z + h / 2.0),
                        sink_hole_size,
                        collection,
                    )
                base_entry = base_carcasses.get(str(cutout.get("module_id") or ""))
                if base_entry:
                    base_obj, base_module = base_entry
                    base_z = float(base_module.get("z_m", 0.1))
                    base_h = float(base_module.get("height_m", 0.72))
                    _apply_rectangular_cutout(
                        base_obj,
                        f"{base_module.get('id')}_{cutout.get('type')}_cabinet",
                        (sink_hole_center[0], sink_hole_center[1], pz + base_z + base_h / 2.0),
                        (sink_hole_size[0], sink_hole_size[1], base_h * 1.15),
                        collection,
                    )
                if not sink_imported:
                    created.extend(
                        _create_sink_placeholder(
                            f"{segment.get('id')}_sink_visible_insert",
                            (sx, sy, pz + z + h + 0.026),
                            sink_size,
                            cutout_mat,
                            sink_bowl_mat,
                            collection,
                        )
                    )
                faucet_x, faucet_y = _surface_point_local(
                    px,
                    py,
                    x,
                    y,
                    cx + cw / 2,
                    max(0.08, cy + 0.015),
                    orientation,
                )
                if sink_asset and _sink_asset_includes_faucet(sink_asset):
                    pass
                elif _appliance_asset(assembly, "faucet"):
                    faucet_size = (0.24, 0.40, 0.36) if orientation == "x" else (0.40, 0.24, 0.36)
                    imported_faucet = _create_or_import_appliance(
                        assembly,
                        "faucet",
                        f"{segment.get('id')}_sink_faucet_asset",
                        (faucet_x, faucet_y, pz + z + h + faucet_size[2] / 2.0),
                        faucet_size,
                        faucet_mat,
                        collection,
                        layout_orientation=orientation,
                        aim_xy=sink_hole_center,
                    )
                    if any(obj.get("kitchen_appliance_role") == "faucet" for obj in imported_faucet):
                        created.extend(imported_faucet)
                    else:
                        created.extend(_create_faucet_placeholder(f"{segment.get('id')}_sink_faucet", (faucet_x, faucet_y, pz + z + h + 0.005), orientation, faucet_mat, collection))
                else:
                    created.extend(_create_faucet_placeholder(f"{segment.get('id')}_sink_faucet", (faucet_x, faucet_y, pz + z + h + 0.005), orientation, faucet_mat, collection))
            elif cutout.get("type") == "entry_handwash":
                created.append(_create_box(f"{segment.get('id')}_{cutout.get('type')}_visual_cutout", (sx, sy, pz + z + h + 0.002), cutout_size, cutout_mat, collection))
                handwash_size = (cd, cw, 0.10) if orientation == "y" else (cw, cd, 0.10)
                created.extend(
                    _create_handwash_placeholder(
                        f"{segment.get('id')}_entry_handwash",
                        (sx, sy, pz + z + h),
                        handwash_size,
                        basin_mat,
                        cutout_mat,
                        collection,
                    )
                )
                faucet_x, faucet_y = _surface_point_local(px, py, x, y, cx + cw / 2, max(0.03, cy - 0.045), orientation)
                created.extend(_create_faucet_placeholder(f"{segment.get('id')}_entry_handwash_faucet", (faucet_x, faucet_y, pz + z + h + 0.005), orientation, faucet_mat, collection))
            elif cutout.get("type") == "cooktop":
                _apply_rectangular_cutout(
                    countertop_obj,
                    f"{segment.get('id')}_{cutout.get('type')}",
                    (sx, sy, pz + z + h / 2.0),
                    (cutout_size[0] * 0.96, cutout_size[1] * 0.96, h * 2.4),
                    collection,
                )
                cooktop_asset = _appliance_asset(assembly, "cooktop")
                cooktop_size = (cd, cw, 0.012) if orientation == "y" else (cw, cd, 0.012)
                cooktop_imported = False
                if cooktop_asset:
                    imported_cooktop_objects = _create_or_import_appliance(
                        assembly,
                        "cooktop",
                        f"{segment.get('id')}_cooktop_asset",
                        (sx, sy, pz + z + h - 0.002),
                        cooktop_size,
                        appliance_mat,
                        collection,
                        layout_orientation=orientation,
                    )
                    created.extend(imported_cooktop_objects)
                    cooktop_imported = any(obj.get("kitchen_appliance_role") == "cooktop" for obj in imported_cooktop_objects)
                    if cooktop_imported:
                        # Some supplier induction panels import as a very flat glass
                        # slab with weak ring contrast. Keep the FBX body, but add
                        # thin visible burner rings so the cooktop reads correctly.
                        _, _, cooktop_h = cooktop_size
                        burner_radius = min(cooktop_size[0], cooktop_size[1]) * 0.13
                        for idx, (dx, dy, radius_scale) in enumerate(
                            [(-0.22, -0.22, 0.92), (0.22, -0.20, 0.72), (-0.22, 0.22, 0.72), (0.22, 0.22, 0.92)],
                            start=1,
                        ):
                            ring = _create_torus(
                                f"{segment.get('id')}_cooktop_asset_visible_ring_{idx}",
                                (sx + dx * cooktop_size[0], sy + dy * cooktop_size[1], pz + z + h + cooktop_h / 2 + 0.005),
                                burner_radius * radius_scale,
                                0.004,
                                burner_mat,
                                collection,
                            )
                            ring["kitchen_appliance_role"] = "cooktop"
                            created.append(ring)
                if not cooktop_imported:
                    created.extend(
                        _create_cooktop_placeholder(
                            f"{segment.get('id')}_cooktop_visible_flush",
                            (sx, sy, pz + z + h + 0.003),
                            cooktop_size,
                            cooktop_mat,
                            burner_mat,
                            collection,
                        )
                    )

    for panel in assembly.get("backsplash_segments") or []:
        x = float(panel.get("x_m", 0.0))
        y = float(panel.get("y_m", 0.0))
        w = float(panel.get("width_m", 0.0))
        h = float(panel.get("height_m", 0.6))
        t = float(panel.get("thickness_m", 0.004))
        z = float(panel.get("z_m", 0.858))
        orientation = panel.get("orientation") or "x"
        if orientation == "y":
            created.append(_create_box(panel.get("id", "backsplash"), world(x + t / 2, y + w / 2, z + h / 2), (t, w, h), backsplash_mat, collection))
        else:
            created.append(_create_box(panel.get("id", "backsplash"), world(x + w / 2, y + t / 2, z + h / 2), (w, t, h), backsplash_mat, collection))

    for module in assembly.get("upper_modules") or []:
        x = float(module.get("x_m", 0.0))
        y = float(module.get("y_m", 0.0))
        w = float(module.get("width_m", 0.6))
        d = float(module.get("depth_m", 0.32))
        h = float(module.get("height_m", 0.72))
        z = float(module.get("z_m", 1.458))
        orientation = module.get("orientation") or "x"
        if module.get("type") in {"hood_cabinet", "hood_wall_mounted", "hood_compact_wall"}:
            cooktop_anchor = _find_cooktop_cutout_center(
                assembly,
                module.get("above_base_module_id"),
                px,
                py,
            )
            hood_width = min(w * 0.96, 0.60)
            hood_depth = min(d * 1.10, 0.38)
            hood_z = pz + z + 0.05
            if cooktop_anchor:
                cooktop_x, cooktop_y, cooktop_w, _, cooktop_orientation = cooktop_anchor
                orientation = cooktop_orientation
                hood_width = min(max(cooktop_w, 0.56), 0.72)
                hx = cooktop_x - (hood_depth / 2.0 if orientation == "y" else hood_width / 2.0)
                hy = cooktop_y - (hood_width / 2.0 if orientation == "y" else hood_depth / 2.0)
            else:
                hx = px + x + (w - hood_width) / 2.0 if orientation == "x" else px + x
                hy = py + y if orientation == "x" else py + y + (w - hood_width) / 2.0

            if module.get("type") == "hood_cabinet":
                backing_height = max(h + 0.16, 0.52)
                backing_thickness = 0.006
                if orientation == "y":
                    created.append(
                        _create_box(
                            f"{module.get('id')}_hood_backsplash_extension",
                            (px + backing_thickness / 2.0, hy + hood_width / 2.0, pz + z + backing_height / 2.0),
                            (backing_thickness, hood_width, backing_height),
                            backsplash_mat,
                            collection,
                        )
                    )
                else:
                    created.append(
                        _create_box(
                            f"{module.get('id')}_hood_backsplash_extension",
                            (hx + hood_width / 2.0, py + backing_thickness / 2.0, pz + z + backing_height / 2.0),
                            (hood_width, backing_thickness, backing_height),
                            backsplash_mat,
                            collection,
                        )
                    )

            hood_height = 0.34 if module.get("type") == "hood_compact_wall" else 0.42
            hood_size = (hood_width, hood_depth, hood_height) if orientation == "x" else (hood_depth, hood_width, hood_height)
            if _appliance_asset(assembly, "hood"):
                center = (
                    hx + (hood_depth / 2.0 if orientation == "y" else hood_width / 2.0),
                    hy + (hood_width / 2.0 if orientation == "y" else hood_depth / 2.0),
                    hood_z + hood_size[2] / 2.0,
                )
                hood_objects = _create_or_import_appliance(
                    assembly,
                    "hood",
                    f"{module.get('id')}_hood_asset",
                    center,
                    hood_size,
                    appliance_mat,
                    collection,
                    layout_orientation=orientation,
                )
                created.extend(hood_objects)
            else:
                created.extend(_create_hood_placeholder(f"{module.get('id')}_hood", hx, hy, hood_width, hood_depth, hood_z, orientation, appliance_mat, cutout_mat, collection))
        elif module.get("type") == "microwave_open_shelf":
            panel = 0.024
            back = 0.018
            if orientation == "y":
                created.append(_create_box(f"{module.get('id')}_bottom", world(x + d / 2, y + w / 2, z + panel / 2), (d, w, panel), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_top", world(x + d / 2, y + w / 2, z + h - panel / 2), (d, w, panel), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_left_side", world(x + d / 2, y + panel / 2, z + h / 2), (d, panel, h), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_right_side", world(x + d / 2, y + w - panel / 2, z + h / 2), (d, panel, h), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_back", world(x + back / 2, y + w / 2, z + h / 2), (back, w, h), body_mat, collection))
            else:
                created.append(_create_box(f"{module.get('id')}_bottom", world(x + w / 2, y + d / 2, z + panel / 2), (w, d, panel), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_top", world(x + w / 2, y + d / 2, z + h - panel / 2), (w, d, panel), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_left_side", world(x + panel / 2, y + d / 2, z + h / 2), (panel, d, h), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_right_side", world(x + w - panel / 2, y + d / 2, z + h / 2), (panel, d, h), body_mat, collection))
                created.append(_create_box(f"{module.get('id')}_back", world(x + w / 2, y + back / 2, z + h / 2), (w, back, h), body_mat, collection))
        else:
            created.append(_create_oriented_box(module.get("id", "upper"), px + x, py + y, w, d, h, pz + z, orientation, body_mat, collection))
            if orientation == "y":
                created.append(_create_box(f"{module.get('id')}_facade", world(x + d + 0.010, y + w / 2, z + h / 2), (0.020, w * 0.96, h * 0.96), facade_mat, collection))
            else:
                created.append(_create_box(f"{module.get('id')}_facade", world(x + w / 2, y + d + 0.010, z + h / 2), (w * 0.96, 0.020, h * 0.96), facade_mat, collection))
    for item in assembly.get("decor_items") or []:
        created.extend(_create_kitchen_decor_item(assembly, item, px, py, pz, decor_mat, faucet_mat, collection))
    _apply_assembly_rotation(created, assembly, (px, py, pz))
    return created


def _apply_assembly_rotation(objects: list[Any], assembly: dict[str, Any], pivot: tuple[float, float, float]) -> None:
    rotation = assembly.get("rotation") or [0.0, 0.0, 0.0]
    if not isinstance(rotation, (list, tuple)) or len(rotation) < 3:
        return
    yaw = float(rotation[2] or 0.0)
    if abs(yaw) <= math.tau:
        yaw_rad = yaw
    else:
        yaw_rad = math.radians(yaw)
    if abs(yaw_rad) < 1e-8:
        return
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    px, py, _ = pivot
    object_set = set(objects)
    roots = [obj for obj in objects if getattr(obj, "parent", None) not in object_set]
    for obj in roots:
        loc = obj.location
        dx = float(loc.x) - px
        dy = float(loc.y) - py
        loc.x = px + dx * cos_y - dy * sin_y
        loc.y = py + dx * sin_y + dy * cos_y
        try:
            obj.rotation_euler.rotate_axis("Z", yaw_rad)
        except Exception:
            obj.rotation_euler.z += yaw_rad
