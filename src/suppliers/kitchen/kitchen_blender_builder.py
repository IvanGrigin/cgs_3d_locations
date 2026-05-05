from __future__ import annotations

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
    obj = bpy.context.object
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
    obj = bpy.context.object
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
    for obj in objects:
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


def _create_or_import_appliance(
    assembly: dict[str, Any],
    role: str,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    fallback_mat,
    collection,
    layout_orientation: str = "x",
) -> list[Any]:
    asset = _appliance_asset(assembly, role)
    if asset:
        objects = _import_asset_objects(asset["asset_local_path"], collection)
        objects = _filter_imported_appliance_objects(role, objects, asset)
        if role == "sink" and objects and _fit_objects_to_footprint_top(
            objects,
            (center[0], center[1]),
            (size[0], size[1]),
            center[2] + size[2] / 2.0,
        ):
            fit_ok = True
        else:
            fit_ok = bool(objects and _fit_objects_to_box(objects, center, size))
        if objects and fit_ok:
            _apply_asset_import_orientation(asset, objects, layout_orientation)
            for obj in objects:
                obj.name = f"{name}_{obj.name}"
                obj["kitchen_appliance_role"] = role
                obj["supplier_unique_key"] = asset.get("unique_key")
                obj["supplier_title"] = asset.get("title")
                if role in {"sink", "fridge"} and getattr(obj, "type", None) == "MESH" and fallback_mat is not None:
                    obj.data.materials.clear()
                    obj.data.materials.append(fallback_mat)
            return objects
    return [_create_box(name, center, size, fallback_mat, collection)]


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
        return _create_or_import_appliance(
            assembly,
            "microwave",
            item.get("id", "decor_microwave"),
            (sx, sy, pz + z + size[2] / 2.0),
            size,
            accent_mat,
            collection,
            layout_orientation=orientation,
        )
    if item_type == "small_kitchen_appliance":
        size = (0.24, 0.24, 0.30)
        if orientation == "y":
            size = (size[1], size[0], size[2])
        return _create_or_import_appliance(
            assembly,
            "small_kitchen_appliance",
            item.get("id", "decor_countertop_appliance"),
            (sx, sy, pz + z + size[2] / 2.0),
            size,
            mat,
            collection,
            layout_orientation=orientation,
        )
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
                created.extend(_create_fridge_placeholder(f"{module.get('id')}_fridge_details", (cx, cy, pz + z + h / 2), fridge_size, orientation, fridge_mat, appliance_mat, collection)[1:])
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
                sink_size = (cd, cw, sink_height) if orientation == "y" else (cw, cd, sink_height)
                sink_hole_size = (cutout_size[0] * 0.78, cutout_size[1] * 0.72, h * 3.0)
                _apply_rectangular_cutout(
                    countertop_obj,
                    f"{segment.get('id')}_{cutout.get('type')}",
                    (sx, sy, pz + z + h / 2.0),
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
                        (sx, sy, pz + base_z + base_h / 2.0),
                        (sink_hole_size[0], sink_hole_size[1], base_h * 1.15),
                        collection,
                    )
                sink_asset = _appliance_asset(assembly, "sink")
                if sink_asset:
                    sink_asset_title = _asset_title(sink_asset)
                    sink_asset_mat = (
                        cutout_mat
                        if any(term in sink_asset_title for term in ("черн", "black", "pvd"))
                        else sink_rim_mat
                    )
                    created.extend(
                        _create_or_import_appliance(
                            assembly,
                            "sink",
                            f"{segment.get('id')}_sink_asset",
                            (sx, sy, pz + z + h + 0.004 - sink_height / 2),
                            sink_size,
                            sink_asset_mat,
                            collection,
                            layout_orientation=orientation,
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
                    min(cy + cd - 0.08, max(0.08, cy + 0.08)),
                    orientation,
                )
                if sink_asset and _sink_asset_includes_faucet(sink_asset):
                    pass
                elif _appliance_asset(assembly, "faucet"):
                    # The current faucet FBX in the supplier catalog has unstable
                    # nested transforms after import, so keep a procedural mixer
                    # instead of silently rendering no faucet.
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
                if cooktop_asset:
                    created.extend(
                        _create_or_import_appliance(
                            assembly,
                            "cooktop",
                            f"{segment.get('id')}_cooktop_asset",
                            (sx, sy, pz + z + h - 0.002),
                            cooktop_size,
                            appliance_mat,
                            collection,
                            layout_orientation=orientation,
                        )
                    )
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
    return created
