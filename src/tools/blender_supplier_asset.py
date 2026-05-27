# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["convert", "proxy", "sanitize"])
    ap.add_argument("--input", default=None)
    ap.add_argument("--texture", default=None)
    ap.add_argument("--width-m", type=float, default=None)
    ap.add_argument("--depth-m", type=float, default=None)
    ap.add_argument("--height-m", type=float, default=None)
    ap.add_argument("--out-glb", default=None)
    ap.add_argument("--out-fbx", default=None)
    return ap.parse_args(argv)


def _clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)


def _import_input(bpy, input_path: Path) -> None:
    suffix = input_path.suffix.lower()

    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(input_path))
        return

    _clear_scene(bpy)

    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(input_path))
            return
        if hasattr(bpy.ops.import_scene, "obj"):  # pragma: no cover
            bpy.ops.import_scene.obj(filepath=str(input_path))  # pragma: no cover
            return  # pragma: no cover
        raise RuntimeError("OBJ import is not available in this Blender build.")  # pragma: no cover

    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(input_path))
        return

    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(input_path))
        return

    raise RuntimeError(f"Unsupported input format for Blender conversion: {suffix}")


def _remove_helper_material_objects(bpy) -> None:
    helper_objects = [
        obj for obj in list(bpy.context.scene.objects)
        if obj.name.lower().startswith("mat_")
    ]
    for obj in helper_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    helper_collections = [
        col for col in list(bpy.data.collections)
        if col.name.lower().startswith("mat_")
    ]
    for collection in helper_collections:
        for parent in list(bpy.data.collections):
            if collection in parent.children:
                parent.children.unlink(collection)
        if collection in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.unlink(collection)
        bpy.data.collections.remove(collection)


def _get_content_objects(bpy) -> list:
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type not in {"CAMERA", "LIGHT"}
    ]


def _compute_scene_bbox(bpy) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    from mathutils import Vector  # type: ignore

    objects = [
        obj for obj in _get_content_objects(bpy)
        if hasattr(obj, "bound_box") and obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}
    ]
    if not objects:
        return None  # pragma: no cover

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]

    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins[0] = min(mins[0], world.x)
            mins[1] = min(mins[1], world.y)
            mins[2] = min(mins[2], world.z)
            maxs[0] = max(maxs[0], world.x)
            maxs[1] = max(maxs[1], world.y)
            maxs[2] = max(maxs[2], world.z)

    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])


def _normalize_import_scale(
    bpy,
    width_m: float | None,
    depth_m: float | None,
    height_m: float | None,
) -> None:
    bbox = _compute_scene_bbox(bpy)
    if not bbox:
        return  # pragma: no cover

    (min_x, min_y, min_z), (max_x, max_y, max_z) = bbox
    current_dims = [max_x - min_x, max_y - min_y, max_z - min_z]
    target_dims = [width_m, depth_m, height_m]

    ratios: list[float] = []
    for current, target in zip(current_dims, target_dims):
        if target is None or target <= 0 or current <= 0:
            continue  # pragma: no cover
        ratios.append(target / current)

    if ratios:
        ratios.sort()
        uniform_scale = ratios[len(ratios) // 2]
        for obj in _get_content_objects(bpy):
            obj.scale = tuple(value * uniform_scale for value in obj.scale)

        bpy.ops.object.select_all(action="DESELECT")
        for obj in _get_content_objects(bpy):
            obj.select_set(True)
        bpy.context.view_layer.objects.active = _get_content_objects(bpy)[0]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bbox = _compute_scene_bbox(bpy)
    if not bbox:
        return  # pragma: no cover

    (min_x, min_y, min_z), (max_x, max_y, _) = bbox
    offset_x = -((min_x + max_x) / 2.0)
    offset_y = -((min_y + max_y) / 2.0)
    offset_z = -min_z

    for obj in _get_content_objects(bpy):
        obj.location.x += offset_x
        obj.location.y += offset_y
        obj.location.z += offset_z


def _build_textured_proxy(
    bpy,
    width_m: float,
    depth_m: float,
    height_m: float,
    texture_path: Path | None,
) -> None:
    _clear_scene(bpy)
    width_m = width_m or 1.0
    depth_m = depth_m or 1.0
    height_m = height_m or 1.0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, max(height_m, 0.01) / 2.0))
    obj = bpy.context.active_object
    obj.name = "SupplierProxy"
    obj.scale = (max(width_m, 0.01), max(depth_m, 0.01), max(height_m, 0.01) / 2.0)

    mat = bpy.data.materials.new(name="SupplierProxyMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")

    if texture_path and texture_path.is_file():
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(texture_path))
        coord = nodes.new(type="ShaderNodeTexCoord")
        mapping = nodes.new(type="ShaderNodeMapping")
        links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    else:
        principled.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)  # pragma: no cover

    if obj.data.materials:
        obj.data.materials[0] = mat  # pragma: no cover
    else:
        obj.data.materials.append(mat)


def _export_outputs(bpy, out_glb: Path | None, out_fbx: Path | None) -> None:
    if out_glb:
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(
            filepath=str(out_glb),
            export_format="GLB",
            use_selection=False,
        )

    if out_fbx:
        out_fbx.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.fbx(
            filepath=str(out_fbx),
            use_selection=False,
            path_mode="COPY",
            embed_textures=True,
        )


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []  # pragma: no cover
    args = _parse_args(argv)

    import bpy  # type: ignore

    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    out_glb = Path(args.out_glb).expanduser().resolve() if args.out_glb else None
    out_fbx = Path(args.out_fbx).expanduser().resolve() if args.out_fbx else None

    if args.mode in {"convert", "sanitize"}:
        if not args.input:
            raise RuntimeError("--input is required for convert/sanitize mode")
        _import_input(bpy, Path(args.input).expanduser().resolve())
        _remove_helper_material_objects(bpy)
        if args.mode == "convert":
            _normalize_import_scale(
                bpy=bpy,
                width_m=args.width_m,
                depth_m=args.depth_m,
                height_m=args.height_m,
            )
    else:
        texture_path = Path(args.texture).expanduser().resolve() if args.texture else None  # pragma: no cover
        _build_textured_proxy(  # pragma: no cover
            bpy=bpy,
            width_m=args.width_m,
            depth_m=args.depth_m,
            height_m=args.height_m,
            texture_path=texture_path,
        )

    _export_outputs(bpy, out_glb=out_glb, out_fbx=out_fbx)


if __name__ == "__main__":
    main()  # pragma: no cover
