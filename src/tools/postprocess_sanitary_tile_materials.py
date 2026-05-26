#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def _set_node_input(node, name: str, value) -> None:
    sock = node.inputs.get(name)
    if sock is None:
        return
    try:
        sock.default_value = value
    except Exception:
        pass


def _make_tile_material(
    name: str,
    *,
    color1: tuple[float, float, float],
    color2: tuple[float, float, float],
    mortar: tuple[float, float, float],
    scale: float,
    brick_width: float,
    row_height: float,
    mortar_size: float,
    roughness: float,
    specular: float,
) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (760, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (520, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-430, 60)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-230, 60)
    links.new(texcoord.outputs["Generated"] if "Generated" in texcoord.outputs else texcoord.outputs["UV"], mapping.inputs["Vector"])

    brick = nodes.new("ShaderNodeTexBrick")
    brick.location = (20, 60)
    links.new(mapping.outputs["Vector"], brick.inputs["Vector"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (20, -170)
    _set_node_input(noise, "Scale", 18.0)
    _set_node_input(noise, "Detail", 9.0)
    _set_node_input(noise, "Roughness", 0.58)
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

    mix = nodes.new("ShaderNodeMix")
    mix.location = (300, 0)
    try:
        mix.data_type = "RGBA"
        mix.factor_mode = "UNIFORM"
    except Exception:
        pass
    if "Factor" in mix.inputs:
        links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    elif "Fac" in mix.inputs:
        links.new(noise.outputs["Fac"], mix.inputs["Fac"])
    if "A" in mix.inputs:
        links.new(brick.outputs["Color"], mix.inputs["A"])
        _set_node_input(mix, "B", (min(color1[0] * 1.08, 1.0), min(color1[1] * 1.06, 1.0), min(color1[2] * 1.04, 1.0), 1.0))
        links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    else:
        links.new(brick.outputs["Color"], bsdf.inputs["Base Color"])

    _set_node_input(brick, "Color1", (color1[0], color1[1], color1[2], 1.0))
    _set_node_input(brick, "Color2", (color2[0], color2[1], color2[2], 1.0))
    _set_node_input(brick, "Mortar", (mortar[0], mortar[1], mortar[2], 1.0))
    _set_node_input(brick, "Scale", scale)
    _set_node_input(brick, "Brick Width", brick_width)
    _set_node_input(brick, "Row Height", row_height)
    _set_node_input(brick, "Mortar Size", mortar_size)
    _set_node_input(brick, "Mortar Smooth", 0.01)

    _set_node_input(bsdf, "Roughness", roughness)
    _set_node_input(bsdf, "Specular IOR Level", specular)
    _set_node_input(bsdf, "Specular", specular)
    _set_node_input(bsdf, "Metallic", 0.0)
    return mat


def _floor_material() -> bpy.types.Material:
    # Local catalog reference: Atlas Concorde Marvel Travertine Tiles Pack
    # (Halo/Navona/Romano sand). This keeps the scene procedural while matching
    # a plausible bathroom porcelain floor instead of a neutral clay fallback.
    return _make_tile_material(
        "MAT_POST_SANITARY_FLOOR_ATLAS_TRAVERTINE_SAND",
        color1=(0.73, 0.64, 0.50),
        color2=(0.58, 0.66, 0.58),
        mortar=(0.34, 0.31, 0.26),
        scale=4.6,
        brick_width=0.82,
        row_height=0.82,
        mortar_size=0.035,
        roughness=0.78,
        specular=0.22,
    )


def _wall_material() -> bpy.types.Material:
    return _make_tile_material(
        "MAT_POST_SANITARY_WALL_SAGE_IVORY_CERAMIC",
        color1=(0.86, 0.84, 0.74),
        color2=(0.48, 0.62, 0.58),
        mortar=(0.46, 0.48, 0.44),
        scale=7.5,
        brick_width=0.90,
        row_height=0.28,
        mortar_size=0.020,
        roughness=0.64,
        specular=0.32,
    )


def _object_material_names(obj: bpy.types.Object) -> list[str]:
    try:
        return [mat.name for mat in obj.data.materials if mat is not None]
    except Exception:
        return []


def _is_floor_object(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return False
    name = obj.name.lower()
    mats = " ".join(_object_material_names(obj)).lower()
    if "floor" in name or "mat_room_floor" in mats or "sanitary_floor" in mats:
        return True
    return False


def _is_wall_object(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return False
    name = obj.name.lower()
    mats = " ".join(_object_material_names(obj)).lower()
    if "mat_room_wall" in mats or "sanitary_wall" in mats:
        return True
    if name.startswith("preview_wall_") or "room_wall" in name or name.endswith("_wall") or ".wall" in name:
        return True
    return False


def _replace_object_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if not obj.data.materials:
        obj.data.materials.append(mat)
        return
    for index in range(len(obj.data.materials)):
        obj.data.materials[index] = mat


def apply_sanitary_tile_postprocess() -> dict[str, int]:
    floor_mat = _floor_material()
    wall_mat = _wall_material()
    floor_count = 0
    wall_count = 0
    for obj in bpy.context.scene.objects:
        if _is_floor_object(obj):
            _replace_object_material(obj, floor_mat)
            floor_count += 1
        elif _is_wall_object(obj):
            _replace_object_material(obj, wall_mat)
            wall_count += 1
    return {"floors": floor_count, "walls": wall_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-blend", default="", help="Optional output .blend path. Defaults to overwriting current file.")
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    args = parser.parse_args(argv)

    report = apply_sanitary_tile_postprocess()
    save_path = Path(args.save_blend).expanduser().resolve() if args.save_blend else Path(bpy.data.filepath).expanduser().resolve()
    if not save_path:
        raise ValueError("No current blend filepath; pass --save-blend")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
    print(f"POSTPROCESS_SANITARY_TILE_MATERIALS {report} saved={save_path}")


if __name__ == "__main__":
    main()
