#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure imported GLB/GLTF world-space bbox in Blender coordinates.")
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out-json", required=True)
    return parser


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def measure_scene() -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[Vector] = []
    mesh_count = 0
    vertex_count = 0
    face_count = 0

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        mesh_count += 1
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            vertex_count += len(mesh.vertices)
            face_count += len(mesh.polygons)
            for vertex in mesh.vertices:
                points.append(evaluated.matrix_world @ vertex.co)
        finally:
            evaluated.to_mesh_clear()

    if not points:
        raise RuntimeError("No mesh vertices found after GLB import")

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    bbox = {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": min(zs),
        "z_max": max(zs),
    }
    size = {
        "x": bbox["x_max"] - bbox["x_min"],
        "y": bbox["y_max"] - bbox["y_min"],
        "z": bbox["z_max"] - bbox["z_min"],
    }
    center = {
        "x": 0.5 * (bbox["x_min"] + bbox["x_max"]),
        "y": 0.5 * (bbox["y_min"] + bbox["y_max"]),
        "z": 0.5 * (bbox["z_min"] + bbox["z_max"]),
    }
    horizontal_long_axis = "x" if size["x"] >= size["y"] else "y"
    return {
        "schema": "glb_bbox_measurement/v1",
        "coordinate_frame": {
            "system": "Blender world after GLB import",
            "right": "+X",
            "forward": "+Y",
            "up": "+Z",
            "units": "Blender units as imported",
        },
        "bbox": {k: round(float(v), 6) for k, v in bbox.items()},
        "center": {k: round(float(v), 6) for k, v in center.items()},
        "size": {k: round(float(v), 6) for k, v in size.items()},
        "horizontal_long_axis": horizontal_long_axis,
        "horizontal_long_axis_direction": "+X/right" if horizontal_long_axis == "x" else "+Y/forward",
        "mesh_count": mesh_count,
        "vertex_count": vertex_count,
        "face_count": face_count,
    }


def main() -> None:
    args = build_cli().parse_args(_argv())
    glb_path = Path(args.glb).expanduser().resolve()
    out_path = Path(args.out_json).expanduser().resolve()
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    bpy.context.view_layer.update()
    report = measure_scene()
    report["glb_path"] = str(glb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
