#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


SUPPORTED_EXTS = {".fbx", ".glb", ".gltf", ".obj"}


def _default_blender() -> str:
    mac = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac.is_file():
        return str(mac)
    return "blender"


def _running_inside_blender() -> bool:
    try:
        import bpy  # type: ignore

        return hasattr(bpy, "ops")
    except Exception:
        return False


def _script_args() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Headless Blender asset -> OBJ + geometry orientation analysis.")
    ap.add_argument("--asset", required=True, help="Input .fbx/.glb/.gltf/.obj")
    ap.add_argument("--out-dir", default="", help="Output directory. Default: <asset>.asset_analysis/")
    ap.add_argument("--out-obj", default="", help="Explicit output OBJ path.")
    ap.add_argument("--out-json", default="", help="Explicit output analysis JSON path.")
    ap.add_argument("--blender", default=_default_blender(), help="Blender binary when launched outside Blender.")
    ap.add_argument("--category", default="", help="Optional semantic category, e.g. chair/wardrobe/desk/bed.")
    ap.add_argument("--no-obj", action="store_true", help="Only write analysis JSON, do not export OBJ.")
    return ap.parse_args(argv)


def _launch_blender(args: argparse.Namespace) -> int:
    cmd = [
        args.blender,
        "--background",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--asset",
        str(Path(args.asset).expanduser().resolve()),
    ]
    for key in ("out_dir", "out_obj", "out_json", "category"):
        value = getattr(args, key)
        if value:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    if args.no_obj:
        cmd.append("--no-obj")
    os.execv(args.blender, cmd)
    return 127


def _clear_scene() -> None:
    import bpy  # type: ignore

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _import_asset(asset: Path) -> list[Any]:
    import bpy  # type: ignore

    before = set(bpy.data.objects)
    ext = asset.suffix.lower()
    if ext == ".fbx":
        try:
            bpy.ops.import_scene.fbx(filepath=str(asset), use_image_search=False)
        except TypeError:
            bpy.ops.import_scene.fbx(filepath=str(asset))
    elif ext in {".glb", ".gltf"}:
        try:
            bpy.ops.import_scene.gltf(filepath=str(asset), import_pack_images=False)
        except TypeError:
            bpy.ops.import_scene.gltf(filepath=str(asset))
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(asset))
        else:
            bpy.ops.import_scene.obj(filepath=str(asset))
    else:
        raise ValueError(f"Unsupported asset extension: {asset.suffix}")
    return [obj for obj in bpy.data.objects if obj not in before]


def _iter_mesh_objects(imported: list[Any]) -> list[Any]:
    import bpy  # type: ignore

    roots = set(imported)
    out = []
    for obj in bpy.data.objects:
        cur = obj
        belongs = obj in roots
        while cur.parent is not None and not belongs:
            cur = cur.parent
            belongs = cur in roots
        if belongs and obj.type == "MESH":
            out.append(obj)
    return out


def _mesh_world_points(mesh_objs: list[Any]) -> list[tuple[float, float, float]]:
    depsgraph = __import__("bpy").context.evaluated_depsgraph_get()  # type: ignore
    points: list[tuple[float, float, float]] = []
    for obj in mesh_objs:
        eo = obj.evaluated_get(depsgraph)
        try:
            mesh = eo.to_mesh()
            mw = obj.matrix_world
            for vertex in mesh.vertices:
                p = mw @ vertex.co
                points.append((float(p.x), float(p.y), float(p.z)))
        finally:
            try:
                eo.to_mesh_clear()
            except Exception:
                pass
    return points


def _mesh_object_reports(mesh_objs: list[Any]) -> list[dict[str, Any]]:
    depsgraph = __import__("bpy").context.evaluated_depsgraph_get()  # type: ignore
    reports: list[dict[str, Any]] = []
    for obj in mesh_objs:
        eo = obj.evaluated_get(depsgraph)
        try:
            mesh = eo.to_mesh()
            mw = obj.matrix_world
            pts = []
            for vertex in mesh.vertices:
                p = mw @ vertex.co
                pts.append((float(p.x), float(p.y), float(p.z)))
        finally:
            try:
                eo.to_mesh_clear()
            except Exception:
                pass
        if not pts:
            continue
        xs, ys, zs = zip(*pts)
        bounds = {
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "z_min": min(zs),
            "z_max": max(zs),
        }
        size = {
            "x": bounds["x_max"] - bounds["x_min"],
            "y": bounds["y_max"] - bounds["y_min"],
            "z": bounds["z_max"] - bounds["z_min"],
        }
        reports.append(
            {
                "name": obj.name,
                "vertex_count": len(pts),
                "bounds": bounds,
                "size": size,
                "center": {
                    "x": (bounds["x_min"] + bounds["x_max"]) * 0.5,
                    "y": (bounds["y_min"] + bounds["y_max"]) * 0.5,
                    "z": (bounds["z_min"] + bounds["z_max"]) * 0.5,
                },
            }
        )
    return reports


def _side_stats(points: list[tuple[float, float, float]], bounds: dict[str, float]) -> dict[str, Any]:
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    sx = max(x_max - x_min, 1e-9)
    sy = max(y_max - y_min, 1e-9)
    band_x = sx * 0.18
    band_y = sy * 0.18
    sides = {
        "+X": [p for p in points if p[0] >= x_max - band_x],
        "-X": [p for p in points if p[0] <= x_min + band_x],
        "+Y": [p for p in points if p[1] >= y_max - band_y],
        "-Y": [p for p in points if p[1] <= y_min + band_y],
    }
    out: dict[str, Any] = {}
    for side, pts in sides.items():
        if not pts:
            out[side] = {"count": 0, "max_z": None, "mean_z": None, "high_point_fraction": 0.0}
            continue
        zs = [p[2] for p in pts]
        z_min, z_max = bounds["z_min"], bounds["z_max"]
        high_threshold = z_min + (z_max - z_min) * 0.62
        out[side] = {
            "count": len(pts),
            "max_z": max(zs),
            "mean_z": sum(zs) / len(zs),
            "high_point_fraction": sum(1 for z in zs if z >= high_threshold) / len(zs),
        }
    return out


def _axis_value(bounds: dict[str, float], axis: str) -> float:
    if axis == "+X":
        return bounds["x_max"]
    if axis == "-X":
        return bounds["x_min"]
    if axis == "+Y":
        return bounds["y_max"]
    if axis == "-Y":
        return bounds["y_min"]
    return 0.0


def _object_side(obj: dict[str, Any], asset_bounds: dict[str, float], tolerance_ratio: float = 0.10) -> str:
    b = obj["bounds"]
    sx = max(asset_bounds["x_max"] - asset_bounds["x_min"], 1e-9)
    sy = max(asset_bounds["y_max"] - asset_bounds["y_min"], 1e-9)
    distances = {
        "+X": abs(asset_bounds["x_max"] - b["x_max"]) / sx,
        "-X": abs(b["x_min"] - asset_bounds["x_min"]) / sx,
        "+Y": abs(asset_bounds["y_max"] - b["y_max"]) / sy,
        "-Y": abs(b["y_min"] - asset_bounds["y_min"]) / sy,
    }
    side, distance = min(distances.items(), key=lambda kv: kv[1])
    return side if distance <= tolerance_ratio else ""


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[idx]


def _side_projection(point: tuple[float, float, float], side: str) -> tuple[float, float]:
    if side in {"+X", "-X"}:
        return point[1], point[2]
    return point[0], point[2]


def _point_outward_value(point: tuple[float, float, float], side: str) -> float:
    if side == "+X":
        return point[0]
    if side == "-X":
        return -point[0]
    if side == "+Y":
        return point[1]
    if side == "-Y":
        return -point[1]
    return 0.0


def _vertex_handle_candidates(points: list[tuple[float, float, float]], bounds: dict[str, float]) -> list[dict[str, Any]]:
    total = {
        "x": max(bounds["x_max"] - bounds["x_min"], 1e-9),
        "y": max(bounds["y_max"] - bounds["y_min"], 1e-9),
        "z": max(bounds["z_max"] - bounds["z_min"], 1e-9),
    }
    out: list[dict[str, Any]] = []
    for side in ("+X", "-X", "+Y", "-Y"):
        values = [_point_outward_value(p, side) for p in points]
        max_v = max(values)
        p90 = _percentile(values, 0.90)
        p95 = _percentile(values, 0.95)
        p98 = _percentile(values, 0.98)
        axis_size = total["x"] if side in {"+X", "-X"} else total["y"]
        protrusion_depth = max_v - p90
        if protrusion_depth < max(axis_size * 0.012, 0.006):
            continue
        band_depth = max(max_v - p98, axis_size * 0.015, 0.008)
        pts = [p for p in points if max_v - _point_outward_value(p, side) <= band_depth]
        if len(pts) < 8:
            continue
        us, zs = zip(*[_side_projection(p, side) for p in pts])
        u_size = max(us) - min(us)
        z_size = max(zs) - min(zs)
        z_center = (max(zs) + min(zs)) * 0.5
        u_total = total["y"] if side in {"+X", "-X"} else total["x"]
        z_ratio = z_size / total["z"]
        u_ratio = u_size / max(u_total, 1e-9)
        center_z_ratio = (z_center - bounds["z_min"]) / total["z"]
        # A protruding handle is usually a compact cluster on the door face, not a whole flat side.
        if not (0.10 <= center_z_ratio <= 0.92 and u_ratio <= 0.45 and z_ratio <= 0.55):
            continue
        compactness = max(0.0, 1.0 - max(u_ratio, z_ratio))
        score = protrusion_depth / axis_size + compactness
        out.append(
            {
                "name": f"vertex_protrusion_{side}",
                "side": side,
                "score": round(score, 6),
                "source": "vertex_protrusion",
                "protrusion_depth_m": round(protrusion_depth, 6),
                "outer_band_depth_m": round(band_depth, 6),
                "outer_vertex_count": len(pts),
                "relative_size": {"u": round(u_ratio, 6), "z": round(z_ratio, 6)},
                "center_z_ratio": round(center_z_ratio, 6),
                "percentiles": {"p90": round(p90, 6), "p95": round(p95, 6), "p98": round(p98, 6), "max": round(max_v, 6)},
            }
        )
    return out


def _handle_analysis(
    category: str,
    object_reports: list[dict[str, Any]],
    bounds: dict[str, float],
    points: list[tuple[float, float, float]],
) -> dict[str, Any]:
    category = category.strip().lower()
    handle_categories = {"wardrobe", "cabinet", "dresser", "nightstand", "side_table", "bedside_table"}
    if category not in handle_categories:
        return {
            "searched": False,
            "reason": "category is not handle-bearing furniture",
            "handle_candidates": [],
            "inferred_front_axis": None,
        }

    total = {
        "x": max(bounds["x_max"] - bounds["x_min"], 1e-9),
        "y": max(bounds["y_max"] - bounds["y_min"], 1e-9),
        "z": max(bounds["z_max"] - bounds["z_min"], 1e-9),
    }
    total_volume = total["x"] * total["y"] * total["z"]
    candidates: list[dict[str, Any]] = []
    for obj in object_reports:
        size = obj["size"]
        b = obj["bounds"]
        volume = max(size["x"] * size["y"] * size["z"], 0.0)
        dims_rel = {axis: size[axis] / total[axis] for axis in ("x", "y", "z")}
        max_rel = max(dims_rel.values())
        min_rel = min(dims_rel.values())
        volume_rel = volume / total_volume
        center_z_rel = (obj["center"]["z"] - bounds["z_min"]) / total["z"]
        side = _object_side(obj, bounds)
        name = str(obj["name"]).lower()
        name_hint = any(token in name for token in ("handle", "knob", "pull", "руч", "maniglia", "griff"))

        is_small_part = volume_rel <= 0.035 and max_rel <= 0.55 and min_rel <= 0.18
        is_handle_height = 0.12 <= center_z_rel <= 0.92
        if not side or not is_handle_height or not (is_small_part or name_hint):
            continue

        slender_bonus = 0.0
        sorted_rel = sorted(dims_rel.values())
        if sorted_rel[0] <= 0.08 and sorted_rel[2] >= 0.16:
            slender_bonus = 0.35
        score = (1.0 - min(volume_rel / 0.035, 1.0)) + slender_bonus + (0.5 if name_hint else 0.0)
        candidates.append(
            {
                "name": obj["name"],
                "side": side,
                "score": round(score, 6),
                "volume_ratio": round(volume_rel, 8),
                "relative_size": {k: round(v, 6) for k, v in dims_rel.items()},
                "center_z_ratio": round(center_z_rel, 6),
                "bounds": b,
            }
        )
    candidates.extend(_vertex_handle_candidates(points, bounds))

    side_scores: dict[str, float] = {}
    for c in candidates:
        side_scores[c["side"]] = side_scores.get(c["side"], 0.0) + float(c["score"])
    front_axis = max(side_scores.items(), key=lambda kv: kv[1])[0] if side_scores else None
    return {
        "searched": True,
        "reason": "small side-mounted components treated as possible handles",
        "handle_candidates": sorted(candidates, key=lambda c: c["score"], reverse=True)[:12],
        "side_scores": {k: round(v, 6) for k, v in sorted(side_scores.items())},
        "inferred_front_axis": front_axis,
        "confidence": "heuristic" if front_axis else "none",
    }


def _opposite_axis(axis: str) -> str:
    return {"+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y"}.get(axis, "")


def _infer_front_axis(category: str, side_stats: dict[str, Any], handle_analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    category = category.strip().lower()
    handle_front_axis = (handle_analysis or {}).get("inferred_front_axis")
    if category in {"wardrobe", "cabinet", "dresser", "nightstand", "side_table", "bedside_table"} and handle_front_axis:
        return {
            "inferred_front_axis": handle_front_axis,
            "confidence": "heuristic",
            "reason": f"front side inferred from handle candidates: {handle_front_axis}",
            "handle_candidate_count": len((handle_analysis or {}).get("handle_candidates") or []),
        }
    scored = []
    for side, stat in side_stats.items():
        if not stat.get("count"):
            continue
        score = float(stat.get("high_point_fraction") or 0.0) * 2.0 + float(stat.get("max_z") or 0.0)
        scored.append((score, side))
    scored.sort(reverse=True)
    high_side = scored[0][1] if scored else ""
    if category in {"chair", "armchair", "stool"} and high_side:
        return {
            "inferred_front_axis": _opposite_axis(high_side),
            "confidence": "heuristic",
            "reason": f"highest side treated as chair back: {high_side}",
            "back_axis": high_side,
        }
    if category in {"bed"} and high_side:
        return {
            "inferred_headboard_axis": high_side,
            "inferred_front_axis": _opposite_axis(high_side),
            "confidence": "heuristic",
            "reason": f"highest side treated as headboard: {high_side}",
        }
    return {
        "inferred_front_axis": None,
        "confidence": "none",
        "reason": "category has no reliable geometry-only front heuristic",
    }


def _export_obj(out_obj: Path) -> None:
    import bpy  # type: ignore

    out_obj.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(out_obj),
            export_selected_objects=True,
            export_materials=False,
            export_uv=False,
            export_normals=False,
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=str(out_obj),
            use_selection=True,
            use_materials=False,
            use_uvs=False,
            use_normals=False,
        )


def _run_in_blender(args: argparse.Namespace) -> int:
    asset = Path(args.asset).expanduser().resolve()
    if asset.suffix.lower() not in SUPPORTED_EXTS:
        raise SystemExit(f"Unsupported asset extension: {asset.suffix}")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else asset.with_suffix("").with_name(asset.stem + ".asset_analysis")
    out_obj = Path(args.out_obj).expanduser().resolve() if args.out_obj else out_dir / f"{asset.stem}.geometry.obj"
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else out_dir / f"{asset.stem}.analysis.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    _clear_scene()
    imported = _import_asset(asset)
    mesh_objs = _iter_mesh_objects(imported)
    points = _mesh_world_points(mesh_objs)
    object_reports = _mesh_object_reports(mesh_objs)
    if not points:
        raise SystemExit(f"No mesh vertices imported from {asset}")

    xs, ys, zs = zip(*points)
    bounds = {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": min(zs),
        "z_max": max(zs),
    }
    size = {
        "x": bounds["x_max"] - bounds["x_min"],
        "y": bounds["y_max"] - bounds["y_min"],
        "z": bounds["z_max"] - bounds["z_min"],
    }
    center = {
        "x": (bounds["x_min"] + bounds["x_max"]) * 0.5,
        "y": (bounds["y_min"] + bounds["y_max"]) * 0.5,
        "z": (bounds["z_min"] + bounds["z_max"]) * 0.5,
    }
    sides = _side_stats(points, bounds)
    handles = _handle_analysis(args.category, object_reports, bounds, points)
    inference = _infer_front_axis(args.category, sides, handles)
    if not args.no_obj:
        _export_obj(out_obj)

    report = {
        "schema": "headless_asset_geometry_analysis/v1",
        "asset": str(asset),
        "category": args.category or None,
        "obj_path": None if args.no_obj else str(out_obj),
        "mesh_object_count": len(mesh_objs),
        "vertex_count": len(points),
        "bounds": bounds,
        "size": size,
        "bounds_center": center,
        "mesh_objects": object_reports,
        "side_stats": sides,
        "handle_analysis": handles,
        "orientation_inference": inference,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "obj": report["obj_path"], "json": str(out_json)}, ensure_ascii=False))
    return 0


def main() -> int:
    args = _parse_args(_script_args())
    if not _running_inside_blender():
        return _launch_blender(args)
    return _run_in_blender(args)


if __name__ == "__main__":
    raise SystemExit(main())
