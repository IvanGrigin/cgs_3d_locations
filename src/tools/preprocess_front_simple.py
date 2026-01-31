#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# ----------------------------
# Math: quaternions (xyzw), Y-up
# ----------------------------

def quat_norm(q: List[float]) -> List[float]:
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0:
        return [0.0, 0.0, 0.0, 1.0]
    return [x/n, y/n, z/n, w/n]

def quat_mul(q1: List[float], q2: List[float]) -> List[float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]

def quat_conj(q: List[float]) -> List[float]:
    x, y, z, w = q
    return [-x, -y, -z, w]

def quat_rotate_vec(q: List[float], v: List[float]) -> List[float]:
    # rotate v by quaternion q: q * (v,0) * conj(q)
    qv = [v[0], v[1], v[2], 0.0]
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:3]

def yaw_from_quat_y_up(q: List[float]) -> float:
    # yaw around Y (Tait-Bryan), quaternion in xyzw
    x, y, z, w = q
    siny_cosp = 2.0 * (w*y + x*z)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def apply_transform(pos: List[float], rot: List[float], scale: List[float], v: List[float]) -> List[float]:
    # v_local -> v_world, elementwise scale, then rotate, then translate
    vs = [v[0]*scale[0], v[1]*scale[1], v[2]*scale[2]]
    vr = quat_rotate_vec(rot, vs)
    return [vr[0] + pos[0], vr[1] + pos[1], vr[2] + pos[2]]

def compose_transform(p1: List[float], r1: List[float], s1: List[float],
                      p2: List[float], r2: List[float], s2: List[float]) -> Tuple[List[float], List[float], List[float]]:
    # T = T1 * T2 (apply T2 then T1)
    r = quat_norm(quat_mul(r1, r2))
    s = [s1[0]*s2[0], s1[1]*s2[1], s1[2]*s2[2]]
    p2s = [p2[0]*s1[0], p2[1]*s1[1], p2[2]*s1[2]]
    p2r = quat_rotate_vec(r1, p2s)
    p = [p1[0] + p2r[0], p1[1] + p2r[1], p1[2] + p2r[2]]
    return p, r, s


# ----------------------------
# Geometry: convex hull (XZ) for floor outline
# ----------------------------

def convex_hull_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # monotonic chain, returns hull in CCW without repeating first point
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def extract_vertices_xyz(mesh_item: Dict[str, Any]) -> List[List[float]]:
    xyz = mesh_item.get("xyz", [])
    if not xyz:
        return []
    out = []
    # xyz is flat array [x,y,z,x,y,z,...], sometimes strings
    for i in range(0, len(xyz), 3):
        try:
            out.append([float(xyz[i]), float(xyz[i+1]), float(xyz[i+2])])
        except Exception:
            break
    return out


# ----------------------------
# Main preprocessing
# ----------------------------

FLOOR_TYPES = {"Floor", "SlabBottom"}
CEILING_TYPES = {"Ceiling", "SlabTop"}


def preprocess_front_json(data: Dict[str, Any]) -> Dict[str, Any]:
    furniture = data.get("furniture", []) or []
    meshes = data.get("mesh", []) or []

    furn_map = {f["uid"]: f for f in furniture if isinstance(f, dict) and "uid" in f}
    mesh_map = {m["uid"]: m for m in meshes if isinstance(m, dict) and "uid" in m}

    scene = data.get("scene", {}) or {}
    scene_p = scene.get("pos", [0.0, 0.0, 0.0])
    scene_r = quat_norm(scene.get("rot", [0.0, 0.0, 0.0, 1.0]))
    scene_s = scene.get("scale", [1.0, 1.0, 1.0])

    out: Dict[str, Any] = {
        "source_uid": data.get("uid"),
        "design_version": data.get("design_version"),
        "code_version": data.get("code_version"),
        "version": data.get("version"),
        "north_vector": data.get("north_vector"),
        "meta": {
            "axis_up": "Y",
            "quat_order": "xyzw",
            "yaw_definition": "rotation_about_Y_in_scene_coords_radians",
            "floor_mesh_types": sorted(FLOOR_TYPES),
            "ceiling_mesh_types": sorted(CEILING_TYPES),
            "note": "All transforms are composed: scene -> room -> child. Positions are in scene/world coords."
        },
        "rooms": []
    }

    for room in (scene.get("room") or []):
        rp = room.get("pos", [0.0, 0.0, 0.0])
        rr = quat_norm(room.get("rot", [0.0, 0.0, 0.0, 1.0]))
        rs = room.get("scale", [1.0, 1.0, 1.0])

        # World transform of room = scene * room
        wp, wr, ws = compose_transform(scene_p, scene_r, scene_s, rp, rr, rs)

        children = room.get("children") or []
        objects: List[Dict[str, Any]] = []

        floor_pts_xz: List[Tuple[float, float]] = []
        floor_y_vals: List[float] = []
        ceiling_y_vals: List[float] = []
        all_y_vals: List[float] = []

        for ch in children:
            ref = ch.get("ref")
            if not ref:
                continue

            cp = ch.get("pos", [0.0, 0.0, 0.0])
            crot_raw = ch.get("rot", [0.0, 0.0, 0.0, 1.0])
            cr = quat_norm(crot_raw) if isinstance(crot_raw, list) and len(crot_raw) == 4 else [0.0, 0.0, 0.0, 1.0]
            cs = ch.get("scale", [1.0, 1.0, 1.0])

            # World transform of child = room_world * child
            cwp, cwr, cws = compose_transform(wp, wr, ws, cp, cr, cs)

            # Furniture instance: ref must exist in furniture.uid
            if ref in furn_map:
                f = furn_map[ref]
                if f.get("valid") is False:
                    continue
                yaw = yaw_from_quat_y_up(cwr)
                objects.append({
                    "ref": ref,
                    "instanceid": ch.get("instanceid"),
                    "jid": f.get("jid"),
                    "sourceCategoryId": f.get("sourceCategoryId"),
                    "title": f.get("title"),
                    "category": f.get("category"),
                    "size": f.get("size"),
                    "bbox_local": f.get("bbox"),
                    "pos": cwp,
                    "rot_quat": cwr,
                    "yaw": yaw
                })

            # Room geometry meshes: ref must exist in mesh.uid
            if ref in mesh_map:
                m = mesh_map[ref]
                verts = extract_vertices_xyz(m)
                if not verts:
                    continue

                mtype = m.get("type", "")
                for v in verts:
                    vw = apply_transform(cwp, cwr, cws, v)
                    all_y_vals.append(vw[1])

                    if mtype in FLOOR_TYPES:
                        floor_pts_xz.append((vw[0], vw[2]))
                        floor_y_vals.append(vw[1])
                    if mtype in CEILING_TYPES:
                        ceiling_y_vals.append(vw[1])

        # Floor summary
        if floor_pts_xz:
            xs = [p[0] for p in floor_pts_xz]
            zs = [p[1] for p in floor_pts_xz]
            bbox_xz = [min(xs), min(zs), max(xs), max(zs)]
            outline = convex_hull_2d(floor_pts_xz)
            floor_y = float(statistics.median(floor_y_vals)) if floor_y_vals else 0.0
        else:
            bbox_xz = None
            outline = []
            floor_y = 0.0

        # Ceiling height
        if ceiling_y_vals:
            ceiling_y = float(max(ceiling_y_vals))
        else:
            ceiling_y = float(max(all_y_vals)) if all_y_vals else floor_y

        out["rooms"].append({
            "room_type": room.get("type"),
            "room_instanceid": room.get("instanceid"),
            "transform_scene": {
                "pos": wp,
                "rot_quat": wr,
                "scale": ws
            },
            "floor": {
                "y": floor_y,
                "ceiling_y": ceiling_y,
                "bbox_xz": bbox_xz,
                "outline_xz": [[float(x), float(z)] for x, z in outline]
            },
            "objects": objects
        })

    return out


def iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.rglob("*.json"))
    return [p for p in files if p.is_file()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a 3D-FRONT scene JSON file OR a directory with json files")
    ap.add_argument("--outdir", default=None, help="Output directory. Default: <input_parent>/_processed")
    ap.add_argument("--suffix", default=".processed.json", help="Output filename suffix")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (input_path.parent / "_processed")
    outdir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(input_path)
    if not files:
        raise SystemExit("No .json files found")

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        processed = preprocess_front_json(data)

        out_name = fp.stem + args.suffix
        out_path = outdir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print(f"[ok] {fp.name} -> {out_path}")

    print(f"Done. Output dir: {outdir}")


if __name__ == "__main__":
    main()