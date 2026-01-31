#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/export_rooms_dataset.py

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Math / transforms
# -----------------------------

def _quat_yaw_xyzw(q: List[float]) -> float:
    """
    Yaw (rotation about +Y) from quaternion q=[x,y,z,w] in a Y-up right-handed system.
    Returns radians in [-pi, pi].
    """
    if not q or len(q) != 4:
        return 0.0
    x, y, z, w = [float(v) for v in q]
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _rotmat_from_quat_xyzw(q: List[float]) -> List[List[float]]:
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return [
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)],
    ]


def _matvec(m: List[List[float]], v: List[float]) -> List[float]:
    return [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
    ]


@dataclass
class Transform:
    pos: List[float]   # 3
    rot: List[float]   # quat xyzw
    scale: List[float] # 3

    def apply_point(self, p: List[float]) -> List[float]:
        s = self.scale if self.scale and len(self.scale) == 3 else [1.0, 1.0, 1.0]
        ps = [p[0]*s[0], p[1]*s[1], p[2]*s[2]]
        R = _rotmat_from_quat_xyzw(self.rot if self.rot and len(self.rot) == 4 else [0,0,0,1])
        pr = _matvec(R, ps)
        return [pr[0] + self.pos[0], pr[1] + self.pos[1], pr[2] + self.pos[2]]

    def compose(self, other: "Transform") -> "Transform":
        s1 = self.scale if self.scale and len(self.scale) == 3 else [1.0, 1.0, 1.0]
        s2 = other.scale if other.scale and len(other.scale) == 3 else [1.0, 1.0, 1.0]
        scale = [s1[0]*s2[0], s1[1]*s2[1], s1[2]*s2[2]]

        x1,y1,z1,w1 = self.rot if self.rot and len(self.rot) == 4 else [0,0,0,1]
        x2,y2,z2,w2 = other.rot if other.rot and len(other.rot) == 4 else [0,0,0,1]
        rot = [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ]

        pos = self.apply_point(other.pos if other.pos and len(other.pos) == 3 else [0.0,0.0,0.0])
        return Transform(pos=pos, rot=rot, scale=scale)


def _get_transform(obj: Dict[str, Any]) -> Transform:
    pos = obj.get("pos") or [0.0, 0.0, 0.0]
    rot = obj.get("rot_quat") or obj.get("rot") or [0.0, 0.0, 0.0, 1.0]
    scale = obj.get("scale") or [1.0, 1.0, 1.0]
    if isinstance(scale, (int, float)):
        scale = [float(scale)] * 3
    return Transform(
        pos=[float(x) for x in pos],
        rot=[float(x) for x in rot],
        scale=[float(x) for x in scale],
    )


# -----------------------------
# Helpers: parsing mesh xyz
# -----------------------------

def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # 3D-FRONT иногда кладёт числа строками
        return float(x.strip())
    raise TypeError(f"cannot convert to float: {type(x)}")


def _xyz_flat_to_vertices(xyz: List[Any]) -> List[Tuple[float, float, float]]:
    """
    xyz: flat list length 3*N with numbers or numeric strings
    """
    if not isinstance(xyz, list) or len(xyz) < 3 or (len(xyz) % 3) != 0:
        return []
    out = []
    for i in range(0, len(xyz), 3):
        out.append((_to_float(xyz[i]), _to_float(xyz[i+1]), _to_float(xyz[i+2])))
    return out


def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid-1] + s[mid])


def _polygon_aabb_from_points_xz(pts_xz: List[Tuple[float, float]]) -> Optional[List[List[float]]]:
    if not pts_xz:
        return None
    xs = [p[0] for p in pts_xz]
    zs = [p[1] for p in pts_xz]
    mnx, mxx = min(xs), max(xs)
    mnz, mxz = min(zs), max(zs)

    # деградация, если все точки почти на линии
    if (mxx - mnx) < 1e-6 or (mxz - mnz) < 1e-6:
        return None

    return [
        [mnx, mnz],
        [mxx, mnz],
        [mxx, mxz],
        [mnx, mxz],
    ]


def _is_floor_like_mesh(m: Dict[str, Any], room_id: str) -> bool:
    """
    Heuristics:
      1) m["type"] contains "floor" (case-insensitive)
      2) or m["material"] contains "floor"
      3) and preferably m["uid"] contains room_id (room-specific geometry)
    """
    t = str(m.get("type") or "").lower()
    mat = str(m.get("material") or "").lower()
    uid = str(m.get("uid") or "")
    if "floor" in t or "floor" in mat:
        if room_id and room_id in uid:
            return True
        # допускаем общий floor, если room-specific не найдём
        return True
    return False


def _polygon_from_raw_floor_mesh_as_rectangle(
    raw: Dict[str, Any],
    room_id: str,
    eps_y: float = 0.03
) -> Tuple[Optional[List[List[float]]], Optional[float]]:
    """
    Если room polygon отсутствует (extension.area нет),
    берём floor triangulation из raw["mesh"] и строим прямоугольник (AABB) по XZ.

    Возвращает:
      (polygon_xz, floor_y)
    """
    meshes = raw.get("mesh") or []
    if not isinstance(meshes, list) or not meshes:
        return None, None

    # 1) сначала ищем floor mesh, привязанный к комнате (uid содержит room_id)
    candidates_room = [m for m in meshes if isinstance(m, dict) and _is_floor_like_mesh(m, room_id) and room_id in str(m.get("uid") or "")]
    # 2) если не нашли — берём любые floor-like
    candidates_any = [m for m in meshes if isinstance(m, dict) and _is_floor_like_mesh(m, room_id)]

    candidates = candidates_room if candidates_room else candidates_any
    if not candidates:
        return None, None

    # объединяем точки из всех подходящих мешей (room-specific может быть несколько частей)
    all_vertices: List[Tuple[float, float, float]] = []
    for m in candidates:
        verts = _xyz_flat_to_vertices(m.get("xyz") or [])
        if verts:
            all_vertices.extend(verts)

    if not all_vertices:
        return None, None

    # floor_y как медиана Y; затем фильтруем точки близкие к полу
    ys = [v[1] for v in all_vertices]
    floor_y = _median(ys)
    pts_xz = [(v[0], v[2]) for v in all_vertices if abs(v[1] - floor_y) <= eps_y]

    # fallback: если фильтр слишком жёсткий (редкие y), используем все
    if len(pts_xz) < 8:
        pts_xz = [(v[0], v[2]) for v in all_vertices]

    poly = _polygon_aabb_from_points_xz(pts_xz)
    return poly, floor_y


# -----------------------------
# Size extraction
# -----------------------------

def _ref_to_jid(ref: Optional[str]) -> Optional[str]:
    if not ref:
        return None
    return ref.split("/", 1)[0].strip() if "/" in ref else ref.strip()


def _bbox6_to_sizes(b: List[float]) -> Optional[List[float]]:
    if len(b) != 6:
        return None
    mnx,mny,mnz,mxx,mxy,mxz = [float(v) for v in b]
    return [abs(mxx-mnx), abs(mxy-mny), abs(mxz-mnz)]


def _normalize_size_any(v: Any) -> Optional[List[float]]:
    if v is None:
        return None
    if isinstance(v, list):
        if len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            return [float(v[0]), float(v[1]), float(v[2])]
        if len(v) == 1 and isinstance(v[0], list):
            return _normalize_size_any(v[0])
        if len(v) == 6 and all(isinstance(x, (int, float)) for x in v):
            return _bbox6_to_sizes(v)
    return None


def _build_furniture_index(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for f in (raw.get("furniture") or []):
        jid = str(f.get("jid")) if f.get("jid") is not None else None
        if jid:
            idx[jid] = f
    return idx


def _infer_attachment_flags(
    title: str,
    y: float,
    floor_y: float,
    ceiling_y: Optional[float],
    eps: float = 0.08
) -> Dict[str, bool]:
    t = (title or "").lower()
    on_floor = abs(y - floor_y) <= eps
    on_ceiling = False
    if ceiling_y is not None:
        on_ceiling = abs(y - ceiling_y) <= eps

    wall_keywords = ("window", "door", "wall", "frame", "baseboard", "switch", "socket")
    on_wall = any(k in t for k in wall_keywords) and (not on_floor) and (not on_ceiling)
    return {"on_floor": bool(on_floor), "on_wall": bool(on_wall), "on_ceiling": bool(on_ceiling)}


# -----------------------------
# Room polygon extraction (processed vs raw)
# -----------------------------

def _polygon_from_processed_room(room: Dict[str, Any]) -> List[List[float]]:
    floor = room.get("floor") or {}
    outline = floor.get("outline_xz") or []
    return [[float(p[0]), float(p[1])] for p in outline if isinstance(p, list) and len(p) == 2]


def _polygon_from_raw_extension_area(raw: Dict[str, Any], room_instanceid: str) -> Optional[List[List[float]]]:
    ext = raw.get("extension") or {}
    area = ext.get("area")
    if not isinstance(area, list):
        return None

    for a in area:
        if not isinstance(a, dict):
            continue
        rid = a.get("roomId") or a.get("instanceid") or a.get("id")
        if str(rid) != str(room_instanceid):
            continue

        pts = a.get("points") or a.get("polygon") or a.get("outline_xz")
        if isinstance(pts, list) and pts:
            poly = []
            for p in pts:
                if isinstance(p, list) and len(p) == 2:
                    poly.append([float(p[0]), float(p[1])])
            return poly if len(poly) >= 3 else None
    return None


# -----------------------------
# Export logic
# -----------------------------

def export_from_processed(inp: Dict[str, Any], out_dir: Path) -> int:
    rooms = inp.get("rooms") or []
    if not isinstance(rooms, list):
        raise ValueError("processed JSON: key 'rooms' must be a list")

    written = 0
    for r in rooms:
        if not isinstance(r, dict):
            continue

        room_type = r.get("room_type") or "Room"
        room_id = r.get("room_instanceid") or "unknown"
        floor = r.get("floor") or {}
        floor_y = float(floor.get("y", 0.0))
        ceiling_y = floor.get("ceiling_y")
        ceiling_y = float(ceiling_y) if ceiling_y is not None else None

        poly = _polygon_from_processed_room(r)

        objects_out = []
        sizes_out: Dict[str, Dict[str, float]] = {}
        placements_out: Dict[str, Any] = {}

        for obj in (r.get("objects") or []):
            if not isinstance(obj, dict):
                continue

            inst = str(obj.get("instanceid") or "")
            title = obj.get("title") or obj.get("category") or obj.get("ref") or ""
            ref = obj.get("ref")
            jid = _ref_to_jid(ref)

            pos = obj.get("pos") or [0.0, 0.0, 0.0]
            pos = [float(pos[0]), float(pos[1]), float(pos[2])]

            yaw = obj.get("yaw")
            if yaw is None:
                yaw = _quat_yaw_xyzw(obj.get("rot_quat") or [0,0,0,1])
            yaw = float(yaw)

            size = _normalize_size_any(obj.get("size")) or _normalize_size_any(obj.get("bbox_local"))

            objects_out.append({
                "id": inst,
                "title": title,
                "ref": ref,
                "jid": jid,
                "sourceCategoryId": obj.get("sourceCategoryId"),
                "category": obj.get("category"),
            })

            if size is not None:
                sizes_out[inst] = {"dx": float(size[0]), "dy": float(size[1]), "dz": float(size[2])}

            flags = _infer_attachment_flags(title=title, y=pos[1], floor_y=floor_y, ceiling_y=ceiling_y)
            placements_out[inst] = {
                "center": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "yaw": yaw,
                **flags
            }

        out = {
            "room": {"type": room_type, "id": room_id},
            "geometry": {
                "polygon_xz": poly,
                "floor_y": floor_y,
                "ceiling_y": ceiling_y,
            },
            "objects": objects_out,
            "sizes": sizes_out,
            "placements": placements_out,
            "meta": inp.get("meta", {}),
            "source_uid": inp.get("source_uid"),
            "design_version": inp.get("design_version"),
            "code_version": inp.get("code_version"),
            "version": inp.get("version"),
        }

        fname = f"{room_id}.json"
        (out_dir / fname).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    return written


def export_from_raw(inp: Dict[str, Any], out_dir: Path) -> int:
    furn_idx = _build_furniture_index(inp)
    scene = inp.get("scene") or {}
    scene_tf = _get_transform(scene)

    rooms = scene.get("room") or []
    if not isinstance(rooms, list):
        raise ValueError("raw JSON: scene.room must be a list")

    written = 0
    for r in rooms:
        if not isinstance(r, dict):
            continue

        room_type = r.get("type") or "Room"
        room_id = str(r.get("instanceid") or "unknown")
        room_tf = _get_transform(r)
        global_room_tf = scene_tf.compose(room_tf)

        # 1) Пытаемся взять polygon комнаты из extension.area
        poly = _polygon_from_raw_extension_area(inp, room_id)

        # 2) Если нет — строим прямоугольник по triangulation floor mesh
        floor_y = 0.0
        ceiling_y = None
        if poly is None:
            poly, floor_y_mesh = _polygon_from_raw_floor_mesh_as_rectangle(inp, room_id, eps_y=0.03)
            if floor_y_mesh is not None:
                floor_y = float(floor_y_mesh)

        if poly is None:
            raise RuntimeError(
                f"Не удалось получить геометрию комнаты room_id={room_id}: "
                f"нет extension.area и не найден floor mesh для построения прямоугольника."
            )

        objects_out = []
        sizes_out: Dict[str, Dict[str, float]] = {}
        placements_out: Dict[str, Any] = {}

        for ch in (r.get("children") or []):
            if not isinstance(ch, dict):
                continue

            inst = str(ch.get("instanceid") or "")
            ref = ch.get("ref")
            jid = _ref_to_jid(ref)
            title = ref or inst

            ch_tf = _get_transform(ch)
            global_obj_tf = global_room_tf.compose(ch_tf)
            center = global_obj_tf.pos
            yaw = _quat_yaw_xyzw(global_obj_tf.rot)

            rb = ch.get("replace_bbox") or {}
            size = None
            if isinstance(rb, dict) and all(k in rb for k in ("xLen", "yLen", "zLen")):
                size = [float(rb["xLen"]), float(rb["yLen"]), float(rb["zLen"])]

            if size is None and jid and jid in furn_idx:
                f = furn_idx[jid]
                size = _normalize_size_any(f.get("size")) or _normalize_size_any(f.get("bbox"))

            objects_out.append({
                "id": inst,
                "title": title,
                "ref": ref,
                "jid": jid,
                "replace_jid": ch.get("replace_jid"),
            })

            if size is not None:
                sizes_out[inst] = {"dx": float(size[0]), "dy": float(size[1]), "dz": float(size[2])}

            flags = _infer_attachment_flags(title=title, y=float(center[1]), floor_y=floor_y, ceiling_y=ceiling_y)
            placements_out[inst] = {
                "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                "yaw": float(yaw),
                **flags
            }

        out = {
            "room": {"type": room_type, "id": room_id},
            "geometry": {
                "polygon_xz": poly,          # для raw без extension.area это AABB-прямоугольник
                "floor_y": floor_y,          # оценка по floor mesh
                "ceiling_y": ceiling_y,
            },
            "objects": objects_out,
            "sizes": sizes_out,
            "placements": placements_out,
            "meta": {
                "axis_up": "Y",
                "quat_order": "xyzw",
                "yaw_definition": "rotation_about_Y_in_scene_coords_radians",
                "units": "meters_assumed",
                "note": "centers are in scene coords after applying scene+room+child transforms",
                "room_polygon_source": "extension.area or floor_mesh_aabb",
            },
            "uid": inp.get("uid"),
            "design_version": inp.get("design_version"),
            "code_version": inp.get("code_version"),
            "version": inp.get("version"),
        }

        fname = f"{room_id}.json"
        (out_dir / fname).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input JSON (raw 3D-FRONT or processed)")
    ap.add_argument("--out-dir", dest="out_dir", default=None, help="output directory (default: <input_dir>/rooms_processed)")
    args = ap.parse_args()

    inp_path = Path(args.inp)
    if not inp_path.exists():
        raise FileNotFoundError(str(inp_path))

    data = json.loads(inp_path.read_text(encoding="utf-8"))

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = inp_path.parent / "rooms_processed"

    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict) and "rooms" in data:
        written = export_from_processed(data, out_dir)
    else:
        written = export_from_raw(data, out_dir)

    print(f"[ok] written {written} room files into: {out_dir}")


if __name__ == "__main__":
    main()