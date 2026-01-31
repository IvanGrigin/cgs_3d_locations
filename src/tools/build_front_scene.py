#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

def _as_mat4(x):
    a = np.array(x, dtype=np.float64)
    if a.shape == (4, 4):
        return a
    if a.size == 16:
        return a.reshape(4, 4)
    return None

def _bbox_center_from_furniture(f):
    # Частый фолбэк: bbox может быть [dx,dy,dz] или [[cx,cy,cz]].
    bb = f.get("bbox", None)
    if bb is None:
        return None
    if isinstance(bb, (list, tuple)) and len(bb) == 1 and isinstance(bb[0], (list, tuple)) and len(bb[0]) == 3:
        return np.array(bb[0], dtype=np.float64)
    # если bbox — это размеры, это не центр
    return None

def _get_translation(f):
    for key in ("position", "pos", "translation", "translate", "t"):
        v = f.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return np.array(v, dtype=np.float64)
    c = _bbox_center_from_furniture(f)
    return c

def _quat_to_mat3(q):
    # q = [x,y,z,w] или [w,x,y,z] — неизвестно.
    # Делаем эвристику: если последний компонент по модулю самый большой -> считаем это w.
    q = np.array(q, dtype=np.float64).reshape(4)
    absq = np.abs(q)
    if absq[3] >= absq[0] and absq[3] >= absq[1] and absq[3] >= absq[2]:
        x, y, z, w = q
    else:
        w, x, y, z = q
    # нормализация
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    # матрица поворота
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def _get_rotation_mat3(f):
    # 1) явная матрица 3x3
    for key in ("rotation_mat", "R"):
        v = f.get(key)
        if isinstance(v, (list, tuple)):
            a = np.array(v, dtype=np.float64)
            if a.shape == (3, 3):
                return a
            if a.size == 9:
                return a.reshape(3, 3)
    # 2) кватернион
    for key in ("rotation", "rot", "quaternion", "quat"):
        v = f.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return _quat_to_mat3(v)
    # 3) Euler (если вдруг встретится)
    for key in ("euler", "rotation_euler"):
        v = f.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            rx, ry, rz = map(float, v)
            return trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
    return np.eye(3)

def _mesh_from_front_piece(piece):
    xyz = np.array(piece["xyz"], dtype=np.float64).reshape(-1, 3)
    faces = np.array(piece["faces"], dtype=np.int64).reshape(-1, 3)
    m = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
    m.metadata["type"] = piece.get("type", "")
    m.metadata["uid"] = piece.get("uid", "")
    m.metadata["material"] = piece.get("material", "")
    return m

def _safe_load_obj(obj_path: Path) -> trimesh.Trimesh:
    # trimesh.load может вернуть Scene; приводим к одному мешу
    loaded = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        # сольём геометрию
        geoms = []
        for g in loaded.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                geoms.append(g)
        if not geoms:
            raise ValueError(f"Empty OBJ scene: {obj_path}")
        return trimesh.util.concatenate(geoms)
    return loaded

def build(scene_json: Path, models_root: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(scene_json.read_text(encoding="utf-8"))

    # 1) room mesh
    room_pieces = []
    for piece in data.get("mesh", []):
        try:
            room_pieces.append(_mesh_from_front_piece(piece))
        except Exception:
            # куски бывают кривые; лучше пропустить, чем падать
            continue

    room_mesh = trimesh.util.concatenate(room_pieces) if room_pieces else trimesh.Trimesh()
    room_glb = out_dir / "room.glb"
    room_mesh.export(room_glb)

    # 2) furniture instances
    instances = []
    furniture_meshes = []

    for f in data.get("furniture", []):
        if not f.get("valid", False):
            continue
        jid = f.get("jid")
        if not jid:
            continue

        model_dir = models_root / jid
        # приоритет normalized_model.obj, иначе raw_model.obj
        obj_path = model_dir / "normalized_model.obj"
        if not obj_path.exists():
            obj_path = model_dir / "raw_model.obj"
        if not obj_path.exists():
            continue

        try:
            mesh = _safe_load_obj(obj_path)
        except Exception:
            continue

        # scale: подгоняем под furniture.size (если есть)
        target_size = f.get("size")
        if isinstance(target_size, (list, tuple)) and len(target_size) == 3:
            target = np.array(target_size, dtype=np.float64)
            extent = np.array(mesh.bounding_box.extents, dtype=np.float64)
            extent = np.maximum(extent, 1e-6)
            s = target / extent
        else:
            s = np.ones(3, dtype=np.float64)

        R = _get_rotation_mat3(f)
        t = _get_translation(f)
        if t is None:
            t = np.zeros(3, dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R @ np.diag(s)
        T[:3, 3] = t

        mesh.apply_transform(T)

        instances.append({
            "uid": f.get("uid"),
            "jid": jid,
            "category": f.get("category", ""),
            "sourceCategoryId": f.get("sourceCategoryId", ""),
            "transform": T.reshape(-1).tolist(),
            "target_size": target_size if target_size is not None else None,
            "bbox_extents": mesh.bounding_box.extents.tolist(),
            "bbox_center": mesh.bounding_box.centroid.tolist(),
        })
        furniture_meshes.append(mesh)

    (out_dir / "objects.json").write_text(
        json.dumps({"uid": data.get("uid"), "instances": instances}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 3) full scene
    full = trimesh.Scene()
    if room_mesh.vertices.size:
        full.add_geometry(room_mesh, node_name="room")
    for i, m in enumerate(furniture_meshes):
        full.add_geometry(m, node_name=f"obj_{i:04d}")

    scene_glb = out_dir / "scene.glb"
    full.export(scene_glb)

    return {
        "room_glb": str(room_glb),
        "scene_glb": str(scene_glb),
        "objects_json": str(out_dir / "objects.json"),
        "num_instances": len(instances),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_json", required=True)
    ap.add_argument("--models_root", required=True)  # .../3D-FUTURE-model
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    res = build(Path(args.scene_json), Path(args.models_root), Path(args.out_dir))
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()