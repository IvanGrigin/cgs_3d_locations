#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# src/tools/preprocess_front_simple.py
#
# Пакетная предобработка 3D-FRONT сцен:
# - читает все *.json из директории input_dir (рекурсивно, опционально)
# - для каждого файла строит "prepared_scene_*.json" (упрощённый формат)
# - пишет в output_dir, сохраняя относительную структуру поддиректорий (если recurse=1)
#
# Пример запуска:
# python3 src/tools/preprocess_front_simple.py \
#   --input_dir data/sourse/3D-FRONT/3D-FRONT \
#   --output_dir data/sourse/3D-FRONT/3D-FRONT-processed \
#   --recurse 0 \
#   --polygon_round_nd 4 \
#   --window_max_dist 1.5
#
# ВАЖНО:
# - Скрипт ожидает формат 3D-FRONT raw json: поля "mesh", "furniture", "scene".
#

import argparse
import json
import math
import collections
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterator


def quat_to_yaw_deg(q: List[float]) -> float:
    """
    q = [x, y, z, w], y-up.
    Возвращает yaw (поворот вокруг оси Y) в градусах.
    """
    x, y, z, w = q
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw * 180.0 / math.pi


def mesh_center_and_yaw_deg(mesh: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Центр меша по среднему координат вершин + оценка ориентации (yaw) по PCA в плоскости XZ.
    """
    xyz = mesh["xyz"]
    pts = [(xyz[i], xyz[i + 1], xyz[i + 2]) for i in range(0, len(xyz), 3)]
    if not pts:
        return 0.0, 0.0, 0.0, 0.0

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)

    xs = [p[0] for p in pts]
    zs = [p[2] for p in pts]
    mx, mz = cx, cz

    cov_xx = sum((x - mx) ** 2 for x in xs) / len(xs)
    cov_zz = sum((z - mz) ** 2 for z in zs) / len(zs)
    cov_xz = sum((x - mx) * (z - mz) for x, z in zip(xs, zs)) / len(xs)

    trace = cov_xx + cov_zz
    det = cov_xx * cov_zz - cov_xz * cov_xz
    disc = max(trace * trace - 4.0 * det, 0.0)
    lam1 = (trace + math.sqrt(disc)) / 2.0

    vx = cov_xz
    vz = lam1 - cov_xx
    if abs(vx) + abs(vz) < 1e-12:
        vx = lam1 - cov_zz
        vz = cov_xz

    # yaw: 0° соответствует направлению +Z, далее по atan2
    yaw = math.atan2(vx, vz) * 180.0 / math.pi
    return cx, cy, cz, yaw


def boundary_polygon_from_floor_meshes(floor_meshes: List[Dict[str, Any]], nd: int = 4) -> List[Dict[str, float]]:
    """
    Периметр комнаты по границе триангуляции пола:
    1) собираем ребра всех треугольников,
    2) ребра с кратностью 1 — граничные,
    3) обходим цикл.
    """
    vid_map: Dict[Tuple[float, float], int] = {}
    vid_pts: List[Tuple[float, float]] = []

    def get_vid(x: float, z: float) -> int:
        k = (round(x, nd), round(z, nd))
        if k not in vid_map:
            vid_map[k] = len(vid_pts)
            vid_pts.append(k)
        return vid_map[k]

    edge_count = collections.Counter()

    for m in floor_meshes:
        xyz = m.get("xyz", [])
        faces = m.get("faces", [])
        if not xyz or not faces:
            continue

        verts_xz = [(xyz[i], xyz[i + 2]) for i in range(0, len(xyz), 3)]
        for i in range(0, len(faces), 3):
            tri = faces[i: i + 3]
            if len(tri) != 3:
                continue
            a, b, c = tri
            if not (0 <= a < len(verts_xz) and 0 <= b < len(verts_xz) and 0 <= c < len(verts_xz)):
                continue

            va = get_vid(*verts_xz[a])
            vb = get_vid(*verts_xz[b])
            vc = get_vid(*verts_xz[c])

            for u, v in ((va, vb), (vb, vc), (vc, va)):
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                edge_count[(u, v)] += 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return []

    adj: Dict[int, List[int]] = collections.defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)

    start = min(adj.keys(), key=lambda i: (vid_pts[i][0], vid_pts[i][1]))
    poly = [start]
    prev = None
    cur = start

    while True:
        neigh = adj[cur]
        if not neigh:
            break

        if prev is None:
            nxt = min(neigh, key=lambda i: (vid_pts[i][0], vid_pts[i][1]))
        else:
            if len(neigh) == 1:
                nxt = neigh[0]
            else:
                nxt = neigh[0] if neigh[1] == prev else neigh[1]

        if nxt == start:
            break

        poly.append(nxt)
        prev, cur = cur, nxt

        if len(poly) > len(adj) + 10:
            break

    return [{"x": vid_pts[i][0], "z": vid_pts[i][1]} for i in poly]


def dist_point_to_segment(px: float, pz: float, x1: float, z1: float, x2: float, z2: float) -> float:
    vx, vz = x2 - x1, z2 - z1
    wx, wz = px - x1, pz - z1
    c1 = wx * vx + wz * vz
    if c1 <= 0:
        return math.hypot(px - x1, pz - z1)
    c2 = vx * vx + vz * vz
    if c2 <= c1:
        return math.hypot(px - x2, pz - z2)
    b = c1 / c2
    bx, bz = x1 + b * vx, z1 + b * vz
    return math.hypot(px - bx, pz - bz)


def room_edge_distance(px: float, pz: float, poly: List[Dict[str, float]]) -> float:
    if not poly:
        return float("inf")
    n = len(poly)
    dmin = float("inf")
    for i in range(n):
        x1, z1 = poly[i]["x"], poly[i]["z"]
        x2, z2 = poly[(i + 1) % n]["x"], poly[(i + 1) % n]["z"]
        dmin = min(dmin, dist_point_to_segment(px, pz, x1, z1, x2, z2))
    return dmin


def iter_scene_nodes(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    yield node
    for ch in node.get("children", []):
        if isinstance(ch, dict):
            yield from iter_scene_nodes(ch)


def convert_one(
    input_path: Path,
    output_path: Path,
    polygon_round_nd: int = 4,
    window_max_dist: float = 1.5,
) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))

    mesh_list = data.get("mesh", [])
    furn_list = data.get("furniture", [])
    scene = data.get("scene", {})

    mesh_by_uid = {m.get("uid"): m for m in mesh_list if isinstance(m, dict) and m.get("uid") is not None}
    furniture_meta = {f.get("uid"): f for f in furn_list if isinstance(f, dict) and f.get("uid") is not None}

    out = {
        "uid": data.get("uid"),
        "north_vector": data.get("north_vector"),
        "rooms": [],
    }

    rooms = scene.get("room", [])
    if not isinstance(rooms, list):
        rooms = []

    window_meshes = [m for m in mesh_list if isinstance(m, dict) and m.get("type") == "Window"]

    for room in rooms:
        if not isinstance(room, dict):
            continue
        nodes = list(iter_scene_nodes(room))
        room_id = room.get("instanceid")
        if room_id is None:
            continue

        mesh_refs = [
            n.get("ref")
            for n in nodes
            if isinstance(n, dict) and str(n.get("instanceid", "")).startswith("mesh/")
        ]
        floor_meshes = [
            mesh_by_uid[r] for r in mesh_refs
            if r in mesh_by_uid and isinstance(mesh_by_uid[r], dict) and mesh_by_uid[r].get("type") == "Floor"
        ]
        polygon = boundary_polygon_from_floor_meshes(floor_meshes, nd=polygon_round_nd)

        objects = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if str(n.get("instanceid", "")).startswith("furniture/"):
                ref = n.get("ref")
                meta = furniture_meta.get(ref, {}) if ref is not None else {}
                jid = meta.get("jid")

                pos = n.get("pos", [0.0, 0.0, 0.0])
                if not isinstance(pos, list) or len(pos) < 3:
                    pos = [0.0, 0.0, 0.0]

                rot = n.get("rot", [0, 0, 0, 1])
                if not isinstance(rot, list) or len(rot) < 4:
                    rot = [0, 0, 0, 1]

                sc = n.get("scale", [1, 1, 1])
                if not isinstance(sc, list) or len(sc) < 3:
                    sc = [1, 1, 1]

                objects.append({
                    "instanceid": n.get("instanceid"),
                    "ref": ref,
                    "jid": jid,  # <-- КЛЮЧЕВО
                    "category": meta.get("category"),
                    "pos": {"x": pos[0], "y": pos[1], "z": pos[2]},
                    "yaw_deg": quat_to_yaw_deg(rot),
                    "scale": sc,
                    "bbox": meta.get("bbox") or meta.get("size"),
                    "valid": meta.get("valid", True),
                })

        doors_out = []
        for d in data.get("extension", {}).get("door", []):
            if not isinstance(d, dict):
                continue
            if d.get("roomId") != room_id:
                continue
            meshes = []
            for ref in d.get("ref", []):
                m = mesh_by_uid.get(ref)
                if m and m.get("type") in ("Door", "Pocket", "Hole"):
                    meshes.append(m)
            if not meshes:
                continue

            centers = [mesh_center_and_yaw_deg(m) for m in meshes]
            cx = sum(c[0] for c in centers) / len(centers)
            cy = sum(c[1] for c in centers) / len(centers)
            cz = sum(c[2] for c in centers) / len(centers)
            yaw = sum(c[3] for c in centers) / len(centers)

            doors_out.append({
                "type": d.get("type"),
                "dir": d.get("dir"),
                "refs": d.get("ref"),
                "center": {"x": cx, "y": cy, "z": cz},
                "yaw_deg": yaw,
            })

        out["rooms"].append({
            "id": room_id,
            "type": room.get("type"),
            "polygon": polygon,
            "doors": doors_out,
            "windows": [],
            "objects": objects,
        })

    for m in window_meshes:
        cx, cy, cz, yaw = mesh_center_and_yaw_deg(m)

        best_room = None
        best_dist = float("inf")
        for r in out["rooms"]:
            d = room_edge_distance(cx, cz, r["polygon"])
            if d < best_dist:
                best_dist = d
                best_room = r

        if best_room is not None and best_dist <= window_max_dist:
            best_room["windows"].append({
                "uid": m.get("uid"),
                "center": {"x": cx, "y": cy, "z": cz},
                "yaw_deg": yaw,
                "dist_to_room": best_dist,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def make_output_path(input_file: Path, input_dir: Path, output_dir: Path, preserve_tree: bool) -> Path:
    """
    Имя: prepared_scene_<stem>.json.
    Если preserve_tree=True — сохраняем относительный путь от input_dir.
    """
    stem = input_file.stem
    out_name = f"prepared_scene_{stem}.json"

    if preserve_tree:
        rel = input_file.relative_to(input_dir)
        rel_parent = rel.parent
        return (output_dir / rel_parent / out_name).resolve()

    return (output_dir / out_name).resolve()


def list_input_jsons(input_dir: Path, recurse: bool) -> List[Path]:
    if recurse:
        files = sorted([p for p in input_dir.rglob("*.json") if p.is_file()])
    else:
        files = sorted([p for p in input_dir.glob("*.json") if p.is_file()])
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Директория с raw 3D-FRONT json файлами")
    ap.add_argument("--output_dir", required=True, help="Директория для 3D-FRONT-processed")
    ap.add_argument("--recurse", type=int, default=0, help="1: рекурсивно по подпапкам; 0: только верхний уровень")
    ap.add_argument("--preserve_tree", type=int, default=0, help="1: сохранять структуру подпапок в output_dir")
    ap.add_argument("--polygon_round_nd", type=int, default=4)
    ap.add_argument("--window_max_dist", type=float, default=1.5)
    ap.add_argument("--skip_existing", type=int, default=0, help="1: пропускать если файл output уже существует")

    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    recurse = bool(int(args.recurse))
    preserve_tree = bool(int(args.preserve_tree))
    skip_existing = bool(int(args.skip_existing))

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir не существует или не директория: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_input_jsons(input_dir, recurse=recurse)
    if not files:
        print(f"Нет *.json в {input_dir} (recurse={int(recurse)})")
        return

    ok = 0
    fail = 0
    for f in files:
        out_path = make_output_path(f, input_dir=input_dir, output_dir=output_dir, preserve_tree=preserve_tree)
        if skip_existing and out_path.exists():
            continue
        try:
            convert_one(
                input_path=f,
                output_path=out_path,
                polygon_round_nd=int(args.polygon_round_nd),
                window_max_dist=float(args.window_max_dist),
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {f} -> {out_path}: {e}")

    print(f"Done. ok={ok}, fail={fail}, out_dir={output_dir}")


if __name__ == "__main__":
    main()
