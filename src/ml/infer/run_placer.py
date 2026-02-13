#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/ml/infer/run_placer.py

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Common IO
# -----------------------------

def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


# -----------------------------
# Room helpers
# -----------------------------

def polygon_bbox(poly_xz: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    xs = [safe_float(p.get("x", 0.0)) for p in poly_xz]
    zs = [safe_float(p.get("z", 0.0)) for p in poly_xz]
    if not xs or not zs:
        return (0.0, 1.0, 0.0, 1.0)
    return (min(xs), max(xs), min(zs), max(zs))


def iso_params_from_bbox(bb: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    x0, x1, z0, z1 = bb
    dx = max(1e-6, x1 - x0)
    dz = max(1e-6, z1 - z0)
    s = max(dx, dz)
    cx = 0.5 * (x0 + x1)
    cz = 0.5 * (z0 + z1)
    return cx, cz, s


# -----------------------------
# objects.json helpers
# -----------------------------

def sample_size_mm(item: Dict[str, Any], rng: random.Random) -> List[int]:
    mn = item.get("min_size_mm", [600, 400, 600])
    mx = item.get("max_size_mm", [1200, 800, 1000])

    def pick(i: int) -> int:
        a = int(mn[i]) if i < len(mn) else 0
        b = int(mx[i]) if i < len(mx) else a
        if b < a:
            a, b = b, a
        if a == b:
            return a
        return rng.randint(a, b)

    return [pick(0), pick(1), pick(2)]


def dims_xz_m_from_size_mm(size_mm: List[int]) -> Tuple[float, float]:
    # В проекте XZ — это план комнаты.
    # В вашем objects.json размеры заданы как [x,y,z] мм.
    # Для AABB в XZ используем (x,z).
    sx = max(0.0, float(size_mm[0]) / 1000.0)
    sz = max(0.0, float(size_mm[2]) / 1000.0)
    return sx, sz


# -----------------------------
# Placement result schema
# -----------------------------

def build_placement_result(seed: Optional[int], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Сохраняем структуру, максимально похожую на то, что ожидает ваш визуализатор:
    # { seed, placements: [ { ...item..., pos:{x,y,z}, yaw_deg, size_mm } ] }
    return {
        "seed": int(seed) if seed is not None else None,
        "placements": items,
    }


# -----------------------------
# Baseline adapters (forest / graph_stat)
# -----------------------------

def _build_room_for_baselines(room_spec: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ForestPlacer / GraphStatPlacer обучались на *.mini.json, где:
      room = { floor_polygon_xz, objects:[ ... ] }
    Мы создаём совместимую структуру "room" из room.json + objects.json.
    """
    poly = room_spec.get("floor_polygon_xz", [])
    if not isinstance(poly, list):
        poly = []

    objs = []
    for it in items:
        iid = it.get("instanceid")
        if iid is None:
            iid = it.get("_instanceid")
        name = str(it.get("name", "UNK"))

        # bbox_world_xy ожидается в формате [xmin, xmax, zmin, zmax] (как у вас в диффузии)
        # Мы пока ставим центр (0,0) — модель вернёт позицию.
        size_mm = it.get("size_mm", [600, 400, 600])
        w, d = dims_xz_m_from_size_mm(size_mm)
        xmin, xmax = -0.5 * w, 0.5 * w
        zmin, zmax = -0.5 * d, 0.5 * d

        objs.append({
            "instanceid": iid,
            "label": name,
            "pos": {"x": 0.0, "z": 0.0},
            "yaw_deg": 0.0,
            "bbox_world_xy": [xmin, xmax, zmin, zmax],
        })

    return {
        "floor_polygon_xz": poly,
        "objects": objs,
    }


def predict_with_forest(model_path: str, room_spec: Dict[str, Any], placed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from src.ml.baselines.forest_placer import ForestPlacer

    placer = ForestPlacer.load(model_path)
    room = _build_room_for_baselines(room_spec, placed_items)

    preds = placer.predict_room(room)
    pred_by_id = {p.get("instanceid"): p for p in preds}

    out = []
    for it in placed_items:
        iid = it.get("_instanceid")
        p = pred_by_id.get(iid)
        if p is None:
            out.append({"x": 0.0, "z": 0.0, "yaw_deg": 0.0})
            continue
        px = float(p["pred"]["pos"]["x"])
        pz = float(p["pred"]["pos"]["z"])
        yaw = float(p["pred"].get("yaw_deg", 0.0)) if isinstance(p.get("pred"), dict) else 0.0
        out.append({"x": px, "z": pz, "yaw_deg": yaw})
    return out


def predict_with_graph_stat(model_path: str, room_spec: Dict[str, Any], placed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from src.ml.baselines.graph_stat_placer import GraphStatPlacer

    placer = GraphStatPlacer.load(model_path)
    room = _build_room_for_baselines(room_spec, placed_items)

    preds = placer.predict_room(room)
    pred_by_id = {p.get("instanceid"): p for p in preds}

    out = []
    for it in placed_items:
        iid = it.get("_instanceid")
        p = pred_by_id.get(iid)
        if p is None:
            out.append({"x": 0.0, "z": 0.0, "yaw_deg": 0.0})
            continue
        px = float(p["pred"]["pos"]["x"])
        pz = float(p["pred"]["pos"]["z"])
        yaw = float(p["pred"].get("yaw_deg", 0.0)) if isinstance(p.get("pred"), dict) else 0.0
        out.append({"x": px, "z": pz, "yaw_deg": yaw})
    return out


# -----------------------------
# Diffusion adapter
# -----------------------------

def predict_with_diffusion(model_path: str, room_spec: Dict[str, Any], placed_items: List[Dict[str, Any]], device: str, steps: int) -> List[Dict[str, Any]]:
    from src.ml.infer.diffusion_placer import DiffusionPlacer

    placer = DiffusionPlacer.load(model_path=model_path, device=device)
    return placer.predict(room_spec=room_spec, placed_items=placed_items, steps=steps)


# -----------------------------
# Main
# -----------------------------

@dataclass
class Args:
    backend: str
    model: str
    room: str
    objects: str
    out: str
    seed: Optional[int]
    device: str
    ddim_steps: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["forest", "graph_stat", "diffusion"])
    ap.add_argument("--model", required=True, help="Path to model file (.pkl or .pt)")
    ap.add_argument("--room", required=True, help="Path to room.json (room-spec)")
    ap.add_argument("--objects", required=True, help="Path to objects.json (generated by run_pipeline.py)")
    ap.add_argument("--out", required=True, help="Path to placement_result.json")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--ddim-steps", type=int, default=50, help="Only for diffusion")
    args_ns = ap.parse_args()

    args = Args(
        backend=str(args_ns.backend),
        model=str(args_ns.model),
        room=str(args_ns.room),
        objects=str(args_ns.objects),
        out=str(args_ns.out),
        seed=int(args_ns.seed) if args_ns.seed is not None else None,
        device=str(args_ns.device),
        ddim_steps=int(args_ns.ddim_steps),
    )

    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    room_spec = load_json(args.room)
    obj_data = load_json(args.objects)

    items = obj_data.get("items", [])
    if not isinstance(items, list) or not items:
        raise SystemExit("[run_placer] objects.json: items is empty")

    # фиксируем instanceid и конкретный size_mm (чтобы Blender ставил именно это)
    placed_items: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        it2 = dict(it)
        it2["_instanceid"] = int(i)
        size_mm = sample_size_mm(it2, rng)
        it2["size_mm"] = size_mm
        placed_items.append(it2)

    if args.backend == "forest":
        preds = predict_with_forest(args.model, room_spec, placed_items)
    elif args.backend == "graph_stat":
        preds = predict_with_graph_stat(args.model, room_spec, placed_items)
    elif args.backend == "diffusion":
        preds = predict_with_diffusion(args.model, room_spec, placed_items, device=args.device, steps=args.ddim_steps)
    else:
        raise SystemExit(f"Unknown backend: {args.backend}")

    if len(preds) != len(placed_items):
        raise SystemExit(f"[run_placer] preds length mismatch: {len(preds)} vs {len(placed_items)}")

    # записываем placements: добавляем pos/yaw_deg на каждый item из objects.json
    out_items: List[Dict[str, Any]] = []
    for it, pr in zip(placed_items, preds):
        x = float(pr.get("x", 0.0))
        z = float(pr.get("z", 0.0))
        yaw_deg = float(pr.get("yaw_deg", 0.0))

        it_out = dict(it)
        it_out["instanceid"] = int(it_out["_instanceid"])
        it_out.pop("_instanceid", None)

        it_out["pos"] = {"x": x, "y": 0.0, "z": z}
        it_out["yaw_deg"] = yaw_deg

        out_items.append(it_out)

    res = build_placement_result(seed=args.seed, items=out_items)
    save_json(args.out, res)

    print(f"[run_placer] OK: backend={args.backend} placements={len(out_items)} out={args.out}")


if __name__ == "__main__":
    main()
