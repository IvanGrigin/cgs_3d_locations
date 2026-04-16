#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.ml.models.repair_diffusion import DiffusionSchedule, RepairDiffusionNet, gather_step
from src.tools.evaluate_unified_scene import (
    box_intersection_volume,
    parse_placements,
    parse_room,
    should_ignore_collision,
)
from src.tools.normalize_scene_format import build_aabb_from_center_size, build_scene_from_room_and_placement, convert_to_scene_v1


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS недоступен")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA недоступна")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def scene_from_inputs(scene_path: str | None, room_path: str | None, placement_path: str | None) -> Dict[str, Any]:
    if scene_path:
        return convert_to_scene_v1(load_json(scene_path))
    if room_path and placement_path:
        return build_scene_from_room_and_placement(load_json(room_path), load_json(placement_path))
    raise ValueError("Нужно передать либо --scene, либо пару --room + --placement")


def room_bbox(room) -> Tuple[float, float, float, float]:
    return room.x_min, room.x_max, room.y_min, room.y_max


def normalize_xy(x: float, y: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, x1, y0, y1 = bounds
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = max((x1 - x0) * 0.5, 1e-6)
    hy = max((y1 - y0) * 0.5, 1e-6)
    return (float((x - cx) / hx), float((y - cy) / hy))


def denormalize_xy(nx: float, ny: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, x1, y0, y1 = bounds
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = max((x1 - x0) * 0.5, 1e-6)
    hy = max((y1 - y0) * 0.5, 1e-6)
    return (float(cx + nx * hx), float(cy + ny * hy))


def normalize_size(size_xy: Tuple[float, float], bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, x1, y0, y1 = bounds
    hx = max((x1 - x0) * 0.5, 1e-6)
    hy = max((y1 - y0) * 0.5, 1e-6)
    return (float(size_xy[0] / hx), float(size_xy[1] / hy))


def detect_bad_indices(room, placements) -> List[int]:
    bad = set()
    for i, p in enumerate(placements):
        if (
            p.x_min < room.x_min or p.x_max > room.x_max or
            p.y_min < room.y_min or p.y_max > room.y_max or
            p.z_min < room.z_min or p.z_max > room.z_max
        ):
            bad.add(i)
        for j, other in enumerate(placements):
            if i >= j:
                continue
            if should_ignore_collision(p, other):
                continue
            if box_intersection_volume(p, other) > 1e-6:
                bad.add(i)
                bad.add(j)
    return sorted(bad)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[RepairDiffusionNet, Dict[str, Any], Dict[str, torch.Tensor]]:
    obj = torch.load(ckpt_path, map_location=device)
    cfg = dict(obj["cfg"])
    model = RepairDiffusionNet(
        num_categories=int(obj["num_categories"]),
        dim=int(cfg.get("dim", 256)),
        num_layers=int(cfg.get("num_layers", 6)),
        num_heads=int(cfg.get("num_heads", 8)),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(obj["model_state"], strict=True)
    model.eval()
    schedule = DiffusionSchedule(
        T=int(cfg.get("T", 200)),
        beta_start=float(cfg.get("beta_start", 1e-4)),
        beta_end=float(cfg.get("beta_end", 2e-2)),
    ).build(device)
    return model, cfg, schedule


def infer_one(
    model: RepairDiffusionNet,
    schedule: Dict[str, torch.Tensor],
    scene: Dict[str, Any],
    meta: Dict[str, Any],
    index: int,
    device: torch.device,
    steps: int,
) -> Dict[str, Any]:
    room = parse_room(scene)
    placements = parse_placements(scene)
    bounds = room_bbox(room)
    num_categories = int(max(meta.get("cat2id", {}).values(), default=0)) + 1
    cat2id = meta.get("cat2id", {})

    N = len(placements)
    context_pos = np.zeros((N, 2), dtype=np.float32)
    context_size = np.zeros((N, 2), dtype=np.float32)
    context_cat = np.zeros((N,), dtype=np.int64)
    context_mask = np.ones((N,), dtype=np.float32)

    for i, p in enumerate(placements):
        context_pos[i] = normalize_xy(p.position_m[0], p.position_m[1], bounds)
        context_size[i] = normalize_size((p.size_m[0], p.size_m[1]), bounds)
        context_cat[i] = int(cat2id.get(p.category, 0))

    target = placements[index]
    target_cat = int(cat2id.get(target.category, 0))
    target_size = np.asarray(context_size[index], dtype=np.float32)
    corruption_type = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    room_h_world = np.array(
        [
            0.5 * (room.x_max - room.x_min),
            0.5 * (room.z_max - room.z_min),
            0.5 * (room.y_max - room.y_min),
        ],
        dtype=np.float32,
    )

    x = torch.tensor(context_pos[index][None, :], dtype=torch.float32, device=device)
    context_pos_t = torch.tensor(context_pos[None, ...], dtype=torch.float32, device=device)
    context_size_t = torch.tensor(context_size[None, ...], dtype=torch.float32, device=device)
    context_cat_t = torch.tensor(context_cat[None, ...], dtype=torch.long, device=device)
    context_mask_t = torch.tensor(context_mask[None, ...], dtype=torch.float32, device=device)
    target_index_t = torch.tensor([index], dtype=torch.long, device=device)
    target_cat_t = torch.tensor([target_cat], dtype=torch.long, device=device)
    target_size_t = torch.tensor(target_size[None, :], dtype=torch.float32, device=device)
    corruption_type_t = torch.tensor(corruption_type[None, :], dtype=torch.float32, device=device)
    room_h_world_t = torch.tensor(room_h_world[None, :], dtype=torch.float32, device=device)

    idx = torch.linspace(int(schedule["abar"].shape[0]) - 1, 0, int(max(1, steps)), device=device).long()
    for k in range(idx.shape[0]):
        t = idx[k].view(1)
        eps = model(
            x_t=x,
            t=t,
            context_pos=context_pos_t,
            context_size=context_size_t,
            context_cat=context_cat_t,
            context_mask=context_mask_t,
            target_index=target_index_t,
            target_cat=target_cat_t,
            target_size=target_size_t,
            corruption_type=corruption_type_t,
            room_h_world=room_h_world_t,
        )
        a_bar = gather_step(schedule["abar"], t)
        sqrt_a_bar = torch.sqrt(a_bar)
        sqrt_1m = torch.sqrt(1.0 - a_bar)
        x0 = (x - sqrt_1m * eps) / torch.clamp(sqrt_a_bar, min=1e-6)
        if k == idx.shape[0] - 1:
            x = x0
            break
        t_prev = idx[k + 1].view(1)
        a_bar_prev = gather_step(schedule["abar"], t_prev)
        x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=0.0)) * eps

    pred = x[0].detach().cpu().numpy()
    pred = np.clip(pred, -1.0, 1.0)
    px, py = denormalize_xy(float(pred[0]), float(pred[1]), bounds)

    out_scene = deepcopy(scene)
    src = deepcopy(out_scene["placements"][index])
    new_pos = [float(px), float(py), float(src["position_m"][2])]
    src["position_m"] = new_pos
    src["aabb"] = build_aabb_from_center_size(new_pos, src["size_m"])
    out_scene["placements"][index] = src
    return out_scene


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer single-object repair with trained diffusion model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True, help="Dataset meta json with cat2id")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--room", default=None)
    ap.add_argument("--placement", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bad-index", type=int, default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    device = pick_device(args.device)
    scene = scene_from_inputs(args.scene, args.room, args.placement)
    meta = load_json(args.meta)
    model, cfg, schedule = load_model(args.model, device)

    room = parse_room(scene)
    placements = parse_placements(scene)
    bad = detect_bad_indices(room, placements)
    if args.bad_index is not None:
        target_idx = int(args.bad_index)
    elif bad:
        target_idx = int(bad[0])
    else:
        target_idx = 0

    out_scene = infer_one(model, schedule, scene, meta, target_idx, device, args.steps)
    out_meta = out_scene.get("meta")
    if not isinstance(out_meta, dict):
        out_meta = {}
        out_scene["meta"] = out_meta
    out_meta["repair_diffusion_infer"] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "target_index": int(target_idx),
        "steps": int(args.steps),
        "model_cfg": cfg,
    }

    save_json(args.out, out_scene)
    print(f"[repair_diffusion_infer] target_idx={target_idx} out={Path(args.out).expanduser().resolve()}")


if __name__ == "__main__":
    main()
