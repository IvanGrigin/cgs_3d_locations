#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.ml.data_py.repair_dataset import FrontRepairDataset, RepairBatch, collate_repair
from src.ml.models.repair_diffusion import (
    DiffusionSchedule,
    RepairDiffusionNet,
    predict_x0_from_eps,
    q_sample,
)


@dataclass
class TrainConfig:
    npz: str
    splits: str
    out_dir: str
    epochs: int = 40
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    train_samples_per_scene: int = 8
    val_samples_per_scene: int = 2
    dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.0
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    w_eps: float = 1.0
    w_x0: float = 0.5
    w_boundary: float = 0.2
    w_collision: float = 0.3
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def move_batch(batch: RepairBatch, device: torch.device) -> RepairBatch:
    return RepairBatch(
        x0_target=batch.x0_target.to(device),
        corrupted_target=batch.corrupted_target.to(device),
        context_pos=batch.context_pos.to(device),
        context_size=batch.context_size.to(device),
        context_cat=batch.context_cat.to(device),
        context_mask=batch.context_mask.to(device),
        target_index=batch.target_index.to(device),
        target_cat=batch.target_cat.to(device),
        target_size=batch.target_size.to(device),
        corruption_type=batch.corruption_type.to(device),
        room_h_world=batch.room_h_world.to(device),
    )


def _pairwise_intersection(
    pred_center: torch.Tensor,
    pred_size: torch.Tensor,
    ctx_center: torch.Tensor,
    ctx_size: torch.Tensor,
) -> torch.Tensor:
    pred_half = pred_size * 0.5
    ctx_half = ctx_size * 0.5
    pred_min = pred_center.unsqueeze(1) - pred_half.unsqueeze(1)
    pred_max = pred_center.unsqueeze(1) + pred_half.unsqueeze(1)
    ctx_min = ctx_center - ctx_half
    ctx_max = ctx_center + ctx_half
    inter_min = torch.maximum(pred_min, ctx_min)
    inter_max = torch.minimum(pred_max, ctx_max)
    inter = torch.clamp(inter_max - inter_min, min=0.0)
    return inter[..., 0] * inter[..., 1]


def boundary_loss(pred_x0: torch.Tensor, target_size: torch.Tensor) -> torch.Tensor:
    half = target_size * 0.5
    low = torch.clamp((-1.0) - (pred_x0 - half), min=0.0)
    high = torch.clamp((pred_x0 + half) - 1.0, min=0.0)
    return (low + high).mean()


def collision_loss(
    pred_x0: torch.Tensor,
    target_size: torch.Tensor,
    context_pos: torch.Tensor,
    context_size: torch.Tensor,
    context_mask: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    inter = _pairwise_intersection(pred_x0, target_size, context_pos, context_size)
    B, N = inter.shape
    target_mask = torch.zeros((B, N), dtype=torch.bool, device=inter.device)
    target_mask.scatter_(1, target_index.view(B, 1), True)
    valid = (context_mask > 0.5) & (~target_mask)
    inter = inter * valid.float()
    denom = valid.float().sum().clamp_min(1.0)
    return inter.sum() / denom


def run_epoch(
    model: RepairDiffusionNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    schedule: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    sums = {
        "loss": 0.0,
        "eps": 0.0,
        "x0": 0.0,
        "boundary": 0.0,
        "collision": 0.0,
        "count": 0.0,
    }

    for batch in loader:
        batch = move_batch(batch, device)
        B = batch.x0_target.shape[0]
        t = torch.randint(0, cfg.T, (B,), device=device)
        noise = torch.randn_like(batch.x0_target)
        x_t = q_sample(batch.x0_target, noise, t, schedule)

        eps_pred = model(
            x_t=x_t,
            t=t,
            context_pos=batch.context_pos,
            context_size=batch.context_size,
            context_cat=batch.context_cat,
            context_mask=batch.context_mask,
            target_index=batch.target_index,
            target_cat=batch.target_cat,
            target_size=batch.target_size,
            corruption_type=batch.corruption_type,
            room_h_world=batch.room_h_world,
        )
        pred_x0 = predict_x0_from_eps(x_t, eps_pred, t, schedule)

        loss_eps = F.mse_loss(eps_pred, noise)
        loss_x0 = F.smooth_l1_loss(pred_x0, batch.x0_target)
        loss_boundary = boundary_loss(pred_x0, batch.target_size)
        loss_collision = collision_loss(
            pred_x0=pred_x0,
            target_size=batch.target_size,
            context_pos=batch.context_pos,
            context_size=batch.context_size,
            context_mask=batch.context_mask,
            target_index=batch.target_index,
        )
        loss = (
            cfg.w_eps * loss_eps +
            cfg.w_x0 * loss_x0 +
            cfg.w_boundary * loss_boundary +
            cfg.w_collision * loss_collision
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        sums["loss"] += float(loss.detach().cpu()) * B
        sums["eps"] += float(loss_eps.detach().cpu()) * B
        sums["x0"] += float(loss_x0.detach().cpu()) * B
        sums["boundary"] += float(loss_boundary.detach().cpu()) * B
        sums["collision"] += float(loss_collision.detach().cpu()) * B
        sums["count"] += float(B)

    denom = max(sums["count"], 1.0)
    return {
        "loss": sums["loss"] / denom,
        "eps": sums["eps"] / denom,
        "x0": sums["x0"] / denom,
        "boundary": sums["boundary"] / denom,
        "collision": sums["collision"] / denom,
    }


def infer_num_categories(npz_path: str) -> int:
    z = np.load(npz_path)
    cat_id = z["cat_id"].astype(np.int64)
    return int(cat_id.max()) + 1


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: RepairDiffusionNet,
    cfg: TrainConfig,
    num_categories: int,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": asdict(cfg),
            "num_categories": int(num_categories),
            "best_val_loss": float(best_val),
            "task": "single_object_repair_diffusion",
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train diffusion model for single-object layout repair")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-samples-per-scene", type=int, default=8)
    ap.add_argument("--val-samples-per-scene", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--beta-start", type=float, default=1e-4)
    ap.add_argument("--beta-end", type=float, default=2e-2)
    ap.add_argument("--w-eps", type=float, default=1.0)
    ap.add_argument("--w-x0", type=float, default=0.5)
    ap.add_argument("--w-boundary", type=float, default=0.2)
    ap.add_argument("--w-collision", type=float, default=0.3)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = ap.parse_args()

    cfg = TrainConfig(
        npz=str(args.npz),
        splits=str(args.splits),
        out_dir=str(args.out_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        train_samples_per_scene=int(args.train_samples_per_scene),
        val_samples_per_scene=int(args.val_samples_per_scene),
        dim=int(args.dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        T=int(args.T),
        beta_start=float(args.beta_start),
        beta_end=float(args.beta_end),
        w_eps=float(args.w_eps),
        w_x0=float(args.w_x0),
        w_boundary=float(args.w_boundary),
        w_collision=float(args.w_collision),
        device=str(args.device),
    )

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    train_ds = FrontRepairDataset(
        npz_path=cfg.npz,
        splits_path=cfg.splits,
        split="train",
        samples_per_scene=cfg.train_samples_per_scene,
        seed=cfg.seed,
    )
    val_ds = FrontRepairDataset(
        npz_path=cfg.npz,
        splits_path=cfg.splits,
        split="val",
        samples_per_scene=cfg.val_samples_per_scene,
        seed=cfg.seed + 1000,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_repair,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_repair,
    )

    num_categories = infer_num_categories(cfg.npz)
    model = RepairDiffusionNet(
        num_categories=num_categories,
        dim=cfg.dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = DiffusionSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end).build(device)

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", asdict(cfg))

    best_val = math.inf
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_epoch(model, train_dl, optimizer, schedule, cfg, device)
        with torch.no_grad():
            val_stats = run_epoch(model, val_dl, None, schedule, cfg, device)

        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        save_json(out_dir / "history.json", {"history": history})

        print(
            f"[repair_diffusion] epoch={epoch:03d} "
            f"train_loss={train_stats['loss']:.6f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"val_x0={val_stats['x0']:.6f} "
            f"val_col={val_stats['collision']:.6f} "
            f"val_bnd={val_stats['boundary']:.6f}"
        )

        save_checkpoint(out_dir / "last.pt", model, cfg, num_categories, best_val)
        if val_stats["loss"] < best_val:
            best_val = float(val_stats["loss"])
            save_checkpoint(out_dir / "best.pt", model, cfg, num_categories, best_val)

    print(f"[repair_diffusion] done best_val={best_val:.6f} out={out_dir}")


if __name__ == "__main__":
    main()
