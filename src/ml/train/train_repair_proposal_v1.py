#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.ml.data_py.repair_proposal_dataset_v1 import (
        ProposalBatch,
        ProposalVocabs,
        RepairProposalDatasetV1,
        as_str,
        collate_proposal,
        encode_sample,
        load_json,
        load_sample_rows,
        model_pose_to_world,
        reconstruct_corrupted_scene,
        build_vocabs,
        room_json_from_scene,
    )
    from src.ml.models.repair_proposal_v1 import RepairProposalNetV1
    from src.ml.data_py.build_repair_corruptions_v1 import (
        aabb_intersection_metrics,
        compute_metrics,
        copy_scene_with_target,
        update_aabb_for_placement,
    )
except ModuleNotFoundError:
    from repair_proposal_dataset_v1 import (  # type: ignore
        ProposalBatch,
        ProposalVocabs,
        RepairProposalDatasetV1,
        as_str,
        collate_proposal,
        encode_sample,
        load_json,
        load_sample_rows,
        model_pose_to_world,
        reconstruct_corrupted_scene,
        build_vocabs,
        room_json_from_scene,
    )
    from repair_proposal_v1 import RepairProposalNetV1  # type: ignore
    from build_repair_corruptions_v1 import aabb_intersection_metrics, compute_metrics, copy_scene_with_target, update_aabb_for_placement  # type: ignore


@dataclass
class TrainConfig:
    samples_jsonl: str
    out_dir: str
    epochs: int = 80
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    dim: int = 192
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    w_pos: float = 1.0
    w_yaw: float = 0.5
    w_boundary: float = 0.2
    w_collision: float = 0.2
    save_every: int = 5
    seed: int = 42
    device: str = "auto"
    resume: str = ""
    limit_train: int = 0
    limit_val: int = 0
    limit_test: int = 0


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train direct ML proposal model for single-object repair")
    ap.add_argument("--samples-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--w-pos", type=float, default=1.0)
    ap.add_argument("--w-yaw", type=float, default=0.5)
    ap.add_argument("--w-boundary", type=float, default=0.2)
    ap.add_argument("--w-collision", type=float, default=0.2)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default="")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    return TrainConfig(**vars(ap.parse_args()))


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
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA недоступна")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS недоступна")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: ProposalBatch, device: torch.device) -> ProposalBatch:
    return ProposalBatch(
        clean_pose=batch.clean_pose.to(device),
        corrupted_pose=batch.corrupted_pose.to(device),
        context_pos=batch.context_pos.to(device),
        context_size=batch.context_size.to(device),
        context_cat=batch.context_cat.to(device),
        context_mask=batch.context_mask.to(device),
        target_index=batch.target_index.to(device),
        target_cat=batch.target_cat.to(device),
        target_size=batch.target_size.to(device),
        corruption_type=batch.corruption_type.to(device),
        room_type=batch.room_type.to(device),
        room_scale=batch.room_scale.to(device),
        corrupted_flags=batch.corrupted_flags.to(device),
    )


def boundary_loss(pred_pose: torch.Tensor, target_size: torch.Tensor, room_scale: torch.Tensor) -> torch.Tensor:
    target_size_norm = target_size[:, :2] / torch.clamp(room_scale[:, :2], min=1e-6)
    half = target_size_norm * 0.5
    low = torch.clamp((-1.0) - (pred_pose[:, :2] - half), min=0.0)
    high = torch.clamp((pred_pose[:, :2] + half) - 1.0, min=0.0)
    return (low + high).mean()


def collision_loss(
    pred_pose: torch.Tensor,
    target_size: torch.Tensor,
    room_scale: torch.Tensor,
    context_pos: torch.Tensor,
    context_size: torch.Tensor,
    context_mask: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    target_size_norm = target_size[:, :2] / torch.clamp(room_scale[:, :2], min=1e-6)
    pred_half = target_size_norm * 0.5
    ctx_half = context_size[:, :, :2] * 0.5
    pred_min = pred_pose[:, None, :2] - pred_half[:, None, :]
    pred_max = pred_pose[:, None, :2] + pred_half[:, None, :]
    ctx_min = context_pos[:, :, :2] - ctx_half
    ctx_max = context_pos[:, :, :2] + ctx_half
    inter_min = torch.maximum(pred_min, ctx_min)
    inter_max = torch.minimum(pred_max, ctx_max)
    inter = torch.clamp(inter_max - inter_min, min=0.0)
    area = inter[..., 0] * inter[..., 1]
    bsz, nctx = area.shape
    target_mask = torch.zeros((bsz, nctx), dtype=torch.bool, device=area.device)
    target_mask.scatter_(1, target_index.view(bsz, 1), True)
    valid = (context_mask > 0.5) & (~target_mask)
    area = area * valid.float()
    return area.sum() / valid.float().sum().clamp_min(1.0)


def run_epoch(
    model: RepairProposalNetV1,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total = {"loss": 0.0, "pos": 0.0, "yaw": 0.0, "boundary": 0.0, "collision": 0.0, "count": 0.0}

    for batch in loader:
        batch = move_batch(batch, device)
        out = model(
            corrupted_pose=batch.corrupted_pose,
            context_pos=batch.context_pos,
            context_size=batch.context_size,
            context_cat=batch.context_cat,
            context_mask=batch.context_mask,
            target_index=batch.target_index,
            target_cat=batch.target_cat,
            target_size=batch.target_size,
            corruption_type=batch.corruption_type,
            room_type=batch.room_type,
            room_scale=batch.room_scale,
            corrupted_flags=batch.corrupted_flags,
        )
        pred = out.clean_pose
        clean = batch.clean_pose

        loss_pos = F.smooth_l1_loss(pred[:, :3], clean[:, :3])
        loss_yaw = F.mse_loss(pred[:, 3:5], clean[:, 3:5])
        loss_boundary = boundary_loss(pred, batch.target_size, batch.room_scale)
        loss_collision = collision_loss(
            pred_pose=pred,
            target_size=batch.target_size,
            room_scale=batch.room_scale,
            context_pos=batch.context_pos,
            context_size=batch.context_size,
            context_mask=batch.context_mask,
            target_index=batch.target_index,
        )
        loss = (
            cfg.w_pos * loss_pos
            + cfg.w_yaw * loss_yaw
            + cfg.w_boundary * loss_boundary
            + cfg.w_collision * loss_collision
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bsz = int(batch.clean_pose.shape[0])
        total["loss"] += float(loss.detach().cpu()) * bsz
        total["pos"] += float(loss_pos.detach().cpu()) * bsz
        total["yaw"] += float(loss_yaw.detach().cpu()) * bsz
        total["boundary"] += float(loss_boundary.detach().cpu()) * bsz
        total["collision"] += float(loss_collision.detach().cpu()) * bsz
        total["count"] += float(bsz)

    denom = max(total["count"], 1.0)
    return {k: round(float(v / denom), 6) for k, v in total.items() if k != "count"}


def clone_json(x: Any) -> Any:
    return json.loads(json.dumps(x))


def footprint_area(target: Dict[str, Any]) -> float:
    size = [float(v) for v in target.get("size_m", [1.0, 1.0, 1.0])]
    return max(size[0] * size[1], 1e-6)


def target_overlap_ratio(clean_target: Dict[str, Any], eval_target: Dict[str, Any]) -> float:
    inter_area_2d, _ = aabb_intersection_metrics(clean_target, eval_target)
    return float(inter_area_2d / max(footprint_area(clean_target), 1e-6))


def evaluate_end_to_end(
    model: RepairProposalNetV1,
    rows: List[dict],
    vocabs: ProposalVocabs,
    device: torch.device,
) -> Dict[str, float]:
    valid = 0
    success = 0
    improved = 0
    pos_errs: List[float] = []
    yaw_errs: List[float] = []
    quality_scores: List[float] = []
    overlap_ratios: List[float] = []
    overlap50 = 0
    overlap80 = 0
    overlap95 = 0

    model.eval()
    with torch.no_grad():
        for row in rows:
            room_ref = as_str(row.get("room_ref")).strip()
            if room_ref:
                room_json = load_json(room_ref)
            else:
                clean_scene = load_json(row["clean_scene_ref"])
                room_json = room_json_from_scene(clean_scene)
            _, corrupted_scene, clean_target = reconstruct_corrupted_scene(row)
            enc = encode_sample(row, room_json, corrupted_scene, clean_target, vocabs)
            batch = ProposalBatch(
                clean_pose=torch.from_numpy(enc["clean_pose"][None, ...]).to(device),
                corrupted_pose=torch.from_numpy(enc["corrupted_pose"][None, ...]).to(device),
                context_pos=torch.from_numpy(enc["context_pos"][None, ...]).to(device),
                context_size=torch.from_numpy(enc["context_size"][None, ...]).to(device),
                context_cat=torch.from_numpy(enc["context_cat"][None, ...]).to(device),
                context_mask=torch.from_numpy(enc["context_mask"][None, ...]).to(device),
                target_index=torch.from_numpy(np.asarray([enc["target_index"]], dtype=np.int64)).to(device),
                target_cat=torch.from_numpy(np.asarray([enc["target_cat"]], dtype=np.int64)).to(device),
                target_size=torch.from_numpy(enc["target_size"][None, ...]).to(device),
                corruption_type=torch.from_numpy(np.asarray([enc["corruption_type"]], dtype=np.int64)).to(device),
                room_type=torch.from_numpy(np.asarray([enc["room_type"]], dtype=np.int64)).to(device),
                room_scale=torch.from_numpy(enc["room_scale"][None, ...]).to(device),
                corrupted_flags=torch.from_numpy(enc["corrupted_flags"][None, ...]).to(device),
            )
            pred = model(
                corrupted_pose=batch.corrupted_pose,
                context_pos=batch.context_pos,
                context_size=batch.context_size,
                context_cat=batch.context_cat,
                context_mask=batch.context_mask,
                target_index=batch.target_index,
                target_cat=batch.target_cat,
                target_size=batch.target_size,
                corruption_type=batch.corruption_type,
                room_type=batch.room_type,
                room_scale=batch.room_scale,
                corrupted_flags=batch.corrupted_flags,
            ).clean_pose[0].detach().cpu().numpy()

            target_idx = int(enc["target_index"])
            corrupted_target = clone_json(corrupted_scene[target_idx])
            world_pos, yaw_deg, yaw_rad = model_pose_to_world(pred, room_json, fallback_z=float(corrupted_target["position_m"][2]))
            new_target = clone_json(corrupted_target)
            new_target["position_m"] = world_pos
            new_target["yaw_deg"] = float(yaw_deg)
            new_target["yaw_rad"] = float(yaw_rad)
            new_target["rotation_deg"] = int(round(float(yaw_deg))) % 360
            new_target["aabb"] = update_aabb_for_placement(new_target)

            repaired_scene = copy_scene_with_target(corrupted_scene, asdict_id(row["target_object_id"]), new_target)
            metrics = compute_metrics(
                clean_target=clean_target,
                eval_target=new_target,
                eval_scene_placements=repaired_scene,
                room_polygon=room_json["floor_polygon_xz"],
            )
            pos_errs.append(float(metrics["position_l2_m"]))
            yaw_errs.append(float(metrics["yaw_abs_error_deg"]))
            quality_scores.append(float(metrics["quality_score"]))
            overlap = target_overlap_ratio(clean_target, new_target)
            overlap_ratios.append(float(overlap))
            if overlap >= 0.50:
                overlap50 += 1
            if overlap >= 0.80:
                overlap80 += 1
            if overlap >= 0.95:
                overlap95 += 1
            if metrics["valid"]:
                valid += 1
            if metrics["valid"] and float(metrics["position_l2_m"]) <= 0.20 and float(metrics["yaw_abs_error_deg"]) <= 15.0:
                success += 1
            if float(metrics["quality_score"]) > float(row["corrupted_metrics"]["quality_score"]):
                improved += 1

    total = max(len(rows), 1)
    return {
        "samples_total": len(rows),
        "valid_rate_after_repair": round(valid / total, 6),
        "success_rate": round(success / total, 6),
        "quality_improved_rate": round(improved / total, 6),
        "target_overlap50_rate": round(overlap50 / total, 6),
        "target_overlap80_rate": round(overlap80 / total, 6),
        "target_overlap95_rate": round(overlap95 / total, 6),
        "mean_position_l2_m": round(float(np.mean(pos_errs)) if pos_errs else 0.0, 6),
        "mean_yaw_abs_error_deg": round(float(np.mean(yaw_errs)) if yaw_errs else 0.0, 6),
        "mean_quality_score": round(float(np.mean(quality_scores)) if quality_scores else 0.0, 6),
        "mean_target_overlap_ratio": round(float(np.mean(overlap_ratios)) if overlap_ratios else 0.0, 6),
    }


def asdict_id(v: str) -> str:
    return str(v)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: RepairProposalNetV1,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    vocabs: ProposalVocabs,
    epoch: int,
    best_val_success: float,
    history: List[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": asdict(cfg),
            "vocabs": {
                "category_vocab": vocabs.category_vocab,
                "corruption_vocab": vocabs.corruption_vocab,
                "room_type_vocab": vocabs.room_type_vocab,
            },
            "epoch": int(epoch),
            "best_val_success": float(best_val_success),
            "history": history,
            "task": "single_object_repair_proposal_v1",
        },
        path,
    )


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def load_partial_state(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    current = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state_dict.items():
        if key not in current or tuple(current[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible[key] = value
    missing = [k for k in current.keys() if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    return {"loaded": sorted(compatible.keys()), "skipped": sorted(skipped), "missing": sorted(missing)}


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)

    train_rows = load_sample_rows(cfg.samples_jsonl, split="train", limit=cfg.limit_train)
    val_rows = load_sample_rows(cfg.samples_jsonl, split="val", limit=cfg.limit_val)
    test_rows = load_sample_rows(cfg.samples_jsonl, split="test", limit=cfg.limit_test)
    vocabs = build_vocabs(train_rows)

    train_ds = RepairProposalDatasetV1(cfg.samples_jsonl, split="train", vocabs=vocabs, limit=cfg.limit_train)
    val_ds = RepairProposalDatasetV1(cfg.samples_jsonl, split="val", vocabs=vocabs, limit=cfg.limit_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_proposal,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_proposal,
    )

    model = RepairProposalNetV1(
        num_categories=len(vocabs.category_vocab),
        num_corruption_types=len(vocabs.corruption_vocab),
        num_room_types=len(vocabs.room_type_vocab),
        dim=cfg.dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[dict] = []
    best_val_success = -1.0
    start_epoch = 1

    if cfg.resume:
        resume_path = Path(cfg.resume).expanduser().resolve()
        obj = load_checkpoint(resume_path)
        resume_info = load_partial_state(model, obj["model_state"])
        can_resume_optimizer = not resume_info["skipped"]
        if can_resume_optimizer:
            optimizer.load_state_dict(obj["optimizer_state"])
            start_epoch = int(obj["epoch"]) + 1
            best_val_success = float(obj.get("best_val_success", -1.0))
            history = list(obj.get("history", []))
            print(f"[repair_proposal_v1] resumed from {resume_path} at epoch={start_epoch}")
        else:
            print(
                f"[repair_proposal_v1] warm-started from {resume_path}; "
                f"skipped={len(resume_info['skipped'])}, optimizer_reset=true"
            )

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, cfg, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, cfg, device)
        val_scene = evaluate_end_to_end(model, val_rows, vocabs, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_scene": val_scene,
        }
        history.append(row)
        print(
            f"[repair_proposal_v1] epoch={epoch}/{cfg.epochs} "
            f"train_loss={train_loss['loss']:.4f} val_loss={val_loss['loss']:.4f} "
            f"val_valid={val_scene['valid_rate_after_repair']:.4f} "
            f"val_success={val_scene['success_rate']:.4f} "
            f"val_overlap50={val_scene['target_overlap50_rate']:.4f} "
            f"val_overlap80={val_scene['target_overlap80_rate']:.4f} "
            f"val_overlap95={val_scene['target_overlap95_rate']:.4f}"
        )

        save_checkpoint(out_dir / "last.pt", model, optimizer, cfg, vocabs, epoch, best_val_success, history)
        if epoch % max(int(cfg.save_every), 1) == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:04d}.pt", model, optimizer, cfg, vocabs, epoch, best_val_success, history)

        if val_scene["success_rate"] > best_val_success:
            best_val_success = float(val_scene["success_rate"])
            save_checkpoint(out_dir / "best.pt", model, optimizer, cfg, vocabs, epoch, best_val_success, history)

    best_ckpt = load_checkpoint(out_dir / "best.pt")
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    final_val = evaluate_end_to_end(model, val_rows, vocabs, device)
    final_test = evaluate_end_to_end(model, test_rows, vocabs, device)
    metrics = {
        "device": str(device),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "best_val_success": round(best_val_success, 6),
        "final_val": final_val,
        "final_test": final_test,
    }
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "config.json", asdict(cfg))
    save_json(out_dir / "history.json", {"epochs": history})

    print(f"[repair_proposal_v1] best_val_success={best_val_success:.4f}")
    print(f"[repair_proposal_v1] final_val_success={final_val['success_rate']:.4f}")
    print(f"[repair_proposal_v1] final_test_success={final_test['success_rate']:.4f}")
    print(f"[repair_proposal_v1] final_val_overlap95={final_val['target_overlap95_rate']:.4f}")
    print(f"[repair_proposal_v1] final_test_overlap95={final_test['target_overlap95_rate']:.4f}")
    print(f"[repair_proposal_v1] wrote checkpoint={out_dir / 'best.pt'}")
    print(f"[repair_proposal_v1] wrote metrics={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
