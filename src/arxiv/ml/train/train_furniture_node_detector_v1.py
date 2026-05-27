#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from src.ml.data_py.corrupted_object_selector_dataset_v1 import (
        CorruptedObjectSelectorDatasetV1,
        SelectorBatch,
        SelectorVocabs,
        build_vocabs,
        collate_selector,
        load_selector_rows,
    )
    from src.ml.models.corrupted_object_selector_v1 import CorruptedObjectSelectorV1
except ModuleNotFoundError:
    from corrupted_object_selector_dataset_v1 import (  # type: ignore
        CorruptedObjectSelectorDatasetV1,
        SelectorBatch,
        SelectorVocabs,
        build_vocabs,
        collate_selector,
        load_selector_rows,
    )
    from corrupted_object_selector_v1 import CorruptedObjectSelectorV1  # type: ignore


@dataclass
class TrainConfig:
    detector_jsonl: str
    out_dir: str
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 192
    emb_dim: int = 24
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    ce_weight: float = 1.0
    bce_weight: float = 0.5
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train general furniture node detector from grouped scene samples")
    ap.add_argument("--detector-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--emb-dim", type=int, default=24)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--bce-weight", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
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
        raise RuntimeError("CUDA not available")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS not available")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: SelectorBatch, device: torch.device) -> SelectorBatch:
    return SelectorBatch(
        features=batch.features.to(device),
        category=batch.category.to(device),
        super_category=batch.super_category.to(device),
        mount_type=batch.mount_type.to(device),
        room_type=batch.room_type.to(device),
        mask=batch.mask.to(device),
        target_index=batch.target_index.to(device),
    )


def round6(v: float) -> float:
    return round(float(v), 6)


def detector_loss(logits: torch.Tensor, target_index: torch.Tensor, mask: torch.Tensor, ce_weight: float, bce_weight: float) -> Dict[str, torch.Tensor]:
    ce_loss = F.cross_entropy(logits, target_index)

    target_matrix = torch.zeros_like(logits)
    target_matrix.scatter_(1, target_index[:, None], 1.0)
    valid = mask > 0.5
    valid_logits = logits[valid]
    valid_targets = target_matrix[valid]
    pos_count = max(float(valid_targets.sum().item()), 1.0)
    neg_count = max(float(valid_targets.numel()) - pos_count, 1.0)
    pos_weight = neg_count / pos_count

    bce_raw = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction="none")
    bce_weights = torch.where(
        valid_targets > 0.5,
        torch.full_like(valid_targets, pos_weight),
        torch.ones_like(valid_targets),
    )
    bce_loss = (bce_raw * bce_weights).sum() / bce_weights.sum().clamp_min(1.0)
    total = float(ce_weight) * ce_loss + float(bce_weight) * bce_loss
    return {
        "loss": total,
        "ce_loss": ce_loss.detach(),
        "bce_loss": bce_loss.detach(),
    }


def run_epoch(
    model: CorruptedObjectSelectorV1,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    ce_weight: float,
    bce_weight: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ce = 0.0
    total_bce = 0.0
    total_count = 0
    top1 = 0
    top3 = 0
    top5 = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(
            features=batch.features,
            category=batch.category,
            super_category=batch.super_category,
            mount_type=batch.mount_type,
            room_type=batch.room_type,
            mask=batch.mask,
        )
        losses = detector_loss(
            logits=out.logits,
            target_index=batch.target_index,
            mask=batch.mask,
            ce_weight=ce_weight,
            bce_weight=bce_weight,
        )
        loss = losses["loss"]
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = int(batch.features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_ce += float(losses["ce_loss"].item()) * batch_size
        total_bce += float(losses["bce_loss"].item()) * batch_size
        total_count += batch_size
        order = torch.argsort(out.logits, dim=-1, descending=True)
        top1 += int((order[:, 0] == batch.target_index).sum().item())
        top3 += int(((order[:, :3] == batch.target_index[:, None]).any(dim=1)).sum().item())
        top5 += int(((order[:, :5] == batch.target_index[:, None]).any(dim=1)).sum().item())
    return {
        "loss": total_loss / max(total_count, 1),
        "ce_loss": total_ce / max(total_count, 1),
        "bce_loss": total_bce / max(total_count, 1),
        "top1": top1 / max(total_count, 1),
        "top3": top3 / max(total_count, 1),
        "top5": top5 / max(total_count, 1),
    }


def save_checkpoint(
    path: Path,
    model: CorruptedObjectSelectorV1,
    cfg: TrainConfig,
    vocabs: SelectorVocabs,
    metrics: Dict[str, float],
    feature_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": asdict(cfg),
            "feature_dim": int(feature_dim),
            "vocabs": {
                "category_vocab": dict(vocabs.category_vocab),
                "super_vocab": dict(vocabs.super_vocab),
                "mount_vocab": dict(vocabs.mount_vocab),
                "room_type_vocab": dict(vocabs.room_type_vocab),
            },
            "model_state": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def copy_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    all_rows = []
    for split in ("train", "val", "test"):
        all_rows.extend(load_selector_rows(cfg.detector_jsonl, split))
    vocabs = build_vocabs(all_rows)

    train_ds = CorruptedObjectSelectorDatasetV1(cfg.detector_jsonl, "train", vocabs=vocabs)
    val_ds = CorruptedObjectSelectorDatasetV1(cfg.detector_jsonl, "val", vocabs=vocabs)
    test_ds = CorruptedObjectSelectorDatasetV1(cfg.detector_jsonl, "test", vocabs=vocabs)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_selector)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_selector)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_selector)

    model = CorruptedObjectSelectorV1(
        feature_dim=train_ds.feature_dim,
        num_categories=len(vocabs.category_vocab),
        num_supers=len(vocabs.super_vocab),
        num_mount_types=len(vocabs.mount_vocab),
        num_room_types=len(vocabs.room_type_vocab),
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_top1 = -1.0
    best_val_top3 = -1.0
    best_top1_metrics: Dict[str, float] = {}
    best_top3_metrics: Dict[str, float] = {}

    for epoch in range(1, int(cfg.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, ce_weight=cfg.ce_weight, bce_weight=cfg.bce_weight)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, ce_weight=cfg.ce_weight, bce_weight=cfg.bce_weight)
        row = {
            "epoch": epoch,
            "train_loss": round6(train_metrics["loss"]),
            "train_ce_loss": round6(train_metrics["ce_loss"]),
            "train_bce_loss": round6(train_metrics["bce_loss"]),
            "train_top1": round6(train_metrics["top1"]),
            "train_top3": round6(train_metrics["top3"]),
            "train_top5": round6(train_metrics["top5"]),
            "val_loss": round6(val_metrics["loss"]),
            "val_ce_loss": round6(val_metrics["ce_loss"]),
            "val_bce_loss": round6(val_metrics["bce_loss"]),
            "val_top1": round6(val_metrics["top1"]),
            "val_top3": round6(val_metrics["top3"]),
            "val_top5": round6(val_metrics["top5"]),
        }
        history.append(row)
        print(
            f"[furniture_node_detector_v1] epoch={epoch}/{cfg.epochs} "
            f"train_loss={row['train_loss']:.4f} train_top1={row['train_top1']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_top1={row['val_top1']:.4f} "
            f"val_top3={row['val_top3']:.4f} val_top5={row['val_top5']:.4f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = float(val_metrics["top1"])
            with torch.no_grad():
                test_metrics = run_epoch(model, test_loader, None, device, ce_weight=cfg.ce_weight, bce_weight=cfg.bce_weight)
            best_top1_metrics = {
                "best_val_top1": round6(best_val_top1),
                "best_val_top3_at_top1": round6(val_metrics["top3"]),
                "best_val_top5_at_top1": round6(val_metrics["top5"]),
                "best_test_top1_at_top1": round6(test_metrics["top1"]),
                "best_test_top3_at_top1": round6(test_metrics["top3"]),
                "best_test_top5_at_top1": round6(test_metrics["top5"]),
            }
            save_checkpoint(out_dir / "best_top1.pt", model, cfg, vocabs, best_top1_metrics, feature_dim=train_ds.feature_dim)

        top3_tie_break = val_metrics["top1"]
        if (val_metrics["top3"] > best_val_top3) or (
            abs(float(val_metrics["top3"]) - best_val_top3) <= 1e-9 and top3_tie_break > best_top1_metrics.get("best_val_top1", -1.0)
        ):
            best_val_top3 = float(val_metrics["top3"])
            with torch.no_grad():
                test_metrics = run_epoch(model, test_loader, None, device, ce_weight=cfg.ce_weight, bce_weight=cfg.bce_weight)
            best_top3_metrics = {
                "best_val_top1_at_top3": round6(val_metrics["top1"]),
                "best_val_top3": round6(best_val_top3),
                "best_val_top5_at_top3": round6(val_metrics["top5"]),
                "best_test_top1_at_top3": round6(test_metrics["top1"]),
                "best_test_top3_at_top3": round6(test_metrics["top3"]),
                "best_test_top5_at_top3": round6(test_metrics["top5"]),
            }
            save_checkpoint(out_dir / "best_top3.pt", model, cfg, vocabs, best_top3_metrics, feature_dim=train_ds.feature_dim)
            copy_checkpoint(out_dir / "best_top3.pt", out_dir / "best.pt")

    metrics = {
        "history": history,
        **best_top1_metrics,
        **best_top3_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[furniture_node_detector_v1] best_val_top1={best_top1_metrics.get('best_val_top1', -1.0):.4f}")
    print(f"[furniture_node_detector_v1] best_test_top1_at_top1={best_top1_metrics.get('best_test_top1_at_top1', -1.0):.4f}")
    print(f"[furniture_node_detector_v1] best_val_top3={best_top3_metrics.get('best_val_top3', -1.0):.4f}")
    print(f"[furniture_node_detector_v1] best_test_top3_at_top3={best_top3_metrics.get('best_test_top3_at_top3', -1.0):.4f}")
    print(f"[furniture_node_detector_v1] wrote checkpoint={out_dir / 'best.pt'}")
    print(f"[furniture_node_detector_v1] wrote metrics={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
