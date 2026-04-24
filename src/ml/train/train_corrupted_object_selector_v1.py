#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
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
    selector_jsonl: str
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
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train corrupted-object selector from grouped repair samples")
    ap.add_argument("--selector-jsonl", required=True)
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


def run_epoch(
    model: CorruptedObjectSelectorV1,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    top1 = 0
    top3 = 0
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
        loss = F.cross_entropy(out.logits, batch.target_index)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * int(batch.features.shape[0])
        total_count += int(batch.features.shape[0])
        order = torch.argsort(out.logits, dim=-1, descending=True)
        top1 += int((order[:, 0] == batch.target_index).sum().item())
        top3 += int(((order[:, :3] == batch.target_index[:, None]).any(dim=1)).sum().item())
    return {
        "loss": total_loss / max(total_count, 1),
        "top1": top1 / max(total_count, 1),
        "top3": top3 / max(total_count, 1),
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


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)

    all_rows = []
    for split in ("train", "val", "test"):
        all_rows.extend(load_selector_rows(cfg.selector_jsonl, split))
    vocabs = build_vocabs(all_rows)

    train_ds = CorruptedObjectSelectorDatasetV1(cfg.selector_jsonl, "train", vocabs=vocabs)
    val_ds = CorruptedObjectSelectorDatasetV1(cfg.selector_jsonl, "val", vocabs=vocabs)
    test_ds = CorruptedObjectSelectorDatasetV1(cfg.selector_jsonl, "test", vocabs=vocabs)

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
    best_metrics = {}

    for epoch in range(1, int(cfg.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device)
        row = {
            "epoch": epoch,
            "train_loss": round6(train_metrics["loss"]),
            "train_top1": round6(train_metrics["top1"]),
            "train_top3": round6(train_metrics["top3"]),
            "val_loss": round6(val_metrics["loss"]),
            "val_top1": round6(val_metrics["top1"]),
            "val_top3": round6(val_metrics["top3"]),
        }
        history.append(row)
        print(
            f"[corrupted_object_selector_v1] epoch={epoch}/{cfg.epochs} "
            f"train_loss={row['train_loss']:.4f} train_top1={row['train_top1']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_top1={row['val_top1']:.4f} val_top3={row['val_top3']:.4f}"
        )
        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = float(val_metrics["top1"])
            with torch.no_grad():
                test_metrics = run_epoch(model, test_loader, None, device)
            best_metrics = {
                "best_val_top1": round6(best_val_top1),
                "best_val_top3": round6(val_metrics["top3"]),
                "best_test_top1": round6(test_metrics["top1"]),
                "best_test_top3": round6(test_metrics["top3"]),
            }
            save_checkpoint(out_dir / "best.pt", model, cfg, vocabs, best_metrics, feature_dim=train_ds.feature_dim)

    metrics = {
        "history": history,
        **best_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[corrupted_object_selector_v1] best_val_top1={best_metrics.get('best_val_top1', 0.0):.4f}")
    print(f"[corrupted_object_selector_v1] best_test_top1={best_metrics.get('best_test_top1', 0.0):.4f}")
    print(f"[corrupted_object_selector_v1] wrote checkpoint={out_dir / 'best.pt'}")
    print(f"[corrupted_object_selector_v1] wrote metrics={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
