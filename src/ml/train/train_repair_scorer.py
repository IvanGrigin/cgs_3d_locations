#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
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
    from src.ml.data_py.repair_ranking_dataset import (
        RankingBatch,
        RankingVocabs,
        RepairRankingDataset,
        build_feature_vector,
        build_vocabs,
        collate_ranking,
        load_ranking_rows,
    )
    from src.ml.models.repair_scorer import RepairScorer
except ModuleNotFoundError:
    from repair_ranking_dataset import (
        RankingBatch,
        RankingVocabs,
        RepairRankingDataset,
        build_feature_vector,
        build_vocabs,
        collate_ranking,
        load_ranking_rows,
    )
    from repair_scorer import RepairScorer


@dataclass
class TrainConfig:
    ranking_jsonl: str
    out_dir: str
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    emb_dim: int = 16
    num_layers: int = 3
    dropout: float = 0.1
    w_score: float = 1.0
    w_best: float = 0.5
    rank_best_weight: float = 0.25
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train candidate scorer for single-object repair ranking")
    ap.add_argument("--ranking-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--w-score", type=float, default=1.0)
    ap.add_argument("--w-best", type=float, default=0.5)
    ap.add_argument("--rank-best-weight", type=float, default=0.25)
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


def move_batch(batch: RankingBatch, device: torch.device) -> RankingBatch:
    return RankingBatch(
        features=batch.features.to(device),
        target_cat=batch.target_cat.to(device),
        target_super=batch.target_super.to(device),
        corruption_type=batch.corruption_type.to(device),
        room_type=batch.room_type.to(device),
        target_score=batch.target_score.to(device),
        target_best=batch.target_best.to(device),
    )


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(np.square(err)))
    rmse = float(math.sqrt(max(mse, 0.0)))
    denom = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 0.0 if denom <= 1e-12 else float(1.0 - np.sum(np.square(err)) / denom)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, len(y_true))

    roc_auc = 0.0
    avg_precision = 0.0
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=np.int64)
        ranks[order] = np.arange(1, len(probs) + 1)
        pos_ranks = ranks[y_true == 1]
        roc_auc = float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

        desc = np.argsort(-probs)
        y_sorted = y_true[desc]
        tp_running = 0
        fp_running = 0
        precisions: List[float] = []
        recalls: List[float] = []
        for label in y_sorted:
            if label == 1:
                tp_running += 1
            else:
                fp_running += 1
            precisions.append(safe_div(tp_running, tp_running + fp_running))
            recalls.append(safe_div(tp_running, n_pos))
        prev_recall = 0.0
        ap = 0.0
        for p, r in zip(precisions, recalls):
            ap += p * max(r - prev_recall, 0.0)
            prev_recall = r
        avg_precision = float(ap)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "positive_rate": safe_div(n_pos, len(y_true)),
    }


def round6(v: float) -> float:
    return round(float(v), 6)


def row_candidate_tensors(row: dict, vocabs: RankingVocabs, device: torch.device) -> Dict[str, torch.Tensor]:
    features = torch.from_numpy(np.stack([build_feature_vector(c) for c in row["candidates"]], axis=0)).to(device)
    target_cat = torch.full(
        (features.shape[0],),
        int(vocabs.target_cat_vocab.get(str(row["target_category"]), 0)),
        dtype=torch.long,
        device=device,
    )
    target_super = torch.full(
        (features.shape[0],),
        int(vocabs.target_super_vocab.get(str(row["target_super_category"]), 0)),
        dtype=torch.long,
        device=device,
    )
    corruption_type = torch.full(
        (features.shape[0],),
        int(vocabs.corruption_vocab.get(str(row["corruption_type"]), 0)),
        dtype=torch.long,
        device=device,
    )
    room_type = torch.full(
        (features.shape[0],),
        int(vocabs.room_type_vocab.get(str(row["room_type"]), 0)),
        dtype=torch.long,
        device=device,
    )
    return {
        "features": features,
        "target_cat": target_cat,
        "target_super": target_super,
        "corruption_type": corruption_type,
        "room_type": room_type,
    }


def evaluate_group_ranking(
    model: RepairScorer,
    rows: List[dict],
    vocabs: RankingVocabs,
    device: torch.device,
    rank_best_weight: float,
) -> Dict[str, float]:
    top1_hits = 0
    top3_hits = 0
    top1_valid = 0
    mean_selected_quality = 0.0
    mean_oracle_quality = 0.0
    mean_quality_gap = 0.0

    with torch.no_grad():
        for row in rows:
            tensors = row_candidate_tensors(row, vocabs, device)
            out = model(**tensors)
            scores = out.quality_score + float(rank_best_weight) * torch.sigmoid(out.best_logit)
            order = torch.argsort(scores, descending=True)
            valid_indices = {i for i, cand in enumerate(row["candidates"]) if bool(cand["label"]["is_valid"])}
            pred_idx = None
            for idx_tensor in order:
                idx = int(idx_tensor.item())
                if idx in valid_indices:
                    pred_idx = idx
                    break
            if pred_idx is None:
                pred_idx = int(order[0].item())
            best_idx = int(row["best_candidate_index"])

            candidate_labels = [float(c["label"]["quality_score"]) for c in row["candidates"]]
            selected_quality = candidate_labels[pred_idx]
            oracle_quality = max(candidate_labels)
            selected_valid = bool(row["candidates"][pred_idx]["label"]["is_valid"])

            top1_hits += int(pred_idx == best_idx)
            top3_hits += int(best_idx in {int(v.item()) for v in order[:3]})
            top1_valid += int(selected_valid)
            mean_selected_quality += selected_quality
            mean_oracle_quality += oracle_quality
            mean_quality_gap += (oracle_quality - selected_quality)

    n = max(len(rows), 1)
    return {
        "top1_hit_rate": round6(top1_hits / n),
        "top3_hit_rate": round6(top3_hits / n),
        "top1_valid_rate": round6(top1_valid / n),
        "mean_selected_quality": round6(mean_selected_quality / n),
        "mean_oracle_quality": round6(mean_oracle_quality / n),
        "mean_quality_gap": round6(mean_quality_gap / n),
    }


def run_epoch(
    model: RepairScorer,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: TrainConfig,
    pos_weight: torch.Tensor,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_score_loss = 0.0
    total_best_loss = 0.0
    total_count = 0
    y_score_true: List[np.ndarray] = []
    y_score_pred: List[np.ndarray] = []
    y_best_true: List[np.ndarray] = []
    y_best_logits: List[np.ndarray] = []

    for batch in loader:
        batch = move_batch(batch, device)
        out = model(
            features=batch.features,
            target_cat=batch.target_cat,
            target_super=batch.target_super,
            corruption_type=batch.corruption_type,
            room_type=batch.room_type,
        )
        loss_score = F.smooth_l1_loss(out.quality_score, batch.target_score)
        loss_best = F.binary_cross_entropy_with_logits(
            out.best_logit,
            batch.target_best,
            pos_weight=pos_weight,
        )
        loss = float(cfg.w_score) * loss_score + float(cfg.w_best) * loss_best

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = int(batch.features.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_score_loss += float(loss_score.detach().cpu()) * bs
        total_best_loss += float(loss_best.detach().cpu()) * bs
        total_count += bs
        y_score_true.append(batch.target_score.detach().cpu().numpy())
        y_score_pred.append(out.quality_score.detach().cpu().numpy())
        y_best_true.append(batch.target_best.detach().cpu().numpy())
        y_best_logits.append(out.best_logit.detach().cpu().numpy())

    y_score_true_np = np.concatenate(y_score_true, axis=0)
    y_score_pred_np = np.concatenate(y_score_pred, axis=0)
    y_best_true_np = np.concatenate(y_best_true, axis=0)
    y_best_logits_np = np.concatenate(y_best_logits, axis=0)

    denom = max(total_count, 1)
    metrics = {
        "loss": round6(total_loss / denom),
        "score_loss": round6(total_score_loss / denom),
        "best_loss": round6(total_best_loss / denom),
    }
    metrics.update({f"reg_{k}": round6(v) for k, v in regression_metrics(y_score_true_np, y_score_pred_np).items()})
    metrics.update({f"cls_{k}": round6(v) for k, v in classification_metrics(y_best_true_np, y_best_logits_np).items()})
    return metrics


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: RepairScorer,
    cfg: TrainConfig,
    vocabs: RankingVocabs,
    feature_dim: int,
    best_val_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": asdict(cfg),
            "vocabs": {
                "target_cat_vocab": vocabs.target_cat_vocab,
                "target_super_vocab": vocabs.target_super_vocab,
                "corruption_vocab": vocabs.corruption_vocab,
                "room_type_vocab": vocabs.room_type_vocab,
            },
            "feature_dim": int(feature_dim),
            "best_val_top1": float(best_val_metric),
            "task": "single_object_repair_scorer",
        },
        path,
    )


def maybe_load_rows(jsonl_path: str, split: str) -> List[dict]:
    try:
        return load_ranking_rows(jsonl_path, split)
    except RuntimeError:
        return []


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)

    train_rows = load_ranking_rows(cfg.ranking_jsonl, "train")
    vocabs = build_vocabs(train_rows)
    train_ds = RepairRankingDataset(cfg.ranking_jsonl, split="train", seed=cfg.seed, vocabs=vocabs)
    val_ds = RepairRankingDataset(cfg.ranking_jsonl, split="val", seed=cfg.seed, vocabs=vocabs)
    val_rows = load_ranking_rows(cfg.ranking_jsonl, "val")
    test_rows = maybe_load_rows(cfg.ranking_jsonl, "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_ranking,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_ranking,
    )

    positives = sum(int(item["target_best"] > 0.5) for item in train_ds.items)
    negatives = max(len(train_ds.items) - positives, 1)
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device).squeeze(0)

    model = RepairScorer(
        feature_dim=train_ds.feature_dim,
        num_target_categories=len(vocabs.target_cat_vocab),
        num_target_supers=len(vocabs.target_super_vocab),
        num_corruption_types=len(vocabs.corruption_vocab),
        num_room_types=len(vocabs.room_type_vocab),
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[dict] = []
    best_val_top1 = -1.0
    best_ckpt = out_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg, pos_weight)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device, cfg, pos_weight)
            group_metrics = evaluate_group_ranking(
                model=model,
                rows=val_rows,
                vocabs=vocabs,
                device=device,
                rank_best_weight=cfg.rank_best_weight,
            )

        merged = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "val_group": group_metrics,
        }
        history.append(merged)
        print(
            f"[repair_scorer] epoch={epoch}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={group_metrics['top1_hit_rate']:.4f} "
            f"val_top1_valid={group_metrics['top1_valid_rate']:.4f}"
        )

        if group_metrics["top1_hit_rate"] > best_val_top1:
            best_val_top1 = float(group_metrics["top1_hit_rate"])
            save_checkpoint(best_ckpt, model, cfg, vocabs, train_ds.feature_dim, best_val_top1)

    history_path = out_dir / "history.json"
    save_json(history_path, {"epochs": history})

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    with torch.no_grad():
        final_val = evaluate_group_ranking(
            model=model,
            rows=val_rows,
            vocabs=vocabs,
            device=device,
            rank_best_weight=cfg.rank_best_weight,
        )
        final_val_candidate = run_epoch(model, val_loader, None, device, cfg, pos_weight)
        if test_rows:
            final_test = evaluate_group_ranking(
                model=model,
                rows=test_rows,
                vocabs=vocabs,
                device=device,
                rank_best_weight=cfg.rank_best_weight,
            )
            test_ds = RepairRankingDataset(cfg.ranking_jsonl, split="test", seed=cfg.seed, vocabs=vocabs)
            test_loader = DataLoader(
                test_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                collate_fn=collate_ranking,
            )
            final_test_candidate = run_epoch(model, test_loader, None, device, cfg, pos_weight)
            test_items = len(test_ds)
        else:
            final_test = {}
            final_test_candidate = {}
            test_items = 0

    metrics = {
        "device": str(device),
        "feature_dim": int(train_ds.feature_dim),
        "train_items": len(train_ds),
        "val_items": len(val_ds),
        "test_items": test_items,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "best_val_top1": round6(best_val_top1),
        "val_candidate": final_val_candidate,
        "val_group": final_val,
        "test_candidate": final_test_candidate,
        "test_group": final_test,
    }
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "config.json", asdict(cfg))

    print(f"[repair_scorer] best_val_top1={metrics['best_val_top1']:.4f}")
    test_top1 = final_test.get("top1_hit_rate")
    if test_top1 is None:
        print(f"[repair_scorer] val_top1={final_val['top1_hit_rate']:.4f} test_top1=n/a")
    else:
        print(f"[repair_scorer] val_top1={final_val['top1_hit_rate']:.4f} test_top1={test_top1:.4f}")
    print(f"[repair_scorer] wrote checkpoint={best_ckpt}")
    print(f"[repair_scorer] wrote metrics={out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
