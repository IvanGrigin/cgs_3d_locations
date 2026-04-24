#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot repair proposal training history")
    ap.add_argument("--history-json", required=True)
    ap.add_argument("--out-png", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.history_json)
    rows: List[Dict[str, Any]] = list(payload.get("epochs") or [])
    if not rows:
        raise RuntimeError(f"No epochs in {args.history_json}")

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]["loss"]) for r in rows]
    val_loss = [float(r["val_loss"]["loss"]) for r in rows]
    val_valid = [float(r["val_scene"]["valid_rate_after_repair"]) for r in rows]
    val_success = [float(r["val_scene"]["success_rate"]) for r in rows]
    val_quality = [float(r["val_scene"]["quality_improved_rate"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(epochs, train_loss, label="train_loss", linewidth=2.0)
    axes[0].plot(epochs, val_loss, label="val_loss", linewidth=2.0)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Repair Proposal Training")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_valid, label="val_valid_rate", linewidth=2.0)
    axes[1].plot(epochs, val_success, label="val_success_rate", linewidth=2.0)
    axes[1].plot(epochs, val_quality, label="val_quality_improved_rate", linewidth=2.0)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    out_png = Path(args.out_png).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"[plot_repair_proposal_history] wrote {out_png}")


if __name__ == "__main__":
    main()
