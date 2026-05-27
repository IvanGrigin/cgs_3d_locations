#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot grid-size sweep metrics from layout heatmap summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METHODS = ["diffuscene", "infinigen", "m3dlayout", "ollama_llm", "procedural"]

PALETTE = {
    "diffuscene": "#ff7f0e",
    "infinigen": "#2ca02c",
    "m3dlayout": "#d62728",
    "ollama_llm": "#9467bd",
    "procedural": "#17becf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grid sweep curves from method_summary_vs_REFERENCE_all.csv.")
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--out-png", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--metric", default="weighted_js_distance")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--grouping", default="class")
    parser.add_argument("--sigma", type=float, default=1.25)
    parser.add_argument("--augmentation", default="rot90_flip")
    parser.add_argument("--title", default="Grid sweep: generators vs 3D-FRONT")
    parser.add_argument("--xlabel", default="Grid size: N x N")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_csv)
    required = {"grouping", "grid", "sigma", "augmentation", "method", args.metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {args.summary_csv}: {missing}")

    plot_df = df[
        (df["grouping"].astype(str) == args.grouping)
        & (pd.to_numeric(df["sigma"], errors="coerce") == float(args.sigma))
        & (df["augmentation"].astype(str) == args.augmentation)
        & (df["method"].astype(str).isin(args.methods))
    ].copy()
    plot_df["grid"] = pd.to_numeric(plot_df["grid"], errors="coerce")
    plot_df[args.metric] = pd.to_numeric(plot_df[args.metric], errors="coerce")
    plot_df = plot_df.dropna(subset=["grid", args.metric]).sort_values(["method", "grid"])

    if plot_df.empty:
        raise SystemExit("No rows left after filtering summary CSV.")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        plot_df.to_csv(args.out_csv, index=False)

    fig, ax = plt.subplots(figsize=(13.5, 7.2), constrained_layout=True)
    for method in args.methods:
        method_df = plot_df[plot_df["method"].astype(str) == method].sort_values("grid")
        if method_df.empty:
            continue
        ax.plot(
            method_df["grid"].to_numpy(),
            method_df[args.metric].to_numpy(),
            marker="o",
            linewidth=2.2,
            markersize=4.2,
            label=method,
            color=PALETTE.get(method),
        )

    ylabel = args.ylabel or f"{args.metric} ↓"
    ax.set_title(args.title, fontsize=17, pad=10)
    ax.set_xlabel(args.xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", frameon=True)
    grids = sorted(int(x) for x in plot_df["grid"].dropna().unique())
    if grids:
        ax.set_xlim(min(grids), max(grids))
        step = 5 if max(grids) - min(grids) >= 20 else 2
        ticks = [g for g in grids if (g - min(grids)) % step == 0]
        if max(grids) not in ticks:
            ticks.append(max(grids))
        ax.set_xticks(ticks)
    fig.savefig(args.out_png, dpi=args.dpi)
    plt.close(fig)

    methods_present = sorted(plot_df["method"].astype(str).unique())
    print(f"[grid-sweep] wrote {args.out_png}")
    if args.out_csv:
        print(f"[grid-sweep] wrote {args.out_csv}")
    print(f"[grid-sweep] methods: {', '.join(methods_present)}")


if __name__ == "__main__":
    main()
