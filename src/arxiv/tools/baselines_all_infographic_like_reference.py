#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scatter plot (5 points) in the same style as the reference image:
- x: CollisionPairRate (↓ лучше)
- y: RMSE_xz (↓ лучше)
- labels placed with explicit pixel offsets
- a short explanatory block under the plot describing each mode
Output: baselines_all_infographic_like_reference.png
"""

import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    # Data (from the provided metrics table)
    points = [
        # name, RMSE_xz (y), CollisionPairRate (x), label_offset(px), label_align
        ("random_feasible+greedy", 4.969091, 0.037595, (0,  -20), dict(ha="left",  va="bottom")),  # 20 px below
        ("forest+greedy",          4.038779, 0.054204, (8,    6), dict(ha="left",  va="bottom")),  # a bit farther
        ("relaxed_cube",           4.993223, 0.093344, (0,  -50), dict(ha="left",  va="bottom")),  # 50 px below
        ("random_feasible",        4.977421, 0.146260, (0,  -80), dict(ha="left",  va="bottom")),  # 80 px below
        ("forest",                 3.912814, 0.330501, (-40,  6), dict(ha="right", va="bottom")),  # slightly left
    ]

    modes_desc = [
        ("forest", "Генерация на основе «forest» (базовый режим)."),
        ("forest+greedy", "«forest» + жадное улучшение (greedy) для снижения коллизий/ошибок."),
        ("relaxed_cube", "Генерация в «relaxed cube» постановке (ослабленные ограничения куба)."),
        ("random_feasible+greedy", "Случайная допустимая расстановка (random_feasible) + greedy-дооптимизация."),
        ("random_feasible", "Случайная допустимая расстановка без greedy-шага."),
    ]

    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    x = np.array([p[2] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    ax.scatter(x, y, s=55)

    ax.set_title("Baselines: trade-off RMSE vs Collisions", fontsize=14)
    ax.set_xlabel("CollisionPairRate (↓ лучше)", fontsize=12)
    ax.set_ylabel("RMSE_xz (↓ лучше)", fontsize=12)
    ax.grid(True, alpha=0.35)

    for name, rmse, cpr, (dx, dy), align in points:
        ax.annotate(
            name,
            xy=(cpr, rmse),
            xytext=(dx, dy),
            textcoords="offset pixels",
            fontsize=11,
            **align
        )

    fig.subplots_adjust(bottom=0.33)

    desc_title = "Что означает каждая точка (режим генерации расстановки):"
    lines = [desc_title] + [f"• {k}: {v}" for k, v in modes_desc]
    fig.text(0.02, 0.02, "\n".join(lines), ha="left", va="bottom", fontsize=10)

    fig.savefig("baselines_all_infographic_like_reference.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
