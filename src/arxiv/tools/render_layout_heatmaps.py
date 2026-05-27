#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render count heatmaps from exported layout distribution objects."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


csv.field_size_limit(1024 * 1024 * 1024)

INDEX_FIELDS = [
    "grouping",
    "creator_family",
    "creator_variant",
    "room_type",
    "class_name",
    "grid_size",
    "n_points",
    "n_rooms",
    "objects_csv",
    "heatmap_png",
]


def sanitize(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text))
    return text.strip("_") or "all"


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib():
    cache_root = Path("/private/tmp/cgs_layout_distribution_matplotlib")
    (cache_root / "mpl").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def group_key(row: dict[str, str], grouping: str) -> tuple[str, str, str, str]:
    creator_family = row.get("creator_family") or "unknown"
    creator_variant = row.get("creator_variant") or "unknown"
    room_type = row.get("room_type") or "__all__"
    class_name = row.get("class_name") or "unknown"
    if grouping == "by_creator":
        return creator_family, creator_variant, "__all__", "__all__"
    if grouping == "by_creator_class":
        return creator_family, creator_variant, "__all__", class_name
    if grouping == "by_creator_room_type":
        return creator_family, creator_variant, room_type, "__all__"
    if grouping == "by_creator_room_type_class":
        return creator_family, creator_variant, room_type, class_name
    raise ValueError(f"Unknown grouping: {grouping}")


def title_for(grouping: str, creator_family: str, creator_variant: str, room_type: str, class_name: str, n_points: int, n_rooms: int) -> str:
    lines = []
    if class_name != "__all__":
        lines.append(class_name)
    else:
        lines.append(creator_family)
    lines.append(f"creator={creator_family} | variant={creator_variant}")
    if room_type != "__all__":
        lines.append(f"room={room_type}")
    if class_name != "__all__":
        lines.append(f"class={class_name}")
    lines.append(f"points={n_points} | rooms={n_rooms}")
    return "\n".join(lines)


def render_heatmap(
    *,
    plt,
    np,
    points: list[tuple[float, float]],
    grid_size: int,
    title: str,
    out_path: Path,
) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    heat, _, _ = np.histogram2d(xs, ys, bins=grid_size, range=[[0, 1], [0, 1]])
    matrix = heat.T

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        matrix,
        origin="lower",
        extent=[0, 1, 0, 1],
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("count")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x_norm")
    ax.set_ylabel("y_norm")
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render layout heatmaps from objects_all.csv.")
    parser.add_argument("--objects-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--max-maps", type=int, default=300)
    parser.add_argument(
        "--groupings",
        nargs="+",
        default=["by_creator_class", "by_creator_room_type_class"],
        choices=["by_creator", "by_creator_class", "by_creator_room_type", "by_creator_room_type_class"],
    )
    parser.add_argument("--creators", nargs="*", default=None, help="Optional creator_family filter.")
    parser.add_argument("--room-types", nargs="*", default=None, help="Optional room_type filter.")
    parser.add_argument("--classes", nargs="*", default=None, help="Optional class_name filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objects_csv = Path(args.objects_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = load_rows(objects_csv)

    creator_filter = set(args.creators or [])
    room_filter = set(args.room_types or [])
    class_filter = set(args.classes or [])

    filtered = []
    for row in rows:
        if row.get("is_trackable_for_distribution") != "1":
            continue
        if creator_filter and row.get("creator_family") not in creator_filter:
            continue
        if room_filter and row.get("room_type") not in room_filter:
            continue
        if class_filter and row.get("class_name") not in class_filter:
            continue
        x = as_float(row.get("x_norm"))
        y = as_float(row.get("y_norm"))
        if x is None or y is None or not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
            continue
        filtered.append(row)

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for grouping in args.groupings:
        for row in filtered:
            grouped[(grouping, *group_key(row, grouping))].append(row)

    candidates = [
        (key, group_rows)
        for key, group_rows in grouped.items()
        if len(group_rows) >= args.min_points
    ]
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    candidates = candidates[: args.max_maps]

    plt, np = setup_matplotlib()
    index_rows: list[dict[str, Any]] = []
    for key, group_rows in candidates:
        grouping, creator_family, creator_variant, room_type, class_name = key
        points = [(float(r["x_norm"]), float(r["y_norm"])) for r in group_rows]
        n_rooms = len({r.get("run_id", "") for r in group_rows})
        title = title_for(grouping, creator_family, creator_variant, room_type, class_name, len(points), n_rooms)
        filename = "__".join([
            sanitize(grouping),
            f"grid_{args.grid_size}",
            sanitize(creator_family),
            sanitize(creator_variant),
            sanitize(room_type),
            sanitize(class_name),
        ]) + ".png"
        out_path = out_dir / grouping / f"grid_{args.grid_size}" / filename
        render_heatmap(
            plt=plt,
            np=np,
            points=points,
            grid_size=args.grid_size,
            title=title,
            out_path=out_path,
        )
        index_rows.append({
            "grouping": grouping,
            "creator_family": creator_family,
            "creator_variant": creator_variant,
            "room_type": room_type,
            "class_name": class_name,
            "grid_size": args.grid_size,
            "n_points": len(points),
            "n_rooms": n_rooms,
            "objects_csv": str(objects_csv),
            "heatmap_png": str(out_path),
        })

    write_csv(out_dir / f"heatmap_index_grid_{args.grid_size}.csv", index_rows, INDEX_FIELDS)
    print(f"Rendered {len(index_rows)} heatmaps to {out_dir}")
    print(out_dir / f"heatmap_index_grid_{args.grid_size}.csv")


if __name__ == "__main__":
    main()
