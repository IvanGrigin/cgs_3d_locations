#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot method distances from exported layout histograms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


csv.field_size_limit(1024 * 1024 * 1024)

POINT_FIELDS = [
    "plot_kind",
    "grid_size",
    "segment",
    "creator_family",
    "x",
    "y",
    "n_objects",
    "n_rooms",
]

DISTANCE_FIELDS = [
    "plot_kind",
    "grid_size",
    "segment",
    "left_creator",
    "right_creator",
    "distance",
    "left_n_objects",
    "right_n_objects",
]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_token(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return "_".join(part for part in text.split() if part)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def js_divergence(p: list[float], q: list[float]) -> float:
    def kl(a: list[float], b: list[float]) -> float:
        total = 0.0
        for av, bv in zip(a, b):
            if av > 0.0 and bv > 0.0:
                total += av * math.log(av / bv, 2)
        return total

    m = [(a + b) * 0.5 for a, b in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def segment_key(row: dict[str, str]) -> str:
    return f"room_type={row.get('room_type') or '__all__'}|class={row.get('class_name') or '__all__'}"


def filter_creator_rows(
    rows: list[dict[str, str]],
    *,
    include_creators: set[str],
    exclude_creators: set[str],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        creator = norm_token(row.get("creator_family") or "unknown")
        if include_creators and creator not in include_creators:
            continue
        if exclude_creators and creator in exclude_creators:
            continue
        copied = dict(row)
        copied["creator_family"] = creator
        out.append(copied)
    return out


def histogram_rows_from_objects(
    object_rows: list[dict[str, str]],
    *,
    grid_sizes: list[int],
    include_creators: set[str],
    exclude_creators: set[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, str]] = []
    for row in object_rows:
        creator = norm_token(row.get("creator_family") or "unknown")
        if include_creators and creator not in include_creators:
            continue
        if exclude_creators and creator in exclude_creators:
            continue
        if str(row.get("is_trackable_for_distribution") or "") not in {"1", "1.0", "true", "True"}:
            continue
        x = safe_float(row.get("x_norm"), default=float("nan"))
        y = safe_float(row.get("y_norm"), default=float("nan"))
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
            continue
        copied = dict(row)
        copied["creator_family"] = creator
        copied["x_norm"] = str(x)
        copied["y_norm"] = str(y)
        filtered.append(copied)

    groupings = [
        ("overall", lambda r: ("__all__", "__all__")),
        ("by_class", lambda r: ("__all__", r.get("class_name") or "__all__")),
        ("by_room_type", lambda r: (r.get("room_type") or "__all__", "__all__")),
    ]
    out: list[dict[str, Any]] = []
    for grid in grid_sizes:
        for grouping, group_key_fn in groupings:
            buckets: dict[tuple[str, str, str, str, str, int, int], int] = defaultdict(int)
            totals: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
            rooms: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
            for row in filtered:
                room_type, class_name = group_key_fn(row)
                key_base = (
                    grouping,
                    row.get("dataset_role") or "unknown",
                    row.get("creator_family") or "unknown",
                    row.get("creator_variant") or row.get("creator_family") or "unknown",
                    room_type,
                    class_name,
                )
                x = safe_float(row.get("x_norm"))
                y = safe_float(row.get("y_norm"))
                cell_x = min(grid - 1, max(0, int(math.floor(x * grid))))
                cell_y = min(grid - 1, max(0, int(math.floor(y * grid))))
                buckets[(*key_base, cell_x, cell_y)] += 1
                totals[key_base] += 1
                rooms[key_base].add(row.get("run_id") or "")
            for key_base, total in sorted(totals.items()):
                grouping_name, role, family, variant, room_type, class_name = key_base
                n_rooms = len(rooms[key_base])
                for cell_y in range(grid):
                    for cell_x in range(grid):
                        count = buckets[(*key_base, cell_x, cell_y)]
                        out.append(
                            {
                                "grouping": grouping_name,
                                "dataset_role": role,
                                "creator_family": family,
                                "creator_variant": variant,
                                "room_type": room_type,
                                "class_name": class_name,
                                "grid_size": grid,
                                "cell_x": cell_x,
                                "cell_y": cell_y,
                                "count": count,
                                "probability": count / max(total, 1),
                                "n_objects": total,
                                "n_rooms": n_rooms,
                            }
                        )
    return out


def histogram_vectors(hist_rows: list[dict[str, str]], grid: int, grouping: str) -> dict[tuple[str, str], dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in hist_rows:
        if row.get("grouping") != grouping:
            continue
        if int(row.get("grid_size") or 0) != grid:
            continue
        creator = row.get("creator_family") or "unknown"
        if creator == "unknown":
            continue
        buckets[(segment_key(row), creator)].append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in buckets.items():
        vec = [0.0] * (grid * grid)
        for row in rows:
            x = int(row["cell_x"])
            y = int(row["cell_y"])
            vec[y * grid + x] = safe_float(row.get("probability"))
        out[key] = {
            "vector": vec,
            "n_objects": int(float(rows[0].get("n_objects") or 0)),
            "n_rooms": int(float(rows[0].get("n_rooms") or 0)),
        }
    return out


def classical_mds(np, names: list[str], distances: dict[tuple[str, str], float]) -> dict[str, tuple[float, float]]:
    n = len(names)
    if n == 0:
        return {}
    if n == 1:
        return {names[0]: (0.0, 0.0)}
    d = np.zeros((n, n), dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            d[i, j] = distances.get((a, b), distances.get((b, a), 0.0))
    d2 = d ** 2
    ident = np.eye(n)
    ones = np.ones((n, n)) / n
    jmat = ident - ones
    bmat = -0.5 * jmat @ d2 @ jmat
    vals, vecs = np.linalg.eigh(bmat)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    coords = np.zeros((n, 2), dtype=float)
    for dim in range(min(2, n)):
        if vals[dim] > 0:
            coords[:, dim] = vecs[:, dim] * math.sqrt(vals[dim])
    return {name: (float(coords[i, 0]), float(coords[i, 1])) for i, name in enumerate(names)}


def plot_points(plt, points: list[dict[str, Any]], title: str, out_path: Path) -> None:
    palette = {
        "3dfront": "#222222",
        "infinigen": "#1f77b4",
        "diffuscene": "#d62728",
        "m3dlayout": "#9467bd",
        "procedural": "#2ca02c",
        "cube": "#2ca02c",
        "random": "#8c564b",
        "relaxed": "#17becf",
        "retrieval": "#ff7f0e",
        "ollama_llm": "#7f7f7f",
    }
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111)
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    if xs and ys:
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        pad_x = max(span_x * 0.18, 0.02)
        pad_y = max(span_y * 0.18, 0.02)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    for point in points:
        name = str(point["creator_family"])
        n = int(point["n_objects"])
        size = 55 + min(260, math.sqrt(max(n, 1)) * 5)
        ax.scatter(
            float(point["x"]),
            float(point["y"]),
            s=size,
            color=palette.get(name, "#555555"),
            alpha=0.84,
            edgecolor="white",
            linewidth=0.9,
        )
        ax.annotate(
            f"{name}\\nn={n}",
            (float(point["x"]), float(point["y"])),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="#999999", linewidth=0.6, alpha=0.35)
    ax.axvline(0, color="#999999", linewidth=0.6, alpha=0.35)
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_plots(
    *,
    hist_rows: list[dict[str, str]],
    out_dir: Path,
    grid_size: int,
    min_objects: int,
    max_class_plots: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plt, np = setup_matplotlib()
    point_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []

    specs = [
        ("overall", "overall", "room_type=__all__|class=__all__", 1),
        ("by_class", "by_class", None, max_class_plots),
        ("by_room_type", "by_room_type", None, max_class_plots),
    ]
    for plot_kind, grouping, forced_segment, limit in specs:
        vectors = histogram_vectors(hist_rows, grid_size, grouping)
        segments = sorted({seg for seg, _ in vectors.keys()})
        if forced_segment is not None:
            segments = [forced_segment]
        else:
            scored = []
            for seg in segments:
                total = sum(vectors[(seg, c)]["n_objects"] for s, c in vectors if s == seg)
                has_gt = (seg, "3dfront") in vectors
                scored.append((0 if has_gt else 1, -total, seg))
            segments = [x[2] for x in sorted(scored)[:limit]]

        for segment in segments:
            creators = sorted(
                c for s, c in vectors.keys()
                if s == segment and vectors[(s, c)]["n_objects"] >= min_objects
            )
            if len(creators) < 2:
                continue
            distances: dict[tuple[str, str], float] = {}
            for i, left in enumerate(creators):
                for right in creators[i + 1:]:
                    d = js_divergence(vectors[(segment, left)]["vector"], vectors[(segment, right)]["vector"])
                    distances[(left, right)] = d
                    distance_rows.append({
                        "plot_kind": plot_kind,
                        "grid_size": grid_size,
                        "segment": segment,
                        "left_creator": left,
                        "right_creator": right,
                        "distance": d,
                        "left_n_objects": vectors[(segment, left)]["n_objects"],
                        "right_n_objects": vectors[(segment, right)]["n_objects"],
                    })
            coords = classical_mds(np, creators, distances)
            points = []
            for creator in creators:
                info = vectors[(segment, creator)]
                x, y = coords[creator]
                row = {
                    "plot_kind": plot_kind,
                    "grid_size": grid_size,
                    "segment": segment,
                    "creator_family": creator,
                    "x": x,
                    "y": y,
                    "n_objects": info["n_objects"],
                    "n_rooms": info["n_rooms"],
                }
                points.append(row)
                point_rows.append(row)
            safe_segment = segment.replace("=", "_").replace("|", "__").replace("/", "_")
            out_path = out_dir / plot_kind / f"grid_{grid_size}__{safe_segment}.png"
            plot_points(plt, points, f"{plot_kind} | {segment} | grid {grid_size} | JS-MDS", out_path)
            index_rows.append({
                "plot_kind": plot_kind,
                "grid_size": grid_size,
                "segment": segment,
                "png": str(out_path),
                "n_methods": len(creators),
            })
    return point_rows, distance_rows, index_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot method distances from layout histograms.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--histograms-csv")
    src.add_argument("--objects-csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--min-objects", type=int, default=30)
    parser.add_argument("--max-class-plots", type=int, default=18)
    parser.add_argument("--include-creators", nargs="*", default=None)
    parser.add_argument("--exclude-creators", nargs="*", default=["unknown"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    include_creators = {norm_token(x) for x in (args.include_creators or [])}
    exclude_creators = {norm_token(x) for x in (args.exclude_creators or [])}
    if args.histograms_csv:
        hist_csv = Path(args.histograms_csv).expanduser().resolve()
        hist_rows = filter_creator_rows(
            read_csv(hist_csv),
            include_creators=include_creators,
            exclude_creators=exclude_creators,
        )
        hist_source = str(hist_csv)
    else:
        objects_csv = Path(args.objects_csv).expanduser().resolve()
        hist_rows = histogram_rows_from_objects(
            read_csv(objects_csv),
            grid_sizes=[args.grid_size],
            include_creators=include_creators,
            exclude_creators=exclude_creators,
        )
        hist_source = str(objects_csv)
        write_csv(
            out_dir / f"histograms_from_objects_grid_{args.grid_size}.csv",
            hist_rows,
            [
                "grouping",
                "dataset_role",
                "creator_family",
                "creator_variant",
                "room_type",
                "class_name",
                "grid_size",
                "cell_x",
                "cell_y",
                "count",
                "probability",
                "n_objects",
                "n_rooms",
            ],
        )
    points, distances, index = build_plots(
        hist_rows=hist_rows,
        out_dir=out_dir,
        grid_size=args.grid_size,
        min_objects=args.min_objects,
        max_class_plots=args.max_class_plots,
    )
    write_csv(out_dir / f"mds_points_grid_{args.grid_size}.csv", points, POINT_FIELDS)
    write_csv(out_dir / f"method_distances_grid_{args.grid_size}.csv", distances, DISTANCE_FIELDS)
    write_csv(out_dir / f"method_distance_plot_index_grid_{args.grid_size}.csv", index, ["plot_kind", "grid_size", "segment", "png", "n_methods"])
    summary = {
        "input_source": hist_source,
        "out_dir": str(out_dir),
        "grid_size": args.grid_size,
        "min_objects": args.min_objects,
        "include_creators": sorted(include_creators),
        "exclude_creators": sorted(exclude_creators),
        "n_points": len(points),
        "n_distances": len(distances),
        "n_plots": len(index),
    }
    (out_dir / f"method_distance_summary_grid_{args.grid_size}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
