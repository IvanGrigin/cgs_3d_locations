#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/tools/plot_layout_heatmap_diff_analysis.py

Назначение:
- сравнить Infinigen и другие методы с 3D-FRONT не общей кучей объектов, а по категориям;
- построить нормализованные heatmap центров объектов в процентах;
- размыть heatmap Gaussian blur и снова нормализовать к 100%;
- построить карты разностей method - 3dfront в процентных пунктах;
- посчитать числовые расстояния по каждой категории;
- сохранить картинки и таблицы, пригодные для ВКР и презентации.

Аугментация:
Абсолютная ориентация комнаты в датасете обычно не является семантически значимой. Одна и та же сцена может
быть повернута на 90/180/270 градусов или отражена по горизонтали/вертикали, оставаясь тем же размещением
относительно стен, углов и центра. Поэтому скрипт умеет применять одинаковую аугментацию ко всем методам:
- горизонтальная симметрия;
- вертикальная симметрия;
- повороты на 90, 180, 270 градусов;
- режим d4: все 8 симметрий квадрата.

Рекомендуемый запуск:

python3 src/tools/plot_layout_heatmap_diff_analysis.py \
  --objects-csv out/layout_distribution_analysis/all_workspace_plus_rtx3060_20260515/objects_all.csv \
  --out-dir out/layout_heatmap_diff_analysis/all_workspace_plus_rtx3060_20260515 \
  --reference 3dfront \
  --exclude-method unknown \
  --classes bed double_bed nightstand wardrobe cabinet chair table sofa tv lamp decor \
  --grouping both \
  --grids 20 32 \
  --sigmas 1.25 \
  --augmentation rot90_flip \
  --min-ref-objects 50 \
  --min-method-objects 30 \
  --class-mode canonical \
  --plot-mode reference \
  --dpi 220

Главные файлы:
- layout_heatmap_diff_report.md
- report_assets/vkr_summary.md
- report_assets/vkr_summary_table.tex
- report_assets/method_summary_for_report.csv
- report_assets/per_category_metrics_for_report.csv
- report_assets/key_figures_index.csv
- report_assets/key_figures/*.png
- tables/heatmap_vs_3dfront_metrics_all.csv
- tables/method_summary_compact_vs_3dfront.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

EPS = 1e-12

REQUIRED_COLUMNS = [
    "creator_family",
    "room_type",
    "class_name",
    "x_norm",
    "y_norm",
    "is_trackable_for_distribution",
]

NUMERIC_COLUMNS = [
    "x_norm",
    "y_norm",
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "size_x_m",
    "size_y_m",
    "size_z_m",
    "yaw_deg",
    "yaw_rad",
    "rotation_deg",
    "aabb_x_min",
    "aabb_x_max",
    "aabb_y_min",
    "aabb_y_max",
    "aabb_z_min",
    "aabb_z_max",
    "room_width_m",
    "room_depth_m",
    "room_area_m2",
    "inside_room_bbox",
    "inside_floor_polygon",
    "distance_to_nearest_wall_m",
    "distance_to_nearest_corner_m",
    "is_near_wall",
    "is_near_corner",
    "is_center_zone",
    "has_valid_aabb",
    "is_small_object",
    "is_trackable_for_distribution",
]

DEFAULT_CLASSES = [
    "bed",
    "double_bed",
    "nightstand",
    "wardrobe",
    "cabinet",
    "chair",
    "table",
    "sofa",
    "tv",
    "lamp",
    "decor",
]

ROOM_ALIASES = {
    "living": "livingroom",
    "living_room": "livingroom",
    "living room": "livingroom",
    "lounge": "livingroom",
    "bed room": "bedroom",
    "bath": "bathroom",
    "toilet": "bathroom",
    "wc": "bathroom",
    "corridor": "hallway",
}


@dataclass(frozen=True)
class HeatmapRecord:
    grouping: str
    group_key: str
    room_type: str
    class_name: str
    method: str
    n_raw: int
    n_augmented: int
    grid: int
    sigma: float
    augmentation: str
    hist_percent: np.ndarray


@dataclass(frozen=True)
class ComparisonRecord:
    grouping: str
    group_key: str
    room_type: str
    class_name: str
    grid: int
    sigma: float
    augmentation: str
    method_a: str
    method_b: str
    n_a: int
    n_b: int
    n_a_augmented: int
    n_b_augmented: int
    tv_distance: float
    l1_percent_points: float
    js_divergence: float
    js_distance: float
    hellinger_distance: float
    cosine_distance: float
    pearson_corr: float
    sliced_wasserstein: float
    centroid_a_x: float
    centroid_a_y: float
    centroid_b_x: float
    centroid_b_y: float
    centroid_shift: float
    max_abs_diff_pp: float
    mean_abs_diff_pp: float
    positive_mass_pp: float
    negative_mass_pp: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Heatmap difference analysis for layout methods against 3D-FRONT."
    )
    parser.add_argument("--objects-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference", default="3dfront")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--exclude-method", nargs="*", default=["unknown"])
    parser.add_argument("--room-types", nargs="*", default=None)
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--class-mode", choices=["raw", "canonical"], default="canonical")
    parser.add_argument("--grouping", choices=["class", "room_class", "both"], default="both")
    parser.add_argument("--grids", nargs="+", type=int, default=[20, 32])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[1.25])
    parser.add_argument(
        "--augmentation",
        choices=["none", "flip", "rot90", "rot90_flip", "d4"],
        default="rot90_flip",
        help="none | flip | rot90 | rot90_flip | d4",
    )
    parser.add_argument("--min-ref-objects", type=int, default=50)
    parser.add_argument("--min-method-objects", type=int, default=30)
    parser.add_argument("--min-objects-for-pairwise", type=int, default=30)
    parser.add_argument("--plot-mode", choices=["none", "reference", "pairwise_all"], default="reference")
    parser.add_argument("--max-comparison-plots", type=int, default=2500)
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--sliced-wasserstein-projections", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--heatmap-cmap", default="viridis")
    parser.add_argument("--diff-cmap", default="coolwarm")
    parser.add_argument("--absdiff-cmap", default="magma")
    parser.add_argument("--top-k-dashboard", type=int, default=30)
    parser.add_argument("--report-grid", type=int, default=None)
    parser.add_argument("--report-sigma", type=float, default=None)
    parser.add_argument("--report-grouping", choices=["class", "room_class"], default="class")
    parser.add_argument(
        "--report-methods",
        nargs="*",
        default=["infinigen", "diffuscene", "retrieval", "random", "ollama_llm", "m3dlayout", "cube", "relaxed"],
    )
    parser.add_argument(
        "--key-figure-methods",
        nargs="*",
        default=["infinigen", "diffuscene", "random", "retrieval"],
    )
    parser.add_argument(
        "--key-figure-classes",
        nargs="*",
        default=["bed", "nightstand", "wardrobe", "cabinet", "chair", "table", "sofa", "tv", "lamp"],
    )
    return parser.parse_args()


def safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_token(value: object) -> str:
    text = safe_str(value).lower().strip()
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_room_type(value: object) -> str:
    room = normalize_token(value)
    return ROOM_ALIASES.get(room, room if room else "unknown_room")


def canonical_class_name(value: object) -> str:
    cls = normalize_token(value)
    if not cls:
        return "unknown_class"

    aliases = {
        "television": "tv",
        "tv_stand": "cabinet",
        "bookcase": "cabinet",
        "bookshelf": "cabinet",
        "shelf": "cabinet",
        "shelving": "cabinet",
        "sideboard": "cabinet",
        "dresser": "cabinet",
        "chest_of_drawers": "cabinet",
        "closet": "wardrobe",
        "armoire": "wardrobe",
        "bedside_table": "nightstand",
        "bedside_cabinet": "nightstand",
        "side_table": "nightstand",
        "coffee_table": "table",
        "dining_table": "table",
        "desk": "table",
        "work_desk": "table",
        "office_desk": "table",
        "couch": "sofa",
        "settee": "sofa",
        "armchair": "chair",
        "stool": "chair",
        "dining_chair": "chair",
        "office_chair": "chair",
        "ceiling_lamp": "lamp",
        "floor_lamp": "lamp",
        "table_lamp": "lamp",
        "wall_lamp": "lamp",
        "wall_light": "lamp",
        "pendant_lamp": "lamp",
        "light": "lamp",
        "lighting": "lamp",
        "wall_art": "decor",
        "wallart": "decor",
        "painting": "decor",
        "picture": "decor",
        "rug": "decor",
        "carpet": "decor",
        "curtain": "decor",
        "plant": "decor",
        "vase": "decor",
        "sculpture": "decor",
        "decoration": "decor",
        "decor_accessory": "decor",
    }
    if cls in aliases:
        return aliases[cls]
    if cls in DEFAULT_CLASSES:
        return cls

    if "nightstand" in cls or "bedside" in cls:
        return "nightstand"
    if "double_bed" in cls or "queen_bed" in cls or "king_bed" in cls:
        return "double_bed"
    if cls == "bed" or cls.endswith("_bed") or "bedfactory" in cls:
        return "bed"
    if "wardrobe" in cls or "closet" in cls or "armoire" in cls:
        return "wardrobe"
    if "cabinet" in cls or "bookcase" in cls or "bookshelf" in cls or "shelf" in cls or "sideboard" in cls or "dresser" in cls:
        return "cabinet"
    if "chair" in cls or "stool" in cls or "armchair" in cls:
        return "chair"
    if "table" in cls or cls.endswith("desk") or "desk" in cls:
        return "table"
    if "sofa" in cls or "couch" in cls:
        return "sofa"
    if cls == "tv" or "television" in cls or re.search(r"(^|_)tv($|_)", cls):
        return "tv"
    if "lamp" in cls or "light" in cls or "lighting" in cls:
        return "lamp"
    if any(k in cls for k in [
        "decor", "wallart", "wall_art", "painting", "picture", "curtain", "rug", "carpet",
        "plant", "vase", "bottle", "bowl", "book", "sculpt", "pillow", "cushion", "blanket"
    ]):
        return "decor"
    return cls


def sanitize_filename(value: str, max_len: int = 180) -> str:
    text = safe_str(value)
    text = re.sub(r"[^A-Za-z0-9А-Яа-я_.=+\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "empty"


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def read_and_filter_objects(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.objects_csv, low_memory=False)
    ensure_required_columns(df)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["method"] = df["creator_family"].map(normalize_token)
    df["room_type_norm"] = df["room_type"].map(normalize_room_type)
    if args.class_mode == "canonical":
        df["class_name_norm"] = df["class_name"].map(canonical_class_name)
    else:
        df["class_name_norm"] = df["class_name"].map(normalize_token)

    df = df[df["is_trackable_for_distribution"] == 1].copy()
    df = df[df["x_norm"].between(0, 1, inclusive="left")].copy()
    df = df[df["y_norm"].between(0, 1, inclusive="left")].copy()

    excluded = {normalize_token(x) for x in (args.exclude_method or [])}
    if excluded:
        df = df[~df["method"].isin(excluded)].copy()

    if args.methods:
        methods = {normalize_token(x) for x in args.methods}
        methods.add(normalize_token(args.reference))
        df = df[df["method"].isin(methods)].copy()

    if args.room_types:
        rooms = {normalize_room_type(x) for x in args.room_types}
        df = df[df["room_type_norm"].isin(rooms)].copy()

    if args.classes:
        if args.class_mode == "canonical":
            classes = {canonical_class_name(x) for x in args.classes}
        else:
            classes = {normalize_token(x) for x in args.classes}
        df = df[df["class_name_norm"].isin(classes)].copy()

    return df


def augmentation_transforms(mode: str) -> List[Tuple[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]]:
    def identity(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, y

    def flip_h(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return 1.0 - x, y

    def flip_v(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, 1.0 - y

    def rot90(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return 1.0 - y, x

    def rot180(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return 1.0 - x, 1.0 - y

    def rot270(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return y, 1.0 - x

    def diag_main(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return y, x

    def diag_anti(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return 1.0 - y, 1.0 - x

    if mode == "none":
        return [("identity", identity)]
    if mode == "flip":
        return [("identity", identity), ("flip_h", flip_h), ("flip_v", flip_v)]
    if mode == "rot90":
        return [("identity", identity), ("rot90", rot90), ("rot180", rot180), ("rot270", rot270)]
    if mode == "rot90_flip":
        return [("identity", identity), ("flip_h", flip_h), ("flip_v", flip_v), ("rot90", rot90), ("rot180", rot180), ("rot270", rot270)]
    if mode == "d4":
        return [("identity", identity), ("flip_h", flip_h), ("flip_v", flip_v), ("rot90", rot90), ("rot180", rot180), ("rot270", rot270), ("diag_main", diag_main), ("diag_anti", diag_anti)]
    raise ValueError(f"Unsupported augmentation mode: {mode}")


def augment_points(x: np.ndarray, y: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _name, transform in augmentation_transforms(mode):
        tx, ty = transform(x, y)
        xs.append(np.clip(tx, 0.0, 1.0 - EPS))
        ys.append(np.clip(ty, 0.0, 1.0 - EPS))
    return np.concatenate(xs), np.concatenate(ys)


def build_hist_percent(x: np.ndarray, y: np.ndarray, grid: int, sigma: float, augmentation: str) -> Tuple[np.ndarray, int]:
    if len(x) == 0:
        return np.zeros((grid, grid), dtype=np.float64), 0

    x_aug, y_aug = augment_points(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), augmentation)
    xi = np.floor(x_aug * grid).astype(np.int64)
    yi = np.floor(y_aug * grid).astype(np.int64)
    xi = np.clip(xi, 0, grid - 1)
    yi = np.clip(yi, 0, grid - 1)

    hist = np.zeros((grid, grid), dtype=np.float64)
    np.add.at(hist, (yi, xi), 1.0)
    total = float(hist.sum())
    if total <= 0:
        return hist, int(len(x_aug))

    hist = hist / total * 100.0
    if sigma > 0:
        if gaussian_filter is None:
            raise RuntimeError("scipy is required for Gaussian blur. Install scipy or run with --sigmas 0.")
        hist = gaussian_filter(hist, sigma=float(sigma), mode="constant", cval=0.0)
        blurred_total = float(hist.sum())
        if blurred_total > 0:
            hist = hist / blurred_total * 100.0
    return hist, int(len(x_aug))


def probability(hist_percent: np.ndarray) -> np.ndarray:
    p = np.asarray(hist_percent, dtype=np.float64).reshape(-1) / 100.0
    total = float(p.sum())
    return p / total if total > 0 else p


def js_divergence_base2(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(float(p.sum()), EPS)
    q = q / max(float(q.sum()), EPS)
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = float(np.sum(p[mask_p] * np.log2(p[mask_p] / np.maximum(m[mask_p], EPS))))
    kl_qm = float(np.sum(q[mask_q] * np.log2(q[mask_q] / np.maximum(m[mask_q], EPS))))
    return float(max(0.5 * (kl_pm + kl_qm), 0.0))


def weighted_wasserstein_1d(values: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0 or q_sum <= 0:
        return float("nan")
    p = p / p_sum
    q = q / q_sum
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    diff = p[order] - q[order]
    if len(v) <= 1:
        return 0.0
    cdf_diff = np.cumsum(diff)
    return float(np.sum(np.abs(cdf_diff[:-1]) * np.diff(v)))


def sliced_wasserstein_grid(p: np.ndarray, q: np.ndarray, grid: int, n_projections: int, seed: int) -> float:
    if float(p.sum()) <= 0 or float(q.sum()) <= 0:
        return float("nan")
    centers = (np.arange(grid, dtype=np.float64) + 0.5) / float(grid)
    xx, yy = np.meshgrid(centers, centers)
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    rng = np.random.default_rng(seed)
    distances = []
    for angle in rng.uniform(0.0, math.pi, size=int(n_projections)):
        direction = np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=np.float64)
        distances.append(weighted_wasserstein_1d(coords @ direction, p, q))
    return float(np.nanmean(distances))


def centroid_from_probability(p: np.ndarray, grid: int) -> Tuple[float, float]:
    p2 = np.asarray(p, dtype=np.float64).reshape(grid, grid)
    total = float(p2.sum())
    if total <= 0:
        return float("nan"), float("nan")
    centers = (np.arange(grid, dtype=np.float64) + 0.5) / float(grid)
    xx, yy = np.meshgrid(centers, centers)
    return float(np.sum(xx * p2) / total), float(np.sum(yy * p2) / total)


def pearson_corr_safe(p: np.ndarray, q: np.ndarray) -> float:
    if len(p) < 2 or float(np.std(p)) <= EPS or float(np.std(q)) <= EPS:
        return float("nan")
    return float(np.corrcoef(p, q)[0, 1])


def cosine_distance_safe(p: np.ndarray, q: np.ndarray) -> float:
    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
    if denom <= EPS:
        return float("nan")
    cos_sim = float(np.dot(p, q) / denom)
    return float(1.0 - max(min(cos_sim, 1.0), -1.0))


def compare_heatmaps(a: HeatmapRecord, b: HeatmapRecord, n_projections: int, seed: int) -> ComparisonRecord:
    p = probability(a.hist_percent)
    q = probability(b.hist_percent)
    diff_pp = b.hist_percent - a.hist_percent
    jsd = js_divergence_base2(p, q)
    ca_x, ca_y = centroid_from_probability(p, a.grid)
    cb_x, cb_y = centroid_from_probability(q, a.grid)

    return ComparisonRecord(
        grouping=a.grouping,
        group_key=a.group_key,
        room_type=a.room_type,
        class_name=a.class_name,
        grid=a.grid,
        sigma=a.sigma,
        augmentation=a.augmentation,
        method_a=a.method,
        method_b=b.method,
        n_a=a.n_raw,
        n_b=b.n_raw,
        n_a_augmented=a.n_augmented,
        n_b_augmented=b.n_augmented,
        tv_distance=0.5 * float(np.sum(np.abs(p - q))),
        l1_percent_points=float(np.sum(np.abs(diff_pp))),
        js_divergence=jsd,
        js_distance=float(math.sqrt(jsd)),
        hellinger_distance=float(math.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))),
        cosine_distance=cosine_distance_safe(p, q),
        pearson_corr=pearson_corr_safe(p, q),
        sliced_wasserstein=sliced_wasserstein_grid(p, q, a.grid, n_projections, seed),
        centroid_a_x=ca_x,
        centroid_a_y=ca_y,
        centroid_b_x=cb_x,
        centroid_b_y=cb_y,
        centroid_shift=float(math.hypot(cb_x - ca_x, cb_y - ca_y)),
        max_abs_diff_pp=float(np.max(np.abs(diff_pp))),
        mean_abs_diff_pp=float(np.mean(np.abs(diff_pp))),
        positive_mass_pp=float(np.sum(np.maximum(diff_pp, 0.0))),
        negative_mass_pp=float(np.sum(np.minimum(diff_pp, 0.0))),
    )


def record_to_dict(r: ComparisonRecord) -> Dict[str, object]:
    return {
        "grouping": r.grouping,
        "group_key": r.group_key,
        "room_type": r.room_type,
        "class_name": r.class_name,
        "grid": r.grid,
        "sigma": r.sigma,
        "augmentation": r.augmentation,
        "method_a": r.method_a,
        "method_b": r.method_b,
        "n_a": r.n_a,
        "n_b": r.n_b,
        "n_a_augmented": r.n_a_augmented,
        "n_b_augmented": r.n_b_augmented,
        "tv_distance": r.tv_distance,
        "l1_percent_points": r.l1_percent_points,
        "js_divergence": r.js_divergence,
        "js_distance": r.js_distance,
        "hellinger_distance": r.hellinger_distance,
        "cosine_distance": r.cosine_distance,
        "pearson_corr": r.pearson_corr,
        "sliced_wasserstein": r.sliced_wasserstein,
        "centroid_a_x": r.centroid_a_x,
        "centroid_a_y": r.centroid_a_y,
        "centroid_b_x": r.centroid_b_x,
        "centroid_b_y": r.centroid_b_y,
        "centroid_shift": r.centroid_shift,
        "max_abs_diff_pp": r.max_abs_diff_pp,
        "mean_abs_diff_pp": r.mean_abs_diff_pp,
        "positive_mass_pp": r.positive_mass_pp,
        "negative_mass_pp": r.negative_mass_pp,
    }


def make_group_columns(df: pd.DataFrame, grouping: str) -> pd.DataFrame:
    out = df.copy()
    if grouping == "class":
        out["analysis_room_type"] = "__all__"
        out["analysis_class_name"] = out["class_name_norm"]
    elif grouping == "room_class":
        out["analysis_room_type"] = out["room_type_norm"]
        out["analysis_class_name"] = out["class_name_norm"]
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")
    out["group_key"] = out["analysis_room_type"] + "__" + out["analysis_class_name"]
    return out


def build_heatmap_records(df: pd.DataFrame, grouping: str, grid: int, sigma: float, augmentation: str) -> Dict[Tuple[str, str], HeatmapRecord]:
    records: Dict[Tuple[str, str], HeatmapRecord] = {}
    cols = ["group_key", "analysis_room_type", "analysis_class_name", "method"]
    for (group_key, room_type, class_name, method), g in df.groupby(cols, dropna=False):
        x = g["x_norm"].to_numpy(dtype=np.float64)
        y = g["y_norm"].to_numpy(dtype=np.float64)
        hist, n_aug = build_hist_percent(x, y, grid, sigma, augmentation)
        records[(safe_str(group_key), safe_str(method))] = HeatmapRecord(
            grouping=grouping,
            group_key=safe_str(group_key),
            room_type=safe_str(room_type),
            class_name=safe_str(class_name),
            method=safe_str(method),
            n_raw=int(len(g)),
            n_augmented=int(n_aug),
            grid=int(grid),
            sigma=float(sigma),
            augmentation=augmentation,
            hist_percent=hist,
        )
    return records


def set_norm_ticks(ax: plt.Axes, grid: int) -> None:
    labels = np.linspace(0.0, 1.0, 6)
    positions = labels * (grid - 1)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels([f"{v:.1f}" for v in labels])
    ax.set_yticklabels([f"{v:.1f}" for v in labels])


def plot_comparison(a: HeatmapRecord, b: HeatmapRecord, comp: ComparisonRecord, out_png: Path, args: argparse.Namespace) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    h_a = a.hist_percent
    h_b = b.hist_percent
    signed = h_b - h_a
    abs_diff = np.abs(signed)
    heat_vmax = float(max(np.max(h_a), np.max(h_b), EPS))
    diff_vmax = float(max(np.max(np.abs(signed)), EPS))
    abs_vmax = float(max(np.max(abs_diff), EPS))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)
    fig.suptitle(
        f"{a.grouping}: room={a.room_type}, class={a.class_name}, grid={a.grid}, sigma={a.sigma:g}, aug={a.augmentation}\n"
        f"{b.method} - {a.method}; TV={comp.tv_distance:.3f}, JS={comp.js_distance:.3f}, SW={comp.sliced_wasserstein:.3f}, n={b.n_raw}/{a.n_raw}",
        fontsize=11,
    )

    panels = [
        (h_a, f"{a.method}: normalized heatmap, n={a.n_raw}", args.heatmap_cmap, 0.0, heat_vmax, "% of objects"),
        (h_b, f"{b.method}: normalized heatmap, n={b.n_raw}", args.heatmap_cmap, 0.0, heat_vmax, "% of objects"),
        (signed, f"Signed difference: {b.method} - {a.method}", args.diff_cmap, -diff_vmax, diff_vmax, "percentage points"),
        (abs_diff, "Absolute difference", args.absdiff_cmap, 0.0, abs_vmax, "percentage points"),
    ]
    for ax, (mat, title, cmap, vmin, vmax, label) in zip(axes.reshape(-1), panels):
        im = ax.imshow(mat, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x_norm")
        ax.set_ylabel("y_norm")
        set_norm_ticks(ax, a.grid)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)


def plot_metric_bar(df: pd.DataFrame, metric: str, title: str, out_png: Path, lower_is_better: bool, top_k: int, dpi: int) -> None:
    if df.empty or metric not in df.columns:
        return
    plot_df = df.dropna(subset=[metric]).sort_values(metric, ascending=lower_is_better).head(top_k)
    if plot_df.empty:
        return
    labels = plot_df["method"].astype(str).tolist()
    values = plot_df[metric].astype(float).to_numpy()
    fig_h = max(4.0, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), constrained_layout=True)
    y = np.arange(len(plot_df))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.3f}", va="center", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def plot_group_distance_heatmap(vs_ref_df: pd.DataFrame, metric: str, out_png: Path, dpi: int) -> None:
    if vs_ref_df.empty or metric not in vs_ref_df.columns:
        return
    data = vs_ref_df.dropna(subset=[metric]).copy()
    if data.empty:
        return
    pivot = data.pivot_table(index="group_key", columns="method_b", values=metric, aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.mean(axis=0).sort_values(ascending=True).index]
    fig_w = max(8.0, 0.7 * len(pivot.columns) + 4.0)
    fig_h = max(5.0, 0.25 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="upper", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Per-category distance to 3D-FRONT: {metric}")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=metric)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def save_count_tables(df: pd.DataFrame, out_dir: Path, grouping: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = (
        df.groupby(["analysis_room_type", "analysis_class_name", "group_key", "method"])
        .size()
        .reset_index(name="n_objects")
        .sort_values(["analysis_room_type", "analysis_class_name", "method"])
    )
    counts.to_csv(out_dir / f"counts_{grouping}.csv", index=False)
    pivot = counts.pivot_table(
        index=["analysis_room_type", "analysis_class_name", "group_key"],
        columns="method",
        values="n_objects",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.to_csv(out_dir / f"counts_pivot_{grouping}.csv", index=False)


def save_heatmap_npy(records: Mapping[Tuple[str, str], HeatmapRecord], out_dir: Path, grid: int, sigma: float) -> None:
    heatmap_dir = out_dir / "heatmaps_npy" / f"grid_{grid}" / f"sigma_{sigma:g}"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for (_group_key, _method), rec in records.items():
        np.save(heatmap_dir / sanitize_filename(f"{rec.grouping}__{rec.group_key}__{rec.method}.percent.npy"), rec.hist_percent)


def summarize_vs_reference(metrics_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    ref_df = metrics_df[metrics_df["method_a"] == reference].copy()
    if ref_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "tv_distance",
        "l1_percent_points",
        "js_divergence",
        "js_distance",
        "hellinger_distance",
        "cosine_distance",
        "sliced_wasserstein",
        "centroid_shift",
        "max_abs_diff_pp",
        "mean_abs_diff_pp",
    ]
    total_ref = ref_df.drop_duplicates(["grouping", "group_key", "grid", "sigma", "augmentation"])["n_a"].sum()
    rows = []
    for method, g in ref_df.groupby("method_b"):
        weights = g["n_a"].astype(float).to_numpy()
        row = {
            "method": method,
            "n_groups": int(len(g)),
            "total_ref_objects_in_compared_groups": int(g["n_a"].sum()),
            "total_method_objects_in_compared_groups": int(g["n_b"].sum()),
            "coverage_ref_weight": float(weights.sum() / max(float(total_ref), EPS)),
            "median_method_objects_per_group": float(g["n_b"].median()),
        }
        for col in metric_cols:
            values = g[col].astype(float).to_numpy()
            valid = np.isfinite(values) & np.isfinite(weights)
            if np.any(valid):
                row[f"weighted_{col}"] = float(np.average(values[valid], weights=weights[valid]))
                row[f"mean_{col}"] = float(np.mean(values[valid]))
                row[f"median_{col}"] = float(np.median(values[valid]))
            else:
                row[f"weighted_{col}"] = float("nan")
                row[f"mean_{col}"] = float("nan")
                row[f"median_{col}"] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["weighted_tv_distance", "weighted_js_distance"], ascending=[True, True])
    return out


def markdown_table(df: pd.DataFrame, max_rows: int = 30, float_digits: int = 4) -> str:
    if df.empty:
        return "\n_No data._\n"
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}f}")
        else:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Shown {max_rows} of {len(df)} rows._")
    return "\n" + "\n".join(lines) + "\n"


def latex_escape(value: object) -> str:
    text = safe_str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    return text


def dataframe_to_latex_table(df: pd.DataFrame, columns: Sequence[str], caption: str, label: str) -> str:
    if df.empty:
        return "% No data\n"
    table = df.loc[:, [c for c in columns if c in df.columns]].copy()
    rename = {
        "method": "Метод",
        "n_groups": "Категорий",
        "coverage_ref_weight": "Покрытие",
        "weighted_tv_distance": "TV",
        "weighted_js_distance": "JS",
        "weighted_sliced_wasserstein": "SW",
        "weighted_centroid_shift": "Смещение",
        "weighted_max_abs_diff_pp": "Max diff, п.п.",
    }
    table = table.rename(columns={c: rename.get(c, c) for c in table.columns})
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{" + latex_escape(caption) + r"}",
        r"\label{" + latex_escape(label) + r"}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        " & ".join(latex_escape(c) for c in table.columns) + r" \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("" if pd.isna(value) else f"{float(value):.3f}")
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def run_one_setting(base_df: pd.DataFrame, grouping: str, grid: int, sigma: float, args: argparse.Namespace, plot_counter: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference = normalize_token(args.reference)
    setting_dir = args.out_dir / f"grouping_{grouping}" / f"grid_{grid}" / f"sigma_{sigma:g}" / f"augmentation_{args.augmentation}"
    setting_dir.mkdir(parents=True, exist_ok=True)

    df = make_group_columns(base_df, grouping)
    save_count_tables(df, setting_dir / "tables", grouping)
    records = build_heatmap_records(df, grouping, grid, sigma, args.augmentation)
    if args.save_npy:
        save_heatmap_npy(records, setting_dir, grid, sigma)

    methods_by_group: Dict[str, List[str]] = {}
    for group_key, method in records.keys():
        methods_by_group.setdefault(group_key, []).append(method)

    pairwise_rows: List[Dict[str, object]] = []
    vs_ref_rows: List[Dict[str, object]] = []
    for group_key in sorted(methods_by_group):
        methods = sorted(set(methods_by_group[group_key]))
        if reference not in methods:
            continue
        ref_rec = records[(group_key, reference)]
        if ref_rec.n_raw < args.min_ref_objects:
            continue

        for method in methods:
            if method == reference:
                continue
            rec = records[(group_key, method)]
            if rec.n_raw < args.min_method_objects:
                continue
            comp = compare_heatmaps(ref_rec, rec, args.sliced_wasserstein_projections, args.random_seed)
            row = record_to_dict(comp)
            pairwise_rows.append(row)
            vs_ref_rows.append(row)
            if args.plot_mode in {"reference", "pairwise_all"} and plot_counter[0] < args.max_comparison_plots:
                png_name = sanitize_filename(f"{grouping}__{group_key}__{method}_minus_{reference}__grid_{grid}__sigma_{sigma:g}__aug_{args.augmentation}.png")
                plot_comparison(ref_rec, rec, comp, setting_dir / "comparisons_vs_reference" / png_name, args)
                plot_counter[0] += 1

        if args.plot_mode == "pairwise_all":
            eligible = [m for m in methods if records[(group_key, m)].n_raw >= args.min_objects_for_pairwise]
            for method_a, method_b in itertools.combinations(eligible, 2):
                if method_a == reference or method_b == reference:
                    continue
                rec_a = records[(group_key, method_a)]
                rec_b = records[(group_key, method_b)]
                comp = compare_heatmaps(rec_a, rec_b, args.sliced_wasserstein_projections, args.random_seed)
                pairwise_rows.append(record_to_dict(comp))
                if plot_counter[0] < args.max_comparison_plots:
                    png_name = sanitize_filename(f"{grouping}__{group_key}__{method_b}_minus_{method_a}__grid_{grid}__sigma_{sigma:g}__aug_{args.augmentation}.png")
                    plot_comparison(rec_a, rec_b, comp, setting_dir / "comparisons_pairwise" / png_name, args)
                    plot_counter[0] += 1

    pairwise_df = pd.DataFrame(pairwise_rows)
    vs_ref_df = pd.DataFrame(vs_ref_rows)
    tables_dir = setting_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pairwise_df.to_csv(tables_dir / f"heatmap_pairwise_metrics_{grouping}_grid_{grid}_sigma_{sigma:g}_aug_{args.augmentation}.csv", index=False)
    vs_ref_df.to_csv(tables_dir / f"heatmap_vs_{reference}_metrics_{grouping}_grid_{grid}_sigma_{sigma:g}_aug_{args.augmentation}.csv", index=False)

    summary_df = summarize_vs_reference(pairwise_df, reference)
    if not summary_df.empty:
        summary_df.insert(0, "augmentation", args.augmentation)
        summary_df.insert(0, "sigma", sigma)
        summary_df.insert(0, "grid", grid)
        summary_df.insert(0, "grouping", grouping)
    summary_df.to_csv(tables_dir / f"method_summary_vs_{reference}_{grouping}_grid_{grid}_sigma_{sigma:g}_aug_{args.augmentation}.csv", index=False)

    dashboard_dir = setting_dir / "dashboard"
    if not summary_df.empty:
        plot_metric_bar(summary_df, "weighted_tv_distance", f"Weighted TV distance to {reference} | {grouping}", dashboard_dir / "ranking_weighted_tv_distance.png", True, args.top_k_dashboard, args.dpi)
        plot_metric_bar(summary_df, "weighted_js_distance", f"Weighted JS distance to {reference} | {grouping}", dashboard_dir / "ranking_weighted_js_distance.png", True, args.top_k_dashboard, args.dpi)
        plot_metric_bar(summary_df, "weighted_sliced_wasserstein", f"Weighted sliced Wasserstein to {reference} | {grouping}", dashboard_dir / "ranking_weighted_sliced_wasserstein.png", True, args.top_k_dashboard, args.dpi)
        plot_metric_bar(summary_df, "coverage_ref_weight", f"Reference coverage by method | {grouping}", dashboard_dir / "ranking_coverage_ref_weight.png", False, args.top_k_dashboard, args.dpi)

    if not vs_ref_df.empty:
        plot_group_distance_heatmap(vs_ref_df, "tv_distance", dashboard_dir / "per_category_tv_distance_to_reference.png", args.dpi)
        plot_group_distance_heatmap(vs_ref_df, "sliced_wasserstein", dashboard_dir / "per_category_sliced_wasserstein_to_reference.png", args.dpi)

    return pairwise_df, summary_df


def write_global_outputs(args: argparse.Namespace, all_pairwise: List[pd.DataFrame], all_summaries: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = args.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pairwise_all = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    summaries_all = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    reference = normalize_token(args.reference)
    pairwise_all.to_csv(tables_dir / "heatmap_pairwise_metrics_all.csv", index=False)
    summaries_all.to_csv(tables_dir / f"method_summary_vs_{reference}_all.csv", index=False)
    if not summaries_all.empty:
        cols = [
            "grouping", "grid", "sigma", "augmentation", "method", "n_groups", "coverage_ref_weight",
            "weighted_tv_distance", "weighted_js_distance", "weighted_sliced_wasserstein", "weighted_centroid_shift",
            "weighted_max_abs_diff_pp", "total_ref_objects_in_compared_groups", "total_method_objects_in_compared_groups"
        ]
        cols = [c for c in cols if c in summaries_all.columns]
        compact = summaries_all[cols].sort_values(["grouping", "grid", "sigma", "weighted_tv_distance"], ascending=[True, True, True, True])
        compact.to_csv(tables_dir / f"method_summary_compact_vs_{reference}.csv", index=False)
    return pairwise_all, summaries_all


def choose_report_setting(args: argparse.Namespace) -> Tuple[int, float, str]:
    report_grid = int(args.report_grid if args.report_grid is not None else max(args.grids))
    report_sigma = float(args.report_sigma if args.report_sigma is not None else max(args.sigmas))
    return report_grid, report_sigma, args.report_grouping


def create_report_assets(args: argparse.Namespace, pairwise_all: pd.DataFrame, summaries_all: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    reference = normalize_token(args.reference)
    report_grid, report_sigma, report_grouping = choose_report_setting(args)
    report_dir = args.out_dir / "report_assets"
    key_fig_dir = report_dir / "key_figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    key_fig_dir.mkdir(parents=True, exist_ok=True)

    if summaries_all.empty:
        return

    summary = summaries_all[
        (summaries_all["grid"] == report_grid)
        & (summaries_all["sigma"] == report_sigma)
        & (summaries_all["grouping"] == report_grouping)
        & (summaries_all["augmentation"] == args.augmentation)
    ].copy()

    report_methods = [normalize_token(m) for m in args.report_methods]
    if not summary.empty:
        summary["method_order"] = summary["method"].map({m: i for i, m in enumerate(report_methods)}).fillna(10000)
        summary = summary.sort_values(["method_order", "weighted_tv_distance"]).drop(columns=["method_order"])
        report_cols = [
            "method", "n_groups", "coverage_ref_weight", "weighted_tv_distance", "weighted_js_distance",
            "weighted_sliced_wasserstein", "weighted_centroid_shift", "weighted_max_abs_diff_pp",
            "total_ref_objects_in_compared_groups", "total_method_objects_in_compared_groups"
        ]
        report_cols = [c for c in report_cols if c in summary.columns]
        summary[report_cols].to_csv(report_dir / "method_summary_for_report.csv", index=False)
        tex_cols = ["method", "n_groups", "coverage_ref_weight", "weighted_tv_distance", "weighted_js_distance", "weighted_sliced_wasserstein", "weighted_centroid_shift", "weighted_max_abs_diff_pp"]
        tex = dataframe_to_latex_table(
            summary[[c for c in tex_cols if c in summary.columns]].head(12),
            tex_cols,
            f"Сравнение распределений расположения объектов с 3D-FRONT по нормализованным тепловым картам (grid={report_grid}, sigma={report_sigma:g}, augmentation={args.augmentation})",
            "tab:layout_heatmap_metrics",
        )
        (report_dir / "vkr_summary_table.tex").write_text(tex, encoding="utf-8")

    vs_ref = pd.DataFrame()
    if not pairwise_all.empty:
        vs_ref = pairwise_all[
            (pairwise_all["method_a"] == reference)
            & (pairwise_all["grid"] == report_grid)
            & (pairwise_all["sigma"] == report_sigma)
            & (pairwise_all["grouping"] == report_grouping)
            & (pairwise_all["augmentation"] == args.augmentation)
        ].copy()
        vs_ref.to_csv(report_dir / "per_category_metrics_for_report.csv", index=False)
        if not vs_ref.empty:
            vs_ref.sort_values(["method_b", "tv_distance"], ascending=[True, False]).to_csv(report_dir / "per_category_deviations_sorted.csv", index=False)
            plot_group_distance_heatmap(vs_ref, "tv_distance", report_dir / "per_category_tv_distance_matrix.png", args.dpi)
            plot_group_distance_heatmap(vs_ref, "sliced_wasserstein", report_dir / "per_category_sliced_wasserstein_matrix.png", args.dpi)

    source_fig_dir = args.out_dir / f"grouping_{report_grouping}" / f"grid_{report_grid}" / f"sigma_{report_sigma:g}" / f"augmentation_{args.augmentation}" / "comparisons_vs_reference"
    key_rows = []
    if source_fig_dir.exists():
        key_methods = [normalize_token(m) for m in args.key_figure_methods]
        key_classes = [canonical_class_name(c) if args.class_mode == "canonical" else normalize_token(c) for c in args.key_figure_classes]
        for method in key_methods:
            for class_name in key_classes:
                if report_grouping == "class":
                    group_key = f"__all____{class_name}"
                    pattern = sanitize_filename(f"{report_grouping}__{group_key}__{method}_minus_{reference}__grid_{report_grid}__sigma_{report_sigma:g}__aug_{args.augmentation}") + "*.png"
                else:
                    pattern = f"*__{class_name}__{method}_minus_{reference}__grid_{report_grid}__sigma_{report_sigma:g}__aug_{args.augmentation}.png"
                matches = sorted(source_fig_dir.glob(pattern))
                if not matches:
                    continue
                src = matches[0]
                dst = key_fig_dir / src.name
                shutil.copy2(src, dst)
                key_rows.append({
                    "method": method,
                    "class_name": class_name,
                    "grouping": report_grouping,
                    "grid": report_grid,
                    "sigma": report_sigma,
                    "augmentation": args.augmentation,
                    "figure_path": str(dst.relative_to(args.out_dir)),
                })
    pd.DataFrame(key_rows).to_csv(report_dir / "key_figures_index.csv", index=False)

    method_counts = df_filtered.groupby("method").size().reset_index(name="trackable_objects").sort_values("trackable_objects", ascending=False)
    method_counts.to_csv(report_dir / "filtered_method_counts.csv", index=False)

    lines = [
        "# Готовые материалы для ВКР и презентации",
        "",
        "## Используемая постановка",
        "",
        f"Сравнение выполнено относительно `{reference}` по нормализованным тепловым картам центров объектов. Основная конфигурация: grouping=`{report_grouping}`, grid=`{report_grid}`, sigma=`{report_sigma:g}`, augmentation=`{args.augmentation}`.",
        "",
        "## Зачем нужна аугментация",
        "",
        "Глобальная ориентация комнаты в датасете не является содержательной характеристикой планировки. Одна и та же сцена может быть повернута на 90, 180 или 270 градусов либо отражена по горизонтали/вертикали, при этом функциональная структура размещения объектов не меняется. Поэтому к 3D-FRONT и ко всем сравниваемым методам применяется одинаковая аугментация. Это снижает зависимость метрики от системы координат и делает сравнение более честным: оценивается не абсолютная ориентация, а политика размещения объектов относительно стен, углов и центральной зоны комнаты.",
        "",
        "## Основная таблица",
        "",
    ]
    if summary.empty:
        lines.append("Нет данных для выбранной report-конфигурации.")
    else:
        cols = ["method", "n_groups", "coverage_ref_weight", "weighted_tv_distance", "weighted_js_distance", "weighted_sliced_wasserstein", "weighted_centroid_shift", "weighted_max_abs_diff_pp"]
        lines.append(markdown_table(summary[[c for c in cols if c in summary.columns]], max_rows=20, float_digits=4))
    lines.extend([
        "",
        "## Файлы для вставки",
        "",
        "- `report_assets/method_summary_for_report.csv` — числовая таблица для отчёта.",
        "- `report_assets/vkr_summary_table.tex` — LaTeX-таблица для текста ВКР.",
        "- `report_assets/per_category_metrics_for_report.csv` — метрики по каждой категории.",
        "- `report_assets/per_category_tv_distance_matrix.png` — матрица TV distance по категориям.",
        "- `report_assets/per_category_sliced_wasserstein_matrix.png` — матрица sliced Wasserstein по категориям.",
        "- `report_assets/key_figures/` — выбранные heatmap-diff картинки для презентации.",
        "",
        "## Формулировка для ВКР",
        "",
        "Для оценки статистической близости планировок к 3D-FRONT были построены нормализованные тепловые карты расположения центров объектов. Для каждой категории объектов сумма значений тепловой карты равна 100%, поэтому сравнение отражает форму пространственного распределения, а не различия в числе объектов. Для уменьшения влияния дискретизации применялось гауссово размытие с последующей повторной нормализацией. Дополнительно использовалась ориентационная аугментация: горизонтальное и вертикальное отражение, а также повороты на 90, 180 и 270 градусов. Это необходимо, поскольку абсолютная ориентация комнаты в датасете произвольна и не должна ухудшать оценку семантически эквивалентной планировки. Близость методов оценивалась с помощью total variation distance, Jensen-Shannon distance и sliced Wasserstein distance.",
    ])
    (report_dir / "vkr_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(args: argparse.Namespace, df_filtered: pd.DataFrame, summaries_all: pd.DataFrame, vs_ref_all: pd.DataFrame) -> None:
    report_path = args.out_dir / "layout_heatmap_diff_report.md"
    method_counts = df_filtered.groupby("method").size().reset_index(name="trackable_objects").sort_values("trackable_objects", ascending=False)
    class_counts = df_filtered.groupby("class_name_norm").size().reset_index(name="trackable_objects").sort_values("trackable_objects", ascending=False)
    room_counts = df_filtered.groupby("room_type_norm").size().reset_index(name="trackable_objects").sort_values("trackable_objects", ascending=False)

    lines = [
        "# Layout heatmap difference analysis",
        "",
        "## Settings",
        "",
        "```json",
        json.dumps({
            "objects_csv": str(args.objects_csv),
            "out_dir": str(args.out_dir),
            "reference": args.reference,
            "classes": args.classes,
            "class_mode": args.class_mode,
            "grouping": args.grouping,
            "grids": args.grids,
            "sigmas": args.sigmas,
            "augmentation": args.augmentation,
            "min_ref_objects": args.min_ref_objects,
            "min_method_objects": args.min_method_objects,
            "plot_mode": args.plot_mode,
        }, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Filtered data counts",
        "",
        f"Filtered object rows: **{len(df_filtered)}**",
        "",
        "### Methods",
        markdown_table(method_counts, max_rows=50),
        "### Classes",
        markdown_table(class_counts, max_rows=70),
        "### Rooms",
        markdown_table(room_counts, max_rows=50),
    ]

    if not summaries_all.empty:
        compact_cols = ["grouping", "grid", "sigma", "augmentation", "method", "n_groups", "coverage_ref_weight", "weighted_tv_distance", "weighted_js_distance", "weighted_sliced_wasserstein", "weighted_centroid_shift", "weighted_max_abs_diff_pp"]
        compact_cols = [c for c in compact_cols if c in summaries_all.columns]
        compact = summaries_all[compact_cols].sort_values(["grouping", "grid", "sigma", "weighted_tv_distance"], ascending=[True, True, True, True])
        lines.extend(["## Method ranking against reference", markdown_table(compact, max_rows=120)])

    if not vs_ref_all.empty:
        cols = ["grouping", "grid", "sigma", "augmentation", "room_type", "class_name", "method_b", "n_a", "n_b", "tv_distance", "js_distance", "sliced_wasserstein", "max_abs_diff_pp", "centroid_shift"]
        cols = [c for c in cols if c in vs_ref_all.columns]
        dev = vs_ref_all[cols].sort_values(["tv_distance", "js_distance"], ascending=[False, False])
        lines.extend(["## Strongest per-category deviations from reference", markdown_table(dev, max_rows=120)])

    lines.extend([
        "## Explanation for thesis text",
        "",
        "The comparison is performed by normalized heatmaps of object centers. For each method and category, the sum of the heatmap equals 100%, so the analysis measures spatial distribution shape rather than absolute object count.",
        "",
        "Gaussian smoothing is applied after histogram construction and followed by renormalization to 100%. Smoothing reduces sensitivity to grid-cell boundaries.",
        "",
        "Orientation augmentation is applied equally to 3D-FRONT and to every compared method. The transforms include vertical and horizontal reflection and rotations by 90, 180 and 270 degrees. This is necessary because absolute room orientation is usually arbitrary; a valid layout may be rotated or mirrored without changing semantic quality.",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.reference = normalize_token(args.reference)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_and_filter_objects(args)
    if df.empty:
        raise RuntimeError("No rows left after filtering.")
    if args.reference not in set(df["method"]):
        available = sorted(df["method"].dropna().unique().tolist())
        raise RuntimeError(f"Reference method '{args.reference}' not found. Available methods: {available}")

    tables_dir = args.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "objects_filtered_for_heatmap_analysis.csv", index=False)

    groupings = ["class", "room_class"] if args.grouping == "both" else [args.grouping]
    all_pairwise: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []
    plot_counter = [0]

    for grouping in groupings:
        for grid in args.grids:
            if grid <= 1:
                raise ValueError(f"Grid must be > 1, got {grid}")
            for sigma in args.sigmas:
                pairwise_df, summary_df = run_one_setting(df, grouping, int(grid), float(sigma), args, plot_counter)
                if not pairwise_df.empty:
                    all_pairwise.append(pairwise_df)
                if not summary_df.empty:
                    all_summaries.append(summary_df)

    pairwise_all, summaries_all = write_global_outputs(args, all_pairwise, all_summaries)
    vs_ref_all = pd.DataFrame()
    if not pairwise_all.empty:
        vs_ref_all = pairwise_all[pairwise_all["method_a"] == args.reference].copy()
        vs_ref_all.to_csv(tables_dir / f"heatmap_vs_{args.reference}_metrics_all.csv", index=False)

    write_report(args, df, summaries_all, vs_ref_all)
    create_report_assets(args, pairwise_all, summaries_all, df)

    print("[done] heatmap difference analysis")
    print(f"[out] {args.out_dir}")
    print(f"[filtered] {len(df)} rows")
    print(f"[augmentation] {args.augmentation}: {[name for name, _ in augmentation_transforms(args.augmentation)]}")
    print(f"[plots] {plot_counter[0]} comparison PNGs")
    print(f"[report] {args.out_dir / 'layout_heatmap_diff_report.md'}")
    print(f"[report_assets] {args.out_dir / 'report_assets'}")
    print(f"[summary] {args.out_dir / 'tables' / f'method_summary_compact_vs_{args.reference}.csv'}")


if __name__ == "__main__":
    main()
