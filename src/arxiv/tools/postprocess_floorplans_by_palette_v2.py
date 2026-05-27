#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_floorplans_by_palette_v2.py

Постобработка crop-кандидатов планировок.

Проблема v1:
    QR-код почти идеально удовлетворяет правилу "white top-1, black top-2".
    Настоящая планировка часто имеет не чисто black, а dark_gray/gray из-за JPEG,
    поэтому строгий top2 == black выбрасывает планы и оставляет QR.

Что делает v2:
    1. Не требует строго top2 == black.
       Вместо этого считает dark_neutral = black + dark_gray.

    2. Отбрасывает QR-коды процедурно:
       - почти квадратная область;
       - много мелких connected components;
       - высокая плотность чёрных модулей по сетке;
       - мало длинных архитектурных H/V-линий.

    3. Требует признаки планировки:
       - белый/почти белый фон;
       - низкая доля цветных пикселей;
       - есть тёмные линии;
       - есть горизонтальные и вертикальные линии;
       - есть длинные сегменты;
       - тёмные пиксели распределены не только как QR/логотип.

Пример запуска:

    python3 src/tools/postprocess_floorplans_by_palette_v2.py \
      --input data/housesru/floorplans_score7_all_pages_parallel/floorplans \
      --out data/housesru/floorplans_score7_all_pages_palette_v2 \
      --mode symlink \
      --preset balanced

Более строгий режим:

    python3 src/tools/postprocess_floorplans_by_palette_v2.py \
      --input data/housesru/floorplans_score7_all_pages_parallel/floorplans \
      --out data/housesru/floorplans_score7_all_pages_palette_v2_strict \
      --mode symlink \
      --preset strict

Диагностика:
    results.jsonl содержит все признаки и причины reject.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


CLASS_NAMES = {
    0: "white",
    1: "black",
    2: "light_gray",
    3: "gray",
    4: "dark_gray",
    5: "red",
    6: "orange",
    7: "yellow",
    8: "green",
    9: "cyan",
    10: "blue",
    11: "purple",
    12: "pink",
    13: "brown",
}


COLOR_CLASSES = {
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
}


@dataclass
class Thresholds:
    white_min: float = 0.45
    dark_neutral_min: float = 0.020
    color_max: float = 0.18
    gray_max: float = 0.55

    min_h_lines: int = 2
    min_v_lines: int = 2
    min_hv_balance: float = 0.18
    min_max_line_length_ratio: float = 0.12
    min_grid_occupancy: float = 0.08
    min_interior_dark_ratio: float = 0.004

    qr_score_max: float = 0.58
    text_score_max: float = 0.88
    frame_score_max: float = 0.82

    min_accept_score: float = 0.90


def thresholds_from_preset(preset: str) -> Thresholds:
    if preset == "recall":
        return Thresholds(
            white_min=0.35,
            dark_neutral_min=0.012,
            color_max=0.25,
            gray_max=0.65,
            min_h_lines=1,
            min_v_lines=1,
            min_hv_balance=0.08,
            min_max_line_length_ratio=0.08,
            min_grid_occupancy=0.05,
            min_interior_dark_ratio=0.002,
            qr_score_max=0.70,
            text_score_max=0.94,
            frame_score_max=0.88,
            min_accept_score=0.55,
        )

    if preset == "strict":
        return Thresholds(
            white_min=0.52,
            dark_neutral_min=0.030,
            color_max=0.10,
            gray_max=0.45,
            min_h_lines=3,
            min_v_lines=3,
            min_hv_balance=0.25,
            min_max_line_length_ratio=0.15,
            min_grid_occupancy=0.10,
            min_interior_dark_ratio=0.006,
            qr_score_max=0.45,
            text_score_max=0.78,
            frame_score_max=0.70,
            min_accept_score=1.15,
        )

    if preset == "balanced":
        return Thresholds()

    raise ValueError(f"unknown preset: {preset}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def rgb_to_hsv_np(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn

    v = mx
    s = np.where(mx == 0, 0.0, diff / np.maximum(mx, 1e-8))

    h = np.zeros_like(mx)

    mask = diff > 1e-8
    rmask = mask & (mx == r)
    gmask = mask & (mx == g)
    bmask = mask & (mx == b)

    h[rmask] = ((g[rmask] - b[rmask]) / diff[rmask]) % 6.0
    h[gmask] = ((b[gmask] - r[gmask]) / diff[gmask]) + 2.0
    h[bmask] = ((r[bmask] - g[bmask]) / diff[bmask]) + 4.0
    h *= 60.0
    h %= 360.0

    return h, s, v


def classify_pixels(rgb: np.ndarray) -> np.ndarray:
    h, s, v = rgb_to_hsv_np(rgb)
    labels = np.full(h.shape, fill_value=3, dtype=np.int16)

    white_mask = (v >= 0.92) & (s <= 0.13)
    black_mask = v <= 0.16

    gray_mask = (s <= 0.16) & (~white_mask) & (~black_mask)

    labels[white_mask] = 0
    labels[black_mask] = 1
    labels[gray_mask & (v >= 0.72)] = 2
    labels[gray_mask & (v >= 0.32) & (v < 0.72)] = 3
    labels[gray_mask & (v < 0.32)] = 4

    color_mask = ~(white_mask | black_mask | gray_mask)

    brown_mask = color_mask & (h >= 15) & (h < 45) & (v < 0.65)
    labels[brown_mask] = 13

    labels[color_mask & (((h >= 0) & (h < 15)) | ((h >= 345) & (h < 360)))] = 5
    labels[color_mask & (h >= 15) & (h < 45) & (~brown_mask)] = 6
    labels[color_mask & (h >= 45) & (h < 75)] = 7
    labels[color_mask & (h >= 75) & (h < 165)] = 8
    labels[color_mask & (h >= 165) & (h < 200)] = 9
    labels[color_mask & (h >= 200) & (h < 260)] = 10
    labels[color_mask & (h >= 260) & (h < 320)] = 11
    labels[color_mask & (h >= 320) & (h < 345)] = 12

    return labels


def shares_from_labels(labels: np.ndarray) -> dict[str, float]:
    flat = labels.reshape(-1)
    total = max(1, flat.size)
    counter = Counter(flat.tolist())
    return {CLASS_NAMES[k]: counter.get(k, 0) / total for k in CLASS_NAMES}


def top_classes(shares: dict[str, float], n: int = 5) -> list[dict[str, Any]]:
    items = sorted(shares.items(), key=lambda x: (-x[1], x[0]))
    return [{"name": name, "share": float(share)} for name, share in items[:n]]


def build_dark_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    # В планировках линии часто не чисто black, а dark gray.
    dark = ((gray <= 185) & (sat <= 190)).astype(np.uint8) * 255

    # Убираем слишком светлый шум.
    dark[gray >= 210] = 0
    return dark


def grid_occupancy(mask: np.ndarray, grid: int = 8, min_cell_dark_ratio: float = 0.010) -> float:
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    occupied = 0
    total = 0

    for gy in range(grid):
        y0 = int(round(gy * h / grid))
        y1 = int(round((gy + 1) * h / grid))
        for gx in range(grid):
            x0 = int(round(gx * w / grid))
            x1 = int(round((gx + 1) * w / grid))
            cell = mask[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            total += 1
            if np.count_nonzero(cell) / cell.size >= min_cell_dark_ratio:
                occupied += 1

    return occupied / max(1, total)


def border_interior_stats(mask: np.ndarray) -> tuple[float, float, float]:
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, 0.0, 0.0

    border_width = max(3, int(round(min(h, w) * 0.08)))

    border = np.zeros((h, w), dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    interior = ~border

    border_area = max(1, int(np.count_nonzero(border)))
    interior_area = max(1, int(np.count_nonzero(interior)))

    border_dark = float(np.count_nonzero(mask[border]) / border_area)
    interior_dark = float(np.count_nonzero(mask[interior]) / interior_area)
    dominance = border_dark / max(1e-6, interior_dark)

    return interior_dark, border_dark, float(dominance)


def connected_component_stats(mask: np.ndarray) -> dict[str, float | int]:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    area = max(1, h * w)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num_labels <= 1:
        return {
            "component_count": 0,
            "largest_cc_ratio": 0.0,
            "small_component_ratio": 0.0,
            "component_density": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    component_count = int(len(areas))
    largest_cc_ratio = float(np.max(areas) / area)

    small_threshold = max(4.0, area * 0.0008)
    small_component_ratio = float(np.mean(areas <= small_threshold))
    component_density = float(min(1.0, component_count / max(1.0, area / 900.0)))

    return {
        "component_count": component_count,
        "largest_cc_ratio": largest_cc_ratio,
        "small_component_ratio": small_component_ratio,
        "component_density": component_density,
    }


def hough_line_stats(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.bitwise_and(edges, mask)

    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return {
            "line_count": 0,
            "h_lines": 0,
            "v_lines": 0,
            "diag_lines": 0,
            "orthogonal_ratio": 0.0,
            "hv_balance": 0.0,
            "max_line_length_ratio": 0.0,
        }

    min_line_length = max(14, int(min(h, w) * 0.12))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=18,
        minLineLength=min_line_length,
        maxLineGap=max(4, int(min(h, w) * 0.025)),
    )

    if lines is None:
        return {
            "line_count": 0,
            "h_lines": 0,
            "v_lines": 0,
            "diag_lines": 0,
            "orthogonal_ratio": 0.0,
            "hv_balance": 0.0,
            "max_line_length_ratio": 0.0,
        }

    line_count = 0
    h_lines = 0
    v_lines = 0
    diag_lines = 0
    max_len = 0.0
    diag_norm = max(1.0, math.hypot(w, h))

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_line_length:
            continue

        line_count += 1
        max_len = max(max_len, length)

        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        d0 = min(abs(angle), abs(angle - 180.0))
        d90 = abs(angle - 90.0)

        if d0 <= 12.0:
            h_lines += 1
        elif d90 <= 12.0:
            v_lines += 1
        else:
            diag_lines += 1

    orth = h_lines + v_lines
    orthogonal_ratio = orth / max(1, line_count)
    hv_balance = min(h_lines, v_lines) / max(1, max(h_lines, v_lines))
    max_line_length_ratio = max_len / diag_norm

    return {
        "line_count": int(line_count),
        "h_lines": int(h_lines),
        "v_lines": int(v_lines),
        "diag_lines": int(diag_lines),
        "orthogonal_ratio": float(orthogonal_ratio),
        "hv_balance": float(hv_balance),
        "max_line_length_ratio": float(max_line_length_ratio),
    }


def estimate_qr_like(
    aspect_ratio: float,
    dark_neutral_share: float,
    grid_occ: float,
    cc: dict[str, float | int],
    lines: dict[str, float | int],
) -> float:
    square_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) / 0.35)
    dense_score = min(1.0, 0.5 * float(cc["small_component_ratio"]) + 0.5 * float(cc["component_density"]))
    dark_score = min(1.0, dark_neutral_share / 0.18)
    no_long_lines = max(0.0, 1.0 - float(lines["max_line_length_ratio"]) / 0.28)
    balanced_modules = min(1.0, grid_occ / 0.45)

    score = (
        0.26 * square_score
        + 0.28 * dense_score
        + 0.20 * dark_score
        + 0.18 * no_long_lines
        + 0.08 * balanced_modules
    )

    # Настоящий квадратный план может иметь длинные H/V линии.
    if int(lines["h_lines"]) >= 3 and int(lines["v_lines"]) >= 3 and float(lines["max_line_length_ratio"]) >= 0.25:
        score *= 0.55

    return float(min(1.0, max(0.0, score)))


def estimate_text_like(
    gray_share: float,
    cc: dict[str, float | int],
    lines: dict[str, float | int],
) -> float:
    h_lines = int(lines["h_lines"])
    v_lines = int(lines["v_lines"])
    horizontal_dominance = h_lines / max(1, h_lines + v_lines)
    no_vertical = 1.0 - min(1.0, v_lines / 3.0)

    score = (
        0.28 * float(cc["small_component_ratio"])
        + 0.20 * float(cc["component_density"])
        + 0.22 * horizontal_dominance
        + 0.18 * no_vertical
        + 0.12 * min(1.0, gray_share / 0.45)
    )

    if v_lines >= 3 and float(lines["hv_balance"]) >= 0.25:
        score *= 0.65

    return float(min(1.0, max(0.0, score)))


def estimate_frame_like(
    grid_occ: float,
    interior_dark: float,
    border_dom: float,
    lines: dict[str, float | int],
) -> float:
    sparse_grid = max(0.0, 1.0 - grid_occ / 0.20)
    weak_interior = max(0.0, 1.0 - interior_dark / 0.020)
    border_heavy = min(1.0, border_dom / 5.0)
    few_lines = max(0.0, 1.0 - (int(lines["h_lines"]) + int(lines["v_lines"])) / 8.0)

    score = 0.30 * sparse_grid + 0.30 * weak_interior + 0.25 * border_heavy + 0.15 * few_lines
    return float(min(1.0, max(0.0, score)))


def analyze_image(path: Path, max_side: int) -> dict[str, Any]:
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    rgb = np.array(img)

    labels = classify_pixels(rgb)
    shares = shares_from_labels(labels)
    top5 = top_classes(shares, n=5)

    white_share = shares["white"]
    black_share = shares["black"]
    light_gray_share = shares["light_gray"]
    gray_share = shares["gray"]
    dark_gray_share = shares["dark_gray"]
    neutral_gray_share = light_gray_share + gray_share + dark_gray_share
    dark_neutral_share = black_share + dark_gray_share
    color_share = sum(shares[name] for name in COLOR_CLASSES)

    dark_mask = build_dark_mask(rgb)
    grid_occ = grid_occupancy(dark_mask)
    interior_dark, border_dark, border_dom = border_interior_stats(dark_mask)
    cc = connected_component_stats(dark_mask)
    lines = hough_line_stats(rgb, dark_mask)

    aspect_ratio = float(rgb.shape[1] / max(1, rgb.shape[0]))

    qr_like = estimate_qr_like(aspect_ratio, dark_neutral_share, grid_occ, cc, lines)
    text_like = estimate_text_like(neutral_gray_share, cc, lines)
    frame_like = estimate_frame_like(grid_occ, interior_dark, border_dom, lines)

    plan_structure_score = (
        0.35 * min(int(lines["h_lines"]) / 4.0, 1.0)
        + 0.35 * min(int(lines["v_lines"]) / 4.0, 1.0)
        + 0.25 * min(float(lines["max_line_length_ratio"]) / 0.30, 1.0)
        + 0.25 * float(lines["hv_balance"])
        + 0.20 * min(grid_occ / 0.35, 1.0)
        + 0.20 * min(interior_dark / 0.05, 1.0)
    )

    palette_score = (
        0.60 * min(white_share / 0.60, 1.0)
        + 0.40 * min(dark_neutral_share / 0.08, 1.0)
        - 0.35 * min(color_share / 0.20, 1.0)
        - 0.15 * min(neutral_gray_share / 0.60, 1.0)
    )

    accept_score = (
        palette_score
        + plan_structure_score
        - 0.90 * qr_like
        - 0.50 * text_like
        - 0.45 * frame_like
    )

    return {
        "image_path": str(path),
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "analysis_width": int(rgb.shape[1]),
        "analysis_height": int(rgb.shape[0]),
        "aspect_ratio": aspect_ratio,
        "shares": shares,
        "top5": top5,
        "white_share": float(white_share),
        "black_share": float(black_share),
        "light_gray_share": float(light_gray_share),
        "gray_share": float(gray_share),
        "dark_gray_share": float(dark_gray_share),
        "neutral_gray_share": float(neutral_gray_share),
        "dark_neutral_share": float(dark_neutral_share),
        "color_share": float(color_share),
        "grid_occupancy": float(grid_occ),
        "interior_dark_ratio": float(interior_dark),
        "border_dark_ratio": float(border_dark),
        "border_dominance": float(border_dom),
        **{k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in cc.items()},
        **{k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in lines.items()},
        "qr_like_score": float(qr_like),
        "text_like_score": float(text_like),
        "frame_like_score": float(frame_like),
        "palette_score": float(palette_score),
        "plan_structure_score": float(plan_structure_score),
        "accept_score": float(accept_score),
    }


def decide(info: dict[str, Any], th: Thresholds) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if info["white_share"] < th.white_min:
        reasons.append(f"white_low:{info['white_share']:.4f}<{th.white_min:.4f}")

    if info["dark_neutral_share"] < th.dark_neutral_min:
        reasons.append(f"dark_neutral_low:{info['dark_neutral_share']:.4f}<{th.dark_neutral_min:.4f}")

    if info["color_share"] > th.color_max:
        reasons.append(f"color_high:{info['color_share']:.4f}>{th.color_max:.4f}")

    if info["neutral_gray_share"] > th.gray_max:
        reasons.append(f"gray_high:{info['neutral_gray_share']:.4f}>{th.gray_max:.4f}")

    if info["h_lines"] < th.min_h_lines:
        reasons.append(f"h_lines_low:{info['h_lines']}<{th.min_h_lines}")

    if info["v_lines"] < th.min_v_lines:
        reasons.append(f"v_lines_low:{info['v_lines']}<{th.min_v_lines}")

    if info["hv_balance"] < th.min_hv_balance:
        reasons.append(f"hv_balance_low:{info['hv_balance']:.4f}<{th.min_hv_balance:.4f}")

    if info["max_line_length_ratio"] < th.min_max_line_length_ratio:
        reasons.append(f"max_line_low:{info['max_line_length_ratio']:.4f}<{th.min_max_line_length_ratio:.4f}")

    if info["grid_occupancy"] < th.min_grid_occupancy:
        reasons.append(f"grid_low:{info['grid_occupancy']:.4f}<{th.min_grid_occupancy:.4f}")

    if info["interior_dark_ratio"] < th.min_interior_dark_ratio:
        reasons.append(f"interior_dark_low:{info['interior_dark_ratio']:.4f}<{th.min_interior_dark_ratio:.4f}")

    if info["qr_like_score"] > th.qr_score_max:
        reasons.append(f"qr_like_high:{info['qr_like_score']:.4f}>{th.qr_score_max:.4f}")

    if info["text_like_score"] > th.text_score_max:
        reasons.append(f"text_like_high:{info['text_like_score']:.4f}>{th.text_score_max:.4f}")

    if info["frame_like_score"] > th.frame_score_max:
        reasons.append(f"frame_like_high:{info['frame_like_score']:.4f}>{th.frame_score_max:.4f}")

    if info["accept_score"] < th.min_accept_score:
        reasons.append(f"accept_score_low:{info['accept_score']:.4f}<{th.min_accept_score:.4f}")

    return len(reasons) == 0, reasons


def materialize(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(mode)


def apply_overrides(th: Thresholds, args: argparse.Namespace) -> Thresholds:
    data = asdict(th)

    for name in data:
        value = getattr(args, name, None)
        if value is not None:
            data[name] = value

    return Thresholds(**data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Palette + structure postprocessing for floorplan crops.")

    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["copy", "symlink", "move"], default="symlink")
    p.add_argument("--preset", choices=["recall", "balanced", "strict"], default="balanced")
    p.add_argument("--max-side", type=int, default=768)

    # Optional threshold overrides.
    p.add_argument("--white-min", dest="white_min", type=float)
    p.add_argument("--dark-neutral-min", dest="dark_neutral_min", type=float)
    p.add_argument("--color-max", dest="color_max", type=float)
    p.add_argument("--gray-max", dest="gray_max", type=float)
    p.add_argument("--min-h-lines", dest="min_h_lines", type=int)
    p.add_argument("--min-v-lines", dest="min_v_lines", type=int)
    p.add_argument("--min-hv-balance", dest="min_hv_balance", type=float)
    p.add_argument("--min-max-line-length-ratio", dest="min_max_line_length_ratio", type=float)
    p.add_argument("--min-grid-occupancy", dest="min_grid_occupancy", type=float)
    p.add_argument("--min-interior-dark-ratio", dest="min_interior_dark_ratio", type=float)
    p.add_argument("--qr-score-max", dest="qr_score_max", type=float)
    p.add_argument("--text-score-max", dest="text_score_max", type=float)
    p.add_argument("--frame-score-max", dest="frame_score_max", type=float)
    p.add_argument("--min-accept-score", dest="min_accept_score", type=float)

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    th = apply_overrides(thresholds_from_preset(args.preset), args)

    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    ensure_dir(accepted_dir)
    ensure_dir(rejected_dir)

    images = iter_images(input_dir)

    results: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0

    for idx, src in enumerate(images, start=1):
        try:
            info = analyze_image(src, max_side=args.max_side)
            ok, reasons = decide(info, th)
        except Exception as exc:
            ok = False
            reasons = [f"analysis_error:{type(exc).__name__}:{exc}"]
            info = {"image_path": str(src)}

        dst = (accepted_dir if ok else rejected_dir) / src.name
        materialize(src, dst, args.mode)

        row = {
            "src": str(src),
            "dst": str(dst),
            "accepted": ok,
            "reasons": reasons,
            **info,
        }
        results.append(row)

        if ok:
            accepted_count += 1
        else:
            rejected_count += 1

        if idx % 200 == 0:
            print(f"processed={idx}/{len(images)} accepted={accepted_count} rejected={rejected_count}", flush=True)

    manifest = {
        "schema": "floorplan_palette_structure_postprocess/v2",
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "mode": args.mode,
        "preset": args.preset,
        "thresholds": asdict(th),
        "total_images": len(images),
        "accepted_images": accepted_count,
        "rejected_images": rejected_count,
        "accepted_dir": str(accepted_dir),
        "rejected_dir": str(rejected_dir),
    }

    ensure_dir(out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("input:", input_dir)
    print("out:", out_dir)
    print("preset:", args.preset)
    print("thresholds:", json.dumps(asdict(th), ensure_ascii=False))
    print("total:", len(images))
    print("accepted:", accepted_count)
    print("rejected:", rejected_count)
    print("accepted_dir:", accepted_dir)
    print("rejected_dir:", rejected_dir)
    print("manifest:", out_dir / "manifest.json")
    print("results:", out_dir / "results.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
