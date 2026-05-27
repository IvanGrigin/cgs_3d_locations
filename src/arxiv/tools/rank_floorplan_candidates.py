#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rank_floorplan_candidates.py

Ранжирование crop-кандидатов планировок по структурным признакам.

Зачем нужен файл
================
Цветовой фильтр плохо отделяет планировки от QR/портретов/текста:
- QR почти всегда white+black;
- старые планировки часто имеют серые линии из-за JPEG;
- фотографии могут иметь белые области.

Поэтому этот скрипт не делит сразу на accepted/rejected по палитре.
Он считает floorplan_score и сортирует кандидаты. Затем сохраняет:
- ranked_all/       — все кандидаты с префиксом ранга и score;
- top_k/            — top-K лучших кандидатов;
- hard_rejects/     — явные QR/фото/текст/рамки;
- results.csv       — таблица признаков;
- results.jsonl     — полная диагностика;
- manifest.json     — параметры запуска.

Главные признаки:
- OpenCV QRCodeDetector;
- доля белого фона;
- доля тёмных нейтральных линий;
- горизонтальные/вертикальные длинные линии;
- баланс H/V линий;
- связность тёмных компонент;
- занятость сетки;
- отличие от текста, QR, рамки/окна, цветного фото.

Пример запуска
==============

    python3 src/tools/rank_floorplan_candidates.py \
      --input data/housesru/floorplans_score7_all_pages_parallel/floorplans \
      --out data/housesru/floorplans_ranked_v1 \
      --top-k 800 \
      --mode symlink \
      --preset balanced

Более мягко:

    python3 src/tools/rank_floorplan_candidates.py \
      --input data/housesru/floorplans_score7_all_pages_parallel/floorplans \
      --out data/housesru/floorplans_ranked_v1_recall \
      --top-k 1200 \
      --mode symlink \
      --preset recall

Более строго:

    python3 src/tools/rank_floorplan_candidates.py \
      --input data/housesru/floorplans_score7_all_pages_parallel/floorplans \
      --out data/housesru/floorplans_ranked_v1_strict \
      --top-k 500 \
      --mode symlink \
      --preset strict

После запуска смотри:
    open data/housesru/floorplans_ranked_v1/top_k
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class Preset:
    # hard reject thresholds
    qr_hard_reject: float
    photo_hard_reject: float
    text_hard_reject: float
    frame_hard_reject: float

    # soft scoring thresholds
    color_soft_max: float
    min_white_for_bonus: float
    min_dark_for_bonus: float

    # line scoring thresholds
    h_line_norm: float
    v_line_norm: float
    long_line_norm: float
    grid_norm: float
    interior_norm: float

    # score weights
    min_score_for_top: float


def get_preset(name: str) -> Preset:
    if name == "recall":
        return Preset(
            qr_hard_reject=0.84,
            photo_hard_reject=0.92,
            text_hard_reject=0.94,
            frame_hard_reject=0.95,
            color_soft_max=0.35,
            min_white_for_bonus=0.25,
            min_dark_for_bonus=0.008,
            h_line_norm=3.0,
            v_line_norm=3.0,
            long_line_norm=0.22,
            grid_norm=0.28,
            interior_norm=0.035,
            min_score_for_top=-999.0,
        )

    if name == "strict":
        return Preset(
            qr_hard_reject=0.55,
            photo_hard_reject=0.72,
            text_hard_reject=0.78,
            frame_hard_reject=0.72,
            color_soft_max=0.14,
            min_white_for_bonus=0.45,
            min_dark_for_bonus=0.025,
            h_line_norm=5.0,
            v_line_norm=5.0,
            long_line_norm=0.35,
            grid_norm=0.42,
            interior_norm=0.060,
            min_score_for_top=0.60,
        )

    if name == "balanced":
        return Preset(
            qr_hard_reject=0.68,
            photo_hard_reject=0.82,
            text_hard_reject=0.86,
            frame_hard_reject=0.84,
            color_soft_max=0.22,
            min_white_for_bonus=0.35,
            min_dark_for_bonus=0.015,
            h_line_norm=4.0,
            v_line_norm=4.0,
            long_line_norm=0.28,
            grid_norm=0.35,
            interior_norm=0.050,
            min_score_for_top=0.10,
        )

    raise ValueError(f"unknown preset: {name}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def safe_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def materialize(src: Path, dst: Path, mode: str, overwrite: bool = True) -> None:
    ensure_dir(dst.parent)

    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def load_rgb(path: Path, max_side: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    orig_size = img.size
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    analysis_size = img.size
    rgb = np.array(img)
    return rgb, orig_size, analysis_size


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

    white_mask = (v >= 0.92) & (s <= 0.14)
    black_mask = v <= 0.16
    gray_mask = (s <= 0.17) & (~white_mask) & (~black_mask)

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
    return {CLASS_NAMES[k]: float(counter.get(k, 0) / total) for k in CLASS_NAMES}


def top_classes(shares: dict[str, float], n: int = 5) -> list[dict[str, Any]]:
    items = sorted(shares.items(), key=lambda x: (-x[1], x[0]))
    return [{"name": k, "share": float(v)} for k, v in items[:n]]


def build_dark_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    # Линии плана могут быть dark gray, не чисто black.
    dark = ((gray <= 190) & (sat <= 200)).astype(np.uint8) * 255
    dark[gray >= 215] = 0

    return dark


def edge_density(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.count_nonzero(edges) / max(1, edges.size))


def gradient_photo_score(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    mean = float(np.mean(mag))
    std = float(np.std(mag))

    # Фото часто имеет мягкие градиенты и цветность; планы — резкие линии.
    smooth = 1.0 - min(1.0, std / 70.0)
    moderate_grad = min(1.0, mean / 35.0)
    return float(max(0.0, min(1.0, 0.55 * smooth + 0.45 * moderate_grad)))


def grid_occupancy(mask: np.ndarray, grid: int = 8, min_cell_dark_ratio: float = 0.010) -> float:
    h, w = mask.shape[:2]
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

    return float(occupied / max(1, total))


def border_interior_stats(mask: np.ndarray) -> tuple[float, float, float]:
    h, w = mask.shape[:2]
    bw = max(3, int(round(min(h, w) * 0.08)))

    border = np.zeros((h, w), dtype=bool)
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
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

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num_labels <= 1:
        return {
            "component_count": 0,
            "largest_cc_ratio": 0.0,
            "small_component_ratio": 0.0,
            "component_density": 0.0,
            "closed_largest_cc_ratio": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    component_count = int(len(areas))
    largest_cc_ratio = float(np.max(areas) / area)

    small_threshold = max(4.0, area * 0.0008)
    small_component_ratio = float(np.mean(areas <= small_threshold))
    component_density = float(min(1.0, component_count / max(1.0, area / 900.0)))

    closed = cv2.morphologyEx(
        m,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    num2, _labels2, stats2, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num2 <= 1:
        closed_largest = 0.0
    else:
        closed_largest = float(np.max(stats2[1:, cv2.CC_STAT_AREA]) / area)

    return {
        "component_count": component_count,
        "largest_cc_ratio": largest_cc_ratio,
        "small_component_ratio": small_component_ratio,
        "component_density": component_density,
        "closed_largest_cc_ratio": closed_largest,
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
            "mean_line_length_ratio": 0.0,
        }

    min_line_length = max(12, int(min(h, w) * 0.10))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=16,
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
            "mean_line_length_ratio": 0.0,
        }

    line_count = 0
    h_lines = 0
    v_lines = 0
    diag_lines = 0
    lengths: list[float] = []
    diag_norm = max(1.0, math.hypot(w, h))

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_line_length:
            continue

        line_count += 1
        lengths.append(length)

        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        d0 = min(abs(angle), abs(angle - 180.0))
        d90 = abs(angle - 90.0)

        if d0 <= 12.0:
            h_lines += 1
        elif d90 <= 12.0:
            v_lines += 1
        else:
            diag_lines += 1

    if not lengths:
        max_len_ratio = 0.0
        mean_len_ratio = 0.0
    else:
        max_len_ratio = max(lengths) / diag_norm
        mean_len_ratio = float(np.mean(lengths) / diag_norm)

    orth = h_lines + v_lines
    orthogonal_ratio = orth / max(1, line_count)
    hv_balance = min(h_lines, v_lines) / max(1, max(h_lines, v_lines))

    return {
        "line_count": int(line_count),
        "h_lines": int(h_lines),
        "v_lines": int(v_lines),
        "diag_lines": int(diag_lines),
        "orthogonal_ratio": float(orthogonal_ratio),
        "hv_balance": float(hv_balance),
        "max_line_length_ratio": float(max_len_ratio),
        "mean_line_length_ratio": float(mean_len_ratio),
    }


def qr_detected(rgb: np.ndarray) -> bool:
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        ok, points = detector.detect(bgr)
        if ok and points is not None:
            return True

        # detectAndDecode иногда ловит то, что detect пропускает.
        value, points, _ = detector.detectAndDecode(bgr)
        return bool(value) or points is not None
    except Exception:
        return False


def estimate_qr_like(
    aspect_ratio: float,
    dark_share: float,
    grid_occ: float,
    cc: dict[str, float | int],
    lines: dict[str, float | int],
    qr_found: bool,
) -> float:
    square_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) / 0.35)
    dense_score = min(1.0, 0.50 * float(cc["small_component_ratio"]) + 0.50 * float(cc["component_density"]))
    dark_score = min(1.0, dark_share / 0.18)
    no_long_lines = max(0.0, 1.0 - float(lines["max_line_length_ratio"]) / 0.30)
    grid_score = min(1.0, grid_occ / 0.50)

    score = (
        0.24 * square_score
        + 0.28 * dense_score
        + 0.18 * dark_score
        + 0.20 * no_long_lines
        + 0.10 * grid_score
    )

    if qr_found:
        score = max(score, 0.98)

    # Квадратная настоящая планировка может иметь сильные длинные H/V линии.
    if int(lines["h_lines"]) >= 4 and int(lines["v_lines"]) >= 4 and float(lines["max_line_length_ratio"]) >= 0.28:
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
    no_vertical = 1.0 - min(1.0, v_lines / 4.0)

    score = (
        0.26 * float(cc["small_component_ratio"])
        + 0.18 * float(cc["component_density"])
        + 0.24 * horizontal_dominance
        + 0.20 * no_vertical
        + 0.12 * min(1.0, gray_share / 0.50)
    )

    if v_lines >= 4 and float(lines["hv_balance"]) >= 0.25:
        score *= 0.65

    return float(min(1.0, max(0.0, score)))


def estimate_frame_like(
    grid_occ: float,
    interior_dark: float,
    border_dom: float,
    lines: dict[str, float | int],
) -> float:
    sparse_grid = max(0.0, 1.0 - grid_occ / 0.22)
    weak_interior = max(0.0, 1.0 - interior_dark / 0.025)
    border_heavy = min(1.0, border_dom / 5.0)
    few_lines = max(0.0, 1.0 - (int(lines["h_lines"]) + int(lines["v_lines"])) / 10.0)

    score = 0.28 * sparse_grid + 0.28 * weak_interior + 0.26 * border_heavy + 0.18 * few_lines
    return float(min(1.0, max(0.0, score)))


def analyze(path: Path, max_side: int, preset: Preset) -> dict[str, Any]:
    rgb, orig_size, analysis_size = load_rgb(path, max_side=max_side)
    h, w = rgb.shape[:2]
    aspect_ratio = float(w / max(1, h))

    labels = classify_pixels(rgb)
    shares = shares_from_labels(labels)

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
    edge = edge_density(rgb)
    photo_grad = gradient_photo_score(rgb)
    qr_found = qr_detected(rgb)

    qr_like = estimate_qr_like(
        aspect_ratio=aspect_ratio,
        dark_share=dark_neutral_share,
        grid_occ=grid_occ,
        cc=cc,
        lines=lines,
        qr_found=qr_found,
    )
    text_like = estimate_text_like(neutral_gray_share, cc, lines)
    frame_like = estimate_frame_like(grid_occ, interior_dark, border_dom, lines)

    color_photo_like = min(1.0, 0.65 * (color_share / max(1e-6, preset.color_soft_max)) + 0.35 * photo_grad)
    color_photo_like = float(min(1.0, max(0.0, color_photo_like)))

    line_score = (
        0.45 * min(int(lines["h_lines"]) / preset.h_line_norm, 1.0)
        + 0.45 * min(int(lines["v_lines"]) / preset.v_line_norm, 1.0)
        + 0.35 * min(float(lines["max_line_length_ratio"]) / preset.long_line_norm, 1.0)
        + 0.25 * min(float(lines["mean_line_length_ratio"]) / (preset.long_line_norm * 0.45), 1.0)
        + 0.30 * float(lines["hv_balance"])
        + 0.15 * float(lines["orthogonal_ratio"])
    )

    topology_score = (
        0.35 * min(grid_occ / preset.grid_norm, 1.0)
        + 0.35 * min(interior_dark / preset.interior_norm, 1.0)
        + 0.20 * min(float(cc["closed_largest_cc_ratio"]) / 0.22, 1.0)
        + 0.10 * min(float(cc["largest_cc_ratio"]) / 0.10, 1.0)
    )

    palette_score = (
        0.30 * min(white_share / max(1e-6, preset.min_white_for_bonus), 1.0)
        + 0.25 * min(dark_neutral_share / max(1e-6, preset.min_dark_for_bonus * 4), 1.0)
        - 0.45 * min(color_share / max(1e-6, preset.color_soft_max), 1.0)
        - 0.15 * min(neutral_gray_share / 0.70, 1.0)
    )

    # Финальная оценка: положительные признаки планировки минус похожесть на мусор.
    floorplan_score = (
        2.00 * line_score
        + 1.35 * topology_score
        + 0.75 * palette_score
        + 0.25 * min(edge / 0.18, 1.0)
        - 1.80 * qr_like
        - 0.95 * text_like
        - 0.85 * frame_like
        - 0.75 * color_photo_like
    )

    hard_reject_reasons: list[str] = []
    if qr_like >= preset.qr_hard_reject:
        hard_reject_reasons.append(f"qr_like:{qr_like:.4f}>={preset.qr_hard_reject:.4f}")
    if color_photo_like >= preset.photo_hard_reject:
        hard_reject_reasons.append(f"photo_like:{color_photo_like:.4f}>={preset.photo_hard_reject:.4f}")
    if text_like >= preset.text_hard_reject:
        hard_reject_reasons.append(f"text_like:{text_like:.4f}>={preset.text_hard_reject:.4f}")
    if frame_like >= preset.frame_hard_reject:
        hard_reject_reasons.append(f"frame_like:{frame_like:.4f}>={preset.frame_hard_reject:.4f}")

    return {
        "src": str(path),
        "name": path.name,
        "orig_width": int(orig_size[0]),
        "orig_height": int(orig_size[1]),
        "analysis_width": int(analysis_size[0]),
        "analysis_height": int(analysis_size[1]),
        "aspect_ratio": aspect_ratio,
        "top5": top_classes(shares, n=5),
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
        "edge_density": float(edge),
        "photo_gradient_score": float(photo_grad),
        "qr_detected": bool(qr_found),
        **{k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in cc.items()},
        **{k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in lines.items()},
        "line_score": float(line_score),
        "topology_score": float(topology_score),
        "palette_score": float(palette_score),
        "qr_like_score": float(qr_like),
        "text_like_score": float(text_like),
        "frame_like_score": float(frame_like),
        "color_photo_like_score": float(color_photo_like),
        "floorplan_score": float(floorplan_score),
        "hard_reject": bool(hard_reject_reasons),
        "hard_reject_reasons": hard_reject_reasons,
    }


def rank_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Hard rejects всегда после non-hard-reject, затем по score.
    return sorted(
        items,
        key=lambda x: (
            bool(x.get("hard_reject", False)),
            -float(x.get("floorplan_score", -9999.0)),
            str(x.get("name", "")),
        ),
    )


def prefixed_name(rank: int, item: dict[str, Any]) -> str:
    score = float(item.get("floorplan_score", -999.0))
    name = safe_name(str(item.get("name", "image.png")))
    return f"{rank:06d}__score_{score:+07.3f}__{name}"


def write_outputs(
    items_ranked: list[dict[str, Any]],
    out_dir: Path,
    top_k: int,
    mode: str,
    min_score_for_top: float,
) -> dict[str, Any]:
    ranked_all_dir = out_dir / "ranked_all"
    top_k_dir = out_dir / "top_k"
    hard_rejects_dir = out_dir / "hard_rejects"

    ensure_dir(ranked_all_dir)
    ensure_dir(top_k_dir)
    ensure_dir(hard_rejects_dir)

    ranked_count = 0
    top_count = 0
    hard_reject_count = 0

    for idx, item in enumerate(items_ranked, start=1):
        src = Path(str(item["src"]))
        out_name = prefixed_name(idx, item)

        # ranked_all: сохраняем всё.
        materialize(src, ranked_all_dir / out_name, mode=mode, overwrite=True)
        item["ranked_path"] = str(ranked_all_dir / out_name)
        ranked_count += 1

        if item.get("hard_reject", False):
            materialize(src, hard_rejects_dir / out_name, mode=mode, overwrite=True)
            item["hard_reject_path"] = str(hard_rejects_dir / out_name)
            hard_reject_count += 1
            continue

        if top_count < top_k and float(item.get("floorplan_score", -9999.0)) >= min_score_for_top:
            materialize(src, top_k_dir / out_name, mode=mode, overwrite=True)
            item["top_k_path"] = str(top_k_dir / out_name)
            top_count += 1

    return {
        "ranked_all_dir": str(ranked_all_dir),
        "top_k_dir": str(top_k_dir),
        "hard_rejects_dir": str(hard_rejects_dir),
        "ranked_count": ranked_count,
        "top_k_count": top_count,
        "hard_reject_count": hard_reject_count,
    }


def write_table(items: list[dict[str, Any]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)

    fieldnames = [
        "rank",
        "name",
        "floorplan_score",
        "hard_reject",
        "hard_reject_reasons",
        "qr_detected",
        "qr_like_score",
        "text_like_score",
        "frame_like_score",
        "color_photo_like_score",
        "line_score",
        "topology_score",
        "palette_score",
        "white_share",
        "dark_neutral_share",
        "color_share",
        "neutral_gray_share",
        "h_lines",
        "v_lines",
        "hv_balance",
        "max_line_length_ratio",
        "mean_line_length_ratio",
        "line_count",
        "grid_occupancy",
        "interior_dark_ratio",
        "closed_largest_cc_ratio",
        "component_count",
        "small_component_ratio",
        "component_density",
        "src",
        "ranked_path",
        "top_k_path",
        "hard_reject_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, item in enumerate(items, start=1):
            row = {key: item.get(key, "") for key in fieldnames}
            row["rank"] = rank
            row["hard_reject_reasons"] = ";".join(item.get("hard_reject_reasons", []))
            writer.writerow(row)


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rank, item in enumerate(items, start=1):
            row = dict(item)
            row["rank"] = rank
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rank floorplan candidates by geometry/topology, not only color.")
    p.add_argument("--input", required=True, help="Directory with crop candidates.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    p.add_argument("--preset", choices=["recall", "balanced", "strict"], default="balanced")
    p.add_argument("--top-k", type=int, default=800)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--include-hard-rejects-in-top", action="store_true")
    p.add_argument("--min-score-for-top", type=float, default=None)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)

    preset = get_preset(args.preset)
    min_score_for_top = preset.min_score_for_top if args.min_score_for_top is None else args.min_score_for_top

    images = iter_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    items: list[dict[str, Any]] = []

    for idx, path in enumerate(images, start=1):
        try:
            item = analyze(path, max_side=args.max_side, preset=preset)
        except Exception as exc:
            item = {
                "src": str(path),
                "name": path.name,
                "floorplan_score": -9999.0,
                "hard_reject": True,
                "hard_reject_reasons": [f"analysis_error:{type(exc).__name__}:{exc}"],
            }

        items.append(item)

        if idx % 200 == 0 or idx == len(images):
            print(f"analyzed={idx}/{len(images)}", flush=True)

    ranked = rank_items(items)

    # Обычно hard rejects не должны попадать в top_k. Но опция оставлена для диагностики.
    if args.include_hard_rejects_in_top:
        for item in ranked:
            item["hard_reject"] = False

    output_info = write_outputs(
        ranked,
        out_dir,
        top_k=args.top_k,
        mode=args.mode,
        min_score_for_top=min_score_for_top,
    )

    write_table(ranked, out_dir / "results.csv")
    write_jsonl(ranked, out_dir / "results.jsonl")

    manifest = {
        "schema": "floorplan_candidate_ranker/v1",
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "mode": args.mode,
        "preset": args.preset,
        "preset_params": asdict(preset),
        "top_k": args.top_k,
        "max_side": args.max_side,
        "min_score_for_top": min_score_for_top,
        "total_images": len(images),
        **output_info,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("input:", input_dir)
    print("out:", out_dir)
    print("total:", len(images))
    print("ranked_all:", output_info["ranked_count"])
    print("top_k:", output_info["top_k_count"])
    print("hard_rejects:", output_info["hard_reject_count"])
    print("manifest:", out_dir / "manifest.json")
    print("results_csv:", out_dir / "results.csv")
    print("results_jsonl:", out_dir / "results.jsonl")
    print("open:", out_dir / "top_k")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
