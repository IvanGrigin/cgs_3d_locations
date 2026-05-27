#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

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

NEUTRAL_CLASSES = {"white", "black", "light_gray", "gray", "dark_gray"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def rgb_to_hsv_np(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    rgb: uint8 array [H, W, 3]
    returns h in [0, 360), s in [0, 1], v in [0, 1]
    """
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
    """
    Возвращает int-коды классов цветов для всех пикселей.
    """
    h, s, v = rgb_to_hsv_np(rgb)

    labels = np.full(h.shape, fill_value=3, dtype=np.int16)  # gray by default

    # Белый
    white_mask = (v >= 0.94) & (s <= 0.10)
    labels[white_mask] = 0

    # Черный
    black_mask = v <= 0.18
    labels[black_mask] = 1

    # Серые
    gray_mask = (s <= 0.12) & (~white_mask) & (~black_mask)
    labels[gray_mask & (v >= 0.75)] = 2   # light_gray
    labels[gray_mask & (v >= 0.35) & (v < 0.75)] = 3  # gray
    labels[gray_mask & (v < 0.35)] = 4    # dark_gray

    # Цветные
    color_mask = ~(white_mask | black_mask | gray_mask)

    # brown / orange / yellow / green / cyan / blue / purple / pink / red
    brown_mask = color_mask & (h >= 15) & (h < 45) & (v < 0.65)
    labels[brown_mask] = 13

    red_mask = color_mask & (((h >= 0) & (h < 15)) | ((h >= 345) & (h < 360)))
    labels[red_mask] = 5

    orange_mask = color_mask & (h >= 15) & (h < 45) & (~brown_mask)
    labels[orange_mask] = 6

    yellow_mask = color_mask & (h >= 45) & (h < 75)
    labels[yellow_mask] = 7

    green_mask = color_mask & (h >= 75) & (h < 165)
    labels[green_mask] = 8

    cyan_mask = color_mask & (h >= 165) & (h < 200)
    labels[cyan_mask] = 9

    blue_mask = color_mask & (h >= 200) & (h < 260)
    labels[blue_mask] = 10

    purple_mask = color_mask & (h >= 260) & (h < 320)
    labels[purple_mask] = 11

    pink_mask = color_mask & (h >= 320) & (h < 345)
    labels[pink_mask] = 12

    return labels


def analyze_image(
    image_path: Path,
    max_side: int = 512,
) -> Dict:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    rgb = np.array(img)

    labels = classify_pixels(rgb)
    flat = labels.reshape(-1).tolist()
    counter = Counter(flat)
    total = len(flat)

    shares = {CLASS_NAMES[k]: counter.get(k, 0) / total for k in CLASS_NAMES.keys()}

    ranked = sorted(
        ((CLASS_NAMES[k], counter.get(k, 0), counter.get(k, 0) / total) for k in CLASS_NAMES.keys()),
        key=lambda x: (-x[1], x[0]),
    )

    top3 = ranked[:3]

    white_share = shares["white"]
    black_share = shares["black"]
    gray_share = shares["light_gray"] + shares["gray"] + shares["dark_gray"]
    colored_share = 1.0 - white_share - black_share - gray_share

    return {
        "image_path": str(image_path),
        "width": int(img.width),
        "height": int(img.height),
        "shares": shares,
        "top3": [
            {"name": name, "count": int(count), "share": float(share)}
            for name, count, share in top3
        ],
        "white_share": float(white_share),
        "black_share": float(black_share),
        "gray_share": float(gray_share),
        "colored_share": float(colored_share),
    }


def decide_accept(
    info: Dict,
    white_min: float,
    black_min: float,
    gray_max: float,
    color_max: float,
    require_top1: str,
    require_top2: str,
    require_top3_neutral: bool,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    top_names = [x["name"] for x in info["top3"]]

    if not top_names:
        reasons.append("empty_top")
        return False, reasons

    if len(top_names) < 2:
        reasons.append("top_less_than_2")
        return False, reasons

    if top_names[0] != require_top1:
        reasons.append(f"top1_is_{top_names[0]}_expected_{require_top1}")

    if top_names[1] != require_top2:
        reasons.append(f"top2_is_{top_names[1]}_expected_{require_top2}")

    if require_top3_neutral:
        for name in top_names:
            if name not in NEUTRAL_CLASSES:
                reasons.append(f"top3_contains_non_neutral_{name}")
                break

    if info["white_share"] < white_min:
        reasons.append(f"white_share_too_low:{info['white_share']:.4f}<{white_min:.4f}")

    if info["black_share"] < black_min:
        reasons.append(f"black_share_too_low:{info['black_share']:.4f}<{black_min:.4f}")

    if info["gray_share"] > gray_max:
        reasons.append(f"gray_share_too_high:{info['gray_share']:.4f}>{gray_max:.4f}")

    if info["colored_share"] > color_max:
        reasons.append(f"colored_share_too_high:{info['colored_share']:.4f}>{color_max:.4f}")

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
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Palette-based postprocessing for extracted floorplan images."
    )
    parser.add_argument("--input", required=True, help="Папка со входными картинками.")
    parser.add_argument("--out", required=True, help="Папка для результата.")
    parser.add_argument(
        "--mode",
        default="symlink",
        choices=["copy", "symlink", "move"],
        help="Как материализовать accepted/rejected.",
    )
    parser.add_argument("--max-side", type=int, default=512, help="Максимальная сторона для анализа.")
    parser.add_argument("--white-min", type=float, default=0.55, help="Минимальная доля white.")
    parser.add_argument("--black-min", type=float, default=0.03, help="Минимальная доля black.")
    parser.add_argument("--gray-max", type=float, default=0.35, help="Максимальная доля gray.")
    parser.add_argument("--color-max", type=float, default=0.10, help="Максимальная доля color.")
    parser.add_argument("--require-top1", default="white", help="Какой класс должен быть top-1.")
    parser.add_argument("--require-top2", default="black", help="Какой класс должен быть top-2.")
    parser.add_argument(
        "--no-require-top3-neutral",
        action="store_true",
        help="Если указано, то top-3 может содержать цветные классы.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    ensure_dir(accepted_dir)
    ensure_dir(rejected_dir)

    images = iter_images(input_dir)

    results = []
    accepted_count = 0
    rejected_count = 0

    for src in images:
        rel_name = src.name

        info = analyze_image(src, max_side=args.max_side)
        ok, reasons = decide_accept(
            info=info,
            white_min=args.white_min,
            black_min=args.black_min,
            gray_max=args.gray_max,
            color_max=args.color_max,
            require_top1=args.require_top1,
            require_top2=args.require_top2,
            require_top3_neutral=not args.no_require_top3_neutral,
        )

        dst = (accepted_dir if ok else rejected_dir) / rel_name
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

    manifest = {
        "schema": "floorplan_palette_postprocess/v1",
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "mode": args.mode,
        "params": {
            "max_side": args.max_side,
            "white_min": args.white_min,
            "black_min": args.black_min,
            "gray_max": args.gray_max,
            "color_max": args.color_max,
            "require_top1": args.require_top1,
            "require_top2": args.require_top2,
            "require_top3_neutral": not args.no_require_top3_neutral,
        },
        "total_images": len(images),
        "accepted_images": accepted_count,
        "rejected_images": rejected_count,
        "accepted_dir": str(accepted_dir),
        "rejected_dir": str(rejected_dir),
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("input:", input_dir)
    print("out:", out_dir)
    print("total:", len(images))
    print("accepted:", accepted_count)
    print("rejected:", rejected_count)
    print("manifest:", out_dir / "manifest.json")
    print("results:", out_dir / "results.jsonl")


if __name__ == "__main__":
    main()