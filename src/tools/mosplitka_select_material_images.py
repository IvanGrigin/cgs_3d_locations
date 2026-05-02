#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select material-safe Mosplitka images from downloaded product galleries.

Mosplitka product JSON-LD galleries usually put the isolated product/material
image first, followed by interior renders, room scenes, and marketing graphics.
For material assignment the conservative default is therefore to keep only
image 01 per product and mark the rest as reference-only.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_index(path: Path) -> int:
    prefix = path.name.split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else 9999


def classify(path: Path, policy: str) -> dict[str, Any]:
    idx = image_index(path)
    with Image.open(path) as image:
        width, height = image.size
    aspect = max(width, height) / max(1, min(width, height))

    keep = idx == 1
    reason = "primary_gallery_image" if keep else "gallery_secondary_reference"

    if policy == "heuristic" and not keep:
        # Secondary images are not auto-kept, but flag likely material crops for
        # manual review: long strips or small product cutouts are often borders.
        if aspect >= 3.0 or min(width, height) <= 350:
            reason = "review_possible_material_closeup"
        elif max(width, height) >= 900 and min(width, height) >= 650:
            reason = "reject_likely_interior_scene"
        else:
            reason = "reject_secondary_gallery_image"

    return {
        "product_dir": path.parent.name,
        "image_file": str(path),
        "image_index": idx,
        "width": width,
        "height": height,
        "aspect": round(aspect, 3),
        "keep_as_material": keep,
        "reason": reason,
    }


def copy_selected(rows: list[dict[str, Any]], out_dir: Path, images_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row["keep_as_material"]:
            continue
        src = Path(row["image_file"])
        product_dir = out_dir / row["product_dir"]
        product_dir.mkdir(parents=True, exist_ok=True)
        dst = product_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        row["selected_path"] = str(dst.relative_to(images_root.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Select material-safe images from Mosplitka downloaded images.")
    parser.add_argument("--images-root", default="data/floor_materials/mosplitka/images_jsonld")
    parser.add_argument("--out-csv", default="data/floor_materials/mosplitka/material_image_selection.csv")
    parser.add_argument("--copy-dir", default="")
    parser.add_argument("--policy", choices=["first-only", "heuristic"], default="first-only")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    paths = [
        path
        for path in sorted(images_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.endswith(".part")
    ]
    rows = [classify(path, args.policy) for path in paths]
    for row in rows:
        row["selected_path"] = ""
    if args.copy_dir:
        copy_selected(rows, Path(args.copy_dir), images_root)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "product_dir",
        "image_file",
        "selected_path",
        "image_index",
        "width",
        "height",
        "aspect",
        "keep_as_material",
        "reason",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    kept = sum(1 for row in rows if row["keep_as_material"])
    print(f"Images: {len(rows)}")
    print(f"Selected material images: {kept}")
    print(f"Saved: {out_csv}")
    if args.copy_dir:
        print(f"Copied selected images to: {args.copy_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
