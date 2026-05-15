#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
package_floorplan_annotation_zip.py

Упаковывает ручную разметку floorplans в ZIP.

Главная проблема:
    good/ и bad/ часто содержат symlink/alias-like файлы, потому что разметка
    запускалась с --mode symlink.

Этот скрипт НЕ кладёт symlink как symlink.
Он разыменовывает ссылки и кладёт в ZIP реальные изображения.

Что попадает в архив:
    good/                       реальные изображения good
    bad/                        реальные изображения bad
    labels.csv
    labels.jsonl
    annotation_manifest.json
    README.txt
    package_manifest.json
    optional extra files:
        --extra data/housesru/floorplans_ranked_v2/results.csv
        --extra ...

Пример:

python3 src/tools/package_floorplan_annotation_zip.py \
  --annotation-dir data/housesru/floorplans_manual_click_v1 \
  --zip-out data/housesru/floorplans_manual_click_v1.zip \
  --extra data/housesru/floorplans_ranked_v2/results.csv \
  --extra data/housesru/floorplans_ranked_v2/manifest.json

Проверка:

unzip -l data/housesru/floorplans_manual_click_v1.zip | head
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_existing(path: Path) -> Path:
    """
    Разыменовывает symlink.
    Для обычных файлов возвращает сам файл.
    """
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        return path


def copy_real_file(src: Path, dst: Path) -> dict[str, Any]:
    ensure_dir(dst.parent)

    real_src = resolve_existing(src)

    if not real_src.exists() or not real_src.is_file():
        return {
            "src": str(src),
            "dst": str(dst),
            "resolved_src": str(real_src),
            "copied": False,
            "reason": "missing_or_not_file",
        }

    shutil.copy2(real_src, dst)

    return {
        "src": str(src),
        "dst": str(dst),
        "resolved_src": str(real_src),
        "copied": True,
        "bytes": dst.stat().st_size,
        "was_symlink": src.is_symlink(),
    }


def copy_tree_real_files(src_dir: Path, dst_dir: Path, *, only_images: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if not src_dir.exists():
        return rows

    for src in sorted(src_dir.rglob("*")):
        if not src.is_file() and not src.is_symlink():
            continue

        if only_images and src.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        rows.append(copy_real_file(src, dst))

    return rows


def copy_aux_file(src: Path, dst_dir: Path) -> dict[str, Any]:
    dst = dst_dir / src.name
    return copy_real_file(src, dst)


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    ensure_dir(zip_path.parent)

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file():
                continue

            arcname = path.relative_to(src_dir).as_posix()

            # Изображения уже сжаты, их лучше писать STORE для скорости и без увеличения CPU.
            if path.suffix.lower() in IMAGE_EXTS:
                zf.write(path, arcname=arcname, compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(path, arcname=arcname)


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def count_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def make_readme(annotation_dir: Path, package_manifest: dict[str, Any]) -> str:
    return f"""Houses.ru floorplan manual annotation package

Source annotation directory:
{annotation_dir}

Archive contents:
- good/                       manually accepted floorplan crops
- bad/                        manually rejected crops
- labels.csv                  tabular labels
- labels.jsonl                JSONL labels
- annotation_manifest.json    original annotation manifest
- package_manifest.json       packaging report
- extra/                      optional extra files supplied via --extra

Important:
The original good/ and bad/ directories may have contained symlinks.
This ZIP contains real copied image files, not symlinks.

Counts:
good_images: {package_manifest.get("good_images")}
bad_images: {package_manifest.get("bad_images")}
total_images: {package_manifest.get("total_images")}
zip_bytes: {package_manifest.get("zip_bytes")}
"""


def build_package(annotation_dir: Path, zip_out: Path, extras: list[Path], keep_staging: bool) -> dict[str, Any]:
    annotation_dir = annotation_dir.expanduser().resolve()
    zip_out = zip_out.expanduser().resolve()

    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation-dir not found: {annotation_dir}")

    good_dir = annotation_dir / "good"
    bad_dir = annotation_dir / "bad"

    with tempfile.TemporaryDirectory(prefix="floorplan_package_") as tmp_name:
        staging = Path(tmp_name) / "floorplan_annotation_package"
        ensure_dir(staging)

        copied_good = copy_tree_real_files(good_dir, staging / "good", only_images=True)
        copied_bad = copy_tree_real_files(bad_dir, staging / "bad", only_images=True)

        copied_aux: list[dict[str, Any]] = []
        for name in ["labels.csv", "labels.jsonl", "annotation_manifest.json"]:
            src = annotation_dir / name
            if src.exists():
                copied_aux.append(copy_aux_file(src, staging))

        extra_dir = staging / "extra"
        copied_extra: list[dict[str, Any]] = []
        for extra in extras:
            extra = extra.expanduser().resolve()
            if extra.exists() and extra.is_file():
                ensure_dir(extra_dir)
                copied_extra.append(copy_real_file(extra, extra_dir / extra.name))

        good_images = sum(1 for r in copied_good if r.get("copied"))
        bad_images = sum(1 for r in copied_bad if r.get("copied"))

        package_manifest = {
            "schema": "floorplan_annotation_zip_package/v1",
            "annotation_dir": str(annotation_dir),
            "zip_out": str(zip_out),
            "good_images": good_images,
            "bad_images": bad_images,
            "total_images": good_images + bad_images,
            "copied_good": copied_good,
            "copied_bad": copied_bad,
            "copied_aux": copied_aux,
            "copied_extra": copied_extra,
            "staging_file_count": count_files(staging),
            "staging_bytes": count_bytes(staging),
        }

        (staging / "package_manifest.json").write_text(
            json.dumps(package_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (staging / "README.txt").write_text(
            make_readme(annotation_dir, package_manifest),
            encoding="utf-8",
        )

        if keep_staging:
            final_staging = zip_out.with_suffix("")
            if final_staging.exists():
                shutil.rmtree(final_staging)
            shutil.copytree(staging, final_staging)
            print("staging kept:", final_staging)

        zip_dir(staging, zip_out)

    package_manifest["zip_bytes"] = zip_out.stat().st_size if zip_out.exists() else 0

    # Обновляем package_manifest рядом с zip.
    sidecar = zip_out.with_suffix(zip_out.suffix + ".manifest.json")
    sidecar.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return package_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package manual floorplan annotation into a ZIP with real files, resolving symlinks.")
    parser.add_argument("--annotation-dir", required=True, help="Directory with labels.csv, labels.jsonl, good/, bad/.")
    parser.add_argument("--zip-out", required=True, help="Output zip path.")
    parser.add_argument("--extra", action="append", default=[], help="Extra file to include under extra/. Can be repeated.")
    parser.add_argument("--keep-staging", action="store_true", help="Also keep unpacked staging directory next to zip.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    manifest = build_package(
        annotation_dir=Path(args.annotation_dir),
        zip_out=Path(args.zip_out),
        extras=[Path(x) for x in args.extra],
        keep_staging=args.keep_staging,
    )

    print("zip:", manifest["zip_out"])
    print("good_images:", manifest["good_images"])
    print("bad_images:", manifest["bad_images"])
    print("total_images:", manifest["total_images"])
    print("zip_bytes:", manifest["zip_bytes"])
    print("sidecar_manifest:", str(Path(manifest["zip_out"]).with_suffix(Path(manifest["zip_out"]).suffix + ".manifest.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
