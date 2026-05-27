#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_housesru_pages_catalog.py

Создаёт единый каталог всех скачанных страниц houses.ru.

Что делает:
1. Сканирует один или несколько корней с журналами:
      data/housesru/all_magazines_v3
      data/housesru/all_magazines
2. Находит все magazine_manifest.json.
3. Берёт только журналы с page_count > 0.
4. Для каждой страницы создаёт файл в едином каталоге:
      data/housesru/all_pages_catalog/pages/

Формат имени:
      housesru__YEAR__SLUG__pPAGE.ext

Примеры:
      housesru__2011__100kk-2011__p0001.jpg
      housesru__2025__100kk-2025__p0018.jpg
      housesru__2025__bs-2-131-2025__p0018.jpg

Почему в имени есть slug:
    Одного года и номера страницы недостаточно, потому что в одном году может быть
    несколько журналов: 100kk-2025, bs-2-131-2025, ko-4-119-2025 и т.д.

Пример запуска:

    python3 src/tools/merge_housesru_pages_catalog.py \
      --sources data/housesru/all_magazines_v3 data/housesru/all_magazines \
      --out data/housesru/all_pages_catalog \
      --mode copy

Быстрее и экономнее по месту на том же диске:

    python3 src/tools/merge_housesru_pages_catalog.py \
      --sources data/housesru/all_magazines_v3 data/housesru/all_magazines \
      --out data/housesru/all_pages_catalog \
      --mode hardlink

Результат:
    data/housesru/all_pages_catalog/
      pages/
        housesru__2011__100kk-2011__p0001.jpg
        ...
      pages_catalog_manifest.json
      pages_catalog.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclasses.dataclass(frozen=True)
class CatalogPage:
    year: int
    slug: str
    page: int
    root_url: str
    source_manifest: str
    source_path: str
    output_path: str
    output_name: str
    bytes: int
    action: str


def safe_slug(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "unknown"


def infer_page_from_name(path: Path) -> Optional[int]:
    m = re.search(r"page[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1])
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_magazine_manifests(sources: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    seen: set[Path] = set()

    for source in sources:
        source = source.expanduser().resolve()
        if not source.exists():
            continue
        for path in source.rglob("magazine_manifest.json"):
            path = path.resolve()
            if path not in seen:
                seen.add(path)
                manifests.append(path)

    return sorted(manifests)


def page_records_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    data = read_json(manifest_path)
    page_count = int(data.get("page_count") or 0)
    if page_count <= 0:
        return []

    year = int(data.get("year") or 0)
    slug = safe_slug(str(data.get("slug") or manifest_path.parent.name))
    root_url = str(data.get("root_url") or "")

    records: list[dict[str, Any]] = []

    pages = data.get("pages")
    if isinstance(pages, list) and pages:
        for item in pages:
            if not isinstance(item, dict):
                continue
            src = item.get("path")
            if not src:
                continue
            src_path = Path(str(src)).expanduser()
            if not src_path.is_absolute():
                # На случай относительного пути в manifest.
                src_path = (manifest_path.parent / src_path).resolve()
            page_num = item.get("page")
            if page_num is None:
                page_num = infer_page_from_name(src_path)
            if page_num is None:
                continue

            records.append(
                {
                    "year": year,
                    "slug": slug,
                    "page": int(page_num),
                    "root_url": root_url,
                    "source_manifest": str(manifest_path),
                    "source_path": str(src_path),
                }
            )
        return records

    # Fallback: если manifest повреждён или без pages[].
    pages_dir = manifest_path.parent / "pages"
    for src_path in sorted(p for p in pages_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        page_num = infer_page_from_name(src_path)
        if page_num is None:
            continue
        records.append(
            {
                "year": year,
                "slug": slug,
                "page": int(page_num),
                "root_url": root_url,
                "source_manifest": str(manifest_path),
                "source_path": str(src_path),
            }
        )

    return records


def choose_best_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Убирает дубли по (year, slug, page).

    Если одна и та же страница есть в нескольких источниках, выбирается файл:
      1. существующий;
      2. с большим размером;
      3. с более предпочтительным расширением: jpg/jpeg, png, webp.
    """
    ext_priority = {
        ".jpg": 30,
        ".jpeg": 30,
        ".png": 20,
        ".webp": 10,
    }

    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for r in records:
        key = (int(r["year"]), str(r["slug"]), int(r["page"]))
        grouped.setdefault(key, []).append(r)

    chosen: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        def score(item: dict[str, Any]) -> tuple[int, int, int, str]:
            p = Path(item["source_path"])
            exists = int(p.exists())
            size = p.stat().st_size if p.exists() else -1
            ext_score = ext_priority.get(p.suffix.lower(), 0)
            return (exists, size, ext_score, str(p))

        best = sorted(items, key=score, reverse=True)[0]
        chosen.append(best)

    return chosen


def make_output_name(year: int, slug: str, page: int, ext: str) -> str:
    ext = ext.lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"
    return f"housesru__{year:04d}__{safe_slug(slug)}__p{page:04d}{ext}"


def materialize(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if not overwrite:
            return "exists"
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_hardlink_failed"

    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return "symlinked"
        except OSError:
            shutil.copy2(src, dst)
            return "copied_symlink_failed"

    raise ValueError(f"unknown mode: {mode}")


def build_catalog(
    sources: list[Path],
    out_dir: Path,
    *,
    mode: str,
    overwrite: bool,
    with_sha1: bool,
) -> list[CatalogPage]:
    manifests = find_magazine_manifests(sources)

    raw_records: list[dict[str, Any]] = []
    for manifest in manifests:
        raw_records.extend(page_records_from_manifest(manifest))

    chosen = choose_best_records(raw_records)

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    result: list[CatalogPage] = []
    skipped_missing = 0

    for r in chosen:
        year = int(r["year"])
        slug = str(r["slug"])
        page = int(r["page"])
        src_path = Path(str(r["source_path"])).expanduser()

        if not src_path.exists() or src_path.suffix.lower() not in IMAGE_EXTS:
            skipped_missing += 1
            continue

        output_name = make_output_name(year, slug, page, src_path.suffix)
        output_path = pages_dir / output_name
        action = materialize(src_path, output_path, mode=mode, overwrite=overwrite)

        item = CatalogPage(
            year=year,
            slug=slug,
            page=page,
            root_url=str(r.get("root_url") or ""),
            source_manifest=str(r.get("source_manifest") or ""),
            source_path=str(src_path),
            output_path=str(output_path),
            output_name=output_name,
            bytes=output_path.stat().st_size if output_path.exists() else 0,
            action=action,
        )
        result.append(item)

    manifest_data: dict[str, Any] = {
        "schema": "housesru_all_pages_catalog/v1",
        "sources": [str(p.expanduser().resolve()) for p in sources],
        "out_dir": str(out_dir),
        "mode": mode,
        "overwrite": overwrite,
        "manifest_count": len(manifests),
        "raw_record_count": len(raw_records),
        "catalog_page_count": len(result),
        "skipped_missing": skipped_missing,
        "magazine_count": len({(x.year, x.slug) for x in result}),
        "pages": [dataclasses.asdict(x) for x in result],
    }

    if with_sha1:
        hashes = {}
        for item in result:
            hashes[item.output_name] = sha1_file(Path(item.output_path))
        manifest_data["sha1"] = hashes

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pages_catalog_manifest.json").write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = out_dir / "pages_catalog.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "slug",
                "page",
                "output_name",
                "output_path",
                "source_path",
                "root_url",
                "bytes",
                "action",
            ],
        )
        writer.writeheader()
        for item in result:
            writer.writerow(
                {
                    "year": item.year,
                    "slug": item.slug,
                    "page": item.page,
                    "output_name": item.output_name,
                    "output_path": item.output_path,
                    "source_path": item.source_path,
                    "root_url": item.root_url,
                    "bytes": item.bytes,
                    "action": item.action,
                }
            )

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge houses.ru magazine page images into one catalog.")
    p.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Один или несколько каталогов со скачанными журналами.",
    )
    p.add_argument("--out", required=True, help="Выходной каталог единого каталога страниц.")
    p.add_argument(
        "--mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="copy — физически копировать; hardlink — экономить место на том же диске; symlink — символические ссылки.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--with-sha1", action="store_true", help="Медленнее: посчитать sha1 для каждого файла.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    sources = [Path(x).expanduser().resolve() for x in args.sources]
    out_dir = Path(args.out).expanduser().resolve()

    result = build_catalog(
        sources,
        out_dir,
        mode=args.mode,
        overwrite=args.overwrite,
        with_sha1=args.with_sha1,
    )

    by_mag: dict[tuple[int, str], int] = {}
    for item in result:
        by_mag[(item.year, item.slug)] = by_mag.get((item.year, item.slug), 0) + 1

    print("catalog:", out_dir)
    print("pages:", len(result))
    print("magazines:", len(by_mag))
    for (year, slug), count in sorted(by_mag.items()):
        print(f"{year} {slug} {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
