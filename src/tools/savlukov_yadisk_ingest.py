#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and normalize Savlukov public Yandex Disk sofa models."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


PUBLIC_RESOURCES_API = "https://cloud-api.yandex.net/v1/disk/public/resources"
DEFAULT_PUBLIC_KEY = "https://disk.yandex.ru/d/8Jqha8s5btjZmg"
DEFAULT_ROOT = Path("data/sourse/suppliers/savlukov")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}


def _slug(value: str) -> str:
    value = value.strip().lower().replace("&", " and ")
    value = re.sub(r"[^0-9a-zа-яё]+", "_", value, flags=re.IGNORECASE)
    return re.sub(r"_+", "_", value).strip("_") or "item"


def _norm_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def fetch_listing(public_key: str, out_path: Path, limit: int = 1000) -> list[dict[str, Any]]:
    session = requests.Session()

    def get(path: str) -> dict[str, Any]:
        resp = session.get(
            PUBLIC_RESOURCES_API,
            headers=HEADERS,
            params={"public_key": public_key, "path": path, "limit": limit},
            timeout=(10, 90),
        )
        resp.raise_for_status()
        return resp.json()

    def walk(path: str) -> list[dict[str, Any]]:
        payload = get(path)
        rows = [payload]
        for item in (payload.get("_embedded") or {}).get("items") or []:
            if item.get("type") == "dir":
                rows.extend(walk(str(item.get("path") or "")))
            elif item.get("type") == "file":
                rows.append(item)
        return rows

    rows = walk("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def load_listing(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def listing_files(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [x for x in rows if x.get("type") == "file"]


def download_files(rows: list[dict[str, Any]], raw_root: Path) -> None:
    session = requests.Session()
    files = listing_files(rows)
    for idx, item in enumerate(files, start=1):
        rel = str(item.get("path") or "").lstrip("/")
        dest = raw_root / rel
        expected = int(item.get("size") or 0)
        if expected > 0 and dest.exists() and dest.stat().st_size == expected:
            print(f"[download] skip {idx}/{len(files)} {rel}")
            continue
        url = _norm_space(item.get("file"))
        if not url:
            raise RuntimeError(f"No download URL for {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"[download] {idx}/{len(files)} {rel} bytes={expected}")
        with session.get(url, headers=HEADERS, stream=True, timeout=(10, 180)) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(dest)


def extract_archives(raw_root: Path, extract_root: Path) -> None:
    seven_zip = shutil.which("7z")
    unar = shutil.which("unar")
    if not seven_zip and not unar:
        raise RuntimeError("7z or unar is required to extract rar/zip archives")
    archives = sorted([p for p in raw_root.rglob("*") if p.suffix.lower() in {".rar", ".zip"}])
    for idx, archive in enumerate(archives, start=1):
        rel = archive.relative_to(raw_root)
        out_dir = extract_root / rel.with_suffix("")
        marker = out_dir / ".extracted.ok"
        if marker.exists():
            print(f"[extract] skip {idx}/{len(archives)} {rel}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[extract] {idx}/{len(archives)} {rel}")
        if archive.suffix.lower() == ".rar" and unar:
            subprocess.run([unar, "-force-overwrite", "-output-directory", str(out_dir), str(archive)], check=True)
        elif seven_zip:
            subprocess.run([seven_zip, "x", "-y", f"-o{out_dir}", str(archive)], check=True)
        elif unar:
            subprocess.run([unar, "-force-overwrite", "-output-directory", str(out_dir), str(archive)], check=True)
        else:
            raise RuntimeError(f"No extractor available for {archive}")
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")


def collect_fbx(raw_root: Path, extract_root: Path) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for root in (raw_root, extract_root):
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".fbx"):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _blender_split_worker(argv: list[str]) -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-slug", required=True)
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(in_path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]

    def descendants(root: Any) -> list[Any]:
        got = [root]
        for child in root.children:
            got.extend(descendants(child))
        return got

    def mesh_bbox(objs: list[Any]) -> tuple[Vector, Vector] | None:
        mesh_objs = [obj for obj in objs if obj.type == "MESH"]
        if not mesh_objs:
            return None
        mins = Vector((float("inf"), float("inf"), float("inf")))
        maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
        for obj in mesh_objs:
            for corner in obj.bound_box:
                world = obj.matrix_world @ Vector(corner)
                mins.x = min(mins.x, world.x)
                mins.y = min(mins.y, world.y)
                mins.z = min(mins.z, world.z)
                maxs.x = max(maxs.x, world.x)
                maxs.y = max(maxs.y, world.y)
                maxs.z = max(maxs.z, world.z)
        return mins, maxs

    roots = [obj for obj in imported if obj.parent is None or obj.parent not in imported]
    candidate_groups: list[tuple[str, list[Any]]] = []
    for root in roots:
        group = descendants(root)
        if any(obj.type == "MESH" for obj in group):
            candidate_groups.append((root.name, group))

    # Many FBX files have every mesh as a root. Split only when roots look like
    # a small number of substantial furniture groups, otherwise keep the file whole.
    if 2 <= len(candidate_groups) <= 4:
        valid = []
        for name, group in candidate_groups:
            bbox = mesh_bbox(group)
            if bbox is None:
                continue
            mins, maxs = bbox
            dims = maxs - mins
            if max(float(dims.x), float(dims.y), float(dims.z)) > 1e-4:
                valid.append((name, group, dims))
        groups = [(name, group) for name, group, _dims in valid] if len(valid) >= 2 else [("full", imported)]
        split_method = "root_groups"
    else:
        groups = [("full", imported)]
        split_method = "whole_scene"

    records = []
    for idx, (name, group) in enumerate(groups, start=1):
        suffix = "" if len(groups) == 1 else f"__part_{idx:02d}_{_slug(name)}"
        out_path = out_dir / f"{args.base_slug}{suffix}.fbx"
        bpy.ops.object.select_all(action="DESELECT")
        export_objs = set(group)
        for obj in group:
            parent = obj.parent
            while parent is not None and parent in imported:
                export_objs.add(parent)
                parent = parent.parent
        for obj in export_objs:
            obj.select_set(True)
        active = next((obj for obj in group if obj.type == "MESH"), None) or group[0]
        bpy.context.view_layer.objects.active = active
        bpy.ops.export_scene.fbx(filepath=str(out_path), use_selection=True, add_leaf_bones=False, path_mode="COPY")
        bbox = mesh_bbox(group)
        dims = None
        if bbox is not None:
            mins, maxs = bbox
            diff = maxs - mins
            dims = [float(diff.x), float(diff.y), float(diff.z)]
        records.append(
            {
                "asset_local_path": str(out_path),
                "source_fbx": str(in_path),
                "group_name": name,
                "group_index": idx,
                "group_count": len(groups),
                "split_method": split_method,
                "mesh_count": sum(1 for obj in group if obj.type == "MESH"),
                "dimensions_scene_units": dims,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def split_fbx_with_blender(fbx_files: list[Path], assets_root: Path, blender_bin: str) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    worker = Path(__file__).resolve()

    def write_unprocessed_manifest(fbx: Path, out_dir: Path, manifest: Path, base_slug: str, error: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fallback = out_dir / f"{base_slug}__unprocessed.fbx"
        shutil.copy2(fbx, fallback)
        manifest.write_text(
            json.dumps(
                [
                    {
                        "asset_local_path": str(fallback),
                        "source_fbx": str(fbx),
                        "group_name": "unprocessed",
                        "group_index": 1,
                        "group_count": 1,
                        "split_method": "copy_unprocessed_import_failed",
                        "mesh_count": None,
                        "dimensions_scene_units": None,
                        "import_error": error,
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    for idx, fbx in enumerate(fbx_files, start=1):
        model_slug = _slug(fbx.parent.name if fbx.parent.name else fbx.stem)
        base_slug = _slug(f"{model_slug}_{fbx.stem}")
        out_dir = assets_root / model_slug
        manifest = out_dir / f"{base_slug}.split_manifest.json"
        if manifest.exists():
            print(f"[split] skip {idx}/{len(fbx_files)} {fbx}")
        else:
            print(f"[split] {idx}/{len(fbx_files)} {fbx}")
            try:
                subprocess.run(
                    [
                        blender_bin,
                        "--background",
                        "--python",
                        str(worker),
                        "--",
                        "blender-worker",
                        "--input",
                        str(fbx),
                        "--output-dir",
                        str(out_dir),
                        "--manifest",
                        str(manifest),
                        "--base-slug",
                        base_slug,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                write_unprocessed_manifest(
                    fbx,
                    out_dir,
                    manifest,
                    base_slug,
                    f"Blender import/export failed with exit code {exc.returncode}",
                )
            if not manifest.exists():
                write_unprocessed_manifest(
                    fbx,
                    out_dir,
                    manifest,
                    base_slug,
                    "Blender did not create a split manifest; likely unsupported FBX variant.",
                )
        manifests.extend(json.loads(manifest.read_text(encoding="utf-8")))
    return manifests


def build_catalog_records(split_records: list[dict[str, Any]], public_key: str) -> list[dict[str, Any]]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    records = []
    for rec in split_records:
        asset = Path(rec["asset_local_path"])
        model_slug = asset.parent.name
        title = model_slug.replace("_", " ").strip().title()
        if rec.get("group_count", 1) > 1:
            title = f"{title} part {rec.get('group_index')}"
        unique = f"savlukov::asset::{asset.stem}"
        records.append(
            {
                "unique_key": unique,
                "source_site": "savlukov",
                "source_db": None,
                "source_url": "https://designers.savlukov.by/models",
                "parsed_at": now,
                "external_id": asset.stem,
                "title": title,
                "brand": "Savlukov",
                "collection": model_slug.replace("_", " "),
                "category_raw": "Мягкая мебель",
                "category_norm": "sofa",
                "product_url": "https://designers.savlukov.by/models",
                "model_link_type": "public_yandex_disk",
                "model_page_url": public_key,
                "model_download_url": None,
                "model_download_landing_url": public_key,
                "model_vendor_url": "https://savlukov.by/product/pryamie-divany/136-priamoy-vegas#params",
                "model_extraction_method": "savlukov_yadisk_blender_split",
                "model_download_filename": asset.name,
                "model_format": "fbx",
                "asset_status": "local",
                "asset_format": "fbx",
                "asset_local_path": str(asset),
                "preview_local_path": None,
                "asset_source_url": rec.get("source_fbx"),
                "price_value": None,
                "price_currency": None,
                "old_price_value": None,
                "style": None,
                "color": None,
                "description": f"Savlukov public 3D model exported to FBX; split method: {rec.get('split_method')}.",
                "dimensions_cm": {
                    "width": None,
                    "depth": None,
                    "height": None,
                    "weight_kg": None,
                    "package_width": None,
                    "package_depth": None,
                    "package_height": None,
                    "packed_weight_kg": None,
                    "volume_m3": None,
                },
                "scheme_url": None,
                "room": "living_room",
                "materials": None,
                "availability": "public",
                "country_brand": "Беларусь",
                "production_country": "Беларусь",
                "tags": ["Savlukov", "sofa", ".fbx"],
                "images": [],
                "related": [],
                "extra": {"savlukov_split": rec},
                "completeness": {
                    "has_title": True,
                    "has_price": False,
                    "has_full_dimensions": False,
                    "has_description": True,
                    "has_category": True,
                    "has_brand": True,
                    "has_model_link": True,
                    "rich_card": False,
                },
            }
        )
    return records


def merge_catalog(catalog_path: Path, records: list[dict[str, Any]]) -> None:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = payload.setdefault("items", [])
    by_key = {item.get("unique_key"): idx for idx, item in enumerate(items)}
    added = 0
    updated = 0
    for rec in records:
        key = rec["unique_key"]
        if key in by_key:
            items[by_key[key]] = rec
            updated += 1
        else:
            items.append(rec)
            added += 1
    payload.setdefault("meta", {}).setdefault("manual_merges", []).append(
        {
            "source": "savlukov_yadisk_ingest",
            "merged_at_unix": time.time(),
            "added": added,
            "updated": updated,
            "item_count_after": len(items),
        }
    )
    if isinstance(payload.get("meta"), dict):
        payload["meta"]["item_count"] = len(items)
    tmp = catalog_path.with_suffix(catalog_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(catalog_path)
    print(f"[catalog] added={added} updated={updated} total={len(items)}")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    if argv and argv[0] == "blender-worker":
        _blender_split_worker(argv[1:])
        return

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public-key", default=DEFAULT_PUBLIC_KEY)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--catalog", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    ap.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender")
    ap.add_argument("--fetch-listing", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--merge-catalog", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    listing_path = root / "savlukov_yadisk_listing.json"
    raw_root = root / "raw"
    extract_root = root / "extracted"
    assets_root = root / "assets"
    manifests_path = root / "savlukov_split_assets.json"

    if args.fetch_listing or not listing_path.exists():
        rows = fetch_listing(args.public_key, listing_path)
    else:
        rows = load_listing(listing_path)
    print(f"[listing] files={len(listing_files(rows))} listing={listing_path}")

    if args.download:
        download_files(rows, raw_root)
    if args.extract:
        extract_archives(raw_root, extract_root)
    if args.split:
        records = split_fbx_with_blender(collect_fbx(raw_root, extract_root), assets_root, args.blender)
        manifests_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[split] records={len(records)} manifest={manifests_path}")
    if args.merge_catalog:
        records = json.loads(manifests_path.read_text(encoding="utf-8"))
        merge_catalog(Path(args.catalog), build_catalog_records(records, args.public_key))


if __name__ == "__main__":
    main()
