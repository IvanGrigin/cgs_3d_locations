from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _safe_json(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        parsed = json.loads(value)
        return parsed if parsed is not None else default
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_products_characteristics(products_csv: Path, properties_csv: Path) -> list[dict[str, Any]]:
    products = _read_csv(products_csv)
    properties_by_url: dict[str, dict[str, str]] = defaultdict(dict)

    if properties_csv.is_file():
        for row in _read_csv(properties_csv):
            url = row.get("product_url") or ""
            key = row.get("property_name") or ""
            value = row.get("property_value") or ""
            if url and key:
                properties_by_url[url][key] = value

    result: list[dict[str, Any]] = []
    for row in products:
        url = row.get("url") or row.get("final_url") or ""
        props_from_row = _safe_json(row.get("properties_json", ""), {})
        if not isinstance(props_from_row, dict):
            props_from_row = {}
        merged_props = dict(props_from_row)
        merged_props.update(properties_by_url.get(url, {}))
        result.append(
            {
                "url": url,
                "final_url": row.get("final_url") or "",
                "sku": row.get("sku") or "",
                "name": row.get("name") or "",
                "brand": row.get("brand") or "",
                "price": row.get("price") or "",
                "price_currency": row.get("price_currency") or "",
                "availability": row.get("availability") or "",
                "description": row.get("description") or "",
                "breadcrumbs": row.get("breadcrumbs") or "",
                "categories": row.get("categories") or "",
                "parse_status": row.get("parse_status") or "",
                "error": row.get("error") or "",
                "properties": merged_props,
                "image_urls": _safe_json(row.get("images_json", ""), []),
                "local_image_paths": _safe_json(row.get("local_image_paths_json", ""), []),
            }
        )
    return result


def write_readme(out_dir: Path) -> None:
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Domlenta Wallpapers Catalog Bundle",
                "",
                "Place this directory at `data/sourse/domlenta_wallpapers/` in the project checkout.",
                "",
                "Main files:",
                "- `products_characteristics.json`: product cards with merged characteristics and image paths.",
                "- `catalog_summary.json`: counts and parse status summary.",
                "- `normalized_wall_materials.jsonl`: normalized wall-material catalog for selector, if exported.",
                "- `products.csv`, `product_properties.csv`, `product_images.csv`: raw scraper tables, if exported.",
                "- `images/`: optional downloaded product images for RGB/k-means color analysis.",
                "",
                "After copying the bundle, refresh normalized colors if images are present:",
                "",
                "```bash",
                "python3 src/tools/run_wall_material_selector.py normalize \\",
                "  --products-csv data/sourse/domlenta_wallpapers/products.csv \\",
                "  --out-jsonl data/sourse/domlenta_wallpapers/normalized_wall_materials.jsonl",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def export_bundle(input_dir: Path, out_dir: Path, include_raw: bool, include_images: bool) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    products_csv = input_dir / "products.csv"
    properties_csv = input_dir / "product_properties.csv"
    if not products_csv.is_file():
        raise FileNotFoundError(products_csv)

    products = build_products_characteristics(products_csv, properties_csv)
    status_counts = Counter(str(row.get("parse_status") or "") for row in products)
    with_images = sum(1 for row in products if row.get("image_urls"))
    with_local_images = sum(1 for row in products if row.get("local_image_paths"))
    summary = {
        "source_dir": str(input_dir),
        "products_total": len(products),
        "parse_status_counts": dict(status_counts),
        "products_with_image_urls": with_images,
        "products_with_local_images": with_local_images,
    }

    _write_json(out_dir / "products_characteristics.json", products)
    _write_json(out_dir / "catalog_summary.json", summary)
    write_readme(out_dir)

    normalized = input_dir / "normalized_wall_materials.jsonl"
    if normalized.is_file():
        shutil.copy2(normalized, out_dir / normalized.name)

    if include_raw:
        for name in [
            "products.csv",
            "products.jsonl",
            "product_urls.csv",
            "product_images.csv",
            "product_properties.csv",
        ]:
            src = input_dir / name
            if src.is_file():
                shutil.copy2(src, out_dir / name)

    if include_images and (input_dir / "images").is_dir():
        target = out_dir / "images"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(input_dir / "images", target)

    return summary


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    source_dir = source_dir.resolve()
    zip_path = zip_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir.parent))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a transferable Domlenta catalog JSON bundle.")
    parser.add_argument("--input-dir", default="data/sourse/domlenta_wallpapers")
    parser.add_argument("--out-dir", default="data/sourse/domlenta_wallpapers/catalog_bundle")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--zip", dest="zip_path", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = export_bundle(
        input_dir=Path(args.input_dir),
        out_dir=Path(args.out_dir),
        include_raw=bool(args.include_raw),
        include_images=bool(args.include_images),
    )
    print(f"Exported: {Path(args.out_dir)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.zip_path:
        zip_dir(Path(args.out_dir), Path(args.zip_path))
        print(f"ZIP: {Path(args.zip_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
