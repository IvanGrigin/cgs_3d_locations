from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ChooseObject.floor_material_normalizer import analyze_floor_material_colors, normalize_domlenta_catalog
from src.ChooseObject.floor_material_selector import FloorMaterialSelector
from src.pipeline.flooring_stage import apply_flooring_to_scene, load_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and select floor covering materials.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="Normalize Domlenta products.csv to JSONL.")
    normalize.add_argument("--products-csv", required=True)
    normalize.add_argument("--out-jsonl", required=True)
    normalize.add_argument("--no-analyze-images", action="store_true", help="Skip average/dominant color extraction.")

    backfill = subparsers.add_parser("backfill-colors", help="Add average/dominant image colors to floor material JSONL tables.")
    backfill.add_argument("--materials", required=True, help="JSONL file or directory with normalized_floor_materials.jsonl files.")
    backfill.add_argument("--force", action="store_true", help="Recompute colors even when average_rgb already exists.")
    backfill.add_argument("--dry-run", action="store_true", help="Report counts without writing files.")
    backfill.add_argument("--all-records", action="store_true", help="Update every JSONL record with local images, not only selectable floor records.")

    image_colors = subparsers.add_parser("image-colors", help="Build a JSONL color table for every image file under a directory.")
    image_colors.add_argument("--root", required=True)
    image_colors.add_argument("--out-jsonl", default=None)
    image_colors.add_argument("--force", action="store_true", help="Recompute colors for paths already present in the output table.")
    image_colors.add_argument("--dry-run", action="store_true", help="Report counts without writing the output table.")

    select = subparsers.add_parser("select", help="Select one floor material for a prompt/style/room.")
    select.add_argument("--materials", required=True)
    select.add_argument("--style-rules", required=True)
    select.add_argument("--prompt", required=True)
    select.add_argument("--style", default=None)
    select.add_argument("--room-type", default=None)
    select.add_argument("--room-description", default=None)
    select.add_argument("--room-id", default="room_001")
    select.add_argument("--out", required=True)
    select.add_argument("--top-k", type=int, default=10)
    select.add_argument("--llm-provider", choices=["none", "ollama"], default="none")
    select.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    select.add_argument("--ollama-model", default="gpt-oss:20b")
    select.add_argument("--ollama-timeout", type=int, default=180)
    select.add_argument("--ollama-temperature", type=float, default=0.0)
    select.add_argument("--ollama-think", default="low")
    select.add_argument("--ollama-num-ctx", type=int, default=8192)
    select.add_argument("--llm-top-n", type=int, default=5)

    apply = subparsers.add_parser("apply-to-scene", help="Apply flooring.selection.v1.json to scene JSON.")
    apply.add_argument("--scene-json", required=True)
    apply.add_argument("--flooring-json", required=True)
    apply.add_argument("--out-scene-json", required=True)
    return parser


def _floor_material_jsonl_paths(path: Path) -> list[Path]:
    path = Path(path).expanduser()
    if path.is_file():
        return [path]
    preferred = sorted(path.rglob("normalized_floor_materials.jsonl"))
    extras = sorted(p for p in path.rglob("*.jsonl") if "surface_materials" in p.name and "test" not in p.name)
    manifests = sorted(path.rglob("image_download_manifest*.jsonl"))
    out: list[Path] = []
    seen: set[Path] = set()
    for file_path in [*preferred, *extras, *manifests]:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(file_path)
    return out


def _all_jsonl_paths(path: Path) -> list[Path]:
    path = Path(path).expanduser()
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.jsonl"))


def _local_image_paths_from_item(item: dict) -> list[str]:
    out: list[str] = []
    for key in ("local_image_paths", "image_paths"):
        raw = item.get(key)
        if isinstance(raw, list):
            out.extend(str(x).strip() for x in raw if str(x).strip())
    for key in ("local_path", "path", "source_path", "selected_path", "image_file"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    for key in ("material_image", "image"):
        image = item.get(key) if isinstance(item.get(key), dict) else {}
        for subkey in ("source_path", "path", "selected_path", "local_path", "image_file"):
            value = image.get(subkey)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for value in out:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def _is_floor_material_item(item: dict) -> bool:
    version = str(item.get("version") or "")
    if version.startswith("floor_material"):
        return True
    if not version.startswith("surface_material"):
        return False
    normalized = item.get("normalized") if isinstance(item.get("normalized"), dict) else {}
    return normalized.get("is_selectable_floor") is True


def _is_image_manifest_item(item: dict) -> bool:
    return bool(item.get("local_path")) and "image_index" in item and "product_url" in item


def backfill_floor_material_colors(path: Path, force: bool = False, dry_run: bool = False, all_records: bool = False) -> dict:
    summary = {
        "files_scanned": 0,
        "rows_scanned": 0,
        "rows_updated": 0,
        "rows_without_colors": 0,
        "updated_files": [],
    }
    paths = _all_jsonl_paths(path) if all_records else _floor_material_jsonl_paths(path)
    for jsonl_path in paths:
        summary["files_scanned"] += 1
        base_dir = jsonl_path.resolve().parent
        changed = False
        out_lines: list[str] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    out_lines.append(line)
                    continue
                summary["rows_scanned"] += 1
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    out_lines.append(line)
                    continue
                if not isinstance(item, dict):
                    out_lines.append(line)
                    continue
                if all_records:
                    if item.get("average_rgb") and not force:
                        out_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                        continue
                    local_paths = _local_image_paths_from_item(item)
                    if not local_paths:
                        out_lines.append(line)
                        continue
                    colors = analyze_floor_material_colors(base_dir, local_paths, k=0)
                    if colors.get("average_rgb"):
                        item["average_rgb"] = colors.get("average_rgb")
                        item["average_hex"] = colors.get("average_hex")
                        item["dominant_colors_rgb"] = colors.get("dominant_colors_rgb") or []
                        item["dominant_colors_hex"] = colors.get("dominant_colors_hex") or []
                        summary["rows_updated"] += 1
                        changed = True
                    else:
                        summary["rows_without_colors"] += 1
                    out_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                    continue
                if _is_image_manifest_item(item):
                    if item.get("average_rgb") and not force:
                        out_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                        continue
                    colors = analyze_floor_material_colors(base_dir, [str(item.get("local_path") or "")], k=0)
                    if colors.get("average_rgb"):
                        item["average_rgb"] = colors.get("average_rgb")
                        item["average_hex"] = colors.get("average_hex")
                        item["dominant_colors_rgb"] = colors.get("dominant_colors_rgb") or []
                        item["dominant_colors_hex"] = colors.get("dominant_colors_hex") or []
                        summary["rows_updated"] += 1
                        changed = True
                    else:
                        summary["rows_without_colors"] += 1
                    out_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                    continue
                if not _is_floor_material_item(item):
                    out_lines.append(line)
                    continue
                if item.get("average_rgb") and not force:
                    out_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                    continue
                colors = analyze_floor_material_colors(base_dir, _local_image_paths_from_item(item))
                if colors.get("average_rgb"):
                    item["average_rgb"] = colors.get("average_rgb")
                    item["average_hex"] = colors.get("average_hex")
                    item["dominant_colors_rgb"] = colors.get("dominant_colors_rgb") or []
                    item["dominant_colors_hex"] = colors.get("dominant_colors_hex") or []
                    summary["rows_updated"] += 1
                    changed = True
                else:
                    summary["rows_without_colors"] += 1
                out_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
        if changed:
            summary["updated_files"].append(str(jsonl_path))
            if not dry_run:
                jsonl_path.write_text("".join(out_lines), encoding="utf-8")
    return summary


def build_image_colors_table(root: Path, out_jsonl: Path | None = None, force: bool = False, dry_run: bool = False) -> dict:
    root = Path(root).expanduser()
    out_jsonl = Path(out_jsonl).expanduser() if out_jsonl else root / "image_colors.jsonl"
    existing: dict[str, dict] = {}
    if out_jsonl.exists() and not force:
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rel = str(item.get("path") or "")
                if rel and item.get("average_rgb"):
                    existing[rel] = item

    rows: list[dict] = []
    stats = {
        "images_scanned": 0,
        "images_with_color": 0,
        "images_without_color": 0,
        "reused_existing": 0,
        "out_jsonl": str(out_jsonl),
    }
    for image_path in sorted(p for p in root.rglob("*") if p.is_file() and _is_image_path(p)):
        rel = str(image_path.relative_to(root))
        stats["images_scanned"] += 1
        if rel in existing:
            rows.append(existing[rel])
            stats["reused_existing"] += 1
            stats["images_with_color"] += 1
            continue
        colors = analyze_floor_material_colors(image_path.parent, [image_path.name], k=0)
        row = {
            "version": "image_color.v1",
            "path": rel,
            "average_rgb": colors.get("average_rgb"),
            "average_hex": colors.get("average_hex"),
            "dominant_colors_rgb": colors.get("dominant_colors_rgb") or [],
            "dominant_colors_hex": colors.get("dominant_colors_hex") or [],
        }
        if row["average_rgb"]:
            stats["images_with_color"] += 1
        else:
            stats["images_without_color"] += 1
        rows.append(row)
    if not dry_run:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "normalize":
        materials = normalize_domlenta_catalog(
            Path(args.products_csv),
            Path(args.out_jsonl),
            analyze_images=not args.no_analyze_images,
        )
        print(f"Loaded products: {len(materials)}")
        print(f"Normalized materials: {len(materials)}")
        print(f"Saved: {Path(args.out_jsonl)}")
        return 0

    if args.command == "backfill-colors":
        summary = backfill_floor_material_colors(
            Path(args.materials),
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            all_records=bool(args.all_records),
        )
        print(f"Files scanned: {summary['files_scanned']}")
        print(f"Rows scanned: {summary['rows_scanned']}")
        print(f"Rows updated: {summary['rows_updated']}")
        print(f"Rows without images/colors: {summary['rows_without_colors']}")
        if summary["updated_files"]:
            print("Updated files:")
            for file_path in summary["updated_files"]:
                print(f"  {file_path}")
        return 0

    if args.command == "image-colors":
        summary = build_image_colors_table(
            Path(args.root),
            out_jsonl=Path(args.out_jsonl) if args.out_jsonl else None,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
        )
        print(f"Images scanned: {summary['images_scanned']}")
        print(f"Images with colors: {summary['images_with_color']}")
        print(f"Images without colors: {summary['images_without_color']}")
        print(f"Reused existing: {summary['reused_existing']}")
        print(f"Output table: {summary['out_jsonl']}")
        return 0

    if args.command == "select":
        selector = FloorMaterialSelector(Path(args.materials), Path(args.style_rules))
        print(f"Loaded materials: {len(selector.materials)}")
        selection = selector.select(
            prompt=args.prompt,
            style=args.style,
            room_type=args.room_type,
            room_description=args.room_description,
            top_k=max(1, args.top_k),
            room_id=args.room_id,
            llm_settings={
                "provider": args.llm_provider,
                "ollama_url": args.ollama_url,
                "ollama_model": args.ollama_model,
                "ollama_timeout": args.ollama_timeout,
                "ollama_temperature": args.ollama_temperature,
                "ollama_think": args.ollama_think,
                "ollama_num_ctx": args.ollama_num_ctx,
                "top_n": args.llm_top_n,
            },
        )
        selector.save_selection(selection, Path(args.out))
        selected = selection.selected_material
        print(f"Candidates after filtering: {selection.filtered_count}")
        if selected:
            print(f"Selected: {selected.sku} | {selected.name}")
            print(f"final_score: {selection.selection_reason.get('final_score')}")
            texture = selection.texture_candidate or {}
            print(f"texture_path: {texture.get('texture_path')}")
            print(f"texture_usable_in_blender: {selection.texture_usable_in_blender}")
            variation = ((texture.get("analysis") or {}).get("color_variation") or {})
            if variation:
                print(f"color_variation_score: {variation.get('variation_score')}")
                print(f"natural_darkening_risk: {variation.get('natural_darkening_risk')}")
                if variation.get("variation_map_path"):
                    print(f"color_variation_map: {variation.get('variation_map_path')}")
            if selection.llm_rerank:
                print(f"llm_rerank: {selection.llm_rerank.get('status')} | {selection.llm_rerank.get('reason')}")
        else:
            print("Selected: none")
        print(f"Saved: {Path(args.out)}")
        return 0

    if args.command == "apply-to-scene":
        scene = load_json(Path(args.scene_json))
        flooring = load_json(Path(args.flooring_json))
        updated = apply_flooring_to_scene(scene, flooring)
        write_json(updated, Path(args.out_scene_json))
        print(f"Saved: {Path(args.out_scene_json)}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
