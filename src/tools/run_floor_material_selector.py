from __future__ import annotations

import argparse
from pathlib import Path

from src.ChooseObject.floor_material_normalizer import normalize_domlenta_catalog
from src.ChooseObject.floor_material_selector import FloorMaterialSelector
from src.pipeline.flooring_stage import apply_flooring_to_scene, load_json, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and select floor covering materials.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="Normalize Domlenta products.csv to JSONL.")
    normalize.add_argument("--products-csv", required=True)
    normalize.add_argument("--out-jsonl", required=True)

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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "normalize":
        materials = normalize_domlenta_catalog(Path(args.products_csv), Path(args.out_jsonl))
        print(f"Loaded products: {len(materials)}")
        print(f"Normalized materials: {len(materials)}")
        print(f"Saved: {Path(args.out_jsonl)}")
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
