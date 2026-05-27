#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script runs the supplier rebind pipeline end to end for an existing scene.
It builds supplier bindings, optionally reranks them with an LLM, acquires assets,
applies only valid supplier replacements, and can render the final scene in Blender.
The goal is to replace fragile manual step chains with one reproducible command.
It supports mixed catalog sources, user preferences, and scene-local asset caches.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from .apply_supplier_bindings import apply_supplier_bindings_to_json
    from .pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
    from .supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog,
        load_supplier_catalog_json,
        read_json,
        write_json,
    )
    from tools.run_procedural_room_supplier import enrich_missing_assets_with_trellis
except ImportError:
    from acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from apply_supplier_bindings import apply_supplier_bindings_to_json
    from pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
    from supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog,
        load_supplier_catalog_json,
        read_json,
        write_json,
    )
    tools_dir = Path(__file__).resolve().parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from run_procedural_room_supplier import enrich_missing_assets_with_trellis


FORCE_SOFT_BED_KEEP_CATEGORIES = {"BlanketFactory", "MattressFactory", "TowelFactory"}
FORCE_CATEGORY_GROUPS = {
    "PillowFactory": "pillow",
    "RugFactory": "rug",
    "BlanketFactory": "blanket",
    "MattressFactory": "mattress",
    "TowelFactory": "towel",
}


def _force_replace_all_targets(targets_path: Path, out_path: Path) -> Path:
    data = read_json(targets_path)
    targets = data.get("targets") if isinstance(data, dict) else None
    if not isinstance(targets, list):
        raise RuntimeError(f"Invalid layout targets JSON: {targets_path}")

    changed = 0
    kept_soft = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        category = str(target.get("category") or "").strip()
        if category in FORCE_CATEGORY_GROUPS:
            target["semantic_group"] = FORCE_CATEGORY_GROUPS[category]
        if category in FORCE_SOFT_BED_KEEP_CATEGORIES:
            target["replacement_policy"] = "keep_generated"
            target["replacement_reason"] = "force_full_supplier_soft_bed_part_suppressed_with_bed"
            kept_soft += 1
            continue
        target["replacement_policy"] = "replace_with_supplier"
        target["replacement_reason"] = "force_full_supplier_pass"
        target["force_replace_with_supplier"] = True
        meta = target.get("meta") if isinstance(target.get("meta"), dict) else {}
        meta["force_replace_with_supplier"] = True
        target["meta"] = meta
        changed += 1

    meta_root = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta_root["force_full_supplier_pass"] = {
        "enabled": True,
        "replace_target_count": changed,
        "soft_bed_keep_count": kept_soft,
        "soft_bed_policy": "suppressed_when_supplier_bed_replaces_anchor",
    }
    data["meta"] = meta_root
    write_json(out_path, data)
    return out_path


def _load_catalog_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    sites = {str(x).strip() for x in (args.site or []) if str(x).strip()} or None
    rows: list[dict[str, Any]] = []
    db_paths = [Path(x).expanduser().resolve() for x in (args.supplier_db or []) if str(x).strip()]
    json_paths = [Path(x).expanduser().resolve() for x in (args.supplier_json or []) if str(x).strip()]
    if db_paths:
        rows.extend(load_supplier_catalog(db_paths, sites=sites, rich_only=bool(args.rich_only)))
    if json_paths:
        rows.extend(load_supplier_catalog_json(json_paths, sites=sites, rich_only=bool(args.rich_only)))
    return rows


def _llm_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "provider": str(args.llm_provider or "none"),
        "ollama_url": str(args.ollama_url or "http://127.0.0.1:11434"),
        "ollama_model": str(args.ollama_model or "gpt-oss:20b"),
        "ollama_timeout": int(args.ollama_timeout or 180),
        "ollama_temperature": float(args.ollama_temperature or 0.0),
        "top_n": int(args.llm_top_n or 5),
    }


def _infer_reference_blend(scene_json_path: Path) -> Path | None:
    try:
        data = read_json(scene_json_path)
    except Exception:
        data = {}

    json_dir = scene_json_path.expanduser().resolve().parent
    meta = data.get("meta") if isinstance(data, dict) else {}
    placement_meta = meta.get("placement_meta") if isinstance(meta, dict) else {}
    raw_scene_blend = str((placement_meta or {}).get("scene_blend") or "").strip()

    candidates: list[Path] = []
    candidates.append(json_dir / "scene_infinigen_clean.blend")
    candidates.append(json_dir / "infinigen_clean_scene.blend")
    if raw_scene_blend:
        raw_path = Path(raw_scene_blend).expanduser()
        candidates.append(raw_path)
        candidates.append(json_dir / raw_path.name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _run_blender_render(args: argparse.Namespace, scene_json_path: Path, *, reference_blend: Path | None = None) -> None:
    if not (args.render or args.save_blend):
        return

    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "Plasement" / "BlenderVisualizePlacement.py").resolve()),
        "--json",
        str(scene_json_path.resolve()),
    ]
    if args.background:
        cmd.append("--background")
    if args.blender:
        cmd += ["--blender", str(Path(args.blender).expanduser())]
    if reference_blend is not None:
        cmd += ["--reference-blend", str(reference_blend.expanduser().resolve())]
    if args.save_blend:
        cmd += ["--save-blend", str(Path(args.save_blend).expanduser().resolve())]
    if args.render:
        cmd += ["--render", str(Path(args.render).expanduser().resolve())]
    if args.force_tint:
        cmd += ["--force-tint"]
    if args.no_pack_assets:
        cmd += ["--no-pack-assets"]
    subprocess.run(cmd, check=True)


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the supplier rebind pipeline end to end for a prepared scene.")
    ap.add_argument("--targets", required=True, help="Path to base_layout_targets.json")
    ap.add_argument("--input-scene-json", required=True, help="Path to scene_infinigen_clean.json or scene.v1.json")
    ap.add_argument("--supplier-db", action="append", default=[], help="Supplier SQLite catalog DB; may be repeated")
    ap.add_argument("--supplier-json", action="append", default=[], help="Supplier catalog JSON export; may be repeated")
    ap.add_argument("--site", action="append", default=None, help="Optional source_site filter; may be repeated")
    ap.add_argument("--rich-only", action="store_true", help="Use only rich supplier cards during matching")
    ap.add_argument("--top-k", type=int, default=8, help="Top-K candidates for heuristic ranking")
    ap.add_argument(
        "--selection-mode",
        default="optimal",
        choices=[
            "cheapest",
            "min_price",
            "lowest_price",
            "cheapest_top20",
            "cheap_top20",
            "optimal",
            "best_match",
            "best_match_v1",
            "best_match_v2",
            "best_visual_reference",
            "best_suitable",
            "most_suitable",
            "legacy_asset_priority",
        ],
        help="Design-aware supplier selection mode.",
    )
    ap.add_argument("--selection-strategy", default="balanced", help="Legacy supplier ordering strategy.")
    ap.add_argument("--user-preferences-json", default=None, help="Optional supplier matcher user preferences JSON")
    ap.add_argument("--llm-provider", choices=["none", "ollama"], default="none", help="Optional supplier matcher reranker")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama URL for supplier reranking")
    ap.add_argument("--ollama-model", default="gpt-oss:20b", help="Ollama model for supplier reranking")
    ap.add_argument("--ollama-timeout", type=int, default=180, help="Ollama timeout for supplier reranking")
    ap.add_argument("--ollama-temperature", type=float, default=0.0, help="Ollama temperature for supplier reranking")
    ap.add_argument("--llm-top-n", type=int, default=5, help="How many heuristic supplier candidates to send to the LLM")
    ap.add_argument("--run-dir", default=None, help="Output run directory; defaults to targets parent")
    ap.add_argument("--bindings-out", default=None, help="Optional bindings output path override")
    ap.add_argument("--assets-bindings-out", default=None, help="Optional enriched bindings output path override")
    ap.add_argument("--scene-out", default=None, help="Optional supplier scene output path override")
    ap.add_argument("--assets-dir", default=None, help="Scene-local supplier assets directory")
    ap.add_argument("--assets-db", default=None, help="Scene-local supplier assets SQLite DB")
    ap.add_argument("--blender", default=None, help="Optional Blender binary path for asset conversion and rendering")
    ap.add_argument("--require-local-asset", action="store_true", help="Apply only replacements with local real mesh assets")
    ap.add_argument(
        "--supplier-asset-fallback-mode",
        choices=["none", "fbx_obj_proxy", "fbx_obj_trellis_proxy"],
        default="none",
        help="Fallback policy for selected candidates without a local real mesh.",
    )
    ap.add_argument(
        "--force-replace-all-targets",
        action="store_true",
        help="Patch layout targets so every standalone object is supplier-replaceable; generated bed soft parts are suppressed with replaced beds.",
    )
    ap.add_argument(
        "--suppress-generated-bedding",
        action="store_true",
        help="Remove generated mattress/blanket/towel/pillow parts around supplier-replaced beds.",
    )
    ap.add_argument("--keep-unresolved-candidates", action="store_true", help="Keep selected candidates after asset acquisition even if no local mesh was found.")
    ap.add_argument("--trellis-generate-missing-assets", action="store_true", help="Use TRELLIS.2 to generate GLBs for selected candidates that still lack local assets.")
    ap.add_argument("--trellis-max-assets", type=int, default=0)
    ap.add_argument("--trellis-skip-categories", default="")
    ap.add_argument("--trellis-ikea-mebelru-images-only", action="store_true")
    ap.add_argument(
        "--trellis-force-all-selected-assets",
        action="store_true",
        help="Generate TRELLIS.2 GLBs for selected supplier candidates even when a local FBX/OBJ/GLB already exists.",
    )
    ap.add_argument(
        "--trellis-force-image-only",
        action="store_true",
        help="Disable direct supplier FBX/OBJ/GLB shortcuts and force TRELLIS.2 image-to-3D generation.",
    )
    ap.add_argument("--trellis-server-host", default="")
    ap.add_argument("--trellis-server-port", type=int, default=28553)
    ap.add_argument("--trellis-server-user", default="root")
    ap.add_argument("--trellis-ssh-key", default="")
    ap.add_argument("--trellis-remote-root", default="/workspace/trellis2_supplier_jobs")
    ap.add_argument("--trellis-remote-trellis-root", default="/workspace/TRELLIS.2")
    ap.add_argument("--trellis-remote-model-dir", default="/workspace/models/TRELLIS.2-4B")
    ap.add_argument("--trellis-remote-python", default="/venv/trellis2/bin/python")
    ap.add_argument("--trellis-remote-worker-root", default="/workspace/trellis2_worker")
    ap.add_argument("--trellis-remote-worker-timeout-sec", type=float, default=1800.0)
    ap.add_argument("--trellis-remote-worker-poll-sec", type=float, default=2.0)
    ap.add_argument("--trellis-remote-persistent-worker", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trellis-remote-text-model-dir", default="")
    ap.add_argument("--trellis-remote-cuda-visible-devices", default="0")
    ap.add_argument("--trellis-multi-mode", default="stochastic", choices=["stochastic", "multidiffusion"])
    ap.add_argument("--trellis-max-images", type=int, default=2)
    ap.add_argument("--trellis-oom-retry-max-images", type=int, default=1)
    ap.add_argument("--trellis-max-candidate-pool", type=int, default=0)
    ap.add_argument("--trellis-disable-after-oom", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trellis-seed", type=int, default=1)
    ap.add_argument("--trellis-sparse-steps", type=int, default=4)
    ap.add_argument("--trellis-slat-steps", type=int, default=4)
    ap.add_argument("--trellis-texture-size", type=int, default=256)
    ap.add_argument("--trellis-simplify", type=float, default=0.98)
    ap.add_argument("--trellis-pipeline-type", type=int, default=512)
    ap.add_argument("--trellis-ss-guidance-strength", type=float, default=7.5)
    ap.add_argument("--trellis-slat-guidance-strength", type=float, default=3.0)
    ap.add_argument("--trellis-decimation-target", type=int, default=50000)
    ap.add_argument("--trellis-pre-export-simplify-target", type=int, default=0)
    ap.add_argument("--trellis-no-remesh", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--trellis-remesh-band", type=int, default=1)
    ap.add_argument("--trellis-remesh-project", type=float, default=0.0)
    ap.add_argument("--trellis-no-webp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trellis-image-size", type=int, default=336)
    ap.add_argument("--trellis-fill-holes-resolution", type=int, default=256)
    ap.add_argument("--trellis-fill-holes-num-views", type=int, default=120)
    ap.add_argument("--trellis-remote-runner-path", default="")
    ap.add_argument("--trellis-vlm-single-object-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trellis-vlm-provider", default="ollama", choices=["ollama", "openai", "openrouter"])
    ap.add_argument("--trellis-vlm-ollama-url", default="http://127.0.0.1:11435")
    ap.add_argument("--trellis-vlm-model", default="llama3.2-vision:11b")
    ap.add_argument("--trellis-vlm-timeout", type=int, default=120)
    ap.add_argument("--trellis-vlm-unload-after-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trellis-text-fallback-if-no-single-image", action="store_true", default=True)
    ap.add_argument("--trellis-progress-log", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trellis-max-failures-per-candidate", type=int, default=2)
    ap.add_argument("--trellis-allow-proxy-fallback", action="store_true")
    ap.add_argument("--background", action="store_true", help="Run Blender render in background mode")
    ap.add_argument("--save-blend", default=None, help="Optional .blend output path for the final supplier scene")
    ap.add_argument("--render", default=None, help="Optional final render path")
    ap.add_argument("--force-tint", action="store_true", help="Pass force tint to Blender scene builder")
    ap.add_argument("--no-pack-assets", action="store_true", help="Do not pack assets when saving .blend")
    ap.add_argument("--manifest-out", default=None, help="Optional manifest JSON with all pipeline outputs")
    add_scene_repair_arguments(ap)
    return ap


def main() -> None:
    args = build_cli().parse_args()

    original_targets_path = Path(args.targets).expanduser().resolve()
    input_scene_path = Path(args.input_scene_json).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else original_targets_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    targets_path = original_targets_path
    if bool(args.force_replace_all_targets):
        targets_path = _force_replace_all_targets(
            original_targets_path,
            run_dir / f"{original_targets_path.stem}.force_full_supplier.json",
        )

    supplier_user_preferences = None
    if str(args.user_preferences_json or "").strip():
        raw = read_json(args.user_preferences_json)
        if not isinstance(raw, dict):
            raise RuntimeError("user preferences JSON must be an object")
        supplier_user_preferences = raw

    rows = _load_catalog_rows(args)
    if not rows:
        raise RuntimeError("No supplier catalog rows were loaded")

    llm_settings = _llm_settings(args)
    suffix = "llm" if str(args.llm_provider or "none").strip().lower() != "none" else "heuristic"
    bindings_out = Path(args.bindings_out).expanduser().resolve() if args.bindings_out else (run_dir / f"base_supplier_bindings.{suffix}.json")
    assets_bindings_out = Path(args.assets_bindings_out).expanduser().resolve() if args.assets_bindings_out else (run_dir / f"{bindings_out.stem}.assets.json")
    scene_out = Path(args.scene_out).expanduser().resolve() if args.scene_out else (run_dir / f"scene_supplier.{suffix}.v1.json")
    assets_dir = Path(args.assets_dir).expanduser().resolve() if args.assets_dir else (run_dir / "supplier_assets")
    assets_db = Path(args.assets_db).expanduser().resolve() if args.assets_db else (run_dir / "supplier_scene_assets.db")

    bindings = build_bindings_with_candidates(
        targets_json_path=targets_path,
        catalog_rows=rows,
        top_k=int(args.top_k),
        selection_strategy=str(args.selection_strategy or "balanced"),
        user_preferences=supplier_user_preferences,
        llm_settings=llm_settings,
        selection_mode=str(args.selection_mode or "optimal"),
    )
    write_json(bindings_out, bindings)

    uses_mesh_or_proxy_fallback = str(args.supplier_asset_fallback_mode or "none") in {"fbx_obj_proxy", "fbx_obj_trellis_proxy"}
    assets_bindings_path = acquire_assets_for_bindings_json(
        bindings_json_path=bindings_out,
        output_json_path=assets_bindings_out,
        db_path=assets_db,
        out_dir=assets_dir,
        blender_bin=args.blender,
        catalog_json_paths=[Path(x).expanduser().resolve() for x in (args.supplier_json or []) if str(x).strip()],
        keep_unresolved_candidates=bool(
            args.keep_unresolved_candidates
            or args.trellis_generate_missing_assets
            or (uses_mesh_or_proxy_fallback and not args.require_local_asset)
        ),
    )

    trellis_generation_report = None
    if bool(args.trellis_generate_missing_assets):
        if not str(args.trellis_server_host or "").strip():
            raise RuntimeError("--trellis-generate-missing-assets requires --trellis-server-host")
        assets_bindings_path, trellis_generation_report = enrich_missing_assets_with_trellis(
            bindings_json_path=Path(assets_bindings_path).expanduser().resolve(),
            output_json_path=run_dir / f"{Path(assets_bindings_path).stem}.trellis.json",
            out_dir=run_dir,
            args=args,
        )

    final_scene_path = apply_supplier_bindings_to_json(
        input_json_path=input_scene_path,
        bindings_json_path=assets_bindings_path,
        output_json_path=scene_out,
        require_local_asset=bool(args.require_local_asset),
        fallback_mode=str(args.supplier_asset_fallback_mode or "none"),
        preserve_generated_bedding=not bool(args.suppress_generated_bedding or args.force_replace_all_targets),
    )
    repaired_scene_path, scene_repair_info = maybe_repair_scene_json(
        args=args,
        scene_json_path=final_scene_path,
        run_dir=run_dir,
        tag=f"supplier_{suffix}",
    )

    reference_blend = _infer_reference_blend(input_scene_path) or _infer_reference_blend(repaired_scene_path)
    _run_blender_render(args, repaired_scene_path, reference_blend=reference_blend)

    final_scene_data = read_json(repaired_scene_path)

    manifest = {
        "targets_json": str(targets_path.resolve()),
        "original_targets_json": str(original_targets_path.resolve()),
        "input_scene_json": str(input_scene_path.resolve()),
        "bindings_json": str(bindings_out.resolve()),
        "assets_bindings_json": str(Path(assets_bindings_path).resolve()),
        "scene_out_json": str(Path(final_scene_path).resolve()),
        "scene_render_json": str(Path(repaired_scene_path).resolve()),
        "assets_dir": str(assets_dir.resolve()),
        "assets_db": str(assets_db.resolve()),
        "catalog_row_count": len(rows),
        "llm_settings": llm_settings,
        "require_local_asset": bool(args.require_local_asset),
        "supplier_asset_fallback_mode": str(args.supplier_asset_fallback_mode or "none"),
        "force_replace_all_targets": bool(args.force_replace_all_targets),
        "trellis_missing_asset_generation": trellis_generation_report,
        "bindings_meta": (read_json(bindings_out).get("meta") or {}),
        "asset_acquisition_meta": (read_json(assets_bindings_path).get("meta") or {}).get("asset_acquisition"),
        "scene_supplier_summary": (final_scene_data.get("meta") or {}).get("supplier_binding_summary"),
    }
    if scene_repair_info is not None:
        manifest["scene_repair"] = scene_repair_info
    manifest_out = Path(args.manifest_out).expanduser().resolve() if args.manifest_out else (run_dir / f"supplier_pipeline.{suffix}.manifest.json")
    write_json(manifest_out, manifest)

    print(f"catalog_rows = {len(rows)}")
    print(f"bindings_json = {bindings_out}")
    print(f"assets_bindings_json = {assets_bindings_path}")
    print(f"scene_out_json = {final_scene_path}")
    if repaired_scene_path != final_scene_path:
        print(f"scene_render_json = {repaired_scene_path}")
    print(f"manifest_json = {manifest_out}")


if __name__ == "__main__":
    main()
