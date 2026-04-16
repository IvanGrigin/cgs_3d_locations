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
    from .supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog,
        load_supplier_catalog_json,
        read_json,
        write_json,
    )
except ImportError:
    from acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from apply_supplier_bindings import apply_supplier_bindings_to_json
    from supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog,
        load_supplier_catalog_json,
        read_json,
        write_json,
    )


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


def _derive_prerepair_scene_path(scene_out_path: Path) -> Path:
    return scene_out_path.with_name(f"{scene_out_path.stem}.prerepair{scene_out_path.suffix}")


def _run_repair_stage(args: argparse.Namespace, scene_json_path: Path, *, output_path: Path) -> Path:
    repair_mode = str(args.repair_mode or "none").strip().lower()
    if repair_mode == "none":
        return scene_json_path

    try:
        from .ml.scene_repair_solver import repair_scene_file
    except ImportError:
        from ml.scene_repair_solver import repair_scene_file

    repaired_path, _ = repair_scene_file(
        scene_path=scene_json_path,
        out_path=output_path,
        mode=repair_mode,
        scope=str(args.repair_scope or "supplier"),
        model_path=args.repair_model,
        meta_path=args.repair_meta,
        device=str(args.repair_device or "auto"),
        infer_steps=int(args.repair_infer_steps or 50),
        local_steps=int(args.repair_local_steps or 7),
        local_samples_per_step=int(args.repair_local_samples_per_step or 96),
        rounds=int(args.repair_rounds or 2),
        max_bad=int(args.repair_max_bad) if args.repair_max_bad is not None else None,
        room_margin=float(args.repair_room_margin or 0.02),
        collision_margin=float(args.repair_collision_margin or 0.012),
        seed=int(args.repair_seed or 0),
    )
    return repaired_path


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the supplier rebind pipeline end to end for a prepared scene.")
    ap.add_argument("--targets", required=True, help="Path to base_layout_targets.json")
    ap.add_argument("--input-scene-json", required=True, help="Path to scene_infinigen_clean.json or scene.v1.json")
    ap.add_argument("--supplier-db", action="append", default=[], help="Supplier SQLite catalog DB; may be repeated")
    ap.add_argument("--supplier-json", action="append", default=[], help="Supplier catalog JSON export; may be repeated")
    ap.add_argument("--site", action="append", default=None, help="Optional source_site filter; may be repeated")
    ap.add_argument("--rich-only", action="store_true", help="Use only rich supplier cards during matching")
    ap.add_argument("--top-k", type=int, default=8, help="Top-K candidates for heuristic ranking")
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
    ap.add_argument("--background", action="store_true", help="Run Blender render in background mode")
    ap.add_argument("--save-blend", default=None, help="Optional .blend output path for the final supplier scene")
    ap.add_argument("--render", default=None, help="Optional final render path")
    ap.add_argument("--force-tint", action="store_true", help="Pass force tint to Blender scene builder")
    ap.add_argument("--no-pack-assets", action="store_true", help="Do not pack assets when saving .blend")
    ap.add_argument("--repair-mode", choices=["none", "auto", "trained", "local"], default="none", help="Optional post-replacement scene repair stage; prefer trained for learned diffusion repair")
    ap.add_argument("--repair-scope", choices=["supplier", "all"], default="supplier", help="Which placements the repair stage may move")
    ap.add_argument("--repair-model", default=None, help="Optional trained repair checkpoint; defaults to src/ml/models/diffusion_model_20260212.pt when present")
    ap.add_argument("--repair-meta", default=None, help="Optional repair meta json with cat2id")
    ap.add_argument("--repair-device", choices=["auto", "cpu", "mps", "cuda"], default="auto", help="Device for trained repair inference")
    ap.add_argument("--repair-infer-steps", type=int, default=50, help="Denoising steps for trained repair inference")
    ap.add_argument("--repair-local-steps", type=int, default=7, help="Noise levels for local debug fallback")
    ap.add_argument("--repair-local-samples-per-step", type=int, default=96, help="Proposal count per local debug fallback step")
    ap.add_argument("--repair-rounds", type=int, default=2, help="How many repair passes to run")
    ap.add_argument("--repair-max-bad", type=int, default=None, help="Repair only first K invalid placements per pass")
    ap.add_argument("--repair-room-margin", type=float, default=0.02, help="Room-boundary tolerance for repair validation")
    ap.add_argument("--repair-collision-margin", type=float, default=0.012, help="Collision margin for repair validation")
    ap.add_argument("--repair-seed", type=int, default=0, help="Seed for local repair fallback")
    ap.add_argument("--manifest-out", default=None, help="Optional manifest JSON with all pipeline outputs")
    return ap


def main() -> None:
    args = build_cli().parse_args()

    targets_path = Path(args.targets).expanduser().resolve()
    input_scene_path = Path(args.input_scene_json).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else targets_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

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
    prerepair_scene_out = _derive_prerepair_scene_path(scene_out) if str(args.repair_mode or "none").strip().lower() != "none" else scene_out
    assets_dir = Path(args.assets_dir).expanduser().resolve() if args.assets_dir else (run_dir / "supplier_assets")
    assets_db = Path(args.assets_db).expanduser().resolve() if args.assets_db else (run_dir / "supplier_scene_assets.db")

    bindings = build_bindings_with_candidates(
        targets_json_path=targets_path,
        catalog_rows=rows,
        top_k=int(args.top_k),
        user_preferences=supplier_user_preferences,
        llm_settings=llm_settings,
    )
    write_json(bindings_out, bindings)

    assets_bindings_path = acquire_assets_for_bindings_json(
        bindings_json_path=bindings_out,
        output_json_path=assets_bindings_out,
        db_path=assets_db,
        out_dir=assets_dir,
        blender_bin=args.blender,
        catalog_json_paths=[Path(x).expanduser().resolve() for x in (args.supplier_json or []) if str(x).strip()],
    )

    final_scene_path = apply_supplier_bindings_to_json(
        input_json_path=input_scene_path,
        bindings_json_path=assets_bindings_path,
        output_json_path=prerepair_scene_out,
        require_local_asset=bool(args.require_local_asset),
    )

    pre_repair_scene_path: Path | None = None
    if str(args.repair_mode or "none").strip().lower() != "none":
        pre_repair_scene_path = Path(final_scene_path).expanduser().resolve()
        final_scene_path = _run_repair_stage(args, pre_repair_scene_path, output_path=scene_out)

    reference_blend = _infer_reference_blend(input_scene_path) or _infer_reference_blend(final_scene_path)
    _run_blender_render(args, final_scene_path, reference_blend=reference_blend)

    final_scene_data = read_json(final_scene_path)

    manifest = {
        "targets_json": str(targets_path.resolve()),
        "input_scene_json": str(input_scene_path.resolve()),
        "bindings_json": str(bindings_out.resolve()),
        "assets_bindings_json": str(Path(assets_bindings_path).resolve()),
        "scene_out_json": str(Path(final_scene_path).resolve()),
        "scene_pre_repair_json": str(pre_repair_scene_path.resolve()) if pre_repair_scene_path else None,
        "assets_dir": str(assets_dir.resolve()),
        "assets_db": str(assets_db.resolve()),
        "catalog_row_count": len(rows),
        "llm_settings": llm_settings,
        "require_local_asset": bool(args.require_local_asset),
        "repair_settings": {
            "mode": str(args.repair_mode or "none"),
            "scope": str(args.repair_scope or "supplier"),
            "model": str(args.repair_model or ""),
            "meta": str(args.repair_meta or ""),
            "device": str(args.repair_device or "auto"),
            "infer_steps": int(args.repair_infer_steps or 50),
            "local_steps": int(args.repair_local_steps or 7),
            "local_samples_per_step": int(args.repair_local_samples_per_step or 96),
            "rounds": int(args.repair_rounds or 2),
            "max_bad": int(args.repair_max_bad) if args.repair_max_bad is not None else None,
            "room_margin": float(args.repair_room_margin or 0.02),
            "collision_margin": float(args.repair_collision_margin or 0.012),
            "seed": int(args.repair_seed or 0),
        },
        "bindings_meta": (read_json(bindings_out).get("meta") or {}),
        "asset_acquisition_meta": (read_json(assets_bindings_path).get("meta") or {}).get("asset_acquisition"),
        "scene_supplier_summary": (final_scene_data.get("meta") or {}).get("supplier_binding_summary"),
        "scene_repair_summary": (final_scene_data.get("meta") or {}).get("scene_repair_solver"),
    }
    manifest_out = Path(args.manifest_out).expanduser().resolve() if args.manifest_out else (run_dir / f"supplier_pipeline.{suffix}.manifest.json")
    write_json(manifest_out, manifest)

    print(f"catalog_rows = {len(rows)}")
    print(f"bindings_json = {bindings_out}")
    print(f"assets_bindings_json = {assets_bindings_path}")
    print(f"scene_out_json = {final_scene_path}")
    print(f"manifest_json = {manifest_out}")


if __name__ == "__main__":
    main()
