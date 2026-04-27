#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/run_pipeline.py

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

try:
    from .acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from .apply_supplier_bindings import apply_supplier_bindings_to_json
    from .layout_targets import create_layout_selection_stub_artifacts
    from .supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog_json,
        read_json as read_supplier_matcher_json,
    )
    from .pipeline_artifacts import (
        build_scene_artifacts,
        choose_scene_for_render,
        normalize_json_artifact,
        run_blender_for_mode,
    )
    from .pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
    from .pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection, write_json as write_flooring_json
    from .pipeline.wall_stage import apply_wall_material_to_scene, run_wall_selection, write_json as write_wall_json
    from .pipeline_config import (
        DEFAULT_LEGO_GENERATION_PRESETS,
        DEFAULT_PATHS_CONFIG,
        ModeOutputs,
        PLACER_SPECS,
        PlacementArtifacts,
        apply_config_defaults,
        build_runtime_paths,
        load_yaml,
        make_mode_run_dir,
        parse_modes,
        project_root_from_config,
        read_prompt_from_args,
        write_json,
    )
    from .pipeline_runners import (
        execute_placer,
        resolve_lego_generation_params,
        run_choose_stage,
        run_lego_generate_from_scratch,
    )
    from .style_profiles import attach_style_hint_to_room_json
    from .style_prompt_analyzer import analyze_prompt_to_style_profile
except ImportError:
    from acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from apply_supplier_bindings import apply_supplier_bindings_to_json
    from layout_targets import create_layout_selection_stub_artifacts
    from supplier_layout_matcher import (
        build_bindings_with_candidates,
        load_supplier_catalog_json,
        read_json as read_supplier_matcher_json,
    )
    from pipeline_artifacts import (
        build_scene_artifacts,
        choose_scene_for_render,
        normalize_json_artifact,
        run_blender_for_mode,
    )
    from pipeline_scene_repair import add_scene_repair_arguments, maybe_repair_scene_json
    from pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection, write_json as write_flooring_json
    from pipeline.wall_stage import apply_wall_material_to_scene, run_wall_selection, write_json as write_wall_json
    from pipeline_config import (
        DEFAULT_LEGO_GENERATION_PRESETS,
        DEFAULT_PATHS_CONFIG,
        ModeOutputs,
        PLACER_SPECS,
        PlacementArtifacts,
        apply_config_defaults,
        build_runtime_paths,
        load_yaml,
        make_mode_run_dir,
        parse_modes,
        project_root_from_config,
        read_prompt_from_args,
        write_json,
    )
    from pipeline_runners import (
        execute_placer,
        resolve_lego_generation_params,
        run_choose_stage,
        run_lego_generate_from_scratch,
    )
    from style_profiles import attach_style_hint_to_room_json
    from style_prompt_analyzer import analyze_prompt_to_style_profile


def _build_layout_selection_stub_for_artifacts(
    *,
    artifacts: PlacementArtifacts,
    run_dir: Path,
    prefix: str = "",
) -> dict[str, str]:
    source_json_path = artifacts.scene_v1 if artifacts.scene_v1 and artifacts.scene_v1.is_file() else artifacts.placement_v1
    return create_layout_selection_stub_artifacts(
        source_json_path=source_json_path,
        run_dir=run_dir,
        prefix=prefix,
    )


def _is_fatal_disk_full_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "remote_disk_full" in text
        or "no space left on device" in text
        or "disk full" in text
    )


def _apply_supplier_bindings_for_artifacts(
    *,
    artifacts: PlacementArtifacts,
    run_dir: Path,
    bindings_json_path: Path,
    require_local_asset: bool,
) -> dict[str, Any]:
    supplier_placement_v1 = run_dir / "placement_supplier.v1.json"
    apply_supplier_bindings_to_json(
        input_json_path=artifacts.placement_v1,
        bindings_json_path=bindings_json_path,
        output_json_path=supplier_placement_v1,
        require_local_asset=require_local_asset,
    )

    supplier_scene_v1 = None
    if artifacts.scene_v1 and artifacts.scene_v1.is_file():
        supplier_scene_v1 = run_dir / "scene_supplier.v1.json"
        apply_supplier_bindings_to_json(
            input_json_path=artifacts.scene_v1,
            bindings_json_path=bindings_json_path,
            output_json_path=supplier_scene_v1,
            require_local_asset=require_local_asset,
        )

    supplier_data = json.loads(supplier_placement_v1.read_text(encoding="utf-8"))
    supplier_summary = ((supplier_data.get("meta") or {}).get("supplier_binding_summary") or {})
    return {
        "bindings_json": str(bindings_json_path.resolve()),
        "placement_v1": str(supplier_placement_v1.resolve()),
        "scene_v1": str(supplier_scene_v1.resolve()) if supplier_scene_v1 else None,
        "require_local_asset": bool(require_local_asset),
        "summary": supplier_summary,
    }


def _acquire_supplier_assets_for_bindings(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    bindings_json_path: Path,
) -> tuple[Path, dict[str, Any]]:
    out_dir = Path(str(args.supplier_assets_dir or (run_dir / "supplier_assets"))).expanduser().resolve()
    db_path = Path(str(args.supplier_assets_db or (run_dir / "supplier_scene_assets.db"))).expanduser().resolve()
    enriched_bindings_path = run_dir / f"{bindings_json_path.stem}.assets.json"
    supplier_catalog_jsons = [Path(x).expanduser().resolve() for x in (args.supplier_catalog_json or []) if str(x).strip()]

    out_path = acquire_assets_for_bindings_json(
        bindings_json_path=bindings_json_path,
        output_json_path=enriched_bindings_path,
        db_path=db_path,
        out_dir=out_dir,
        blender_bin=args.supplier_assets_blender or args.blender,
        catalog_json_paths=supplier_catalog_jsons,
    )
    asset_data = json.loads(out_path.read_text(encoding="utf-8"))
    summary = ((asset_data.get("meta") or {}).get("asset_acquisition") or {})
    return out_path, {
        "bindings_json": str(out_path.resolve()),
        "db_path": str(db_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "summary": summary,
    }


def _resolve_supplier_bindings_json(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    layout_targets_json_path: str,
    supplier_user_preferences_json: str | None = None,
) -> Path | None:
    explicit = str(args.supplier_bindings_json or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    supplier_catalog_jsons = [Path(x).expanduser().resolve() for x in (args.supplier_catalog_json or []) if str(x).strip()]
    if not supplier_catalog_jsons:
        return None

    sites = {str(x).strip() for x in (args.supplier_site or []) if str(x).strip()} or None
    catalog_rows = load_supplier_catalog_json(
        supplier_catalog_jsons,
        sites=sites,
        rich_only=bool(args.supplier_rich_only),
    )

    supplier_user_preferences: dict[str, Any] | None = None
    supplier_preferences_path = str(
        supplier_user_preferences_json
        or getattr(args, "supplier_user_preferences_json", "")
        or ""
    ).strip()
    if supplier_preferences_path:
        raw = read_supplier_matcher_json(supplier_preferences_path)
        if not isinstance(raw, dict):
            raise RuntimeError("supplier user preferences JSON must be an object")
        supplier_user_preferences = raw

    supplier_llm_provider = str(getattr(args, "supplier_llm_provider", "none") or "none").strip().lower()
    llm_settings = {
        "provider": supplier_llm_provider,
        "ollama_url": str(getattr(args, "supplier_ollama_url", None) or args.ollama_url or "http://127.0.0.1:11434"),
        "ollama_model": str(getattr(args, "supplier_ollama_model", None) or args.ollama_model or "gpt-oss:20b"),
        "ollama_timeout": int(getattr(args, "supplier_ollama_timeout", None) or args.ollama_timeout or 180),
        "ollama_temperature": float(getattr(args, "supplier_ollama_temperature", None) or 0.0),
        "top_n": int(getattr(args, "supplier_llm_top_n", None) or min(max(int(args.supplier_top_k), 1), 5)),
    }

    out_suffix = "llm" if supplier_llm_provider != "none" else "heuristic"
    out_path = run_dir / f"base_supplier_bindings.{out_suffix}.json"
    result = build_bindings_with_candidates(
        targets_json_path=Path(layout_targets_json_path).expanduser().resolve(),
        catalog_rows=catalog_rows,
        top_k=int(args.supplier_top_k),
        user_preferences=supplier_user_preferences,
        llm_settings=llm_settings,
    )
    write_json(out_path, result)
    return out_path


def _flooring_style_label(style_profile: dict[str, Any]) -> str | None:
    raw = str(style_profile.get("style_label") or "").strip().lower().replace("-", "_")
    aliases = {
        "modern": "contemporary",
        "industrial": "loft",
        "classicism": "classic",
        "neoclassical": "classic",
        "wabi_sabi": "japandi",
        "mid_century_modern": "contemporary",
        "art_deco": "classic",
        "rustic": "classic",
        "coastal": "scandinavian",
    }
    return aliases.get(raw, raw or None)


def _flooring_room_type(style_profile: dict[str, Any], scene_json_path: Path) -> str | None:
    raw = str(style_profile.get("room_type") or "").strip().lower().replace(" ", "_")
    aliases = {
        "bedroom": "bedroom",
        "livingroom": "living_room",
        "living_room": "living_room",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "diningroom": "living_room",
        "dining_room": "living_room",
    }
    if raw in aliases:
        return aliases[raw]
    try:
        data = json.loads(scene_json_path.read_text(encoding="utf-8"))
        room = data.get("room") if isinstance(data, dict) else {}
        if isinstance(room, dict):
            scene_room = str(room.get("room_type") or "").strip().lower()
            return aliases.get(scene_room, scene_room or None)
    except Exception:
        return None
    return None


def _maybe_apply_flooring_to_scene(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scene_json_path: Path,
    prompt_text: str,
    style_profile: dict[str, Any],
    room_id: str,
    suffix: str,
) -> tuple[Path, dict[str, Any] | None]:
    if bool(getattr(args, "no_flooring", False)):
        return scene_json_path, None

    materials_path = Path(str(getattr(args, "flooring_materials", "") or "")).expanduser()
    style_rules_path = Path(str(getattr(args, "flooring_style_rules", "") or "")).expanduser()
    if not materials_path.is_absolute():
        materials_path = (Path.cwd() / materials_path).resolve()
    if not style_rules_path.is_absolute():
        style_rules_path = (Path.cwd() / style_rules_path).resolve()

    if not materials_path.is_file():
        print(f"⏭ flooring: каталог не найден, пропуск: {materials_path}")
        return scene_json_path, None
    if not style_rules_path.is_file():
        print(f"⏭ flooring: правила стилей не найдены, пропуск: {style_rules_path}")
        return scene_json_path, None

    selection_path = run_dir / f"flooring.selection{suffix}.v1.json"
    scene_out_path = run_dir / f"{scene_json_path.stem}.flooring.v1.json"
    style = _flooring_style_label(style_profile)
    room_type = _flooring_room_type(style_profile, scene_json_path)
    llm_settings = {
        "provider": str(getattr(args, "flooring_llm_provider", "ollama") or "ollama"),
        "ollama_url": str(getattr(args, "flooring_ollama_url", None) or getattr(args, "ollama_url", None) or "http://127.0.0.1:11434"),
        "ollama_model": str(getattr(args, "flooring_ollama_model", None) or getattr(args, "ollama_model", None) or "gpt-oss:20b"),
        "ollama_timeout": int(getattr(args, "flooring_ollama_timeout", None) or getattr(args, "ollama_timeout", None) or 180),
        "ollama_temperature": float(getattr(args, "flooring_ollama_temperature", 0.0) or 0.0),
        "ollama_num_ctx": int(getattr(args, "flooring_ollama_num_ctx", 8192) or 8192),
        "top_n": int(getattr(args, "flooring_llm_top_n", 5) or 5),
    }

    flooring_prompt_text = _flooring_prompt_for_selector(prompt_text, style_profile, run_dir)
    print("🧱 flooring: подбор покрытия пола")
    selection = run_flooring_selection(
        prompt=flooring_prompt_text,
        style=style,
        room_type=room_type,
        room_description=str(style_profile.get("style_hint") or ""),
        room_id=room_id,
        materials_path=materials_path,
        style_rules_path=style_rules_path,
        out_path=selection_path,
        top_k=int(getattr(args, "flooring_top_k", 10) or 10),
        llm_settings=llm_settings,
    )

    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    scene_with_flooring = apply_flooring_to_scene(scene, selection)
    write_flooring_json(scene_with_flooring, scene_out_path)
    selected = selection.get("selected_material") or {}
    texture = selection.get("texture_candidate") or {}
    print(
        "🧱 flooring selected: "
        f"{selected.get('sku')} | {selected.get('name')} | "
        f"texture={texture.get('texture_abs_path') or texture.get('texture_path')} | "
        f"usable={bool(texture.get('usable_in_blender'))}"
    )
    return scene_out_path, {
        "selection_json": str(selection_path.resolve()),
        "scene_v1": str(scene_out_path.resolve()),
        "selected_sku": selected.get("sku"),
        "selected_name": selected.get("name"),
        "texture_path": texture.get("texture_abs_path") or texture.get("texture_path"),
        "texture_usable_in_blender": bool(texture.get("usable_in_blender")),
        "llm_rerank": selection.get("llm_rerank"),
    }


def _maybe_apply_wall_material_to_scene(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scene_json_path: Path,
    prompt_text: str,
    style_profile: dict[str, Any],
    room_id: str,
    suffix: str,
) -> tuple[Path, dict[str, Any] | None]:
    if bool(getattr(args, "no_wall_material", False)):
        return scene_json_path, None

    materials_path = Path(str(getattr(args, "wall_materials", "") or "")).expanduser()
    if not materials_path.is_absolute():
        materials_path = (Path.cwd() / materials_path).resolve()
    if not materials_path.is_file():
        print(f"⏭ wall material: каталог не найден, пропуск: {materials_path}")
        return scene_json_path, None

    selection_path = run_dir / f"wall_material.selection{suffix}.v1.json"
    scene_out_path = run_dir / f"{scene_json_path.stem}.wall_material.v1.json"
    style = _flooring_style_label(style_profile)
    room_type = _flooring_room_type(style_profile, scene_json_path)
    llm_settings = {
        "provider": str(getattr(args, "wall_llm_provider", "ollama") or "ollama"),
        "ollama_url": str(getattr(args, "wall_ollama_url", None) or getattr(args, "ollama_url", None) or "http://127.0.0.1:11434"),
        "ollama_model": str(getattr(args, "wall_ollama_model", None) or getattr(args, "ollama_model", None) or "gpt-oss:20b"),
        "ollama_timeout": int(getattr(args, "wall_ollama_timeout", None) or getattr(args, "ollama_timeout", None) or 180),
        "ollama_temperature": float(getattr(args, "wall_ollama_temperature", 0.0) or 0.0),
        "ollama_num_ctx": int(getattr(args, "wall_ollama_num_ctx", 8192) or 8192),
        "top_n": int(getattr(args, "wall_llm_top_n", 5) or 5),
    }

    wall_prompt_text = _flooring_prompt_for_selector(prompt_text, style_profile, run_dir)
    print("🧱 wall material: подбор покрытия стен")
    selection = run_wall_selection(
        prompt=wall_prompt_text,
        style=style,
        room_type=room_type,
        room_description=str(style_profile.get("style_hint") or ""),
        room_id=room_id,
        materials_path=materials_path,
        out_path=selection_path,
        top_k=int(getattr(args, "wall_top_k", 10) or 10),
        llm_settings=llm_settings,
    )

    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    scene_with_wall = apply_wall_material_to_scene(scene, selection)
    write_wall_json(scene_with_wall, scene_out_path)
    selected = selection.get("selected_material") or {}
    print(
        "🧱 wall material selected: "
        f"{selected.get('sku')} | {selected.get('name')} | "
        f"avg={selected.get('average_hex') or selected.get('average_rgb')}"
    )
    return scene_out_path, {
        "selection_json": str(selection_path.resolve()),
        "scene_v1": str(scene_out_path.resolve()),
        "selected_sku": selected.get("sku"),
        "selected_name": selected.get("name"),
        "average_rgb": selected.get("average_rgb"),
        "average_hex": selected.get("average_hex"),
        "dominant_colors_hex": selected.get("dominant_colors_hex"),
        "llm_rerank": selection.get("llm_rerank"),
    }


def _flooring_prompt_for_selector(prompt_text: str, style_profile: dict[str, Any], run_dir: Path) -> str:
    parts = [str(prompt_text or "").strip()]
    style_hint = str(style_profile.get("style_hint") or "").strip()
    if style_hint:
        parts.append(f"Style/color context from style LLM: {style_hint}")
    preferred_colors = style_profile.get("preferred_colors")
    if isinstance(preferred_colors, list) and preferred_colors:
        parts.append("Preferred room colors: " + ", ".join(str(x) for x in preferred_colors if str(x).strip()))
    material_family = style_profile.get("material_family")
    if isinstance(material_family, list) and material_family:
        parts.append("Preferred materials: " + ", ".join(str(x) for x in material_family if str(x).strip()))
    meta_path = run_dir / "infinigen_clean_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            style_label = str(meta.get("style_label") or "").strip()
            room_semantic = str(meta.get("room_semantic") or "").strip()
            if style_label or room_semantic:
                parts.append(
                    "Infinigen generated scene context: "
                    f"style={style_label or 'unknown'}, room={room_semantic or 'unknown'}. "
                    "Choose a floor color/material that harmonizes with the generated Infinigen interior."
                )
        except Exception:
            pass
    return "\n".join(part for part in parts if part).strip() or str(prompt_text or "")


def run_pipeline_for_mode(
    cfg_runtime: dict[str, str],
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    prompt_text: str,
    style_profile_template: dict[str, Any],
) -> ModeOutputs:
    print(f"\n====== РЕЖИМ {layout_mode.upper()} ======")
    print(f"📁 mode_run_dir: {run_dir}")

    placer_spec = PLACER_SPECS[args.placer]
    chooser_required = bool(placer_spec.get("requires_object_selection", True))
    chooser_seed = int.from_bytes(secrets.token_bytes(8), "big")
    style_profile = deepcopy(style_profile_template)
    style_profile_path = run_dir / "style_profile.json"
    write_json(style_profile_path, style_profile)

    original_room_path = Path(room_path).expanduser().resolve()
    styled_room_path = run_dir / "room.style.v1.json"
    room_data = json.loads(original_room_path.read_text(encoding="utf-8"))
    styled_room_data = attach_style_hint_to_room_json(room_data, style_profile)
    write_json(styled_room_path, styled_room_data)
    effective_room_path = str(styled_room_path.resolve())

    chooser_prompt_text = str(style_profile.get("chooser_prompt") or prompt_text).strip() or prompt_text
    effective_prompt_text = chooser_prompt_text
    style_supplier_preferences = style_profile.get("supplier_preferences")
    style_supplier_preferences_path: Optional[Path] = None
    if isinstance(style_supplier_preferences, dict):
        style_supplier_preferences_path = run_dir / "style_supplier_preferences.json"
        write_json(style_supplier_preferences_path, style_supplier_preferences)

    (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (run_dir / "prompt.styled.txt").write_text(effective_prompt_text, encoding="utf-8")
    (run_dir / "chooser_prompt.txt").write_text(chooser_prompt_text, encoding="utf-8")

    objects_path: Optional[Path] = None
    normalized_objects_path: Optional[Path] = None
    if chooser_required:
        objects_path = run_choose_stage(
            args=args,
            cfg_runtime=cfg_runtime,
            room_path=effective_room_path,
            prompt_text=chooser_prompt_text,
            run_dir=run_dir,
            seed=chooser_seed,
        )

        normalized_objects_path = run_dir / "objects.v1.json"
        normalize_json_artifact(
            cfg_runtime=cfg_runtime,
            input_path=objects_path,
            output_path=normalized_objects_path,
            target="objects",
        )
    else:
        print(f"⏭ Пропуск chooser для placer={args.placer}")

    run_manifest = {
        "room": effective_room_path,
        "room_original": str(original_room_path),
        "prompt": prompt_text,
        "prompt_styled": effective_prompt_text,
        "chooser_prompt": chooser_prompt_text,
        "chooser_seed": chooser_seed,
        "placer": args.placer,
        "layout_mode": layout_mode,
        "run_dir": str(run_dir),
        "style_profile_json": str(style_profile_path.resolve()),
        "style_room_json": str(styled_room_path.resolve()),
        "style": {
            "style_label": style_profile.get("style_label"),
            "room_type": style_profile.get("room_type"),
            "confidence": style_profile.get("confidence"),
            "style_hint": style_profile.get("style_hint"),
        },
        "supplier_preferences_json": (
            str(Path(args.supplier_user_preferences_json).expanduser().resolve())
            if str(getattr(args, "supplier_user_preferences_json", "") or "").strip()
            else str(style_supplier_preferences_path.resolve()) if style_supplier_preferences_path else None
        ),
        "objects_legacy": str(objects_path.resolve()) if objects_path else None,
        "objects_v1": str(normalized_objects_path.resolve()) if normalized_objects_path else None,
        "chooser_llm": {
            "provider": "ollama",
            "url": args.ollama_url,
            "model": args.ollama_model,
            "models": list(args.ollama_models) if getattr(args, "ollama_models", None) else [args.ollama_model],
            "timeout": args.ollama_timeout,
            "temperature": args.ollama_temperature,
            "max_attempts": args.ollama_max_attempts,
        },
        "plan_llm": {
            "models": list(args.plan_models),
            "think": args.plan_think,
            "temperature": args.plan_temperature,
        },
        "critic_llm": {
            "models": list(args.critic_models),
            "think": args.critic_think,
            "temperature": args.critic_temperature,
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, run_manifest)

    if args.placer == "lego_gen":
        if normalized_objects_path is None:
            raise RuntimeError("placer=lego_gen требует objects.v1.json, но chooser stage был пропущен")
        lego_artifacts = run_lego_generate_from_scratch(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=effective_room_path,
            objects_v1_path=normalized_objects_path,
            run_dir=run_dir,
        )
        lego_selection_stub = _build_layout_selection_stub_for_artifacts(
            artifacts=lego_artifacts,
            run_dir=run_dir,
            prefix="lego_gen",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["lego_gen"] = {
            "enabled": True,
            "placement_legacy": str(lego_artifacts.placement_legacy.resolve()),
            "placement_v1": str(lego_artifacts.placement_v1.resolve()),
            "scene_v1": str(lego_artifacts.scene_v1.resolve()) if lego_artifacts.scene_v1 else None,
            "scene_legacy": str(lego_artifacts.scene_legacy.resolve()) if lego_artifacts.scene_legacy else None,
            "layout_targets_json": lego_selection_stub["layout_targets_json"],
            "supplier_bindings_stub_json": lego_selection_stub["supplier_bindings_stub_json"],
            "scene_pricing_stub_json": lego_selection_stub["scene_pricing_stub_json"],
        }

        supplier_scene_for_render: Optional[Path] = None
        supplier_bindings_path = _resolve_supplier_bindings_json(
            args=args,
            run_dir=run_dir,
            layout_targets_json_path=lego_selection_stub["layout_targets_json"],
            supplier_user_preferences_json=(
                str(style_supplier_preferences_path.resolve())
                if style_supplier_preferences_path and not str(getattr(args, "supplier_user_preferences_json", "") or "").strip()
                else None
            ),
        )
        if supplier_bindings_path:
            supplier_bindings_path, supplier_assets_info = _acquire_supplier_assets_for_bindings(
                args=args,
                run_dir=run_dir,
                bindings_json_path=supplier_bindings_path,
            )
            supplier_info = _apply_supplier_bindings_for_artifacts(
                artifacts=lego_artifacts,
                run_dir=run_dir,
                bindings_json_path=supplier_bindings_path,
                require_local_asset=bool(args.supplier_require_local_asset),
            )
            manifest["supplier_rebind"] = supplier_info
            manifest["supplier_assets"] = supplier_assets_info
            if supplier_info.get("scene_v1"):
                supplier_scene_for_render = Path(str(supplier_info["scene_v1"])).expanduser().resolve()
        base_scene_for_render = choose_scene_for_render(lego_artifacts)
        base_scene_for_render, base_repair_info = maybe_repair_scene_json(
            args=args,
            scene_json_path=base_scene_for_render,
            run_dir=run_dir,
            tag="lego_gen_base",
        )
        if base_repair_info is not None:
            manifest["scene_repair_base"] = base_repair_info
        base_scene_for_render, base_flooring_info = _maybe_apply_flooring_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=base_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".lego_gen_base",
        )
        if base_flooring_info is not None:
            manifest["flooring_base"] = base_flooring_info
            if isinstance(manifest.get("lego_gen"), dict):
                manifest["lego_gen"]["scene_v1_flooring"] = base_flooring_info.get("scene_v1")
        base_scene_for_render, base_wall_info = _maybe_apply_wall_material_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=base_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".lego_gen_base",
        )
        if base_wall_info is not None:
            manifest["wall_material_base"] = base_wall_info
            if isinstance(manifest.get("lego_gen"), dict):
                manifest["lego_gen"]["scene_v1_wall_material"] = base_wall_info.get("scene_v1")
        if supplier_scene_for_render and supplier_scene_for_render.is_file():
            supplier_scene_for_render, supplier_repair_info = maybe_repair_scene_json(
                args=args,
                scene_json_path=supplier_scene_for_render,
                run_dir=run_dir,
                tag="lego_gen_supplier",
            )
            if supplier_repair_info is not None:
                manifest["scene_repair_supplier"] = supplier_repair_info
            supplier_scene_for_render, supplier_flooring_info = _maybe_apply_flooring_to_scene(
                args=args,
                run_dir=run_dir,
                scene_json_path=supplier_scene_for_render,
                prompt_text=prompt_text,
                style_profile=style_profile,
                room_id="room_001",
                suffix=".lego_gen_supplier",
            )
            if supplier_flooring_info is not None:
                manifest["flooring_supplier"] = supplier_flooring_info
                if isinstance(manifest.get("supplier_rebind"), dict):
                    manifest["supplier_rebind"]["scene_v1_flooring"] = supplier_flooring_info.get("scene_v1")
            supplier_scene_for_render, supplier_wall_info = _maybe_apply_wall_material_to_scene(
                args=args,
                run_dir=run_dir,
                scene_json_path=supplier_scene_for_render,
                prompt_text=prompt_text,
                style_profile=style_profile,
                room_id="room_001",
                suffix=".lego_gen_supplier",
            )
            if supplier_wall_info is not None:
                manifest["wall_material_supplier"] = supplier_wall_info
                if isinstance(manifest.get("supplier_rebind"), dict):
                    manifest["supplier_rebind"]["scene_v1_wall_material"] = supplier_wall_info.get("scene_v1")
        write_json(manifest_path, manifest)

        if args.skip_blender:
            print(f"⏭ Пропуск Blender для режима {layout_mode}")
            print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
            return ModeOutputs(base_artifacts=lego_artifacts, lego_artifacts=None)

        run_blender_for_mode(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=effective_room_path,
            run_dir=run_dir,
            layout_mode=layout_mode,
            scene_json_path=base_scene_for_render,
            variant_suffix="lego_gen",
        )

        if supplier_scene_for_render and supplier_scene_for_render.is_file():
            run_blender_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=effective_room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                scene_json_path=supplier_scene_for_render,
                variant_suffix="lego_gen_supplier",
            )

        print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
        return ModeOutputs(base_artifacts=lego_artifacts, lego_artifacts=None)

    placement_out = run_dir / f"placement_{layout_mode}.json"
    base_artifacts: Optional[PlacementArtifacts] = None
    placement_attempts = 1 if args.placer == "ollama_llm" else int(args.max_attempts)

    for attempt in range(1, placement_attempts + 1):
        print(f"\n---------- ПОПЫТКА {attempt} ({layout_mode}) ----------")
        try:
            attempt_seed = int.from_bytes(secrets.token_bytes(8), "big")

            attempt_info = {
                "attempt": attempt,
                "attempt_seed": attempt_seed,
                "chooser_seed": chooser_seed,
                "layout_mode": layout_mode,
                "placer": args.placer,
                "objects_path": str(objects_path.resolve()) if objects_path else None,
                "objects_v1_path": str(normalized_objects_path.resolve()) if normalized_objects_path else None,
                "placement_legacy_path": str(placement_out.resolve()),
            }
            write_json(run_dir / f"attempt_{attempt:02d}.json", attempt_info)

            execute_placer(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=effective_room_path,
                objects_path=objects_path,
                layout_mode=layout_mode,
                seed=attempt_seed,
                out_path=placement_out,
                run_dir=run_dir,
                prompt_text=effective_prompt_text,
            )

            base_artifacts = build_scene_artifacts(
                cfg_runtime=cfg_runtime,
                room_path=effective_room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                placement_out=placement_out,
                variant_suffix="",
            )
            base_selection_stub = _build_layout_selection_stub_for_artifacts(
                artifacts=base_artifacts,
                run_dir=run_dir,
                prefix="base",
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["base"] = {
                "placement_legacy": str(base_artifacts.placement_legacy.resolve()),
                "placement_v1": str(base_artifacts.placement_v1.resolve()),
                "scene_v1": str(base_artifacts.scene_v1.resolve()) if base_artifacts.scene_v1 else None,
                "scene_legacy": str(base_artifacts.scene_legacy.resolve()) if base_artifacts.scene_legacy else None,
                "layout_targets_json": base_selection_stub["layout_targets_json"],
                "supplier_bindings_stub_json": base_selection_stub["supplier_bindings_stub_json"],
                "scene_pricing_stub_json": base_selection_stub["scene_pricing_stub_json"],
            }
            write_json(manifest_path, manifest)

            print(f"✅ placement stage success: {layout_mode}")
            break

        except Exception as e:
            print(f"❌ placement stage failed on attempt {attempt}: {e}")
            if _is_fatal_disk_full_error(e):
                raise RuntimeError(
                    "Placement aborted due to full disk on the remote/local worker. "
                    "Free space and rerun."
                ) from e
            if attempt >= placement_attempts:
                raise

    if base_artifacts is None:
        raise RuntimeError(f"Не удалось получить base placement для режима {layout_mode}")

    supplier_scene_for_render: Optional[Path] = None
    supplier_bindings_path = _resolve_supplier_bindings_json(
        args=args,
        run_dir=run_dir,
        layout_targets_json_path=base_selection_stub["layout_targets_json"],
        supplier_user_preferences_json=(
            str(style_supplier_preferences_path.resolve())
            if style_supplier_preferences_path and not str(getattr(args, "supplier_user_preferences_json", "") or "").strip()
            else None
        ),
    )
    if supplier_bindings_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        supplier_bindings_path, supplier_assets_info = _acquire_supplier_assets_for_bindings(
            args=args,
            run_dir=run_dir,
            bindings_json_path=supplier_bindings_path,
        )
        supplier_info = _apply_supplier_bindings_for_artifacts(
            artifacts=base_artifacts,
            run_dir=run_dir,
            bindings_json_path=supplier_bindings_path,
            require_local_asset=bool(args.supplier_require_local_asset),
        )
        manifest["supplier_rebind"] = supplier_info
        manifest["supplier_assets"] = supplier_assets_info
        if supplier_info.get("scene_v1"):
            supplier_scene_for_render = Path(str(supplier_info["scene_v1"])).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_scene_for_render = choose_scene_for_render(base_artifacts)
    base_scene_for_render, base_repair_info = maybe_repair_scene_json(
        args=args,
        scene_json_path=base_scene_for_render,
        run_dir=run_dir,
        tag="base",
    )
    if base_repair_info is not None:
        manifest["scene_repair_base"] = base_repair_info
    base_scene_for_render, base_flooring_info = _maybe_apply_flooring_to_scene(
        args=args,
        run_dir=run_dir,
        scene_json_path=base_scene_for_render,
        prompt_text=prompt_text,
        style_profile=style_profile,
        room_id="room_001",
        suffix=".base",
    )
    if base_flooring_info is not None:
        manifest["flooring_base"] = base_flooring_info
        if isinstance(manifest.get("base"), dict):
            manifest["base"]["scene_v1_flooring"] = base_flooring_info.get("scene_v1")
    base_scene_for_render, base_wall_info = _maybe_apply_wall_material_to_scene(
        args=args,
        run_dir=run_dir,
        scene_json_path=base_scene_for_render,
        prompt_text=prompt_text,
        style_profile=style_profile,
        room_id="room_001",
        suffix=".base",
    )
    if base_wall_info is not None:
        manifest["wall_material_base"] = base_wall_info
        if isinstance(manifest.get("base"), dict):
            manifest["base"]["scene_v1_wall_material"] = base_wall_info.get("scene_v1")
    if supplier_scene_for_render and supplier_scene_for_render.is_file():
        supplier_scene_for_render, supplier_repair_info = maybe_repair_scene_json(
            args=args,
            scene_json_path=supplier_scene_for_render,
            run_dir=run_dir,
            tag="supplier",
        )
        if supplier_repair_info is not None:
            manifest["scene_repair_supplier"] = supplier_repair_info
        supplier_scene_for_render, supplier_flooring_info = _maybe_apply_flooring_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=supplier_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".supplier",
        )
        if supplier_flooring_info is not None:
            manifest["flooring_supplier"] = supplier_flooring_info
            if isinstance(manifest.get("supplier_rebind"), dict):
                manifest["supplier_rebind"]["scene_v1_flooring"] = supplier_flooring_info.get("scene_v1")
        supplier_scene_for_render, supplier_wall_info = _maybe_apply_wall_material_to_scene(
            args=args,
            run_dir=run_dir,
            scene_json_path=supplier_scene_for_render,
            prompt_text=prompt_text,
            style_profile=style_profile,
            room_id="room_001",
            suffix=".supplier",
        )
        if supplier_wall_info is not None:
            manifest["wall_material_supplier"] = supplier_wall_info
            if isinstance(manifest.get("supplier_rebind"), dict):
                manifest["supplier_rebind"]["scene_v1_wall_material"] = supplier_wall_info.get("scene_v1")
    write_json(manifest_path, manifest)

    if args.skip_blender:
        print(f"⏭ Пропуск Blender для режима {layout_mode}")
        print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
        return ModeOutputs(base_artifacts=base_artifacts, lego_artifacts=None)

    run_blender_for_mode(
        cfg_runtime=cfg_runtime,
        args=args,
        room_path=effective_room_path,
        run_dir=run_dir,
        layout_mode=layout_mode,
        scene_json_path=base_scene_for_render,
        variant_suffix="",
    )

    if supplier_scene_for_render and supplier_scene_for_render.is_file():
        run_blender_for_mode(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=effective_room_path,
            run_dir=run_dir,
            layout_mode=layout_mode,
            scene_json_path=supplier_scene_for_render,
            variant_suffix="supplier",
        )

    print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
    return ModeOutputs(base_artifacts=base_artifacts, lego_artifacts=None)


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("items", nargs="*")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)

    p.add_argument("--paths-config", default=DEFAULT_PATHS_CONFIG)
    p.add_argument("--room", default="__USE_CFG_DEFAULT__")

    p.add_argument("--prepared-info", default=None)
    p.add_argument("--future-root", default=None)

    p.add_argument("--placer", default=None)
    p.add_argument("--ml-model", default=None)
    p.add_argument("--ml-device", default=None)
    p.add_argument("--diffusion-steps", type=int, default=None)
    p.add_argument("--max-attempts", type=int, default=None)

    p.add_argument("--save-blend", default=None)
    p.add_argument("--render", default=None)
    p.add_argument("--blender", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--skip-blender", action="store_true")
    p.add_argument("--no-bbox-fallback", action="store_true", help="Disable default bbox fallback for items without a resolved/imported mesh")
    p.add_argument("--no-import-glb", action="store_true", help="Compat flag, ignored by current Blender scene builder")

    p.add_argument("--run-dir", default=None)
    p.add_argument("--keep-tmp", action="store_true")

    p.add_argument("--remote-runner", default=None)
    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=None)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)
    p.add_argument("--remote-conda-env", default=None)
    p.add_argument("--remote-infinigen-src", default=None)

    p.add_argument("--ollama-url", default=None)
    p.add_argument("--ollama-model", default=None)
    p.add_argument("--ollama-models", nargs="*", default=None)
    p.add_argument("--ollama-timeout", type=int, default=None)
    p.add_argument("--ollama-temperature", type=float, default=None)
    p.add_argument("--ollama-max-attempts", type=int, default=None)
    p.add_argument("--style-llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--style-ollama-url", default=None)
    p.add_argument("--style-ollama-model", default=None)
    p.add_argument("--style-ollama-models", nargs="*", default=None)
    p.add_argument("--style-ollama-timeout", type=int, default=None)
    p.add_argument("--style-ollama-temperature", type=float, default=None)
    p.add_argument("--style-llm-max-attempts", type=int, default=None)
    p.add_argument("--style-llm-think", choices=["low", "medium", "high"], default=None)
    p.add_argument("--style-llm-debug-dir", default=None)

    p.add_argument("--plan-model", default=None)
    p.add_argument("--plan-models", nargs="*", default=None)
    p.add_argument("--plan-think", choices=["none", "low"], default=None)
    p.add_argument("--llm-think", choices=["none", "low"], default=None)
    p.add_argument("--plan-temperature", type=float, default=None)

    p.add_argument("--critic-model", default=None)
    p.add_argument("--critic-models", nargs="*", default=None)
    p.add_argument("--critic-think", choices=["none", "low"], default=None)
    p.add_argument("--critic-temperature", type=float, default=None)
    p.add_argument("--max-scene-attempts", type=int, default=None)

    p.add_argument("--modes", default=None)
    p.add_argument("--supplier-bindings-json", default=None, help="Optional supplier_bindings json to apply after placement")
    p.add_argument("--supplier-catalog-json", action="append", default=[], help="Supplier catalog export JSON for automatic binding search; can be repeated")
    p.add_argument("--supplier-site", action="append", default=None, help="Optional supplier source_site filter for automatic binding search")
    p.add_argument("--supplier-top-k", type=int, default=5, help="Top-K candidates for automatic supplier matcher")
    p.add_argument("--supplier-rich-only", action="store_true", help="Use only rich supplier cards during automatic binding search")
    p.add_argument("--supplier-user-preferences-json", default=None, help="Optional JSON with supplier matcher user preferences")
    p.add_argument("--supplier-llm-provider", choices=["none", "ollama"], default="none", help="Optional LLM reranker for supplier matcher")
    p.add_argument("--supplier-ollama-url", default=None, help="Optional Ollama URL override for supplier matcher reranking")
    p.add_argument("--supplier-ollama-model", default=None, help="Optional Ollama model override for supplier matcher reranking")
    p.add_argument("--supplier-ollama-timeout", type=int, default=None, help="Optional timeout override in seconds for supplier matcher reranking")
    p.add_argument("--supplier-ollama-temperature", type=float, default=None, help="Optional temperature override for supplier matcher reranking")
    p.add_argument("--supplier-llm-top-n", type=int, default=None, help="How many top heuristic supplier candidates to send to the supplier LLM reranker")
    p.add_argument("--supplier-require-local-asset", action="store_true", help="Apply supplier replacement only for bindings with local downloaded assets")
    p.add_argument("--supplier-assets-dir", default=None, help="Directory for scene-specific downloaded supplier assets")
    p.add_argument("--supplier-assets-db", default=None, help="SQLite DB for scene-specific downloaded supplier assets")
    p.add_argument("--supplier-assets-blender", default=None, help="Optional Blender binary for supplier asset conversion")

    p.add_argument("--no-flooring", action="store_true", help="Disable supplier floor covering selection and Blender floor texture application")
    p.add_argument("--flooring-materials", default="data/sourse/obi_floor_coverings_cards/normalized_floor_materials.jsonl")
    p.add_argument("--flooring-style-rules", default="config/flooring_style_rules.json")
    p.add_argument("--flooring-top-k", type=int, default=10)
    p.add_argument("--flooring-llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--flooring-ollama-url", default=None)
    p.add_argument("--flooring-ollama-model", default=None)
    p.add_argument("--flooring-ollama-timeout", type=int, default=None)
    p.add_argument("--flooring-ollama-temperature", type=float, default=0.0)
    p.add_argument("--flooring-ollama-num-ctx", type=int, default=8192)
    p.add_argument("--flooring-llm-top-n", type=int, default=5)
    p.add_argument("--no-wall-material", action="store_true", help="Disable supplier wall covering selection")
    p.add_argument("--wall-materials", default="data/sourse/domlenta_wallpapers/normalized_wall_materials.jsonl")
    p.add_argument("--wall-top-k", type=int, default=10)
    p.add_argument("--wall-llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--wall-ollama-url", default=None)
    p.add_argument("--wall-ollama-model", default=None)
    p.add_argument("--wall-ollama-timeout", type=int, default=None)
    p.add_argument("--wall-ollama-temperature", type=float, default=0.0)
    p.add_argument("--wall-ollama-num-ctx", type=int, default=8192)
    p.add_argument("--wall-llm-top-n", type=int, default=5)

    p.add_argument("--lego-postprocess", action="store_true")
    p.add_argument("--infinigen-src", default=None)
    p.add_argument("--lego-modes", default=None)
    p.add_argument("--lego-repo", default=None)
    p.add_argument("--lego-python", default=None)
    p.add_argument("--lego-helper-script", default=None)
    p.add_argument("--lego-tmp-root", default=None)
    p.add_argument("--lego-checkpoint-bedroom", default=None)
    p.add_argument("--lego-checkpoint-livingroom", default=None)
    p.add_argument("--lego-room-type", choices=["auto", "bedroom", "livingroom"], default="auto")
    p.add_argument("--lego-render-policy", choices=["base_only", "lego_only", "both"], default="both")
    p.add_argument("--lego-failure-policy", choices=["skip", "raise"], default="skip")
    p.add_argument(
        "--lego-generation-preset",
        choices=sorted(DEFAULT_LEGO_GENERATION_PRESETS.keys()),
        default=None,
    )
    p.add_argument("--lego-method", choices=["direct_map_once", "direct_map", "grad_nonoise", "grad_noise"], default=None)
    p.add_argument("--lego-outer-passes", type=int, default=None)
    p.add_argument("--lego-num-restarts", type=int, default=None)
    p.add_argument("--lego-init-pos-noise-std", type=float, default=None)
    p.add_argument("--lego-init-ang-noise-deg", type=float, default=None)
    p.add_argument("--lego-init-scene-mode", choices=["perturb", "random_full"], default=None)

    add_scene_repair_arguments(p)

    return p


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    cfg_path = Path(args.paths_config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    cfg_base_dir = project_root_from_config(cfg, cfg_path)

    apply_config_defaults(args, cfg, cfg_base_dir)
    cfg_runtime = build_runtime_paths(cfg, cfg_base_dir)

    room_path = os.path.abspath((args.room or cfg_runtime["DEFAULT_ROOM_JSON"]).strip())
    modes = parse_modes(args, cfg)

    print(f"📦 modes: {', '.join(modes)}")
    print(f"🧭 paths-config: {cfg_path}")
    print(f"🤖 json ollama models: {', '.join(args.ollama_models)}")
    print(f"🧠 plan ollama models: {', '.join(args.plan_models)}")
    print(f"🧐 critic ollama models: {', '.join(args.critic_models)}")
    print(f"🧩 plan/critic/json think: {args.plan_think}/{args.critic_think}/{args.llm_think}")

    style_models = [str(x).strip() for x in (getattr(args, "style_ollama_models", None) or args.ollama_models or []) if str(x).strip()]
    if not style_models:
        style_models = [str(getattr(args, "style_ollama_model", None) or args.ollama_model or "gpt-oss:20b").strip()]
    style_think = str(getattr(args, "style_llm_think", None) or "").strip().lower()
    if style_think not in {"low", "medium", "high"}:
        style_think = "low"
    style_temperature = getattr(args, "style_ollama_temperature", None)
    if style_temperature is None:
        style_temperature = args.ollama_temperature if args.ollama_temperature is not None else 0.0
    print(f"🎨 style llm: provider={args.style_llm_provider}, models={', '.join(style_models)}")

    if args.lego_postprocess:
        lego_cfg = resolve_lego_generation_params(args)
        print(
            "🧩 lego generation: "
            f"preset={lego_cfg['preset']}, "
            f"method={lego_cfg['method']}, "
            f"init_scene_mode={lego_cfg['init_scene_mode']}, "
            f"outer_passes={lego_cfg['outer_passes']}, "
            f"num_restarts={lego_cfg['num_restarts']}, "
            f"init_pos_noise_std={lego_cfg['init_pos_noise_std']}, "
            f"init_ang_noise_deg={lego_cfg['init_ang_noise_deg']}"
        )

    prompt_text = read_prompt_from_args(args)
    style_profile_template = analyze_prompt_to_style_profile(
        prompt_text=prompt_text,
        room_path=room_path,
        provider=str(getattr(args, "style_llm_provider", "ollama") or "ollama"),
        ollama_url=str(getattr(args, "style_ollama_url", None) or args.ollama_url or "http://127.0.0.1:11434"),
        ollama_models=style_models,
        timeout_sec=int(getattr(args, "style_ollama_timeout", None) or args.ollama_timeout or 180),
        temperature=float(style_temperature),
        max_attempts=int(getattr(args, "style_llm_max_attempts", None) or args.ollama_max_attempts or 4),
        think=style_think,
        debug_dir=str(getattr(args, "style_llm_debug_dir", None) or ""),
    )
    print(
        "🎯 style selected: "
        f"{style_profile_template.get('style_label')} "
        f"(room={style_profile_template.get('room_type')}, "
        f"confidence={float(style_profile_template.get('confidence') or 0.0):.2f})"
    )
    created_run_dirs: list[Path] = []

    try:
        for layout_mode in modes:
            mode_run_dir, _ = make_mode_run_dir(cfg_runtime["TMP_ROOT"], layout_mode, args.run_dir)
            created_run_dirs.append(mode_run_dir)

            run_pipeline_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=room_path,
                run_dir=mode_run_dir,
                layout_mode=layout_mode,
                prompt_text=prompt_text,
                style_profile_template=style_profile_template,
            )

        print("\n✅ ВСЕ РЕЖИМЫ ОТРАБОТАЛИ УСПЕШНО")

    finally:
        if not args.keep_tmp and not args.run_dir:
            for p in created_run_dirs:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"🗑 Удалён run_dir: {p}")


if __name__ == "__main__":
    main()
