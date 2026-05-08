#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from ml.lego_seed_scene import (
    build_seed_scene_and_placement,
    crop_last_object,
    room_area_m2,
    sort_objects_for_generation,
    total_objects_footprint_m2,
)
from pipeline_artifacts import (
    build_scene_artifacts,
    choose_scene_for_render,
    copy_tree_contents,
    normalize_json_artifact,
    read_json,
    run_blender_for_mode,
    sync_objects_to_legacy_input,
    write_json,
)
from pipeline_config import (
    DEFAULT_LEGO_GENERATION_PRESETS,
    PLACER_SPECS,
    PlacementArtifacts,
)


_CAMEL_RE_1 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_BLEND_NAME_SUFFIX_RE = re.compile(r"\([^)]*\)$")

_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed", "king size bed", "king-size bed", "double bed", "single bed", "kids bed"),
    "nightstand": ("nightstand", "bedside", "bed side"),
    "wardrobe": ("wardrobe", "closet"),
    "dresser": ("drawer chest", "chest of drawers", "corner cabinet", "dresser", "single cabinet", "children cabinet"),
    "desk": ("desk", "simple desk", "dressing table", "vanity"),
    "tv_stand": ("tv stand", "television stand"),
    "armchair": ("armchair", "easy chair"),
    "chair": ("chair", "dining chair", "office chair", "lounge chair", "dressing chair", "cafe chair"),
    "sofa": ("sofa", "loveseat", "l shaped sofa", "l-shaped sofa", "chaise longue sofa", "lazy sofa"),
    "coffee_table": ("coffee table",),
    "side_table": ("side table", "corner side table", "corner table", "round end table"),
    "lamp_ceiling": ("ceiling lamp", "ceiling light", "ceilinglight", "ceiling light factory"),
    "lamp_pendant": ("pendant lamp", "pendant light", "pendantlamp"),
    "lamp_floor": ("floor lamp", "floorlight"),
    "lamp_wall": ("wall lamp", "wall light"),
    "wall_art": ("wall art", "wallart", "picture"),
    "rug": ("rug",),
    "shelf": ("shelf", "bookcase", "bookshelf", "large shelf", "simple bookcase", "cell shelf"),
    "plant": ("plant", "plant container", "large plant container"),
    "mirror": ("mirror",),
    "monitor": ("monitor", "tv", "tv monitor"),
}


def _semantic_text(value: Any) -> str:
    s = str(value or "").strip()
    s = _CAMEL_RE_1.sub(r"\1 \2", s)
    s = s.replace("Factory", " ")
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = _NON_ALNUM_RE.sub(" ", s.lower())
    return " ".join(s.split())


def _semantic_group(name: Any, category: Any = None, constraints: Optional[dict[str, Any]] = None) -> str:
    text = " ".join(x for x in [_semantic_text(name), _semantic_text(category)] if x).strip()
    for group, aliases in _SEMANTIC_ALIASES.items():
        if any(alias in text for alias in aliases):
            return group

    mount_type = None
    if isinstance(constraints, dict):
        mount_type = constraints.get("mount_type")
    mount_type_text = _semantic_text(mount_type)
    if mount_type_text == "ceiling":
        return "lamp_ceiling"
    if mount_type_text == "wall":
        return "lamp_wall"
    return text or "unknown"


def _extract_size_m(obj: dict[str, Any]) -> list[float]:
    size_m = obj.get("size_m")
    if isinstance(size_m, list) and len(size_m) == 3:
        return [float(size_m[0]), float(size_m[1]), float(size_m[2])]

    mins = obj.get("min_size_mm")
    maxs = obj.get("max_size_mm")
    if isinstance(mins, list) and isinstance(maxs, list) and len(mins) == 3 and len(maxs) == 3:
        return [0.0005 * (float(a) + float(b)) for a, b in zip(mins, maxs)]

    asset_meta = obj.get("asset_meta") or {}
    if all(k in asset_meta for k in ("size_x", "size_y", "size_z")):
        return [float(asset_meta["size_x"]), float(asset_meta["size_y"]), float(asset_meta["size_z"])]

    return [0.0, 0.0, 0.0]


def _extract_asset_block_from_selected_object(obj: dict[str, Any]) -> dict[str, Any]:
    if isinstance(obj.get("asset"), dict):
        return deepcopy(obj["asset"])
    asset_meta = deepcopy(obj.get("asset_meta") or {})
    asset: dict[str, Any] = {
        "source": obj.get("asset_source"),
        "model_id": asset_meta.get("model_id") or obj.get("future_jid") or obj.get("model_jid"),
        "mesh_path": obj.get("mesh_path"),
        "mesh_fit_mode": obj.get("mesh_fit_mode"),
        "mesh_texture_dirs": deepcopy(obj.get("mesh_texture_dirs") or []),
    }
    return {k: v for k, v in asset.items() if v is not None}


def _semantic_name_from_blend_object_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.split(".", 1)[0].strip()
    text = _BLEND_NAME_SUFFIX_RE.sub("", text).strip()
    return text


def _load_selected_objects(objects_path: Path) -> list[dict[str, Any]]:
    data = read_json(objects_path)
    src_items = data.get("objects")
    if not isinstance(src_items, list):
        src_items = data.get("items")
    if not isinstance(src_items, list):
        raise RuntimeError(f"Некорректный objects JSON: {objects_path}")

    out: list[dict[str, Any]] = []
    for idx, obj in enumerate(src_items):
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name") or obj.get("category") or f"object_{idx}")
        category = str(obj.get("category") or name)
        constraints = deepcopy(obj.get("constraints") or {})
        out.append(
            {
                "id": str(obj.get("id") or f"sel_{idx:04d}"),
                "name": name,
                "category": category,
                "size_m": _extract_size_m(obj),
                "constraints": constraints,
                "asset": _extract_asset_block_from_selected_object(obj),
                "color": deepcopy(obj.get("color") or [0.7, 0.7, 0.7]),
                "meta": deepcopy(obj.get("meta") or {}),
                "source": deepcopy(obj.get("source") or {}),
                "group": _semantic_group(name, category, constraints),
            }
        )
    return out


def _load_generated_placements(placement_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = read_json(placement_path)
    placements = data.get("placements")
    if not isinstance(placements, list):
        placements = data.get("items")
    if not isinstance(placements, list):
        raise RuntimeError(f"Некорректный placement JSON: {placement_path}")

    out: list[dict[str, Any]] = []
    for idx, obj in enumerate(placements):
        if not isinstance(obj, dict):
            continue
        size_m = obj.get("size_m") if isinstance(obj.get("size_m"), list) else [0.0, 0.0, 0.0]
        constraints = deepcopy(obj.get("constraints") or {})
        source = obj.get("source") or {}
        semantic_name = _semantic_name_from_blend_object_name(source.get("blend_object_name"))
        semantic_category = semantic_name or str(obj.get("category") or obj.get("name") or f"generated_{idx}")
        semantic_display_name = semantic_name or str(obj.get("name") or obj.get("category") or f"generated_{idx}")
        out.append(
            {
                "index": idx,
                "raw": obj,
                "name": semantic_display_name,
                "category": semantic_category,
                "size_m": [float(size_m[0]), float(size_m[1]), float(size_m[2])] if len(size_m) == 3 else [0.0, 0.0, 0.0],
                "constraints": constraints,
                "group": _semantic_group(semantic_display_name, semantic_category, constraints),
            }
        )
    return data, out


def _same_family(group_a: str, group_b: str) -> bool:
    if group_a == group_b:
        return True
    lamp_groups = {"lamp_ceiling", "lamp_pendant", "lamp_floor", "lamp_wall"}
    table_groups = {"coffee_table", "side_table", "desk"}
    chair_groups = {"chair", "armchair"}
    shelf_groups = {"shelf", "tv_stand"}
    storage_groups = {"dresser", "wardrobe"}
    families = [lamp_groups, table_groups, chair_groups, shelf_groups, storage_groups]
    return any(group_a in fam and group_b in fam for fam in families)


def _size_distance(size_a: list[float], size_b: list[float]) -> float:
    eps = 1e-6
    vals = []
    for a, b in zip(size_a, size_b):
        aa = max(float(a), eps)
        bb = max(float(b), eps)
        vals.append(abs(math.log(aa / bb)))
    return sum(vals) / max(len(vals), 1)


def _match_score(selected: dict[str, Any], generated: dict[str, Any]) -> float:
    score = -5.0 * _size_distance(selected["size_m"], generated["size_m"])
    if selected["group"] == generated["group"]:
        score += 1000.0
    elif _same_family(selected["group"], generated["group"]):
        score += 200.0
    strict_groups: dict[str, set[str]] = {
        "bed": {"bed"},
        "nightstand": {"nightstand", "side_table"},
        "wardrobe": {"wardrobe", "dresser"},
    }
    allowed_groups = strict_groups.get(selected["group"])
    if allowed_groups is not None and generated["group"] not in allowed_groups:
        score -= 500.0
    return score


def _aabb_center(aabb: dict[str, Any]) -> list[float]:
    return [
        0.5 * (float(aabb["x_min"]) + float(aabb["x_max"])),
        0.5 * (float(aabb["y_min"]) + float(aabb["y_max"])),
        0.5 * (float(aabb["z_min"]) + float(aabb["z_max"])),
    ]


def _rotation_aware_world_size(size_m: list[float], rotation_deg: float) -> list[float]:
    sx, sy, sz = [max(float(v), 1e-6) for v in size_m]
    theta = math.radians(float(rotation_deg or 0.0))
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    return [
        sx * c + sy * s,
        sx * s + sy * c,
        sz,
    ]


def _apply_selected_geometry(
    item: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    selected_size = [max(float(v), 1e-6) for v in selected["size_m"]]
    rotation_deg = float(item.get("rotation_deg", item.get("yaw_deg", 0.0)) or 0.0)
    world_size = _rotation_aware_world_size(selected_size, rotation_deg)
    constraints = selected.get("constraints") or {}
    source_aabb = item.get("aabb") or {}

    if isinstance(source_aabb, dict) and all(k in source_aabb for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
        cx, cy, cz = _aabb_center(source_aabb)
        z_min_prev = float(source_aabb["z_min"])
        z_max_prev = float(source_aabb["z_max"])
    else:
        pos = item.get("position_m") or [0.0, 0.0, 0.0]
        cx, cy, cz = [float(pos[i]) for i in range(3)]
        z_min_prev = cz - 0.5 * world_size[2]
        z_max_prev = cz + 0.5 * world_size[2]

    mount_type = _semantic_text(constraints.get("mount_type"))
    is_ceiling = mount_type == "ceiling"
    is_wall = mount_type == "wall"

    if is_ceiling:
        z_max = z_max_prev
        z_min = z_max - world_size[2]
    elif is_wall:
        z_min = cz - 0.5 * world_size[2]
        z_max = cz + 0.5 * world_size[2]
    else:
        z_min = z_min_prev
        z_max = z_min + world_size[2]

    item["size_m"] = selected_size
    item["position_m"] = [
        cx,
        cy,
        0.5 * (z_min + z_max),
    ]
    item["aabb"] = {
        "x_min": cx - 0.5 * world_size[0],
        "x_max": cx + 0.5 * world_size[0],
        "y_min": cy - 0.5 * world_size[1],
        "y_max": cy + 0.5 * world_size[1],
        "z_min": z_min,
        "z_max": z_max,
    }


def _build_generated_placeholder_item(
    generated_raw: dict[str, Any],
    *,
    placer_name: str,
    out_idx: int,
) -> dict[str, Any]:
    item = deepcopy(generated_raw)
    source = deepcopy(item.get("source") or {})
    semantic_name = _semantic_name_from_blend_object_name(source.get("blend_object_name"))
    name = str(semantic_name or item.get("name") or item.get("category") or f"generated_{out_idx}")
    category = str(semantic_name or item.get("category") or item.get("name") or name)
    item["id"] = f"obj_{out_idx:04d}"
    item["name"] = name
    item["category"] = category
    item["asset"] = {}
    source["placement_source"] = placer_name
    source["placeholder_bbox"] = True
    item["source"] = source
    meta = deepcopy(item.get("meta") or {})
    meta["placeholder_bbox"] = True
    meta["placeholder_reason"] = "generated_object_without_selected_asset"
    item["meta"] = meta
    item["color"] = deepcopy(item.get("color") or [0.85, 0.25, 0.25])
    return item


def _strip_binding_annotations(raw: dict[str, Any]) -> dict[str, Any]:
    item = deepcopy(raw)
    source = raw.get("source") or {}
    if isinstance(source, dict):
        source = deepcopy(source)
        source.pop("selected_object_id", None)
        source.pop("selected_object_name", None)
        source.pop("placeholder_bbox", None)
        item["source"] = source
    meta = item.get("meta") or {}
    if isinstance(meta, dict):
        meta = deepcopy(meta)
        meta.pop("selected_object_meta", None)
        meta.pop("placeholder_bbox", None)
        meta.pop("placeholder_reason", None)
        item["meta"] = meta
    return item


def rebind_generated_layout_to_selected_objects(
    placement_path: Path,
    objects_path: Optional[Path],
    *,
    placer_name: str,
) -> None:
    if objects_path is None:
        return

    selected = _load_selected_objects(objects_path)
    if not selected:
        return

    data, generated = _load_generated_placements(placement_path)
    if not generated:
        raise RuntimeError(f"{placer_name}: placement не содержит объектов")
    if len(generated) < len(selected):
        raise RuntimeError(
            f"{placer_name}: генератор вернул меньше объектов ({len(generated)}), "
            f"чем выбрано пользователем ({len(selected)})"
        )

    same_group_counts = {
        sel["id"]: sum(1 for gen in generated if gen["group"] == sel["group"])
        for sel in selected
    }
    selected_order = sorted(
        selected,
        key=lambda sel: (
            same_group_counts[sel["id"]],
            -(sel["size_m"][0] * sel["size_m"][1] * sel["size_m"][2]),
            sel["name"],
        ),
    )

    remaining = set(range(len(generated)))
    assignments: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for sel in selected_order:
        ranked = sorted(
            (
                (_match_score(sel, generated[idx]), idx)
                for idx in remaining
            ),
            reverse=True,
        )
        if not ranked:
            raise RuntimeError(f"{placer_name}: не осталось placement-слотов для {sel['name']}")
        best_score, best_idx = ranked[0]
        best = generated[best_idx]
        if sel["group"] != best["group"] and not _same_family(sel["group"], best["group"]):
            warnings.append(
                f"{sel['name']} -> {best['name']} (fallback group match: {sel['group']} -> {best['group']})"
            )
        assignments[sel["id"]] = best
        remaining.remove(best_idx)

    rebound: list[dict[str, Any]] = []
    for out_idx, sel in enumerate(selected):
        gen = assignments[sel["id"]]["raw"]
        item = deepcopy(gen)
        item["id"] = f"obj_{out_idx:04d}"
        item["name"] = sel["name"]
        item["category"] = sel["category"]
        item["constraints"] = deepcopy(sel["constraints"])
        item["asset"] = deepcopy(sel["asset"])
        item["color"] = deepcopy(sel["color"])
        _apply_selected_geometry(item, sel)
        source = deepcopy(item.get("source") or {})
        source["placement_source"] = placer_name
        source["selected_object_id"] = sel["id"]
        source["selected_object_name"] = sel["name"]
        source.pop("placeholder_bbox", None)
        item["source"] = source
        meta = deepcopy(item.get("meta") or {})
        meta["selected_object_meta"] = deepcopy(sel["meta"])
        meta.pop("placeholder_bbox", None)
        meta.pop("placeholder_reason", None)
        item["meta"] = meta
        rebound.append(item)

    remaining_generated = [
        _strip_binding_annotations(generated[idx]["raw"])
        for idx in sorted(remaining)
    ]
    for raw in remaining_generated:
        rebound.append(
            _build_generated_placeholder_item(
                raw,
                placer_name=placer_name,
                out_idx=len(rebound),
            )
        )

    data["placements"] = rebound
    data["placer"] = placer_name
    meta = deepcopy(data.get("meta") or {})
    meta["selected_object_count"] = len(selected)
    meta["generated_object_count"] = len(generated)
    meta["placeholder_bbox_count"] = len(remaining_generated)
    if warnings:
        meta["asset_binding_warnings"] = warnings
    data["meta"] = meta
    write_json(placement_path, data)


def run_choose_stage(
    args: argparse.Namespace,
    cfg_runtime: dict[str, Any],
    room_path: str,
    prompt_text: str,
    run_dir: Path,
    seed: int,
) -> Path:
    out_objects = run_dir / "objects.json"
    chooser_llm_debug_dir = run_dir / "llm_choose_debug"

    cmd = [
        sys.executable,
        cfg_runtime["CHOOSER_SCRIPT"],
        "--room-json",
        os.path.abspath(room_path),
        "--prompt",
        prompt_text,
        "--prepared-info",
        os.path.abspath(args.prepared_info),
        "--future-root",
        os.path.abspath(args.future_root),
        "--out",
        str(out_objects.resolve()),
        "--run-dir",
        str(run_dir.resolve()),
        "--seed",
        str(int(seed)),
        "--llm-provider",
        str(getattr(args, "chooser_llm_provider", "ollama") or "ollama"),
        "--ollama-url",
        args.ollama_url,
        "--ollama-model",
        args.ollama_model,
        "--ollama-timeout",
        str(int(args.ollama_timeout)),
        "--ollama-temperature",
        str(float(args.ollama_temperature)),
        "--llm-max-attempts",
        str(int(args.ollama_max_attempts)),
        "--llm-think",
        "low",
        "--llm-debug-dir",
        str(chooser_llm_debug_dir.resolve()),
    ]

    if str(getattr(args, "chooser_llm_provider", "ollama") or "ollama").strip().lower() == "none":
        cmd.append("--disable-llm")

    if getattr(args, "ollama_models", None):
        cmd.append("--ollama-models")
        cmd.extend(args.ollama_models)

    print("▶ Выбор предметов:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_objects


def patch_objects_seed(objects_path: Path, seed: int) -> None:
    data = read_json(objects_path)
    if not isinstance(data, dict):
        raise RuntimeError(f"Некорректный objects.json: {objects_path}")
    data["seed"] = int(seed)
    write_json(objects_path, data)


def run_cube_placer(
    cfg_runtime: dict[str, Any],
    room_path: str,
    objects_path: Path,
    layout_mode: str,
    out_path: Path,
    attempt_seed: int,
) -> None:
    patch_objects_seed(objects_path, attempt_seed)
    sync_objects_to_legacy_input(objects_path, cfg_runtime["LEGACY_OBJECTS_JSON"])
    cube_input = f"{os.path.abspath(room_path)}\n{str(objects_path.resolve())}\n{layout_mode}\n"
    subprocess.run([sys.executable, cfg_runtime["CUBE_SCRIPT"]], input=cube_input, text=True, check=True)

    legacy_out = Path(cfg_runtime["LEGACY_PLACEMENT_JSON"]).resolve()
    if not legacy_out.is_file():
        raise RuntimeError(f"Cube placer не создал {legacy_out}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_out, out_path)


def run_ml_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    layout_mode: str,
    seed: int,
    out_path: Path,
) -> None:
    if not room_path.lower().endswith(".json"):
        raise RuntimeError("ML placer требует room-spec .json")

    if not args.ml_model:
        raise RuntimeError(f"--ml-model обязателен для placer={args.placer}")

    cmd = [
        sys.executable,
        cfg_runtime["ML_PLACER_SCRIPT"],
        "--backend",
        args.placer,
        "--model",
        os.path.abspath(args.ml_model),
        "--room",
        os.path.abspath(room_path),
        "--objects",
        str(objects_path.resolve()),
        "--out",
        str(out_path.resolve()),
        "--device",
        args.ml_device,
        "--seed",
        str(int(seed)),
    ]

    if args.placer == "diffusion":
        cmd += ["--ddim-steps", str(int(args.diffusion_steps))]

    if layout_mode:
        cmd += ["--mode", layout_mode]

    print("▶ Запуск ML-расстановщика:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_layout_refiner_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    layout_mode: str,
    seed: int,
    out_path: Path,
    run_dir: Path,
) -> None:
    if not room_path.lower().endswith(".json"):
        raise RuntimeError("layout_refiner требует room-spec .json")

    if not args.ml_model:
        raise RuntimeError("--ml-model обязателен для placer=layout_refiner")

    debug_image = run_dir / f"layout_refiner_debug_{layout_mode}.png"
    infer_log = run_dir / f"layout_refiner_{layout_mode}.log"

    cmd = [
        sys.executable,
        cfg_runtime["LAYOUT_REFINER_SCRIPT"],
        "--checkpoint",
        os.path.abspath(args.ml_model),
        "--room",
        os.path.abspath(room_path),
        "--objects",
        str(objects_path.resolve()),
        "--out",
        str(out_path.resolve()),
        "--device",
        args.ml_device,
        "--init-mode",
        layout_mode,
        "--seed",
        str(int(seed)),
        "--debug-image",
        str(debug_image.resolve()),
        "--print-summary",
    ]

    print("▶ Запуск layout_refiner:\n ", " ".join(cmd))
    with infer_log.open("w", encoding="utf-8") as log_f:
        subprocess.run(cmd, check=True, stdout=log_f, stderr=subprocess.STDOUT)

    print(f"📄 Лог layout_refiner: {infer_log}")
    if debug_image.is_file():
        print(f"🖼 Debug image: {debug_image}")


def run_diffuscene_remote_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    out_path: Path,
    run_dir: Path,
) -> None:
    remote_run_name = run_dir.name
    remote_artifacts_dir = Path(cfg_runtime["TMP_ROOT"]).resolve() / remote_run_name

    cmd = [
        sys.executable,
        cfg_runtime["DIFFUSCENE_REMOTE_SCRIPT"],
        "--room",
        os.path.abspath(room_path),
        "--objects",
        str(objects_path.resolve()),
        "--out",
        str(out_path.resolve()),
        "--run-name",
        remote_run_name,
        "--remote-runner",
        os.path.abspath(args.remote_runner),
    ]

    if getattr(args, "remote_host", None):
        cmd += ["--remote-host", str(args.remote_host)]
    if getattr(args, "remote_port", None):
        cmd += ["--remote-port", str(int(args.remote_port))]
    if getattr(args, "remote_user", None):
        cmd += ["--remote-user", str(args.remote_user)]
    if getattr(args, "remote_key", None):
        cmd += ["--remote-key", str(Path(args.remote_key).expanduser())]

    print("▶ Запуск DiffuScene remote placer:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not out_path.is_file():
        raise RuntimeError(f"DiffuScene remote не создал итоговый placement: {out_path}")

    local_mode_artifacts = run_dir / "diffuscene_remote_artifacts"
    if remote_artifacts_dir.is_dir():
        copy_tree_contents(remote_artifacts_dir, local_mode_artifacts)
        print(f"📥 Артефакты DiffuScene -> {local_mode_artifacts}")


def run_ollama_llm_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    layout_mode: str,
    out_path: Path,
    prompt_text: str,
) -> None:
    if not room_path.lower().endswith(".json"):
        raise RuntimeError("ollama_llm placer требует room-spec .json")

    cmd = [
        sys.executable,
        cfg_runtime["OLLAMA_LLM_SCRIPT"],
        "--room",
        os.path.abspath(room_path),
        "--objects",
        str(objects_path.resolve()),
        "--out",
        str(out_path.resolve()),
        "--mode",
        layout_mode,
        "--design-brief",
        prompt_text,
        "--ollama-url",
        args.ollama_url,
        "--ollama-model",
        args.ollama_model,
        "--timeout",
        str(int(args.ollama_timeout)),
        "--temperature",
        str(float(args.ollama_temperature)),
        "--max-llm-attempts",
        str(int(args.ollama_max_attempts)),
        "--max-scene-attempts",
        str(int(args.max_scene_attempts)),
    ]

    if getattr(args, "ollama_models", None):
        cmd.append("--ollama-models")
        cmd.extend(args.ollama_models)
    if getattr(args, "plan_model", None):
        cmd += ["--plan-model", str(args.plan_model)]
    if getattr(args, "plan_models", None):
        cmd.append("--plan-models")
        cmd.extend(args.plan_models)
    if getattr(args, "critic_model", None):
        cmd += ["--critic-model", str(args.critic_model)]
    if getattr(args, "critic_models", None):
        cmd.append("--critic-models")
        cmd.extend(args.critic_models)
    if getattr(args, "plan_temperature", None) is not None:
        cmd += ["--plan-temperature", str(float(args.plan_temperature))]
    if getattr(args, "critic_temperature", None) is not None:
        cmd += ["--critic-temperature", str(float(args.critic_temperature))]
    if getattr(args, "plan_think", None):
        cmd += ["--plan-think", str(args.plan_think)]
    if getattr(args, "critic_think", None):
        cmd += ["--critic-think", str(args.critic_think)]
    if getattr(args, "llm_think", None):
        cmd += ["--llm-think", str(args.llm_think)]

    print("▶ Запуск Ollama LLM-расстановщика:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_m3dlayout_clean(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Optional[Path],
    prompt_text: str,
    seed: int,
    out_path: Path,
    model_type: str,
) -> None:
    script = cfg_runtime.get("M3DLAYOUT_SCRIPT")
    if not script:
        raise RuntimeError("Не задан local.scripts.m3dlayout / runtime M3DLAYOUT_SCRIPT")
    if not getattr(args, "remote_host", None) or not getattr(args, "remote_user", None):
        raise RuntimeError("Для M3DLayout нужны remote-host и remote-user")

    cmd = [
        sys.executable,
        script,
        "--room",
        os.path.abspath(room_path),
        "--prompt",
        prompt_text,
        "--seed",
        str(int(seed)),
        "--model-type",
        model_type,
        "--out",
        str(out_path.resolve()),
        "--remote-host",
        str(args.remote_host),
        "--remote-user",
        str(args.remote_user),
    ]
    if getattr(args, "remote_port", None):
        cmd += ["--remote-port", str(int(args.remote_port))]
    if getattr(args, "remote_key", None):
        cmd += ["--remote-key", str(Path(args.remote_key).expanduser())]
    if getattr(args, "remote_conda_env", None):
        cmd += ["--remote-conda-env", str(args.remote_conda_env)]
    print("▶ Запуск M3DLayout:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    rebind_generated_layout_to_selected_objects(out_path, objects_path, placer_name=f"m3dlayout_{'ar' if model_type == 'autoregressive' else 'diffusion'}")


def run_infinigen_clean(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Optional[Path],
    seed: int,
    out_path: Path,
    run_dir: Path,
) -> None:
    script = cfg_runtime.get("INFINIGEN_CLEAN_SCRIPT")
    if not script:
        raise RuntimeError("Не задан local.scripts.infinigen_clean / runtime INFINIGEN_CLEAN_SCRIPT")

    cmd = [
        sys.executable,
        script,
        "--room",
        os.path.abspath(room_path),
        "--seed",
        str(int(seed)),
        "--out",
        str(out_path.resolve()),
        "--run-dir",
        str(run_dir.resolve()),
    ]
    style_profile_path = run_dir / "style_profile.json"
    if style_profile_path.is_file():
        cmd += ["--style-profile", str(style_profile_path.resolve())]
    if getattr(args, "infinigen_src", None):
        cmd += ["--infinigen-src", str(Path(args.infinigen_src).expanduser())]
    if getattr(args, "remote_host", None):
        cmd += ["--remote-host", str(args.remote_host)]
    if getattr(args, "remote_port", None):
        cmd += ["--remote-port", str(int(args.remote_port))]
    if getattr(args, "remote_user", None):
        cmd += ["--remote-user", str(args.remote_user)]
    if getattr(args, "remote_key", None):
        cmd += ["--remote-key", str(Path(args.remote_key).expanduser())]
    if getattr(args, "remote_conda_env", None):
        cmd += ["--remote-conda-env", str(args.remote_conda_env)]
    if getattr(args, "remote_infinigen_src", None):
        cmd += ["--remote-infinigen-src", str(args.remote_infinigen_src)]
    if getattr(args, "infinigen_task", None):
        cmd += ["--infinigen-task", str(args.infinigen_task)]
    if getattr(args, "infinigen_configs", None):
        cmd.append("--infinigen-configs")
        cmd.extend(str(x) for x in args.infinigen_configs)
    print("▶ Запуск Infinigen clean:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    _validate_infinigen_clean_artifacts(out_path=out_path, run_dir=run_dir)
    if bool(getattr(args, "infinigen_rebind_selected_objects", False)):
        rebind_generated_layout_to_selected_objects(out_path, objects_path, placer_name="infinigen_clean")


def _validate_infinigen_clean_artifacts(*, out_path: Path, run_dir: Path) -> None:
    blend_path = run_dir / "infinigen_clean_scene.blend"
    if not blend_path.is_file() or blend_path.stat().st_size <= 0:
        raise RuntimeError(f"infinigen_clean: full scene blend was not downloaded: {blend_path}")

    placement = read_json(out_path)
    placements = placement.get("placements") if isinstance(placement, dict) else None
    if not isinstance(placements, list) or not placements:
        raise RuntimeError(f"infinigen_clean: extracted placement is empty: {out_path}")

    summary_path = run_dir / "inventory_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        raw_count = int((summary.get("raw_real_object_count") or 0) if isinstance(summary, dict) else 0)
        core_count = int((summary.get("real_object_count") or 0) if isinstance(summary, dict) else 0)
        if raw_count <= 0:
            raise RuntimeError(f"infinigen_clean: inventory is empty: {summary_path}")
        print(f"✅ Infinigen clean inventory: raw_real_object_count={raw_count}, core_furniture_count={core_count}")


def infer_lego_room_type(room_path: str) -> str:
    name = Path(room_path).name.lower()
    candidates = [name, str(room_path).lower()]
    joined = " ".join(candidates)
    if "living" in joined:
        return "livingroom"
    if "bedroom" in joined:
        return "bedroom"
    return "bedroom"


def build_ssh_base(args: argparse.Namespace) -> list[str]:
    if not args.remote_host or not args.remote_user:
        raise RuntimeError("Для LEGO-Net нужны remote-host и remote-user")
    cmd = ["ssh"]
    if args.remote_port:
        cmd += ["-p", str(int(args.remote_port))]
    if args.remote_key:
        cmd += ["-i", str(Path(args.remote_key).expanduser())]
    cmd.append(f"{args.remote_user}@{args.remote_host}")
    return cmd


def build_scp_base(args: argparse.Namespace) -> list[str]:
    if not args.remote_host or not args.remote_user:
        raise RuntimeError("Для LEGO-Net нужны remote-host и remote-user")
    cmd = ["scp"]
    if args.remote_port:
        cmd += ["-P", str(int(args.remote_port))]
    if args.remote_key:
        cmd += ["-i", str(Path(args.remote_key).expanduser())]
    return cmd


def run_cmd(cmd: list[str]) -> None:
    print("▶", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ssh_run(args: argparse.Namespace, remote_command: str) -> None:
    cmd = build_ssh_base(args) + [remote_command]
    run_cmd(cmd)


def scp_upload(args: argparse.Namespace, local_path: Path, remote_path: str) -> None:
    remote_spec = f"{args.remote_user}@{args.remote_host}:{remote_path}"
    cmd = build_scp_base(args) + [str(local_path.resolve()), remote_spec]
    run_cmd(cmd)


def scp_download(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_spec = f"{args.remote_user}@{args.remote_host}:{remote_path}"
    cmd = build_scp_base(args) + [remote_spec, str(local_path.resolve())]
    run_cmd(cmd)


def parse_csv_set(raw: str) -> set[str]:
    return {x.strip() for x in (raw or "").split(",") if x.strip()}


def resolve_lego_generation_params(args: argparse.Namespace) -> dict[str, Any]:
    preset_name = str(args.lego_generation_preset or "gen_medium").strip()
    if preset_name not in DEFAULT_LEGO_GENERATION_PRESETS:
        raise RuntimeError(
            f"Неизвестный lego generation preset: {preset_name}. "
            f"Допустимые: {sorted(DEFAULT_LEGO_GENERATION_PRESETS)}"
        )

    cfg = dict(DEFAULT_LEGO_GENERATION_PRESETS[preset_name])
    cfg["preset"] = preset_name

    if args.lego_method is not None:
        cfg["method"] = str(args.lego_method)
    if args.lego_outer_passes is not None:
        cfg["outer_passes"] = int(args.lego_outer_passes)
    if args.lego_num_restarts is not None:
        cfg["num_restarts"] = int(args.lego_num_restarts)
    if args.lego_init_pos_noise_std is not None:
        cfg["init_pos_noise_std"] = float(args.lego_init_pos_noise_std)
    if args.lego_init_ang_noise_deg is not None:
        cfg["init_ang_noise_deg"] = float(args.lego_init_ang_noise_deg)
    if args.lego_init_scene_mode is not None:
        cfg["init_scene_mode"] = str(args.lego_init_scene_mode)

    return cfg


def require_objects_path(objects_path: Optional[Path], placer: str) -> Path:
    if objects_path is None:
        raise RuntimeError(f"placer={placer} требует objects.json, но chooser stage был пропущен")
    return objects_path


def run_lego_generate_from_scratch(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_v1_path: Path,
    run_dir: Path,
) -> PlacementArtifacts:
    if not args.lego_postprocess:
        raise RuntimeError("Для placer=lego_gen нужен --lego-postprocess")

    room_json = read_json(room_path)
    objects_v1 = read_json(objects_v1_path)

    room_type = args.lego_room_type
    if room_type == "auto":
        room_type = infer_lego_room_type(room_path)

    if room_type not in {"bedroom", "livingroom"}:
        raise RuntimeError(f"Неподдерживаемый room_type для lego_gen: {room_type}")

    if not args.lego_repo:
        raise RuntimeError("Не задан --lego-repo")
    if not args.lego_python:
        raise RuntimeError("Не задан --lego-python")
    if not args.lego_helper_script:
        raise RuntimeError("Не задан --lego-helper-script")
    if not args.lego_tmp_root:
        raise RuntimeError("Не задан --lego-tmp-root")

    checkpoint = args.lego_checkpoint_bedroom if room_type == "bedroom" else args.lego_checkpoint_livingroom
    if not checkpoint:
        raise RuntimeError(f"Не задан checkpoint для room_type={room_type}")

    gen_cfg = resolve_lego_generation_params(args)
    room_area = room_area_m2(room_json)
    objects_v1 = sort_objects_for_generation(objects_v1)

    out_dir = run_dir / "lego_gen"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_scene_tries = max(1, int(args.max_attempts or 8))
    fill_ratio_limit = 0.65
    reduction_index = 0

    working_objects = objects_v1
    while True:
        objs = working_objects.get("objects") or []
        if not objs:
            raise RuntimeError("lego_gen: закончились объекты, не удалось построить валидную сцену")

        footprint = total_objects_footprint_m2(working_objects)
        manifest = {
            "reduction_index": reduction_index,
            "object_count": len(objs),
            "room_area_m2": room_area,
            "objects_footprint_m2": footprint,
            "fill_ratio": (footprint / room_area) if room_area > 1e-9 else None,
        }
        write_json(out_dir / f"reduction_{reduction_index:02d}.json", manifest)

        if room_area <= 1e-9 or footprint / room_area <= fill_ratio_limit:
            break

        working_objects = crop_last_object(working_objects)
        reduction_index += 1

    seed_scene, seed_placement = build_seed_scene_and_placement(room_json, working_objects)

    local_seed_scene = out_dir / "seed_scene.json"
    local_seed_placement = out_dir / "seed_placement.json"
    write_json(local_seed_scene, seed_scene)
    write_json(local_seed_placement, seed_placement)

    remote_root = str(Path(args.lego_tmp_root).rstrip("/"))
    remote_run_dir = f"{remote_root}/{run_dir.name}"
    remote_seed_scene = f"{remote_run_dir}/seed_scene.json"
    remote_seed_placement = f"{remote_run_dir}/seed_placement.json"
    remote_out_dir = f"{remote_run_dir}/out"

    ssh_run(args, f"mkdir -p {shlex.quote(remote_out_dir)}")
    scp_upload(args, local_seed_scene, remote_seed_scene)
    scp_upload(args, local_seed_placement, remote_seed_placement)

    helper_cmd = [
        shlex.quote(args.lego_python),
        shlex.quote(args.lego_helper_script),
        "--repo-root", shlex.quote(args.lego_repo),
        "--room-type", shlex.quote(room_type),
        "--checkpoint", shlex.quote(checkpoint),
        "--seed-scene", shlex.quote(remote_seed_scene),
        "--seed-placement", shlex.quote(remote_seed_placement),
        "--out-dir", shlex.quote(remote_out_dir),
        "--method", shlex.quote(gen_cfg["method"]),
        "--init-scene-mode", shlex.quote(gen_cfg["init_scene_mode"]),
        "--num-restarts", str(int(gen_cfg["num_restarts"])),
        "--outer-passes", str(int(gen_cfg["outer_passes"])),
        "--init-pos-noise-std", str(float(gen_cfg["init_pos_noise_std"])),
        "--init-ang-noise-deg", str(float(gen_cfg["init_ang_noise_deg"])),
        "--max-scene-tries", str(int(max_scene_tries)),
    ]
    ssh_run(args, " ".join(helper_cmd))

    local_remote_out = out_dir / "remote_out"
    local_remote_out.mkdir(parents=True, exist_ok=True)
    scp_download(args, f"{remote_out_dir}/placement.json", local_remote_out / "placement.json")

    out_placement_legacy = out_dir / "placement_lego_gen.json"
    shutil.copy2(local_remote_out / "placement.json", out_placement_legacy)

    out_placement_v1 = out_dir / "placement_lego_gen.v1.json"
    normalize_json_artifact(
        cfg_runtime=cfg_runtime,
        input_path=out_placement_legacy,
        output_path=out_placement_v1,
        target="placement",
    )

    out_scene_v1 = out_dir / "scene_lego_gen.v1.json"
    out_scene_legacy = out_dir / "scene_lego_gen.json"
    if room_path.lower().endswith(".json"):
        from pipeline_artifacts import build_normalized_scene_artifact, merge_room_spec_and_placements

        build_normalized_scene_artifact(
            cfg_runtime=cfg_runtime,
            room_path=room_path,
            placement_path=out_placement_legacy,
            output_path=out_scene_v1,
        )
        merge_room_spec_and_placements(room_path, str(out_placement_legacy.resolve()), str(out_scene_legacy.resolve()))
    else:
        out_scene_v1 = None  # type: ignore[assignment]
        out_scene_legacy = None  # type: ignore[assignment]

    print(
        "✅ LEGO-Net refined "
        f"room_type={room_type}, preset={gen_cfg['preset']}, "
        f"init_scene_mode={gen_cfg['init_scene_mode']}, "
        f"restarts={gen_cfg['num_restarts']}, "
        f"pos_noise={gen_cfg['init_pos_noise_std']}, "
        f"ang_noise={gen_cfg['init_ang_noise_deg']}"
    )

    return PlacementArtifacts(
        placement_legacy=out_placement_legacy,
        placement_v1=out_placement_v1,
        scene_v1=out_scene_v1 if isinstance(out_scene_v1, Path) and out_scene_v1.is_file() else None,
        scene_legacy=out_scene_legacy if isinstance(out_scene_legacy, Path) and out_scene_legacy.is_file() else None,
    )


def execute_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Optional[Path],
    layout_mode: str,
    seed: int,
    out_path: Path,
    run_dir: Path,
    prompt_text: str,
) -> None:
    spec = PLACER_SPECS[args.placer]

    if spec["requires_ml_model"] and not args.ml_model:
        raise RuntimeError(f"--ml-model обязателен для placer={args.placer}")

    runner = spec["runner"]

    if runner == "cube":
        objects_path = require_objects_path(objects_path, args.placer)
        run_cube_placer(cfg_runtime, room_path, objects_path, layout_mode, out_path, seed)
        return

    if runner == "layout_refiner":
        objects_path = require_objects_path(objects_path, args.placer)
        run_layout_refiner_placer(cfg_runtime, args, room_path, objects_path, layout_mode, seed, out_path, run_dir)
        return

    if runner == "ml_generic":
        objects_path = require_objects_path(objects_path, args.placer)
        run_ml_placer(cfg_runtime, args, room_path, objects_path, layout_mode, seed, out_path)
        return

    if runner == "diffuscene_remote":
        objects_path = require_objects_path(objects_path, args.placer)
        run_diffuscene_remote_placer(cfg_runtime, args, room_path, objects_path, out_path, run_dir)
        return

    if runner == "ollama_llm":
        objects_path = require_objects_path(objects_path, args.placer)
        run_ollama_llm_placer(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=room_path,
            objects_path=objects_path,
            layout_mode=layout_mode,
            out_path=out_path,
            prompt_text=prompt_text,
        )
        return

    if runner == "m3dlayout_ar":
        run_m3dlayout_clean(cfg_runtime, args, room_path, objects_path, prompt_text, seed, out_path, model_type="autoregressive")
        return

    if runner == "m3dlayout_diffusion":
        run_m3dlayout_clean(cfg_runtime, args, room_path, objects_path, prompt_text, seed, out_path, model_type="diffusion")
        return

    if runner == "infinigen_clean":
        run_infinigen_clean(cfg_runtime, args, room_path, objects_path, seed, out_path, run_dir)
        return

    if runner == "lego_gen":
        raise RuntimeError("lego_gen не должен вызываться через execute_placer(); используй отдельную ветку в run_pipeline_for_mode")

    raise RuntimeError(f"Неизвестный runner для placer={args.placer}: {runner}")
