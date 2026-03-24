#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from ml.lego_seed_scene import (
    room_area_m2,
    total_objects_footprint_m2,
    sort_objects_for_generation,
    crop_last_object,
    build_seed_scene_and_placement,
)


DEFAULT_PATHS_CONFIG = "config/paths.yaml"

DEFAULT_MODES_BY_PLACER = {
    "cube": ["random", "relaxed"],
    "forest": ["random", "relaxed"],
    "graph_stat": ["random", "relaxed"],
    "diffusion": ["random", "relaxed"],
    "layout_refiner": ["random", "relaxed"],
    "diffuscene_remote": ["diffuscene"],
    "ollama_llm": ["llm"],
    "lego_gen": ["lego"],
}

DEFAULT_LEGO_GENERATION_PRESETS = {
    "reconstruct": {
        "init_scene_mode": "perturb",
        "num_restarts": 8,
        "init_pos_noise_std": 0.12,
        "init_ang_noise_deg": 20.0,
        "outer_passes": 1,
        "method": "grad_noise",
    },
    "gen_soft": {
        "init_scene_mode": "random_full",
        "num_restarts": 8,
        "init_pos_noise_std": 0.12,
        "init_ang_noise_deg": 20.0,
        "outer_passes": 1,
        "method": "grad_noise",
    },
    "gen_medium": {
        "init_scene_mode": "random_full",
        "num_restarts": 16,
        "init_pos_noise_std": 0.18,
        "init_ang_noise_deg": 35.0,
        "outer_passes": 1,
        "method": "grad_noise",
    },
    "gen_hard": {
        "init_scene_mode": "random_full",
        "num_restarts": 24,
        "init_pos_noise_std": 0.24,
        "init_ang_noise_deg": 60.0,
        "outer_passes": 1,
        "method": "grad_noise",
    },
    "gen_extreme": {
        "init_scene_mode": "random_full",
        "num_restarts": 32,
        "init_pos_noise_std": 0.30,
        "init_ang_noise_deg": 90.0,
        "outer_passes": 1,
        "method": "grad_noise",
    },
    "refine_light": {
        "init_scene_mode": "perturb",
        "num_restarts": 6,
        "init_pos_noise_std": 0.05,
        "init_ang_noise_deg": 8.0,
        "outer_passes": 1,
        "method": "grad_noise",
    },
}

PLACER_SPECS = {
    "cube": {
        "runner": "cube",
        "requires_ml_model": False,
        "supports_layout_mode": True,
        "mode_semantics": "direct",
    },
    "forest": {
        "runner": "ml_generic",
        "requires_ml_model": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "graph_stat": {
        "runner": "ml_generic",
        "requires_ml_model": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "diffusion": {
        "runner": "ml_generic",
        "requires_ml_model": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "layout_refiner": {
        "runner": "layout_refiner",
        "requires_ml_model": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "diffuscene_remote": {
        "runner": "diffuscene_remote",
        "requires_ml_model": False,
        "supports_layout_mode": False,
        "mode_semantics": None,
    },
    "ollama_llm": {
        "runner": "ollama_llm",
        "requires_ml_model": False,
        "supports_layout_mode": True,
        "mode_semantics": "prompt_hint",
    },
    "lego_gen": {
        "runner": "lego_gen",
        "requires_ml_model": False,
        "supports_layout_mode": False,
        "mode_semantics": None,
    },
}


@dataclass
class PlacementArtifacts:
    placement_legacy: Path
    placement_v1: Path
    scene_v1: Optional[Path]
    scene_legacy: Optional[Path]


@dataclass
class ModeOutputs:
    base_artifacts: PlacementArtifacts
    lego_artifacts: Optional[PlacementArtifacts]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"YAML-конфиг не найден: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Некорректный YAML-конфиг: {p}")
    return data


def get_nested(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def must_get(cfg: dict[str, Any], path: str) -> Any:
    value = get_nested(cfg, path, None)
    if value is None:
        raise KeyError(f"В YAML отсутствует обязательный ключ: {path}")
    return value


def resolve_local_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def project_root_from_config(cfg: dict[str, Any], cfg_path: Path) -> Path:
    root = get_nested(cfg, "project.root", None)
    if root:
        return Path(root).expanduser().resolve()
    return cfg_path.parent.parent.resolve()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def merge_room_spec_and_placements(room_json_path: str, placement_json_path: str, out_json_path: str) -> None:
    room_p = Path(room_json_path).expanduser().resolve()
    pl_p = Path(placement_json_path).expanduser().resolve()
    out_p = Path(out_json_path).expanduser().resolve()

    room = json.loads(room_p.read_text(encoding="utf-8"))
    pl = json.loads(pl_p.read_text(encoding="utf-8"))

    placements = pl.get("placements") or pl.get("items") or []
    if not isinstance(placements, list):
        placements = []

    scene = dict(room)
    scene["placements"] = placements

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_mode_run_dir(tmp_root: str, mode: str, run_dir_arg: Optional[str]) -> tuple[Path, bool]:
    if run_dir_arg:
        p = Path(run_dir_arg).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p, True

    run_hash = secrets.token_urlsafe(7).replace("-", "").replace("_", "").lower()
    p = Path(tmp_root).expanduser().resolve() / f"{mode}_{now_stamp()}_{run_hash}"
    p.mkdir(parents=True, exist_ok=True)
    return p, False


def read_prompt_from_args(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return str(args.prompt)
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    if args.items:
        return "Need to place the following objects: " + ", ".join(args.items)
    raise RuntimeError("Нужно передать либо positional items, либо --prompt, либо --prompt-file")


def normalize_prompt_for_object_choice(prompt_text: str, room_path: str) -> str:
    raw = (prompt_text or "").strip()
    low = raw.lower()

    room_type = None
    if "спаль" in low or "bedroom" in low:
        room_type = "bedroom"
    elif "гостин" in low or "living" in low:
        room_type = "living"
    elif "детск" in low or "kids" in low or "children" in low:
        room_type = "kids"
    elif "кабинет" in low or "office" in low:
        room_type = "office"

    if room_type is None:
        room_name = Path(room_path).name.lower()
        if "bedroom" in room_name:
            room_type = "bedroom"
        elif "living" in room_name:
            room_type = "living"

    items: list[tuple[str, int]] = []

    def add(cat: str, count: int = 1) -> None:
        for i, (c, n) in enumerate(items):
            if c == cat:
                items[i] = (c, max(n, count))
                return
        items.append((cat, count))

    if any(x in low for x in ["двуспаль", "double bed", "king bed", "king-size bed", "кровать", "bed"]):
        add("King-size Bed", 1)
    if any(x in low for x in ["шкаф", "wardrobe", "closet", "гардероб"]):
        add("Wardrobe", 1)
    if any(x in low for x in ["стол", "desk", "table", "письменный стол", "рабочий стол"]):
        add("Desk", 1)
    if any(x in low for x in ["стул", "chair", "кресло"]):
        add("Chair", 1)
    if any(x in low for x in ["тумба", "nightstand", "bedside"]):
        add("Nightstand", 1)
    if any(x in low for x in ["комод", "drawer chest", "chest of drawers", "dresser"]):
        add("Drawer Chest / Corner cabinet", 1)
    if any(x in low for x in ["туалетный столик", "dressing table", "vanity"]):
        add("Dressing Table", 1)
    if any(x in low for x in ["стеллаж", "bookcase", "bookshelf", "armoire"]):
        add("Bookcase / jewelry Armoire", 1)
    if any(x in low for x in ["светильник", "люстра", "lamp", "light"]):
        add("Ceiling Lamp", 1)

    if room_type == "bedroom":
        has_bed = any(c == "King-size Bed" for c, _ in items)
        has_storage = any(c in {"Wardrobe", "Drawer Chest / Corner cabinet"} for c, _ in items)
        if not has_bed:
            add("King-size Bed", 1)
        if not has_storage:
            add("Wardrobe", 1)

    if not items:
        return raw

    lines = [
        raw,
        "",
        "Normalized specification for LLM-based object selection.",
        f"ROOM_TYPE: {room_type or 'unknown'}",
        "REQUIRED_ITEMS:",
    ]
    lines.extend(f"- {cat} x{count}" for cat, count in items)
    lines.extend(
        [
            "",
            "The model must select objects strictly according to REQUIRED_ITEMS.",
            "Explicitly requested user items must not be dropped without a strong reason.",
            "If an item is explicitly requested by the user, it must appear in the final JSON items list.",
        ]
    )
    return "\n".join(lines)


def sync_objects_to_legacy_input(objects_path: Path, legacy_objects_json: str) -> None:
    dst = Path(legacy_objects_json).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(objects_path, dst)


def blender_outputs_for_mode(
    args: argparse.Namespace,
    run_dir: Path,
    mode: str,
    variant_suffix: str = "",
) -> tuple[Optional[str], Optional[str]]:
    suffix = f"_{variant_suffix}" if variant_suffix else ""

    if args.save_blend:
        p = Path(args.save_blend).expanduser().resolve()
        if p.suffix.lower() == ".blend":
            blend = str(p.with_name(f"{p.stem}_{mode}{suffix}.blend"))
        else:
            blend = str(p)
    else:
        blend = str((run_dir / f"scene_{mode}{suffix}.blend").resolve())

    if args.render:
        p = Path(args.render).expanduser().resolve()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            render = str(p.with_name(f"{p.stem}_{mode}{suffix}{p.suffix}"))
        else:
            render = str(p)
    else:
        render = str((run_dir / f"render_{mode}{suffix}.png").resolve())

    return blend, render


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def parse_modes(args: argparse.Namespace, cfg: dict[str, Any]) -> list[str]:
    raw = args.modes.strip() if args.modes else ""
    if raw:
        modes = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        modes = list(
            get_nested(cfg, f"modes_by_placer.{args.placer}", None)
            or DEFAULT_MODES_BY_PLACER.get(args.placer, ["random", "relaxed"])
        )

    if not modes:
        raise RuntimeError("Список режимов пуст")

    seen = set()
    uniq = []
    for m in modes:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


def normalize_json_artifact(
    cfg_runtime: dict[str, str],
    input_path: Path,
    output_path: Path,
    target: str,
) -> None:
    cmd = [
        sys.executable,
        cfg_runtime["NORMALIZE_JSON_SCRIPT"],
        "--input",
        str(input_path.expanduser().resolve()),
        "--output",
        str(output_path.expanduser().resolve()),
        "--target",
        target,
    ]
    print("▶ Нормализация JSON:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_normalized_scene_artifact(
    cfg_runtime: dict[str, str],
    room_path: str,
    placement_path: Path,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        cfg_runtime["NORMALIZE_JSON_SCRIPT"],
        "--room",
        str(Path(room_path).expanduser().resolve()),
        "--placement",
        str(placement_path.expanduser().resolve()),
        "--output",
        str(output_path.expanduser().resolve()),
        "--target",
        "scene",
    ]
    print("▶ Сборка канонического scene.v1:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def apply_config_defaults(args: argparse.Namespace, cfg: dict[str, Any], cfg_base_dir: Path) -> None:
    if args.room == "__USE_CFG_DEFAULT__":
        args.room = resolve_local_path(get_nested(cfg, "local.room.default_json"), cfg_base_dir)

    if args.prepared_info is None:
        args.prepared_info = resolve_local_path(get_nested(cfg, "local.data.prepared_info"), cfg_base_dir)

    if args.future_root is None:
        args.future_root = resolve_local_path(get_nested(cfg, "local.data.future_root"), cfg_base_dir)

    if args.remote_runner is None:
        args.remote_runner = resolve_local_path(get_nested(cfg, "local.scripts.remote_runner_sh"), cfg_base_dir)

    if args.blender is None:
        blender_bin = get_nested(cfg, "local.blender.binary", None)
        args.blender = resolve_local_path(blender_bin, cfg_base_dir) if blender_bin else None

    if args.max_attempts is None:
        args.max_attempts = int(get_nested(cfg, "defaults.max_attempts", 30))

    if args.placer is None:
        args.placer = str(get_nested(cfg, "defaults.placer", "cube"))

    if args.ml_device is None:
        args.ml_device = str(get_nested(cfg, "defaults.ml_device", "cpu"))

    if args.diffusion_steps is None:
        args.diffusion_steps = int(get_nested(cfg, "defaults.diffusion_steps", 50))

    if args.ollama_url is None:
        args.ollama_url = str(get_nested(cfg, "defaults.ollama.url", "http://127.0.0.1:11434"))

    cfg_models = get_nested(cfg, "defaults.ollama.models", None)
    if getattr(args, "ollama_models", None) is None:
        if isinstance(cfg_models, list):
            args.ollama_models = [str(x).strip() for x in cfg_models if str(x).strip()]
        else:
            args.ollama_models = None

    if args.ollama_model is None:
        args.ollama_model = str(get_nested(cfg, "defaults.ollama.model", "gpt-oss:20b"))

    if not getattr(args, "ollama_models", None):
        args.ollama_models = [str(args.ollama_model).strip()]

    args.ollama_models = [str(x).strip() for x in args.ollama_models if str(x).strip()]
    if not args.ollama_models:
        args.ollama_models = ["gpt-oss:20b"]
    args.ollama_model = args.ollama_models[0]

    if args.ollama_timeout is None:
        args.ollama_timeout = int(get_nested(cfg, "defaults.ollama.timeout", 300))

    if args.ollama_temperature is None:
        args.ollama_temperature = float(get_nested(cfg, "defaults.ollama.temperature", 0.1))

    if args.ollama_max_attempts is None:
        args.ollama_max_attempts = int(get_nested(cfg, "defaults.ollama.max_attempts", 8))

    if args.plan_model is None:
        args.plan_model = args.ollama_model

    if getattr(args, "plan_models", None) is None:
        args.plan_models = list(args.ollama_models) if getattr(args, "ollama_models", None) else [args.plan_model]

    args.plan_models = [str(x).strip() for x in args.plan_models if str(x).strip()]
    if not args.plan_models:
        args.plan_models = [str(args.plan_model).strip()]

    if args.plan_temperature is None:
        args.plan_temperature = float(args.ollama_temperature)

    if args.plan_think is None:
        args.plan_think = "low"

    if args.llm_think is None:
        args.llm_think = "none"

    if args.critic_model is None:
        args.critic_model = args.plan_model

    if getattr(args, "critic_models", None) is None:
        args.critic_models = list(args.plan_models) if getattr(args, "plan_models", None) else [args.critic_model]

    args.critic_models = [str(x).strip() for x in args.critic_models if str(x).strip()]
    if not args.critic_models:
        args.critic_models = [str(args.critic_model).strip()]

    if args.critic_temperature is None:
        args.critic_temperature = float(args.plan_temperature)

    if args.critic_think is None:
        args.critic_think = "low"

    if args.max_scene_attempts is None:
        args.max_scene_attempts = int(get_nested(cfg, "defaults.ollama.max_scene_attempts", 10))

    remote_host = get_nested(cfg, "remote.ssh.host", None)
    remote_port = get_nested(cfg, "remote.ssh.port", None)
    remote_user = get_nested(cfg, "remote.ssh.user", None)
    remote_key = get_nested(cfg, "remote.ssh.key", None)

    if getattr(args, "remote_host", None) is None and remote_host is not None:
        args.remote_host = str(remote_host)
    if getattr(args, "remote_port", None) is None and remote_port is not None:
        args.remote_port = int(remote_port)
    if getattr(args, "remote_user", None) is None and remote_user is not None:
        args.remote_user = str(remote_user)
    if getattr(args, "remote_key", None) is None and remote_key is not None:
        args.remote_key = str(Path(remote_key).expanduser())

    if args.ml_model is None:
        model_from_cfg = get_nested(cfg, f"local.ml_models.{args.placer}", None)
        if model_from_cfg:
            args.ml_model = resolve_local_path(model_from_cfg, cfg_base_dir)

    if args.lego_repo is None:
        args.lego_repo = get_nested(cfg, "remote.lego_net.repo_root", None)

    if args.lego_python is None:
        args.lego_python = get_nested(cfg, "remote.lego_net.python", None)

    if args.lego_helper_script is None:
        args.lego_helper_script = get_nested(cfg, "remote.lego_net.helper_script", None)

    if args.lego_tmp_root is None:
        args.lego_tmp_root = get_nested(cfg, "remote.lego_net.tmp_root", None)

    if args.lego_checkpoint_bedroom is None:
        args.lego_checkpoint_bedroom = get_nested(cfg, "remote.lego_net.checkpoint_bedroom", None)

    if args.lego_checkpoint_livingroom is None:
        args.lego_checkpoint_livingroom = get_nested(cfg, "remote.lego_net.checkpoint_livingroom", None)

    if args.lego_modes is None:
        args.lego_modes = get_nested(cfg, "local.lego.modes", "random,relaxed")

    if args.lego_generation_preset is None:
        args.lego_generation_preset = str(get_nested(cfg, "local.lego.generation_preset", "gen_medium"))

    if args.lego_method is None:
        args.lego_method = str(get_nested(cfg, "local.lego.method", "grad_noise"))

    if args.lego_outer_passes is None:
        cfg_val = get_nested(cfg, "local.lego.outer_passes", None)
        if cfg_val is not None:
            args.lego_outer_passes = int(cfg_val)

    if args.lego_num_restarts is None:
        cfg_val = get_nested(cfg, "local.lego.num_restarts", None)
        if cfg_val is not None:
            args.lego_num_restarts = int(cfg_val)

    if args.lego_init_pos_noise_std is None:
        cfg_val = get_nested(cfg, "local.lego.init_pos_noise_std", None)
        if cfg_val is not None:
            args.lego_init_pos_noise_std = float(cfg_val)

    if args.lego_init_ang_noise_deg is None:
        cfg_val = get_nested(cfg, "local.lego.init_ang_noise_deg", None)
        if cfg_val is not None:
            args.lego_init_ang_noise_deg = float(cfg_val)

    if args.lego_init_scene_mode is None:
        cfg_val = get_nested(cfg, "local.lego.init_scene_mode", None)
        if cfg_val is not None:
            args.lego_init_scene_mode = str(cfg_val)


def build_runtime_paths(cfg: dict[str, Any], cfg_base_dir: Path) -> dict[str, str]:
    runtime: dict[str, str] = {}

    runtime["CHOOSER_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.chooser"), cfg_base_dir)
    runtime["CUBE_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.cube"), cfg_base_dir)
    runtime["BLENDER_VIS_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.blender_visualize"), cfg_base_dir)
    runtime["ML_PLACER_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.ml_placer"), cfg_base_dir)
    runtime["LAYOUT_REFINER_SCRIPT"] = resolve_local_path(
        must_get(cfg, "local.scripts.layout_refiner_infer"),
        cfg_base_dir,
    )
    runtime["DIFFUSCENE_REMOTE_SCRIPT"] = resolve_local_path(
        must_get(cfg, "local.scripts.diffuscene_remote"),
        cfg_base_dir,
    )
    runtime["OLLAMA_LLM_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.ollama_layout"), cfg_base_dir)

    runtime["DEFAULT_ROOM_GLB"] = resolve_local_path(must_get(cfg, "local.room.default_glb"), cfg_base_dir)
    runtime["DEFAULT_ROOM_JSON"] = resolve_local_path(must_get(cfg, "local.room.default_json"), cfg_base_dir)

    runtime["LEGACY_OBJECTS_JSON"] = resolve_local_path(must_get(cfg, "local.input.objects_json"), cfg_base_dir)
    runtime["LEGACY_PLACEMENT_JSON"] = resolve_local_path(must_get(cfg, "local.output.placement_json"), cfg_base_dir)

    runtime["TMP_ROOT"] = resolve_local_path(must_get(cfg, "local.output.tmp_root"), cfg_base_dir)

    runtime["NORMALIZE_JSON_SCRIPT"] = resolve_local_path(
        must_get(cfg, "local.scripts.normalize_scene_format"),
        cfg_base_dir,
    )

    return runtime


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
        "ollama",
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


def resolve_lego_generation_params(args: argparse.Namespace) -> dict[str, Any]:
    preset_name = str(args.lego_generation_preset or "gen_medium").strip()
    if preset_name not in DEFAULT_LEGO_GENERATION_PRESETS:
        raise RuntimeError(
            f"Неизвестный lego generation preset: {preset_name}. "
            f"Допустимые: {sorted(DEFAULT_LEGO_GENERATION_PRESETS)}"
        )

    base = dict(DEFAULT_LEGO_GENERATION_PRESETS[preset_name])

    if args.lego_method is not None:
        base["method"] = str(args.lego_method)

    if args.lego_outer_passes is not None:
        base["outer_passes"] = int(args.lego_outer_passes)

    if args.lego_num_restarts is not None:
        base["num_restarts"] = int(args.lego_num_restarts)

    if args.lego_init_pos_noise_std is not None:
        base["init_pos_noise_std"] = float(args.lego_init_pos_noise_std)

    if args.lego_init_ang_noise_deg is not None:
        base["init_ang_noise_deg"] = float(args.lego_init_ang_noise_deg)

    if args.lego_init_scene_mode is not None:
        base["init_scene_mode"] = str(args.lego_init_scene_mode)

    base["preset"] = preset_name
    return base


def infer_lego_room_type(room_path: str) -> Optional[str]:
    name = Path(room_path).name.lower()

    if "living" in name:
        return "livingroom"
    if "bedroom" in name:
        return "bedroom"

    try:
        room = read_json(room_path)
    except Exception:
        return None

    candidates = []

    for key in ("room_type", "type", "category", "name", "label"):
        v = room.get(key)
        if isinstance(v, str):
            candidates.append(v.lower())

    meta = room.get("meta")
    if isinstance(meta, dict):
        for key in ("room_type", "type", "category", "name", "label"):
            v = meta.get(key)
            if isinstance(v, str):
                candidates.append(v.lower())

    room_block = room.get("room")
    if isinstance(room_block, dict):
        for key in ("room_type", "type", "category", "name", "label"):
            v = room_block.get(key)
            if isinstance(v, str):
                candidates.append(v.lower())

    joined = " ".join(candidates)
    if "living" in joined:
        return "livingroom"
    if "bedroom" in joined:
        return "bedroom"

    return None


def choose_scene_for_render(artifacts: PlacementArtifacts) -> Path:
    if artifacts.scene_v1 and artifacts.scene_v1.is_file():
        return artifacts.scene_v1
    if artifacts.scene_legacy and artifacts.scene_legacy.is_file():
        return artifacts.scene_legacy
    raise RuntimeError("Нет доступного scene-артефакта для рендера")


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

    while True:
        objs = objects_v1.get("objects") or []
        if not objs:
            raise RuntimeError("lego_gen: закончились объекты, не удалось построить валидную сцену")

        footprint = total_objects_footprint_m2(objects_v1)

        manifest = {
            "reduction_index": reduction_index,
            "object_count": len(objs),
            "room_area_m2": room_area,
            "objects_footprint_m2": footprint,
            "fill_ratio": (footprint / room_area) if room_area > 1e-9 else None,
        }
        write_json(out_dir / f"reduction_{reduction_index:02d}.json", manifest)

        if room_area <= 1e-9:
            raise RuntimeError("Площадь комнаты некорректна")

        if footprint > room_area * fill_ratio_limit:
            print(
                f"⚠️ lego_gen: footprint={footprint:.3f} > limit={room_area * fill_ratio_limit:.3f}, "
                f"удаляем последний объект",
                flush=True,
            )
            objects_v1 = crop_last_object(objects_v1)
            reduction_index += 1
            continue

        for attempt in range(1, max_scene_tries + 1):
            seed = int.from_bytes(secrets.token_bytes(8), "big")

            seed_scene_v1, seed_placement_v1 = build_seed_scene_and_placement(
                room_json=room_json,
                objects_v1=objects_v1,
                seed=seed,
            )

            local_seed_scene = out_dir / f"seed_scene_r{reduction_index:02d}_a{attempt:02d}.v1.json"
            local_seed_placement = out_dir / f"seed_placement_r{reduction_index:02d}_a{attempt:02d}.v1.json"
            local_seed_legacy = out_dir / f"seed_placement_r{reduction_index:02d}_a{attempt:02d}.json"

            write_json(local_seed_scene, seed_scene_v1)
            write_json(local_seed_placement, seed_placement_v1)
            write_json(
                local_seed_legacy,
                {
                    "placements": deepcopy(seed_scene_v1["placements"]),
                    "placer": "lego_gen",
                    "mode": "random_full",
                },
            )

            remote_run_name = f"{room_type}_lego_gen_{secrets.token_urlsafe(8).replace('-', '').replace('_', '').lower()}"
            remote_run_dir = f"{args.lego_tmp_root.rstrip('/')}/{remote_run_name}"

            remote_room = f"{remote_run_dir}/room.json"
            remote_in_scene_v1 = f"{remote_run_dir}/scene_input.v1.json"
            remote_in_placement_v1 = f"{remote_run_dir}/placement_input.v1.json"
            remote_in_placement_legacy = f"{remote_run_dir}/placement_input_legacy.json"

            remote_out_scene_v1 = f"{remote_run_dir}/scene_lego.v1.json"
            remote_out_scene_legacy = f"{remote_run_dir}/scene_lego.json"
            remote_out_placement_v1 = f"{remote_run_dir}/placement_lego.v1.json"
            remote_out_placement_legacy = f"{remote_run_dir}/placement_lego.json"

            ssh_run(args, f"mkdir -p {shlex.quote(remote_run_dir)}")
            scp_upload(args, Path(room_path), remote_room)
            scp_upload(args, local_seed_scene, remote_in_scene_v1)
            scp_upload(args, local_seed_placement, remote_in_placement_v1)
            scp_upload(args, local_seed_legacy, remote_in_placement_legacy)

            remote_cmd_parts = [
                shlex.quote(args.lego_python),
                shlex.quote(args.lego_helper_script),
                "--room-type", shlex.quote(room_type),
                "--mode", shlex.quote("lego_gen"),
                "--room", shlex.quote(remote_room),
                "--in-placement-legacy", shlex.quote(remote_in_placement_legacy),
                "--in-placement-v1", shlex.quote(remote_in_placement_v1),
                "--in-scene-v1", shlex.quote(remote_in_scene_v1),
                "--out-placement-legacy", shlex.quote(remote_out_placement_legacy),
                "--out-placement-v1", shlex.quote(remote_out_placement_v1),
                "--out-scene-legacy", shlex.quote(remote_out_scene_legacy),
                "--out-scene-v1", shlex.quote(remote_out_scene_v1),
                "--checkpoint", shlex.quote(checkpoint),
                "--lego-repo", shlex.quote(args.lego_repo),
                "--outer-passes", shlex.quote(str(int(gen_cfg["outer_passes"]))),
                "--method", shlex.quote(str(gen_cfg["method"])),
                "--num-restarts", shlex.quote(str(int(gen_cfg["num_restarts"]))),
                "--init-pos-noise-std", shlex.quote(str(float(gen_cfg["init_pos_noise_std"]))),
                "--init-ang-noise-deg", shlex.quote(str(float(gen_cfg["init_ang_noise_deg"]))),
                "--init-scene-mode", shlex.quote(str(gen_cfg["init_scene_mode"])),
            ]

            try:
                ssh_run(args, " ".join(remote_cmd_parts))
            except Exception as e:
                print(
                    f"⚠️ lego_gen remote attempt failed: reduction={reduction_index}, attempt={attempt}, err={e}",
                    flush=True,
                )
                continue

            out_placement_legacy = out_dir / f"placement_lego_r{reduction_index:02d}_a{attempt:02d}.json"
            out_placement_v1 = out_dir / f"placement_lego_r{reduction_index:02d}_a{attempt:02d}.v1.json"
            out_scene_legacy = out_dir / f"scene_lego_r{reduction_index:02d}_a{attempt:02d}.json"
            out_scene_v1 = out_dir / f"scene_lego_r{reduction_index:02d}_a{attempt:02d}.v1.json"

            try:
                scp_download(args, remote_out_placement_legacy, out_placement_legacy)
                scp_download(args, remote_out_placement_v1, out_placement_v1)
                scp_download(args, remote_out_scene_legacy, out_scene_legacy)
                scp_download(args, remote_out_scene_v1, out_scene_v1)
            except Exception as e:
                print(
                    f"⚠️ lego_gen download failed: reduction={reduction_index}, attempt={attempt}, err={e}",
                    flush=True,
                )
                continue

            final_scene = read_json(out_scene_v1)
            print(json.dumps(final_scene, ensure_ascii=False, indent=2), flush=True)

            return PlacementArtifacts(
                placement_legacy=out_placement_legacy,
                placement_v1=out_placement_v1,
                scene_v1=out_scene_v1,
                scene_legacy=out_scene_legacy,
            )

        print(f"⚠️ lego_gen: не удалось найти валидную сцену для {len(objs)} объектов, удаляем последний", flush=True)
        objects_v1 = crop_last_object(objects_v1)
        reduction_index += 1


def maybe_run_lego_postprocess(
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    base_artifacts: PlacementArtifacts,
) -> Optional[PlacementArtifacts]:
    if not args.lego_postprocess:
        return None

    enabled_modes = parse_csv_set(args.lego_modes)
    if layout_mode not in enabled_modes:
        print(f"⏭ LEGO-Net skip: mode={layout_mode} not in {sorted(enabled_modes)}")
        return None

    if args.lego_room_type == "auto":
        room_type = infer_lego_room_type(room_path)
    else:
        room_type = args.lego_room_type

    if room_type not in {"bedroom", "livingroom"}:
        print(f"⏭ LEGO-Net skip: room_type for {room_path} is unsupported")
        return None

    if base_artifacts.scene_v1 is None or not base_artifacts.scene_v1.is_file():
        print("⏭ LEGO-Net skip: scene.v1.json is missing")
        return None

    if not args.lego_repo:
        raise RuntimeError("Не задан remote.lego_net.repo_root / --lego-repo")
    if not args.lego_python:
        raise RuntimeError("Не задан remote.lego_net.python / --lego-python")
    if not args.lego_helper_script:
        raise RuntimeError("Не задан remote.lego_net.helper_script / --lego-helper-script")
    if not args.lego_tmp_root:
        raise RuntimeError("Не задан remote.lego_net.tmp_root / --lego-tmp-root")

    checkpoint = None
    if room_type == "bedroom":
        checkpoint = args.lego_checkpoint_bedroom
    elif room_type == "livingroom":
        checkpoint = args.lego_checkpoint_livingroom

    if not checkpoint:
        raise RuntimeError(f"Не задан checkpoint для room_type={room_type}")

    gen_cfg = resolve_lego_generation_params(args)

    out_dir = run_dir / "lego_post"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = gen_cfg["preset"]

    out_placement_legacy = out_dir / f"placement_{layout_mode}_lego_{suffix}.json"
    out_placement_v1 = out_dir / f"placement_{layout_mode}_lego_{suffix}.v1.json"
    out_scene_legacy = out_dir / f"scene_{layout_mode}_lego_{suffix}.json"
    out_scene_v1 = out_dir / f"scene_{layout_mode}_lego_{suffix}.v1.json"

    remote_run_name = f"{room_type}_{layout_mode}_{secrets.token_urlsafe(8).replace('-', '').replace('_', '').lower()}"
    remote_run_dir = f"{args.lego_tmp_root.rstrip('/')}/{remote_run_name}"

    remote_room = f"{remote_run_dir}/room.json"
    remote_in_placement_legacy = f"{remote_run_dir}/placement_input_legacy.json"
    remote_in_placement_v1 = f"{remote_run_dir}/placement_input.v1.json"
    remote_in_scene_v1 = f"{remote_run_dir}/scene_input.v1.json"

    remote_out_placement_legacy = f"{remote_run_dir}/placement_lego.json"
    remote_out_placement_v1 = f"{remote_run_dir}/placement_lego.v1.json"
    remote_out_scene_legacy = f"{remote_run_dir}/scene_lego.json"
    remote_out_scene_v1 = f"{remote_run_dir}/scene_lego.v1.json"

    ssh_run(args, f"mkdir -p {shlex.quote(remote_run_dir)}")

    scp_upload(args, Path(room_path), remote_room)
    scp_upload(args, base_artifacts.placement_legacy, remote_in_placement_legacy)
    scp_upload(args, base_artifacts.placement_v1, remote_in_placement_v1)
    scp_upload(args, base_artifacts.scene_v1, remote_in_scene_v1)

    remote_cmd_parts = [
        shlex.quote(args.lego_python),
        shlex.quote(args.lego_helper_script),
        "--room-type", shlex.quote(room_type),
        "--mode", shlex.quote(layout_mode),
        "--room", shlex.quote(remote_room),
        "--in-placement-legacy", shlex.quote(remote_in_placement_legacy),
        "--in-placement-v1", shlex.quote(remote_in_placement_v1),
        "--in-scene-v1", shlex.quote(remote_in_scene_v1),
        "--out-placement-legacy", shlex.quote(remote_out_placement_legacy),
        "--out-placement-v1", shlex.quote(remote_out_placement_v1),
        "--out-scene-legacy", shlex.quote(remote_out_scene_legacy),
        "--out-scene-v1", shlex.quote(remote_out_scene_v1),
        "--checkpoint", shlex.quote(checkpoint),
        "--lego-repo", shlex.quote(args.lego_repo),
        "--outer-passes", shlex.quote(str(int(gen_cfg["outer_passes"]))),
        "--method", shlex.quote(str(gen_cfg["method"])),
        "--num-restarts", shlex.quote(str(int(gen_cfg["num_restarts"]))),
        "--init-pos-noise-std", shlex.quote(str(float(gen_cfg["init_pos_noise_std"]))),
        "--init-ang-noise-deg", shlex.quote(str(float(gen_cfg["init_ang_noise_deg"]))),
        "--init-scene-mode", shlex.quote(str(gen_cfg["init_scene_mode"])),
    ]
    ssh_run(args, " ".join(remote_cmd_parts))

    scp_download(args, remote_out_placement_legacy, out_placement_legacy)
    scp_download(args, remote_out_placement_v1, out_placement_v1)
    scp_download(args, remote_out_scene_legacy, out_scene_legacy)
    scp_download(args, remote_out_scene_v1, out_scene_v1)

    print(
        "✅ LEGO-Net refined "
        f"mode={layout_mode}, room_type={room_type}, "
        f"preset={gen_cfg['preset']}, init_scene_mode={gen_cfg['init_scene_mode']}, "
        f"restarts={gen_cfg['num_restarts']}, "
        f"pos_noise={gen_cfg['init_pos_noise_std']}, "
        f"ang_noise={gen_cfg['init_ang_noise_deg']}"
    )

    return PlacementArtifacts(
        placement_legacy=out_placement_legacy,
        placement_v1=out_placement_v1,
        scene_v1=out_scene_v1,
        scene_legacy=out_scene_legacy if out_scene_legacy.is_file() else None,
    )


def execute_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
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
        run_cube_placer(cfg_runtime, room_path, objects_path, layout_mode, out_path, seed)
        return

    if runner == "layout_refiner":
        run_layout_refiner_placer(cfg_runtime, args, room_path, objects_path, layout_mode, seed, out_path, run_dir)
        return

    if runner == "ml_generic":
        run_ml_placer(cfg_runtime, args, room_path, objects_path, layout_mode, seed, out_path)
        return

    if runner == "diffuscene_remote":
        run_diffuscene_remote_placer(cfg_runtime, args, room_path, objects_path, out_path, run_dir)
        return

    if runner == "ollama_llm":
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

    if runner == "lego_gen":
        raise RuntimeError("lego_gen не должен вызываться через execute_placer(); используй отдельную ветку в run_pipeline_for_mode")

    raise RuntimeError(f"Неизвестный runner для placer={args.placer}: {runner}")


def build_scene_artifacts(
    cfg_runtime: dict[str, Any],
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    placement_out: Path,
    variant_suffix: str = "",
) -> PlacementArtifacts:
    suffix = f"_{variant_suffix}" if variant_suffix else ""

    normalized_placement_path = run_dir / f"placement{suffix}.v1.json"
    normalize_json_artifact(
        cfg_runtime=cfg_runtime,
        input_path=placement_out,
        output_path=normalized_placement_path,
        target="placement",
    )

    scene_v1_path = None
    scene_legacy_path = None

    if room_path.lower().endswith(".json"):
        scene_v1_path = run_dir / f"scene{suffix}.v1.json"
        build_normalized_scene_artifact(
            cfg_runtime=cfg_runtime,
            room_path=room_path,
            placement_path=placement_out,
            output_path=scene_v1_path,
        )

        scene_legacy_path = run_dir / f"scene_{layout_mode}{suffix}.json"
        merge_room_spec_and_placements(room_path, str(placement_out.resolve()), str(scene_legacy_path.resolve()))

    return PlacementArtifacts(
        placement_legacy=placement_out,
        placement_v1=normalized_placement_path,
        scene_v1=scene_v1_path,
        scene_legacy=scene_legacy_path,
    )


def run_blender_for_mode(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    scene_json_path: Path,
    variant_suffix: str = "",
) -> None:
    if not scene_json_path.is_file():
        raise RuntimeError(f"Scene JSON not found for Blender: {scene_json_path}")

    blend_out, render_out = blender_outputs_for_mode(args, run_dir, layout_mode, variant_suffix=variant_suffix)

    glb_for_arg = os.path.abspath(cfg_runtime["DEFAULT_ROOM_GLB"])
    cmd = [
        sys.executable,
        cfg_runtime["BLENDER_VIS_SCRIPT"],
        "--glb",
        glb_for_arg,
        "--json",
        str(scene_json_path.resolve()),
    ]

    if args.blender:
        cmd += ["--blender", args.blender]
    if args.headless:
        cmd.append("--background")
    if args.no_import_glb:
        cmd.append("--no-import-glb")
    if blend_out:
        cmd += ["--save-blend", str(Path(blend_out).resolve())]
    if render_out:
        cmd += ["--render", str(Path(render_out).resolve())]

    print("▶ Запуск Blender-визуализатора:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_pipeline_for_mode(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    layout_mode: str,
    prompt_text: str,
) -> ModeOutputs:
    print(f"\n====== РЕЖИМ {layout_mode.upper()} ======")
    print(f"📁 mode_run_dir: {run_dir}")

    chooser_seed = int.from_bytes(secrets.token_bytes(8), "big")
    chooser_prompt_text = prompt_text
    (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (run_dir / "chooser_prompt.txt").write_text(chooser_prompt_text, encoding="utf-8")

    objects_path = run_choose_stage(
        args=args,
        cfg_runtime=cfg_runtime,
        room_path=room_path,
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

    run_manifest = {
        "room": room_path,
        "prompt": prompt_text,
        "chooser_prompt": chooser_prompt_text,
        "chooser_seed": chooser_seed,
        "placer": args.placer,
        "layout_mode": layout_mode,
        "run_dir": str(run_dir),
        "objects_legacy": str(objects_path.resolve()),
        "objects_v1": str(normalized_objects_path.resolve()),
        "chooser_llm": {
            "provider": "ollama",
            "url": args.ollama_url,
            "model": args.ollama_model,
            "models": list(args.ollama_models) if getattr(args, "ollama_models", None) else [args.ollama_model],
            "timeout": int(args.ollama_timeout),
            "temperature": float(args.ollama_temperature),
            "max_attempts": int(args.ollama_max_attempts),
            "debug_dir": str((run_dir / "llm_choose_debug").resolve()),
        },
        "placement_llm": {
            "plan_model": args.plan_model,
            "plan_models": list(args.plan_models) if getattr(args, "plan_models", None) else [args.plan_model],
            "critic_model": args.critic_model,
            "critic_models": list(args.critic_models) if getattr(args, "critic_models", None) else [args.critic_model],
            "json_model": args.ollama_model,
            "json_models": list(args.ollama_models) if getattr(args, "ollama_models", None) else [args.ollama_model],
            "plan_temperature": float(args.plan_temperature),
            "critic_temperature": float(args.critic_temperature),
            "json_temperature": float(args.ollama_temperature),
            "plan_think": args.plan_think,
            "critic_think": args.critic_think,
            "json_think": args.llm_think,
            "timeout": int(args.ollama_timeout),
            "max_attempts": int(args.ollama_max_attempts),
            "max_scene_attempts": int(args.max_scene_attempts),
        },
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    if args.placer == "lego_gen":
        lego_artifacts = run_lego_generate_from_scratch(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=room_path,
            objects_v1_path=normalized_objects_path,
            run_dir=run_dir,
        )

        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["lego_gen"] = {
            "enabled": True,
            "placement_legacy": str(lego_artifacts.placement_legacy.resolve()),
            "placement_v1": str(lego_artifacts.placement_v1.resolve()),
            "scene_v1": str(lego_artifacts.scene_v1.resolve()) if lego_artifacts.scene_v1 else None,
            "scene_legacy": str(lego_artifacts.scene_legacy.resolve()) if lego_artifacts.scene_legacy else None,
        }
        write_json(manifest_path, manifest)

        if args.skip_blender:
            print(f"⏭ Пропуск Blender для режима {layout_mode}")
            print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
            return ModeOutputs(base_artifacts=lego_artifacts, lego_artifacts=None)

        run_blender_for_mode(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=room_path,
            run_dir=run_dir,
            layout_mode=layout_mode,
            scene_json_path=choose_scene_for_render(lego_artifacts),
            variant_suffix="lego_gen",
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
                "objects_path": str(objects_path.resolve()),
                "objects_v1_path": str(normalized_objects_path.resolve()),
                "placement_legacy_path": str(placement_out.resolve()),
            }
            write_json(run_dir / f"attempt_{attempt:02d}.json", attempt_info)

            execute_placer(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=room_path,
                objects_path=objects_path,
                layout_mode=layout_mode,
                seed=attempt_seed,
                out_path=placement_out,
                run_dir=run_dir,
                prompt_text=prompt_text,
            )

            base_artifacts = build_scene_artifacts(
                cfg_runtime=cfg_runtime,
                room_path=room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                placement_out=placement_out,
                variant_suffix="",
            )

            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["base"] = {
                "placement_legacy": str(base_artifacts.placement_legacy.resolve()),
                "placement_v1": str(base_artifacts.placement_v1.resolve()),
                "scene_v1": str(base_artifacts.scene_v1.resolve()) if base_artifacts.scene_v1 else None,
                "scene_legacy": str(base_artifacts.scene_legacy.resolve()) if base_artifacts.scene_legacy else None,
            }
            write_json(manifest_path, manifest)

            print(f"✅ placement stage success: {layout_mode}")
            break

        except subprocess.CalledProcessError:
            print(f"⚠️ Неудачная попытка placement ({layout_mode}), пересборка...")
        except Exception as e:
            print(f"❌ Ошибка placement ({layout_mode}): {e}")

    if base_artifacts is None:
        raise RuntimeError(f"Не удалось собрать placement в режиме {layout_mode}")

    lego_artifacts: Optional[PlacementArtifacts] = None
    lego_error: Optional[str] = None

    if args.lego_postprocess:
        try:
            lego_cfg = resolve_lego_generation_params(args)

            lego_artifacts = maybe_run_lego_postprocess(
                args=args,
                room_path=room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                base_artifacts=base_artifacts,
            )

            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["lego"] = {
                "enabled": True,
                "success": lego_artifacts is not None,
                "generation_preset": lego_cfg["preset"],
                "method": lego_cfg["method"],
                "outer_passes": int(lego_cfg["outer_passes"]),
                "num_restarts": int(lego_cfg["num_restarts"]),
                "init_pos_noise_std": float(lego_cfg["init_pos_noise_std"]),
                "init_ang_noise_deg": float(lego_cfg["init_ang_noise_deg"]),
                "init_scene_mode": lego_cfg["init_scene_mode"],
                "placement_legacy": str(lego_artifacts.placement_legacy.resolve()) if lego_artifacts else None,
                "placement_v1": str(lego_artifacts.placement_v1.resolve()) if lego_artifacts else None,
                "scene_v1": str(lego_artifacts.scene_v1.resolve()) if lego_artifacts and lego_artifacts.scene_v1 else None,
                "scene_legacy": str(lego_artifacts.scene_legacy.resolve()) if lego_artifacts and lego_artifacts.scene_legacy else None,
            }
            write_json(manifest_path, manifest)

        except Exception as e:
            lego_error = str(e)
            print(f"⚠️ LEGO postprocess failed ({layout_mode}): {lego_error}")

            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["lego"] = {
                "enabled": True,
                "success": False,
                "error": lego_error,
            }
            write_json(manifest_path, manifest)

            if args.lego_failure_policy == "raise":
                raise

    if args.skip_blender:
        print(f"⏭ Пропуск Blender для режима {layout_mode}")
        print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
        return ModeOutputs(base_artifacts=base_artifacts, lego_artifacts=lego_artifacts)

    try:
        if args.lego_render_policy == "base_only":
            run_blender_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                scene_json_path=choose_scene_for_render(base_artifacts),
                variant_suffix="base",
            )

        elif args.lego_render_policy == "lego_only":
            render_artifacts = lego_artifacts if lego_artifacts is not None else base_artifacts
            render_suffix = "lego" if lego_artifacts is not None else "base_fallback"
            run_blender_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                scene_json_path=choose_scene_for_render(render_artifacts),
                variant_suffix=render_suffix,
            )

        else:
            run_blender_for_mode(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=room_path,
                run_dir=run_dir,
                layout_mode=layout_mode,
                scene_json_path=choose_scene_for_render(base_artifacts),
                variant_suffix="base",
            )

            if lego_artifacts is not None:
                run_blender_for_mode(
                    cfg_runtime=cfg_runtime,
                    args=args,
                    room_path=room_path,
                    run_dir=run_dir,
                    layout_mode=layout_mode,
                    scene_json_path=choose_scene_for_render(lego_artifacts),
                    variant_suffix="lego",
                )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Placement для режима {layout_mode} успешно построен, "
            f"но Blender/рендер завершился ошибкой: {e}"
        ) from e

    print(f"\n✅ УСПЕХ! РЕЖИМ {layout_mode}")
    return ModeOutputs(base_artifacts=base_artifacts, lego_artifacts=lego_artifacts)


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Пайплайн: prompt -> выбор предметов -> расстановка -> optional LEGO-Net(server) -> Blender")

    p.add_argument("--paths-config", default=DEFAULT_PATHS_CONFIG)
    p.add_argument("items", nargs="*")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)

    p.add_argument("--room", default="__USE_CFG_DEFAULT__")
    p.add_argument("--prepared-info", default=None)
    p.add_argument("--future-root", default=None)

    p.add_argument("--run-dir", default=None)
    p.add_argument("--keep-tmp", dest="keep_tmp", action="store_true", default=True)
    p.add_argument("--no-keep-tmp", dest="keep_tmp", action="store_false")

    p.add_argument("--blender", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-import-glb", action="store_true")
    p.add_argument("--save-blend", default=None)
    p.add_argument("--render", default=None)
    p.add_argument("--skip-blender", action="store_true")
    p.add_argument("--max-attempts", type=int, default=None)

    p.add_argument(
        "--placer",
        choices=["cube", "forest", "graph_stat", "diffusion", "layout_refiner", "diffuscene_remote", "ollama_llm", "lego_gen"],
        default=None,
    )
    p.add_argument("--ml-model", default=None)
    p.add_argument("--ml-device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--diffusion-steps", type=int, default=None)

    p.add_argument("--remote-runner", default=None)
    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=None)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)

    p.add_argument("--ollama-url", default=None)
    p.add_argument("--ollama-model", default=None)
    p.add_argument("--ollama-models", nargs="*", default=None)
    p.add_argument("--ollama-timeout", type=int, default=None)
    p.add_argument("--ollama-temperature", type=float, default=None)
    p.add_argument("--ollama-max-attempts", type=int, default=None)

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

    p.add_argument("--lego-postprocess", action="store_true")
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