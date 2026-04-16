#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


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
    "m3dlayout_ar": ["m3dlayout_ar"],
    "m3dlayout_diffusion": ["m3dlayout_diffusion"],
    "infinigen_clean": ["infinigen_clean"],
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
        "requires_object_selection": True,
        "supports_layout_mode": True,
        "mode_semantics": "direct",
    },
    "forest": {
        "runner": "ml_generic",
        "requires_ml_model": True,
        "requires_object_selection": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "graph_stat": {
        "runner": "ml_generic",
        "requires_ml_model": True,
        "requires_object_selection": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "diffusion": {
        "runner": "ml_generic",
        "requires_ml_model": True,
        "requires_object_selection": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "layout_refiner": {
        "runner": "layout_refiner",
        "requires_ml_model": True,
        "requires_object_selection": True,
        "supports_layout_mode": True,
        "mode_semantics": "init_mode",
    },
    "diffuscene_remote": {
        "runner": "diffuscene_remote",
        "requires_ml_model": False,
        "requires_object_selection": True,
        "supports_layout_mode": False,
        "mode_semantics": None,
    },
    "ollama_llm": {
        "runner": "ollama_llm",
        "requires_ml_model": False,
        "requires_object_selection": True,
        "supports_layout_mode": True,
        "mode_semantics": "prompt_hint",
    },
    "lego_gen": {
        "runner": "lego_gen",
        "requires_ml_model": False,
        "requires_object_selection": True,
        "supports_layout_mode": False,
        "mode_semantics": None,
    },
    "m3dlayout_ar": {
        "runner": "m3dlayout_ar",
        "requires_ml_model": False,
        "requires_object_selection": True,
        "supports_layout_mode": False,
        "mode_semantics": None,
    },
    "m3dlayout_diffusion": {
        "runner": "m3dlayout_diffusion",
        "requires_ml_model": False,
        "requires_object_selection": True,
        "supports_layout_mode": False,
        "mode_semantics": None,
    },
    "infinigen_clean": {
        "runner": "infinigen_clean",
        "requires_ml_model": False,
        "requires_object_selection": True,
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

    if getattr(args, "remote_conda_env", None) is None:
        if args.placer in {"m3dlayout_ar", "m3dlayout_diffusion"}:
            cfg_val = get_nested(cfg, "remote.m3dlayout.conda_env", None)
            if cfg_val is not None:
                args.remote_conda_env = str(cfg_val)
        elif args.placer == "infinigen_clean":
            cfg_val = get_nested(cfg, "remote.infinigen.conda_env", None)
            if cfg_val is not None:
                args.remote_conda_env = str(cfg_val)

    if getattr(args, "infinigen_src", None) is None:
        cfg_val = get_nested(cfg, "local.infinigen.src", None)
        if cfg_val is not None:
            args.infinigen_src = resolve_local_path(str(cfg_val), cfg_base_dir)

    if getattr(args, "remote_infinigen_src", None) is None:
        cfg_val = get_nested(cfg, "remote.infinigen.src", None)
        if cfg_val is not None:
            args.remote_infinigen_src = str(cfg_val)

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
    runtime["OLLAMA_LLM_SCRIPT"] = resolve_local_path(
        must_get(cfg, "local.scripts.ollama_layout"),
        cfg_base_dir,
    )
    runtime["M3DLAYOUT_SCRIPT"] = resolve_local_path(
        get_nested(cfg, "local.scripts.m3dlayout", "src/Plasement/run_m3dlayout.py"),
        cfg_base_dir,
    )
    runtime["INFINIGEN_CLEAN_SCRIPT"] = resolve_local_path(
        get_nested(cfg, "local.scripts.infinigen_clean", "src/Plasement/run_infinigen_clean.py"),
        cfg_base_dir,
    )

    runtime["DEFAULT_ROOM_GLB"] = resolve_local_path(must_get(cfg, "local.room.default_glb"), cfg_base_dir)
    runtime["DEFAULT_ROOM_JSON"] = resolve_local_path(must_get(cfg, "local.room.default_json"), cfg_base_dir)

    runtime["LEGACY_OBJECTS_JSON"] = resolve_local_path(must_get(cfg, "local.input.objects_json"), cfg_base_dir)
    runtime["LEGACY_PLACEMENT_JSON"] = resolve_local_path(must_get(cfg, "local.output.placement_json"), cfg_base_dir)

    runtime["TMP_ROOT"] = resolve_local_path(must_get(cfg, "local.output.tmp_root"), cfg_base_dir)

    runtime["NORMALIZE_JSON_SCRIPT"] = resolve_local_path(
        must_get(cfg, "local.scripts.normalize_scene_format"),
        cfg_base_dir,
    )

    runtime["LEGO_POSTPROCESS_SCRIPT"] = resolve_local_path(
        must_get(cfg, "local.scripts.lego_postprocess"),
        cfg_base_dir,
    )

    return runtime


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
