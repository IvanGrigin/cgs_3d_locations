#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/run_quality_search.py

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import yaml

from pipeline_config import PLACER_SPECS
from pipeline_runners import run_infinigen_clean, run_m3dlayout_clean


# ------------------------------------------------------------
# Конфиг
# ------------------------------------------------------------
DEFAULT_PATHS_CONFIG = "config/paths.yaml"

DEFAULT_QUALITY_THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 0.95]

# Формат: <placer>:<mode>
DEFAULT_STAGE_SEQUENCE = [
    ("cube", "random"),
    ("cube", "relaxed"),
    ("ollama_llm", "llm"),
]


# ------------------------------------------------------------
# YAML / config utils
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# IO / misc
# ------------------------------------------------------------
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def quality_label(x: float) -> str:
    return f"q_{x:.2f}".replace(".", "_")


def parse_quality_thresholds(raw: Optional[str]) -> list[float]:
    if not raw:
        return list(DEFAULT_QUALITY_THRESHOLDS)
    vals = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        v = float(s)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Некорректный threshold: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("Список quality thresholds пуст")
    return vals


def parse_stage_sequence(raw: Optional[str]) -> list[tuple[str, str]]:
    if not raw:
        return list(DEFAULT_STAGE_SEQUENCE)

    out: list[tuple[str, str]] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        if ":" not in s:
            raise ValueError(f"Некорректный stage-sequence элемент: {s}. Ожидалось placer:mode")
        placer, mode = s.split(":", 1)
        placer = placer.strip()
        mode = mode.strip()
        if not placer or not mode:
            raise ValueError(f"Некорректный stage-sequence элемент: {s}")
        if placer not in PLACER_SPECS:
            raise ValueError(
                f"Неизвестный placer в stage-sequence: {placer}. "
                f"Допустимые: {', '.join(sorted(PLACER_SPECS.keys()))}"
            )
        out.append((placer, mode))

    if not out:
        raise ValueError("Список stage sequence пуст")
    return out


def make_search_run_dir(tmp_root: str, run_dir_arg: Optional[str]) -> tuple[Path, bool]:
    if run_dir_arg:
        p = Path(run_dir_arg).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p, True

    run_hash = secrets.token_urlsafe(7).replace("-", "").replace("_", "").lower()
    p = Path(tmp_root).expanduser().resolve() / f"quality_search_{now_stamp()}_{run_hash}"
    p.mkdir(parents=True, exist_ok=True)
    return p, False


def read_prompt_from_args(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return str(args.prompt)
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    if args.items:
        return "Нужно разместить следующие предметы: " + ", ".join(args.items)
    raise RuntimeError("Нужно передать либо positional items, либо --prompt, либо --prompt-file")


def sync_objects_to_legacy_input(objects_path: Path, legacy_objects_json: str) -> None:
    dst = Path(legacy_objects_json).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(objects_path, dst)


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


def read_total_score(evaluation_path: Path) -> float:
    data = load_json(evaluation_path)
    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict):
        raise RuntimeError(f"В evaluation.json нет aggregate: {evaluation_path}")
    score = aggregate.get("total_score")
    if score is None:
        raise RuntimeError(f"В evaluation.json нет aggregate.total_score: {evaluation_path}")
    return float(score)


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


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
class RunLogger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.log_path = root / "run.log"
        self.jsonl_path = root / "search_summary.jsonl"

    def log(self, msg: str) -> None:
        line = f"[{iso_now()}] {msg}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_json(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.jsonl_path, payload)


# ------------------------------------------------------------
# Config defaults
# ------------------------------------------------------------
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

    if args.ml_device is None:
        args.ml_device = str(get_nested(cfg, "defaults.ml_device", "cpu"))

    if args.diffusion_steps is None:
        args.diffusion_steps = int(get_nested(cfg, "defaults.diffusion_steps", 50))

    if args.ollama_url is None:
        args.ollama_url = str(get_nested(cfg, "defaults.ollama.url", "http://127.0.0.1:11434"))

    if args.ollama_model is None:
        args.ollama_model = str(get_nested(cfg, "defaults.ollama.model", "gpt-oss:20b"))

    if args.ollama_timeout is None:
        args.ollama_timeout = int(get_nested(cfg, "defaults.ollama.timeout", 300))

    if args.ollama_temperature is None:
        args.ollama_temperature = float(get_nested(cfg, "defaults.ollama.temperature", 0.1))

    if args.ollama_max_attempts is None:
        args.ollama_max_attempts = int(get_nested(cfg, "defaults.ollama.max_attempts", 8))

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


def build_runtime_paths(cfg: dict[str, Any], cfg_base_dir: Path) -> dict[str, str]:
    runtime: dict[str, str] = {}

    runtime["CHOOSER_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.chooser"), cfg_base_dir)
    runtime["CUBE_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.cube"), cfg_base_dir)
    runtime["BLENDER_VIS_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.blender_visualize"), cfg_base_dir)
    runtime["ML_PLACER_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.ml_placer"), cfg_base_dir)
    runtime["DIFFUSCENE_REMOTE_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.diffuscene_remote"), cfg_base_dir)
    runtime["OLLAMA_LLM_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.ollama_layout"), cfg_base_dir)
    runtime["NORMALIZE_JSON_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.normalize_scene_format"), cfg_base_dir)
    runtime["M3DLAYOUT_SCRIPT"] = resolve_local_path(
        get_nested(cfg, "local.scripts.m3dlayout", "src/Plasement/run_m3dlayout.py"),
        cfg_base_dir,
    )
    runtime["INFINIGEN_CLEAN_SCRIPT"] = resolve_local_path(
        get_nested(cfg, "local.scripts.infinigen_clean", "src/Plasement/run_infinigen_clean.py"),
        cfg_base_dir,
    )

    evaluate_scene_script = get_nested(cfg, "local.scripts.evaluate_unified_scene", None)
    if evaluate_scene_script is None:
        evaluate_scene_script = "src/tools/evaluate_unified_scene.py"
    runtime["EVALUATE_SCENE_SCRIPT"] = resolve_local_path(evaluate_scene_script, cfg_base_dir)

    model_info_json = get_nested(cfg, "local.data.model_info", None)
    if model_info_json is None:
        model_info_json = "data/sourse/3D-FRONT/3D-FUTURE-model/model_info.json"
    runtime["MODEL_INFO_JSON"] = resolve_local_path(model_info_json, cfg_base_dir)

    runtime["DEFAULT_ROOM_JSON"] = resolve_local_path(must_get(cfg, "local.room.default_json"), cfg_base_dir)

    runtime["LEGACY_OBJECTS_JSON"] = resolve_local_path(must_get(cfg, "local.input.objects_json"), cfg_base_dir)
    runtime["LEGACY_PLACEMENT_JSON"] = resolve_local_path(must_get(cfg, "local.output.placement_json"), cfg_base_dir)

    runtime["TMP_ROOT"] = resolve_local_path(must_get(cfg, "local.output.tmp_root"), cfg_base_dir)
    return runtime


def derive_effective_args_for_placer(args: argparse.Namespace, placer: str) -> argparse.Namespace:
    eff = argparse.Namespace(**vars(args))
    cfg = getattr(args, "_cfg", {})
    cfg_base_dir = getattr(args, "_cfg_base_dir", None)

    if getattr(eff, "remote_conda_env", None) is None:
        if placer in {"m3dlayout_ar", "m3dlayout_diffusion"}:
            cfg_val = get_nested(cfg, "remote.m3dlayout.conda_env", None)
            if cfg_val is not None:
                eff.remote_conda_env = str(cfg_val)
        elif placer == "infinigen_clean":
            cfg_val = get_nested(cfg, "remote.infinigen.conda_env", None)
            if cfg_val is not None:
                eff.remote_conda_env = str(cfg_val)

    if getattr(eff, "infinigen_src", None) is None:
        cfg_val = get_nested(cfg, "local.infinigen.src", None)
        if cfg_val is not None and cfg_base_dir is not None:
            eff.infinigen_src = resolve_local_path(str(cfg_val), cfg_base_dir)

    if getattr(eff, "remote_infinigen_src", None) is None:
        cfg_val = get_nested(cfg, "remote.infinigen.src", None)
        if cfg_val is not None:
            eff.remote_infinigen_src = str(cfg_val)

    return eff


# ------------------------------------------------------------
# External scripts wrappers
# ------------------------------------------------------------
def normalize_json_artifact(
    cfg_runtime: dict[str, str],
    input_path: Path,
    output_path: Path,
    target: str,
    logger: RunLogger,
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
    logger.log("▶ Нормализация JSON: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_normalized_scene_artifact(
    cfg_runtime: dict[str, str],
    room_path: str,
    placement_path: Path,
    output_path: Path,
    logger: RunLogger,
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
    logger.log("▶ Сборка scene.v1: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def evaluate_scene_artifact(
    cfg_runtime: dict[str, str],
    scene_v1_path: Path,
    evaluation_out_path: Path,
    prompt_style: Optional[str],
    generation_time_sec: Optional[float],
    logger: RunLogger,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        cfg_runtime["EVALUATE_SCENE_SCRIPT"],
        "--scene",
        str(scene_v1_path.resolve()),
        "--model-info",
        cfg_runtime["MODEL_INFO_JSON"],
        "--output",
        str(evaluation_out_path.resolve()),
    ]
    if prompt_style:
        cmd += ["--prompt-style", prompt_style]
    if generation_time_sec is not None:
        cmd += ["--generation-time-sec", str(float(generation_time_sec))]

    logger.log("▶ Оценка сцены: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return load_json(evaluation_out_path)


def blender_outputs_for_candidate(
    args: argparse.Namespace,
    candidate_dir: Path,
    placer: str,
    mode: str,
    render_tag: str = "instance",
) -> tuple[Optional[str], Optional[str]]:
    suffix = f"{placer}_{mode}_{render_tag}"

    if args.save_blend:
        p = Path(args.save_blend).expanduser().resolve()
        if p.suffix.lower() == ".blend":
            blend = str(p.with_name(f"{p.stem}_{suffix}.blend"))
        else:
            blend = str(p)
    else:
        blend = str((candidate_dir / f"scene_{suffix}.blend").resolve())

    if args.render:
        p = Path(args.render).expanduser().resolve()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            render = str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))
        else:
            render = str(p)
    else:
        render = str((candidate_dir / f"render_{suffix}.png").resolve())

    return blend, render


# ------------------------------------------------------------
# Chooser / placers / blender
# ------------------------------------------------------------
def run_choose_stage(
    args: argparse.Namespace,
    cfg_runtime: dict[str, Any],
    room_path: str,
    prompt_text: str,
    run_dir: Path,
    seed: int,
    logger: RunLogger,
) -> Path:
    out_objects = run_dir / "objects.json"

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
    ]

    logger.log("▶ Выбор предметов: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_objects


def run_cube_placer(
    cfg_runtime: dict[str, Any],
    room_path: str,
    objects_path: Path,
    mode: str,
    out_path: Path,
    logger: RunLogger,
) -> None:
    sync_objects_to_legacy_input(objects_path, cfg_runtime["LEGACY_OBJECTS_JSON"])
    cube_input = f"{os.path.abspath(room_path)}\n{str(objects_path.resolve())}\n{mode}\n"
    logger.log(f"▶ Cube placer mode={mode}")
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
    placer: str,
    mode: str,
    seed: int,
    out_path: Path,
    logger: RunLogger,
) -> None:
    if not room_path.lower().endswith(".json"):
        raise RuntimeError("ML placer требует room-spec .json")

    ml_model = args.ml_model
    if not ml_model:
        model_from_cfg = get_nested(args._cfg, f"local.ml_models.{placer}", None)
        if model_from_cfg:
            ml_model = resolve_local_path(model_from_cfg, args._cfg_base_dir)

    if not ml_model:
        raise RuntimeError(f"--ml-model обязателен для placer={placer}")

    cmd = [
        sys.executable,
        cfg_runtime["ML_PLACER_SCRIPT"],
        "--backend",
        placer,
        "--model",
        os.path.abspath(ml_model),
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

    if placer == "diffusion":
        cmd += ["--ddim-steps", str(int(args.diffusion_steps))]
    if mode:
        cmd += ["--mode", mode]

    logger.log("▶ ML placer: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_diffuscene_remote_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    mode: str,
    seed: int,
    out_path: Path,
    run_dir: Path,
    logger: RunLogger,
) -> None:
    del seed
    del mode

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

    logger.log("▶ DiffuScene remote placer: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not out_path.is_file():
        raise RuntimeError(f"DiffuScene remote не создал итоговый placement: {out_path}")

    local_mode_artifacts = run_dir / "diffuscene_remote_artifacts"
    if remote_artifacts_dir.is_dir():
        copy_tree_contents(remote_artifacts_dir, local_mode_artifacts)
        logger.log(f"📥 Артефакты DiffuScene -> {local_mode_artifacts}")
    else:
        logger.log(f"⚠️ Папка remote-артефактов не найдена: {remote_artifacts_dir}")


def run_ollama_llm_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    mode: str,
    out_path: Path,
    logger: RunLogger,
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
        mode,
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
    ]

    logger.log("▶ Ollama LLM placer: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_blender_for_candidate(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    candidate_dir: Path,
    placer: str,
    mode: str,
    placement_path: Path,
    logger: RunLogger,
    render_tag: str = "instance",
) -> tuple[Optional[str], Optional[str]]:
    is_room_json = room_path.lower().endswith(".json")

    if is_room_json:
        scene_json = candidate_dir / f"scene_{placer}_{mode}_{render_tag}.json"
        merge_room_spec_and_placements(room_path, str(placement_path.resolve()), str(scene_json.resolve()))
        scene_json_for_blender = scene_json
    else:
        scene_json_for_blender = placement_path

    blend_out, render_out = blender_outputs_for_candidate(args, candidate_dir, placer, mode, render_tag=render_tag)

    cmd = [
        sys.executable,
        cfg_runtime["BLENDER_VIS_SCRIPT"],
        "--json",
        str(scene_json_for_blender.resolve()),
    ]

    if args.blender:
        cmd += ["--blender", args.blender]
    if args.headless:
        cmd.append("--background")

    if blend_out:
        Path(blend_out).resolve().parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--save-blend", str(Path(blend_out).resolve())]
    if render_out:
        Path(render_out).resolve().parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--render", str(Path(render_out).resolve())]

    logger.log("▶ Blender render: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return blend_out, render_out


# ------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------
def run_single_candidate(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    prompt_text: str,
    candidate_dir: Path,
    placer: str,
    mode: str,
    prompt_style: Optional[str],
    logger: RunLogger,
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_start = time.perf_counter()

    chooser_seed = int.from_bytes(secrets.token_bytes(8), "big")
    attempt_seed = int.from_bytes(secrets.token_bytes(8), "big")
    placer_args = derive_effective_args_for_placer(args, placer)
    placer_spec = PLACER_SPECS.get(placer)
    chooser_required = bool(placer_spec.get("requires_object_selection", True)) if placer_spec else True

    manifest = {
        "created_at": iso_now(),
        "room": room_path,
        "prompt": prompt_text,
        "placer": placer,
        "mode": mode,
        "chooser_seed": chooser_seed,
        "attempt_seed": attempt_seed,
        "candidate_dir": str(candidate_dir.resolve()),
    }
    write_json(candidate_dir / "run_manifest.json", manifest)

    objects_path: Optional[Path] = None
    normalized_objects_path: Optional[Path] = None
    t_choose = 0.0
    t_norm_objects = 0.0
    if chooser_required:
        t0 = time.perf_counter()
        objects_path = run_choose_stage(
            args=args,
            cfg_runtime=cfg_runtime,
            room_path=room_path,
            prompt_text=prompt_text,
            run_dir=candidate_dir,
            seed=chooser_seed,
            logger=logger,
        )
        t_choose = time.perf_counter() - t0

        normalized_objects_path = candidate_dir / "objects.v1.json"
        t0 = time.perf_counter()
        normalize_json_artifact(
            cfg_runtime=cfg_runtime,
            input_path=objects_path,
            output_path=normalized_objects_path,
            target="objects",
            logger=logger,
        )
        t_norm_objects = time.perf_counter() - t0

    placement_out = candidate_dir / f"placement_{placer}_{mode}.json"
    normalized_placement_path = candidate_dir / "placement.v1.json"
    normalized_scene_path = candidate_dir / "scene.v1.json"
    evaluation_path = candidate_dir / "evaluation.json"

    # placement
    t0 = time.perf_counter()
    if placer == "cube":
        if objects_path is None:
            raise RuntimeError(f"placer={placer} требует chooser objects")
        run_cube_placer(
            cfg_runtime=cfg_runtime,
            room_path=room_path,
            objects_path=objects_path,
            mode=mode,
            out_path=placement_out,
            logger=logger,
        )
    elif placer == "diffuscene_remote":
        if objects_path is None:
            raise RuntimeError(f"placer={placer} требует chooser objects")
        run_diffuscene_remote_placer(
            cfg_runtime=cfg_runtime,
            args=placer_args,
            room_path=room_path,
            objects_path=objects_path,
            mode=mode,
            seed=attempt_seed,
            out_path=placement_out,
            run_dir=candidate_dir,
            logger=logger,
        )
    elif placer == "ollama_llm":
        if objects_path is None:
            raise RuntimeError(f"placer={placer} требует chooser objects")
        run_ollama_llm_placer(
            cfg_runtime=cfg_runtime,
            args=placer_args,
            room_path=room_path,
            objects_path=objects_path,
            mode=mode,
            out_path=placement_out,
            logger=logger,
        )
    elif placer == "m3dlayout_ar":
        run_m3dlayout_clean(
            cfg_runtime=cfg_runtime,
            args=placer_args,
            room_path=room_path,
            objects_path=objects_path,
            prompt_text=prompt_text,
            seed=attempt_seed,
            out_path=placement_out,
            model_type="autoregressive",
        )
    elif placer == "m3dlayout_diffusion":
        run_m3dlayout_clean(
            cfg_runtime=cfg_runtime,
            args=placer_args,
            room_path=room_path,
            objects_path=objects_path,
            prompt_text=prompt_text,
            seed=attempt_seed,
            out_path=placement_out,
            model_type="diffusion",
        )
    elif placer == "infinigen_clean":
        run_infinigen_clean(
            cfg_runtime=cfg_runtime,
            args=placer_args,
            room_path=room_path,
            objects_path=objects_path,
            seed=attempt_seed,
            out_path=placement_out,
            run_dir=candidate_dir,
        )
    else:
        if objects_path is None:
            raise RuntimeError(f"placer={placer} требует chooser objects")
        run_ml_placer(
            cfg_runtime=cfg_runtime,
            args=placer_args,
            room_path=room_path,
            objects_path=objects_path,
            placer=placer,
            mode=mode,
            seed=attempt_seed,
            out_path=placement_out,
            logger=logger,
        )
    t_place = time.perf_counter() - t0

    # normalize placement
    t0 = time.perf_counter()
    normalize_json_artifact(
        cfg_runtime=cfg_runtime,
        input_path=placement_out,
        output_path=normalized_placement_path,
        target="placement",
        logger=logger,
    )
    t_norm_placement = time.perf_counter() - t0

    # build scene
    t0 = time.perf_counter()
    if room_path.lower().endswith(".json"):
        build_normalized_scene_artifact(
            cfg_runtime=cfg_runtime,
            room_path=room_path,
            placement_path=placement_out,
            output_path=normalized_scene_path,
            logger=logger,
        )
    else:
        raise RuntimeError("Оценка качества сейчас поддерживается только для room-spec .json, а не для .glb")
    t_scene = time.perf_counter() - t0

    # evaluate
    generation_time = time.perf_counter() - candidate_start
    t0 = time.perf_counter()
    evaluation = evaluate_scene_artifact(
        cfg_runtime=cfg_runtime,
        scene_v1_path=normalized_scene_path,
        evaluation_out_path=evaluation_path,
        prompt_style=prompt_style,
        generation_time_sec=generation_time,
        logger=logger,
    )
    t_evaluate = time.perf_counter() - t0
    total_score = read_total_score(evaluation_path)

    # render instance
    blend_instance = None
    render_instance = None
    t_blender_instance = 0.0
    if not args.skip_blender:
        t0 = time.perf_counter()
        blend_instance, render_instance = run_blender_for_candidate(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=room_path,
            candidate_dir=candidate_dir,
            placer=placer,
            mode=mode,
            placement_path=placement_out,
            logger=logger,
            render_tag="instance",
        )
        t_blender_instance = time.perf_counter() - t0

    candidate_total_time = time.perf_counter() - candidate_start

    timing = {
        "choose_sec": t_choose,
        "normalize_objects_sec": t_norm_objects,
        "placement_sec": t_place,
        "normalize_placement_sec": t_norm_placement,
        "build_scene_sec": t_scene,
        "evaluate_sec": t_evaluate,
        "blender_instance_sec": t_blender_instance,
        "generation_sec": generation_time,
        "total_sec": candidate_total_time,
    }

    manifest = load_json(candidate_dir / "run_manifest.json")
    manifest.update({
        "objects_legacy": str(objects_path.resolve()) if objects_path else None,
        "objects_v1": str(normalized_objects_path.resolve()) if normalized_objects_path else None,
        "placement_legacy": str(placement_out.resolve()),
        "placement_v1": str(normalized_placement_path.resolve()),
        "scene_v1": str(normalized_scene_path.resolve()),
        "evaluation": str(evaluation_path.resolve()),
        "total_score": total_score,
        "timing": timing,
        "blender_instance": {
            "blend_path": str(Path(blend_instance).resolve()) if blend_instance else None,
            "render_path": str(Path(render_instance).resolve()) if render_instance else None,
        },
    })
    write_json(candidate_dir / "run_manifest.json", manifest)

    return {
        "candidate_dir": candidate_dir,
        "objects_path": objects_path,
        "objects_v1_path": normalized_objects_path,
        "placement_legacy_path": placement_out,
        "placement_v1_path": normalized_placement_path,
        "scene_v1_path": normalized_scene_path,
        "evaluation_path": evaluation_path,
        "evaluation": evaluation,
        "total_score": total_score,
        "placer": placer,
        "mode": mode,
        "timing": timing,
        "blender_instance": {
            "blend_path": blend_instance,
            "render_path": render_instance,
        },
    }


# ------------------------------------------------------------
# Stats / table
# ------------------------------------------------------------
def _fmt_num(x: Optional[float], ndigits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{ndigits}f}"


def print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print()
    print(title)
    if not rows:
        print("(нет данных)")
        return

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    def render_row(vals: list[Any]) -> str:
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals))

    print(render_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(render_row(row))


def collect_final_statistics(search_summary: dict[str, Any]) -> dict[str, Any]:
    all_candidates: list[dict[str, Any]] = []
    accepted_thresholds: list[dict[str, Any]] = []

    for res in search_summary.get("results", []):
        tried = res.get("tried_candidates", [])
        for item in tried:
            if item.get("event") == "candidate_finished":
                all_candidates.append(item)
        if res.get("accepted"):
            accepted_thresholds.append(res)

    scores = [float(x["score"]) for x in all_candidates if x.get("score") is not None]
    times_total = [float(x["timing"]["total_sec"]) for x in all_candidates if x.get("timing")]
    times_generation = [float(x["timing"]["generation_sec"]) for x in all_candidates if x.get("timing")]

    by_stage: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for x in all_candidates:
        by_stage[(x["placer"], x["mode"])].append(x)

    stage_stats = []
    for (placer, mode), arr in sorted(by_stage.items()):
        arr_scores = [float(x["score"]) for x in arr if x.get("score") is not None]
        arr_total_times = [float(x["timing"]["total_sec"]) for x in arr if x.get("timing")]
        arr_gen_times = [float(x["timing"]["generation_sec"]) for x in arr if x.get("timing")]
        stage_stats.append({
            "placer": placer,
            "mode": mode,
            "runs": len(arr),
            "avg_score": mean(arr_scores) if arr_scores else None,
            "best_score": max(arr_scores) if arr_scores else None,
            "avg_total_sec": mean(arr_total_times) if arr_total_times else None,
            "avg_generation_sec": mean(arr_gen_times) if arr_gen_times else None,
        })

    best_candidate = max(all_candidates, key=lambda x: float(x["score"])) if all_candidates else None
    worst_candidate = min(all_candidates, key=lambda x: float(x["score"])) if all_candidates else None

    return {
        "total_candidates": len(all_candidates),
        "accepted_thresholds_count": len(accepted_thresholds),
        "avg_score": mean(scores) if scores else None,
        "best_score": max(scores) if scores else None,
        "worst_score": min(scores) if scores else None,
        "avg_total_sec": mean(times_total) if times_total else None,
        "avg_generation_sec": mean(times_generation) if times_generation else None,
        "best_candidate": best_candidate,
        "worst_candidate": worst_candidate,
        "stage_stats": stage_stats,
    }


def print_final_console_report(search_summary: dict[str, Any]) -> None:
    stats = collect_final_statistics(search_summary)

    overall_rows = [[
        stats["total_candidates"],
        stats["accepted_thresholds_count"],
        _fmt_num(stats["avg_score"]),
        _fmt_num(stats["best_score"]),
        _fmt_num(stats["worst_score"]),
        _fmt_num(stats["avg_generation_sec"]),
        _fmt_num(stats["avg_total_sec"]),
    ]]
    print_table(
        "ИТОГОВАЯ СТАТИСТИКА ПО ВСЕМУ ЗАПУСКУ",
        [
            "Всего кандидатов",
            "Пройдено порогов",
            "Средний score",
            "Лучший score",
            "Худший score",
            "Среднее generation_sec",
            "Среднее total_sec",
        ],
        overall_rows,
    )

    threshold_rows = []
    for res in search_summary.get("results", []):
        threshold_rows.append([
            res.get("threshold"),
            "yes" if res.get("accepted") else "no",
            res.get("num_tried"),
            res.get("cycles_used"),
            _fmt_num(res.get("accepted_score")),
            _fmt_num(res.get("best_score")),
            _fmt_num(res.get("threshold_total_sec")),
        ])
    print_table(
        "СТАТИСТИКА ПО ПОРОГАМ",
        [
            "Threshold",
            "Accepted",
            "Кандидатов",
            "Циклов",
            "Accepted score",
            "Best score",
            "Threshold sec",
        ],
        threshold_rows,
    )

    stage_rows = []
    for st in stats["stage_stats"]:
        stage_rows.append([
            st["placer"],
            st["mode"],
            st["runs"],
            _fmt_num(st["avg_score"]),
            _fmt_num(st["best_score"]),
            _fmt_num(st["avg_generation_sec"]),
            _fmt_num(st["avg_total_sec"]),
        ])
    print_table(
        "СТАТИСТИКА ПО ГЕНЕРАТОРАМ",
        [
            "Placer",
            "Mode",
            "Запусков",
            "Средний score",
            "Лучший score",
            "Среднее generation_sec",
            "Среднее total_sec",
        ],
        stage_rows,
    )

    if stats["best_candidate"]:
        best = stats["best_candidate"]
        print()
        print("ЛУЧШИЙ КАНДИДАТ")
        print(f"score={_fmt_num(float(best['score']))}")
        print(f"placer={best['placer']} mode={best['mode']}")
        print(f"dir={best['candidate_dir']}")

    if stats["worst_candidate"]:
        worst = stats["worst_candidate"]
        print()
        print("ХУДШИЙ КАНДИДАТ")
        print(f"score={_fmt_num(float(worst['score']))}")
        print(f"placer={worst['placer']} mode={worst['mode']}")
        print(f"dir={worst['candidate_dir']}")


# ------------------------------------------------------------
# Search loop
# ------------------------------------------------------------
def run_quality_search(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    prompt_text: str,
    search_root: Path,
    thresholds: list[float],
    stage_sequence: list[tuple[str, str]],
    prompt_style: Optional[str],
    logger: RunLogger,
) -> None:
    search_start = time.perf_counter()

    summary_path = search_root / "search_summary.json"
    search_summary: dict[str, Any] = {
        "created_at": iso_now(),
        "room": room_path,
        "prompt": prompt_text,
        "thresholds": thresholds,
        "stage_sequence": [{"placer": p, "mode": m} for p, m in stage_sequence],
        "search_root": str(search_root.resolve()),
        "results": [],
    }
    write_json(summary_path, search_summary)

    global_candidate_counter = 0

    for threshold in thresholds:
        threshold_start = time.perf_counter()
        threshold_dir = search_root / quality_label(threshold)
        threshold_dir.mkdir(parents=True, exist_ok=True)

        logger.log(f"==============================")
        logger.log(f"🎯 ЦЕЛЕВОЕ КАЧЕСТВО >= {threshold:.2f}")
        logger.log(f"📁 threshold_dir: {threshold_dir}")
        logger.log(f"==============================")

        accepted_result: Optional[dict[str, Any]] = None
        best_result: Optional[dict[str, Any]] = None
        tried_candidates: list[dict[str, Any]] = []

        cycle_index = 0
        while accepted_result is None and cycle_index < int(args.max_candidates_per_threshold):
            cycle_index += 1
            logger.log(f"########## ЦИКЛ {cycle_index} ДЛЯ ПОРОГА {threshold:.2f} ##########")

            for placer, mode in stage_sequence:
                global_candidate_counter += 1
                candidate_dir = threshold_dir / f"candidate_{global_candidate_counter:04d}_{placer}_{mode}"

                try:
                    result = run_single_candidate(
                        cfg_runtime=cfg_runtime,
                        args=args,
                        room_path=room_path,
                        prompt_text=prompt_text,
                        candidate_dir=candidate_dir,
                        placer=placer,
                        mode=mode,
                        prompt_style=prompt_style,
                        logger=logger,
                    )

                    score = float(result["total_score"])
                    candidate_meta = {
                        "ts": iso_now(),
                        "event": "candidate_finished",
                        "threshold": threshold,
                        "candidate_dir": str(candidate_dir.resolve()),
                        "placer": placer,
                        "mode": mode,
                        "score": score,
                        "evaluation_path": str(result["evaluation_path"].resolve()),
                        "scene_v1_path": str(result["scene_v1_path"].resolve()),
                        "timing": result["timing"],
                        "blender_instance": result["blender_instance"],
                    }
                    tried_candidates.append(candidate_meta)
                    logger.log_json(candidate_meta)

                    if best_result is None or score > float(best_result["total_score"]):
                        best_result = result
                        write_json(
                            threshold_dir / "best_so_far.json",
                            {
                                "threshold": threshold,
                                "score": score,
                                "placer": placer,
                                "mode": mode,
                                "candidate_dir": str(candidate_dir.resolve()),
                                "evaluation_path": str(result["evaluation_path"].resolve()),
                                "scene_v1_path": str(result["scene_v1_path"].resolve()),
                                "timing": result["timing"],
                            },
                        )

                    logger.log(f"📊 score={score:.6f} | threshold={threshold:.2f} | placer={placer} | mode={mode}")

                    if score >= threshold:
                        accepted_result = result

                        accepted_dir = threshold_dir / "accepted_target"
                        accepted_dir.mkdir(parents=True, exist_ok=True)

                        accepted_blend = None
                        accepted_render = None
                        accepted_blender_sec = 0.0
                        if not args.skip_blender:
                            t0 = time.perf_counter()
                            accepted_blend, accepted_render = run_blender_for_candidate(
                                cfg_runtime=cfg_runtime,
                                args=args,
                                room_path=room_path,
                                candidate_dir=accepted_dir,
                                placer=placer,
                                mode=mode,
                                placement_path=result["placement_legacy_path"],
                                logger=logger,
                                render_tag=f"accepted_{quality_label(threshold)}",
                            )
                            accepted_blender_sec = time.perf_counter() - t0

                        accepted_payload = {
                            "threshold": threshold,
                            "score": score,
                            "placer": placer,
                            "mode": mode,
                            "candidate_dir": str(candidate_dir.resolve()),
                            "evaluation_path": str(result["evaluation_path"].resolve()),
                            "scene_v1_path": str(result["scene_v1_path"].resolve()),
                            "timing": result["timing"],
                            "accepted_target_render": {
                                "blend_path": str(Path(accepted_blend).resolve()) if accepted_blend else None,
                                "render_path": str(Path(accepted_render).resolve()) if accepted_render else None,
                                "blender_sec": accepted_blender_sec,
                            },
                        }
                        write_json(threshold_dir / "accepted.json", accepted_payload)
                        logger.log_json({
                            "ts": iso_now(),
                            "event": "threshold_accepted",
                            **accepted_payload,
                        })

                        logger.log(f"✅ ПОРОГ {threshold:.2f} ДОСТИГНУТ: score={score:.6f}")
                        break

                except subprocess.CalledProcessError as e:
                    err_payload = {
                        "ts": iso_now(),
                        "event": "candidate_error",
                        "threshold": threshold,
                        "candidate_dir": str(candidate_dir.resolve()),
                        "placer": placer,
                        "mode": mode,
                        "error_type": "CalledProcessError",
                        "returncode": e.returncode,
                    }
                    write_json(candidate_dir / "error.json", err_payload)
                    tried_candidates.append(err_payload)
                    logger.log_json(err_payload)
                    logger.log(f"⚠️ subprocess error | placer={placer} | mode={mode} | returncode={e.returncode}")

                except Exception as e:
                    err_payload = {
                        "ts": iso_now(),
                        "event": "candidate_error",
                        "threshold": threshold,
                        "candidate_dir": str(candidate_dir.resolve()),
                        "placer": placer,
                        "mode": mode,
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                    write_json(candidate_dir / "error.json", err_payload)
                    tried_candidates.append(err_payload)
                    logger.log_json(err_payload)
                    logger.log(f"❌ error | placer={placer} | mode={mode} | {e}")

            if accepted_result is not None:
                break

        threshold_total_sec = time.perf_counter() - threshold_start

        threshold_result = {
            "threshold": threshold,
            "accepted": accepted_result is not None,
            "accepted_candidate_dir": str(accepted_result["candidate_dir"].resolve()) if accepted_result else None,
            "accepted_score": float(accepted_result["total_score"]) if accepted_result else None,
            "best_score": float(best_result["total_score"]) if best_result else None,
            "best_candidate_dir": str(best_result["candidate_dir"].resolve()) if best_result else None,
            "num_tried": len(tried_candidates),
            "cycles_used": cycle_index,
            "threshold_total_sec": threshold_total_sec,
            "tried_candidates": tried_candidates,
        }
        search_summary["results"].append(threshold_result)
        search_summary["total_elapsed_sec"] = time.perf_counter() - search_start
        write_json(summary_path, search_summary)

        logger.log_json({
            "ts": iso_now(),
            "event": "threshold_finished",
            **threshold_result,
        })

        if accepted_result is None:
            raise RuntimeError(
                f"Не удалось достичь качества >= {threshold:.2f} "
                f"за {args.max_candidates_per_threshold} циклов stage-sequence"
            )

    search_summary["finished_at"] = iso_now()
    search_summary["total_elapsed_sec"] = time.perf_counter() - search_start
    write_json(summary_path, search_summary)
    logger.log("\n✅ ВСЕ ПОРОГИ КАЧЕСТВА ДОСТИГНУТЫ")
    print_final_console_report(search_summary)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Поиск сцен по quality thresholds: random -> relaxed -> llm с подробной статистикой"
    )

    p.add_argument("--paths-config", default=DEFAULT_PATHS_CONFIG, help="YAML-файл со всеми путями проекта")

    p.add_argument(
        "items",
        nargs="*",
        help="Опциональный список предметов. Если нет --prompt, будет собран текстовый prompt из items.",
    )
    p.add_argument("--prompt", default=None, help="Текстовый prompt для генерации набора предметов")
    p.add_argument("--prompt-file", default=None, help="Файл с prompt")
    p.add_argument("--prompt-style", default=None, help="Стиль из промпта для evaluator, например Modern")

    p.add_argument("--room", default="__USE_CFG_DEFAULT__", help="Путь комнаты (.json room-spec или .glb)")
    p.add_argument("--prepared-info", default=None)
    p.add_argument("--future-root", default=None)

    p.add_argument("--run-dir", default=None, help="Корневая папка quality-search")
    p.add_argument("--keep-tmp", dest="keep_tmp", action="store_true", default=True)
    p.add_argument("--no-keep-tmp", dest="keep_tmp", action="store_false")

    p.add_argument("--blender", default=None, help="Путь к бинарю Blender")
    p.add_argument("--headless", action="store_true", help="Запуск Blender без GUI")
    p.add_argument("--no-import-glb", action="store_true", help="Compat flag, ignored by current Blender scene builder")
    p.add_argument("--save-blend", default=None, help="Базовый путь для .blend")
    p.add_argument("--render", default=None, help="Базовый путь для render")
    p.add_argument("--skip-blender", action="store_true", help="Не запускать Blender")

    p.add_argument(
        "--max-candidates-per-threshold",
        type=int,
        default=20,
        help="Сколько циклов stage-sequence максимум разрешено на один порог качества",
    )

    p.add_argument("--ml-model", default=None)
    p.add_argument("--ml-device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--diffusion-steps", type=int, default=None)

    p.add_argument("--remote-runner", default=None)
    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=None)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)
    p.add_argument("--remote-conda-env", default=None)
    p.add_argument("--infinigen-src", default=None)
    p.add_argument("--remote-infinigen-src", default=None)

    p.add_argument("--ollama-url", default=None)
    p.add_argument("--ollama-model", default=None)
    p.add_argument("--ollama-timeout", type=int, default=None)
    p.add_argument("--ollama-temperature", type=float, default=None)
    p.add_argument("--ollama-max-attempts", type=int, default=None)

    p.add_argument(
        "--quality-thresholds",
        default="0.6,0.7,0.8,0.9,0.95",
        help="Пороги качества через запятую",
    )
    p.add_argument(
        "--stage-sequence",
        default="cube:random,cube:relaxed,ollama_llm:llm",
        help="Последовательность генераторов через запятую, формат placer:mode",
    )

    return p


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    cfg_path = Path(args.paths_config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    cfg_base_dir = project_root_from_config(cfg, cfg_path)

    args._cfg = cfg
    args._cfg_base_dir = cfg_base_dir

    apply_config_defaults(args, cfg, cfg_base_dir)
    cfg_runtime = build_runtime_paths(cfg, cfg_base_dir)

    room_path = os.path.abspath((args.room or cfg_runtime["DEFAULT_ROOM_JSON"]).strip())
    prompt_text = read_prompt_from_args(args)
    thresholds = parse_quality_thresholds(args.quality_thresholds)
    stage_sequence = parse_stage_sequence(args.stage_sequence)

    search_root, explicit_run_dir = make_search_run_dir(cfg_runtime["TMP_ROOT"], args.run_dir)
    logger = RunLogger(search_root)

    logger.log(f"🧭 paths-config: {cfg_path}")
    logger.log(f"📁 search_root: {search_root}")
    logger.log(f"🎯 thresholds: {thresholds}")
    logger.log(f"🔁 stage_sequence: {stage_sequence}")

    created_dirs: list[Path] = [search_root]

    try:
        run_quality_search(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path=room_path,
            prompt_text=prompt_text,
            search_root=search_root,
            thresholds=thresholds,
            stage_sequence=stage_sequence,
            prompt_style=args.prompt_style,
            logger=logger,
        )
    finally:
        if not args.keep_tmp and not explicit_run_dir:
            for p in created_dirs:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"🗑 Удалён search_root: {p}")


if __name__ == "__main__":
    main()
