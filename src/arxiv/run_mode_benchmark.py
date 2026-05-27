#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/run_mode_benchmark.py

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import yaml

from pipeline_config import PLACER_SPECS
from pipeline_runners import run_infinigen_clean, run_m3dlayout_clean


# ============================================================
# Конфиг
# ============================================================
DEFAULT_PATHS_CONFIG = "config/paths.yaml"

# Пользовательские режимы бенчмарка -> (placer, mode)
MODE_ALIASES: dict[str, tuple[str, str]] = {
    "random": ("cube", "random"),
    "relaxed": ("cube", "relaxed"),
    "llm": ("ollama_llm", "llm"),
    "m3dlayout_ar": ("m3dlayout_ar", "m3dlayout_ar"),
    "m3dlayout_diffusion": ("m3dlayout_diffusion", "m3dlayout_diffusion"),
    "infinigen_clean": ("infinigen_clean", "infinigen_clean"),
}

DEFAULT_MODES = ["random", "relaxed", "llm"]
DEFAULT_COUNT_PER_MODE = 1000


# ============================================================
# YAML / config utils
# ============================================================
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


# ============================================================
# IO / misc
# ============================================================
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# Logging
# ============================================================
class RunLogger:
    def __init__(self, root: Path, quiet_console: bool = False) -> None:
        self.root = root
        self.quiet_console = quiet_console
        self.log_path = root / "run.log"

    def log(self, msg: str, force: bool = False) -> None:
        line = f"[{iso_now()}] {msg}"
        if (not self.quiet_console) or force:
            print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================
# Config defaults
# ============================================================
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


# ============================================================
# Subprocess helper
# ============================================================
def run_logged_subprocess(
    cmd: list[str],
    stdout_path: Path,
    stderr_path: Path,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        return subprocess.run(
            cmd,
            input=input_text,
            text=True,
            stdout=out,
            stderr=err,
            check=True,
        )


# ============================================================
# External scripts wrappers
# ============================================================
def normalize_json_artifact(
    cfg_runtime: dict[str, str],
    input_path: Path,
    output_path: Path,
    target: str,
    logs_dir: Path,
    tag: str,
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
    run_logged_subprocess(
        cmd=cmd,
        stdout_path=logs_dir / f"{tag}.stdout.log",
        stderr_path=logs_dir / f"{tag}.stderr.log",
    )


def build_normalized_scene_artifact(
    cfg_runtime: dict[str, str],
    room_path: str,
    placement_path: Path,
    output_path: Path,
    logs_dir: Path,
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
    run_logged_subprocess(
        cmd=cmd,
        stdout_path=logs_dir / "build_scene.stdout.log",
        stderr_path=logs_dir / "build_scene.stderr.log",
    )


def evaluate_scene_artifact(
    cfg_runtime: dict[str, str],
    scene_v1_path: Path,
    evaluation_out_path: Path,
    prompt_style: Optional[str],
    generation_time_sec: Optional[float],
    logs_dir: Path,
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

    run_logged_subprocess(
        cmd=cmd,
        stdout_path=logs_dir / "evaluate.stdout.log",
        stderr_path=logs_dir / "evaluate.stderr.log",
    )
    return load_json(evaluation_out_path)


# ============================================================
# Chooser / placers
# ============================================================
def run_choose_stage(
    args: argparse.Namespace,
    cfg_runtime: dict[str, Any],
    room_path: str,
    prompt_text: str,
    run_dir: Path,
    seed: int,
    logs_dir: Path,
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

    run_logged_subprocess(
        cmd=cmd,
        stdout_path=logs_dir / "choose.stdout.log",
        stderr_path=logs_dir / "choose.stderr.log",
    )
    return out_objects


def run_cube_placer(
    cfg_runtime: dict[str, Any],
    room_path: str,
    objects_path: Path,
    mode: str,
    out_path: Path,
    logs_dir: Path,
) -> None:
    sync_objects_to_legacy_input(objects_path, cfg_runtime["LEGACY_OBJECTS_JSON"])

    cube_input = f"{os.path.abspath(room_path)}\n{str(objects_path.resolve())}\n{mode}\n"
    cmd = [sys.executable, cfg_runtime["CUBE_SCRIPT"]]

    run_logged_subprocess(
        cmd=cmd,
        input_text=cube_input,
        stdout_path=logs_dir / "placer.stdout.log",
        stderr_path=logs_dir / "placer.stderr.log",
    )

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
    logs_dir: Path,
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

    run_logged_subprocess(
        cmd=cmd,
        stdout_path=logs_dir / "placer.stdout.log",
        stderr_path=logs_dir / "placer.stderr.log",
    )


def run_diffuscene_remote_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    mode: str,
    seed: int,
    out_path: Path,
    run_dir: Path,
    logs_dir: Path,
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

    run_logged_subprocess(
        cmd=cmd,
        stdout_path=logs_dir / "placer.stdout.log",
        stderr_path=logs_dir / "placer.stderr.log",
    )

    if not out_path.is_file():
        raise RuntimeError(f"DiffuScene remote не создал итоговый placement: {out_path}")

    local_mode_artifacts = run_dir / "diffuscene_remote_artifacts"
    if remote_artifacts_dir.is_dir():
        local_mode_artifacts.mkdir(parents=True, exist_ok=True)
        for item in remote_artifacts_dir.iterdir():
            target = local_mode_artifacts / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)


def run_ollama_llm_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    mode: str,
    out_path: Path,
    logs_dir: Path,
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

    run_logged_subprocess(
        cmd=cmd,
        stdout_path=logs_dir / "placer.stdout.log",
        stderr_path=logs_dir / "placer.stderr.log",
    )


# ============================================================
# Flatten evaluation for CSV
# ============================================================
def flatten_for_csv(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten_for_csv(v, key))
            elif isinstance(v, list):
                out[key] = json.dumps(v, ensure_ascii=False)
            else:
                out[key] = v
        return out

    out[prefix or "value"] = obj
    return out


def rebuild_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> None:
    rows = read_jsonl(jsonl_path)
    if not rows:
        return

    flat_rows = [flatten_for_csv(row) for row in rows]
    fieldnames: list[str] = []
    seen = set()

    priority = [
        "ts",
        "candidate_idx",
        "mode_label",
        "placer",
        "mode",
        "status",
        "score",
        "error_stage",
        "error_type",
        "error_message",
        "timing.generation_sec",
        "timing.total_sec",
        "paths.candidate_dir",
        "paths.scene_v1_path",
        "paths.evaluation_path",
    ]

    for key in priority:
        for row in flat_rows:
            if key in row and key not in seen:
                fieldnames.append(key)
                seen.add(key)

    for row in flat_rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


# ============================================================
# Summary / distributions
# ============================================================
def percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    p = max(0.0, min(1.0, p))
    pos = p * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def histogram_counts(values: list[float], bins: int, vmin: float, vmax: float) -> list[dict[str, Any]]:
    if bins <= 0:
        return []
    if vmax <= vmin:
        vmax = vmin + 1.0
    step = (vmax - vmin) / bins
    counts = [0] * bins

    for x in values:
        if x < vmin:
            idx = 0
        elif x >= vmax:
            idx = bins - 1
        else:
            idx = int((x - vmin) / step)
            idx = max(0, min(bins - 1, idx))
        counts[idx] += 1

    out = []
    for i in range(bins):
        a = vmin + i * step
        b = a + step
        out.append({
            "bin_index": i,
            "from": a,
            "to": b,
            "count": counts[i],
        })
    return out


def build_mode_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [r for r in records if r.get("event") == "candidate_finished"]
    errors = [r for r in records if r.get("event") == "candidate_error"]

    scores = sorted([float(r["score"]) for r in finished if r.get("score") is not None])
    gen_times = sorted([float(r["timing"]["generation_sec"]) for r in finished if r.get("timing")])
    total_times = sorted([float(r["timing"]["total_sec"]) for r in finished if r.get("timing")])

    best = max(finished, key=lambda x: float(x["score"])) if finished else None

    return {
        "updated_at": iso_now(),
        "num_records": len(records),
        "num_finished": len(finished),
        "num_errors": len(errors),
        "success_rate": (len(finished) / len(records)) if records else 0.0,
        "score": {
            "mean": mean(scores) if scores else None,
            "min": min(scores) if scores else None,
            "p05": percentile(scores, 0.05),
            "p25": percentile(scores, 0.25),
            "p50": percentile(scores, 0.50),
            "p75": percentile(scores, 0.75),
            "p90": percentile(scores, 0.90),
            "p95": percentile(scores, 0.95),
            "p99": percentile(scores, 0.99),
            "max": max(scores) if scores else None,
            "histogram_10": histogram_counts(scores, bins=10, vmin=0.0, vmax=1.0),
        },
        "generation_time_sec": {
            "mean": mean(gen_times) if gen_times else None,
            "min": min(gen_times) if gen_times else None,
            "p50": percentile(gen_times, 0.50),
            "p90": percentile(gen_times, 0.90),
            "p95": percentile(gen_times, 0.95),
            "max": max(gen_times) if gen_times else None,
        },
        "total_time_sec": {
            "mean": mean(total_times) if total_times else None,
            "min": min(total_times) if total_times else None,
            "p50": percentile(total_times, 0.50),
            "p90": percentile(total_times, 0.90),
            "p95": percentile(total_times, 0.95),
            "max": max(total_times) if total_times else None,
        },
        "best_candidate": best,
    }


def print_mode_summary_console(mode_label: str, summary: dict[str, Any]) -> None:
    score = summary.get("score", {})
    gtime = summary.get("generation_time_sec", {})
    print()
    print(f"=== MODE {mode_label.upper()} ===")
    print(f"finished={summary.get('num_finished')} errors={summary.get('num_errors')} success_rate={summary.get('success_rate', 0.0):.4f}")
    print(
        "score: "
        f"mean={score.get('mean')!s} "
        f"p50={score.get('p50')!s} "
        f"p90={score.get('p90')!s} "
        f"p95={score.get('p95')!s} "
        f"max={score.get('max')!s}"
    )
    print(
        "generation_sec: "
        f"mean={gtime.get('mean')!s} "
        f"p50={gtime.get('p50')!s} "
        f"p90={gtime.get('p90')!s} "
        f"max={gtime.get('max')!s}"
    )
    best = summary.get("best_candidate")
    if isinstance(best, dict):
        print(f"best_score={best.get('score')} best_dir={best.get('paths', {}).get('candidate_dir')}")


# ============================================================
# Best candidate links
# ============================================================
def _remove_path_if_exists(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def save_best_links(mode_dir: Path, candidate_record: dict[str, Any]) -> None:
    best_info_path = mode_dir / "best_candidate.json"
    write_json(best_info_path, candidate_record)

    scene_src = Path(candidate_record["paths"]["scene_v1_path"]).resolve()
    eval_src = Path(candidate_record["paths"]["evaluation_path"]).resolve()
    manifest_src = Path(candidate_record["paths"]["manifest_path"]).resolve()

    link_targets = {
        mode_dir / "best_scene.v1.json": scene_src,
        mode_dir / "best_evaluation.json": eval_src,
        mode_dir / "best_run_manifest.json": manifest_src,
    }

    for link_path, src_path in link_targets.items():
        _remove_path_if_exists(link_path)
        try:
            link_path.symlink_to(src_path)
        except Exception:
            shutil.copy2(src_path, link_path)


# ============================================================
# Candidate generation
# ============================================================
def run_single_candidate(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    prompt_text: str,
    candidate_dir: Path,
    placer: str,
    mode: str,
    mode_label: str,
    prompt_style: Optional[str],
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ensure_dir(candidate_dir / "logs")
    candidate_start = time.perf_counter()

    chooser_seed = int.from_bytes(secrets.token_bytes(8), "big")
    attempt_seed = int.from_bytes(secrets.token_bytes(8), "big")
    placer_args = derive_effective_args_for_placer(args, placer)
    placer_spec = PLACER_SPECS.get(placer)
    chooser_required = bool(placer_spec.get("requires_object_selection", True)) if placer_spec else True

    manifest_path = candidate_dir / "run_manifest.json"
    manifest = {
        "created_at": iso_now(),
        "room": room_path,
        "prompt": prompt_text,
        "mode_label": mode_label,
        "placer": placer,
        "mode": mode,
        "chooser_seed": chooser_seed,
        "attempt_seed": attempt_seed,
        "candidate_dir": str(candidate_dir.resolve()),
    }
    write_json(manifest_path, manifest)

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
            logs_dir=logs_dir,
        )
        t_choose = time.perf_counter() - t0

        normalized_objects_path = candidate_dir / "objects.v1.json"
        t0 = time.perf_counter()
        normalize_json_artifact(
            cfg_runtime=cfg_runtime,
            input_path=objects_path,
            output_path=normalized_objects_path,
            target="objects",
            logs_dir=logs_dir,
            tag="normalize_objects",
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
            logs_dir=logs_dir,
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
            logs_dir=logs_dir,
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
            logs_dir=logs_dir,
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
            logs_dir=logs_dir,
        )
    t_place = time.perf_counter() - t0

    # normalize placement
    t0 = time.perf_counter()
    normalize_json_artifact(
        cfg_runtime=cfg_runtime,
        input_path=placement_out,
        output_path=normalized_placement_path,
        target="placement",
        logs_dir=logs_dir,
        tag="normalize_placement",
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
            logs_dir=logs_dir,
        )
    else:
        raise RuntimeError("Бенчмарк поддерживает только room-spec .json")
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
        logs_dir=logs_dir,
    )
    t_evaluate = time.perf_counter() - t0
    total_score = read_total_score(evaluation_path)

    candidate_total_time = time.perf_counter() - candidate_start

    timing = {
        "choose_sec": t_choose,
        "normalize_objects_sec": t_norm_objects,
        "placement_sec": t_place,
        "normalize_placement_sec": t_norm_placement,
        "build_scene_sec": t_scene,
        "evaluate_sec": t_evaluate,
        "generation_sec": generation_time,
        "total_sec": candidate_total_time,
    }

    manifest = load_json(manifest_path)
    manifest.update({
        "objects_legacy": str(objects_path.resolve()) if objects_path else None,
        "objects_v1": str(normalized_objects_path.resolve()) if normalized_objects_path else None,
        "placement_legacy": str(placement_out.resolve()),
        "placement_v1": str(normalized_placement_path.resolve()),
        "scene_v1": str(normalized_scene_path.resolve()),
        "evaluation": str(evaluation_path.resolve()),
        "total_score": total_score,
        "timing": timing,
        "logs_dir": str(logs_dir.resolve()),
    })
    write_json(manifest_path, manifest)

    return {
        "candidate_dir": candidate_dir,
        "manifest_path": manifest_path,
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
        "mode_label": mode_label,
        "timing": timing,
        "logs_dir": logs_dir,
    }


# ============================================================
# Benchmark loop
# ============================================================
def candidate_record_success(candidate_idx: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": iso_now(),
        "event": "candidate_finished",
        "candidate_idx": candidate_idx,
        "mode_label": result["mode_label"],
        "placer": result["placer"],
        "mode": result["mode"],
        "status": "ok",
        "score": float(result["total_score"]),
        "timing": result["timing"],
        "paths": {
            "candidate_dir": str(result["candidate_dir"].resolve()),
            "manifest_path": str(result["manifest_path"].resolve()),
            "scene_v1_path": str(result["scene_v1_path"].resolve()),
            "evaluation_path": str(result["evaluation_path"].resolve()),
            "placement_v1_path": str(result["placement_v1_path"].resolve()),
            "logs_dir": str(result["logs_dir"].resolve()),
        },
        "evaluation": result["evaluation"],
    }


def candidate_record_error(
    candidate_idx: int,
    mode_label: str,
    placer: str,
    mode: str,
    candidate_dir: Path,
    error_stage: str,
    error_type: str,
    error_message: str,
    returncode: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "ts": iso_now(),
        "event": "candidate_error",
        "candidate_idx": candidate_idx,
        "mode_label": mode_label,
        "placer": placer,
        "mode": mode,
        "status": "error",
        "score": None,
        "error_stage": error_stage,
        "error_type": error_type,
        "error_message": error_message,
        "returncode": returncode,
        "paths": {
            "candidate_dir": str(candidate_dir.resolve()),
        },
    }


def run_mode_benchmark(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    prompt_text: str,
    bench_root: Path,
    mode_label: str,
    placer: str,
    mode: str,
    prompt_style: Optional[str],
    logger: RunLogger,
) -> None:
    mode_dir = ensure_dir(bench_root / mode_label)
    results_jsonl = mode_dir / "results.jsonl"
    results_csv = mode_dir / "results.csv"
    summary_json = mode_dir / "summary.json"
    existing_records = read_jsonl(results_jsonl)
    start_idx = len(existing_records)

    logger.log(f"=== MODE {mode_label.upper()} | placer={placer} mode={mode} ===", force=True)

    best_record: Optional[dict[str, Any]] = None
    for rec in existing_records:
        if rec.get("event") == "candidate_finished":
            if best_record is None or float(rec["score"]) > float(best_record["score"]):
                best_record = rec

    for i in range(start_idx + 1, int(args.count_per_mode) + 1):
        candidate_dir = ensure_dir(mode_dir / f"candidate_{i:04d}_{mode_label}")

        try:
            result = run_single_candidate(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=room_path,
                prompt_text=prompt_text,
                candidate_dir=candidate_dir,
                placer=placer,
                mode=mode,
                mode_label=mode_label,
                prompt_style=prompt_style,
            )
            record = candidate_record_success(i, result)
            append_jsonl(results_jsonl, record)

            if best_record is None or float(record["score"]) > float(best_record["score"]):
                best_record = record
                save_best_links(mode_dir, best_record)

            if (i % int(args.progress_every) == 0) or (i == 1) or (i == int(args.count_per_mode)):
                logger.log(
                    f"[{mode_label}] {i}/{args.count_per_mode} | score={record['score']:.6f} | best={best_record['score']:.6f}",
                    force=True,
                )

        except subprocess.CalledProcessError as e:
            record = candidate_record_error(
                candidate_idx=i,
                mode_label=mode_label,
                placer=placer,
                mode=mode,
                candidate_dir=candidate_dir,
                error_stage="subprocess",
                error_type="CalledProcessError",
                error_message=f"subprocess returncode={e.returncode}",
                returncode=e.returncode,
            )
            append_jsonl(results_jsonl, record)
            logger.log(f"[{mode_label}] {i}/{args.count_per_mode} | subprocess error rc={e.returncode}", force=True)

        except KeyboardInterrupt:
            logger.log(f"[{mode_label}] interrupted by user at run {i}", force=True)
            rebuild_csv_from_jsonl(results_jsonl, results_csv)
            summary = build_mode_summary(read_jsonl(results_jsonl))
            write_json(summary_json, summary)
            raise

        except Exception as e:
            record = candidate_record_error(
                candidate_idx=i,
                mode_label=mode_label,
                placer=placer,
                mode=mode,
                candidate_dir=candidate_dir,
                error_stage="python",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            append_jsonl(results_jsonl, record)
            logger.log(f"[{mode_label}] {i}/{args.count_per_mode} | error={type(e).__name__}: {e}", force=True)

        # CSV и summary обновляем после каждого запуска, чтобы при прерывании всё осталось
        rows_now = read_jsonl(results_jsonl)
        rebuild_csv_from_jsonl(results_jsonl, results_csv)
        summary = build_mode_summary(rows_now)
        write_json(summary_json, summary)

    final_summary = build_mode_summary(read_jsonl(results_jsonl))
    write_json(summary_json, final_summary)
    print_mode_summary_console(mode_label, final_summary)


# ============================================================
# Общая сводка
# ============================================================
def build_global_summary(bench_root: Path, modes: list[str]) -> dict[str, Any]:
    per_mode: dict[str, Any] = {}
    rows = []

    for mode_label in modes:
        summary_path = bench_root / mode_label / "summary.json"
        if not summary_path.is_file():
            continue
        summary = load_json(summary_path)
        per_mode[mode_label] = summary

        score = summary.get("score", {})
        gtime = summary.get("generation_time_sec", {})
        rows.append({
            "mode": mode_label,
            "num_finished": summary.get("num_finished"),
            "num_errors": summary.get("num_errors"),
            "success_rate": summary.get("success_rate"),
            "score_mean": score.get("mean"),
            "score_p50": score.get("p50"),
            "score_p90": score.get("p90"),
            "score_p95": score.get("p95"),
            "score_max": score.get("max"),
            "gen_mean": gtime.get("mean"),
            "gen_p90": gtime.get("p90"),
        })

    return {
        "updated_at": iso_now(),
        "bench_root": str(bench_root.resolve()),
        "modes": modes,
        "rows": rows,
        "per_mode": per_mode,
    }


def save_global_summary(bench_root: Path, modes: list[str]) -> None:
    summary = build_global_summary(bench_root, modes)
    write_json(bench_root / "summary.json", summary)

    csv_path = bench_root / "summary.csv"
    rows = summary["rows"]
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def print_global_summary_console(bench_root: Path, modes: list[str]) -> None:
    summary = build_global_summary(bench_root, modes)
    print()
    print("=== GLOBAL SUMMARY ===")
    for row in summary["rows"]:
        print(
            f"{row['mode']}: "
            f"finished={row['num_finished']} "
            f"errors={row['num_errors']} "
            f"success_rate={row['success_rate']:.4f} "
            f"score_mean={row['score_mean']} "
            f"score_p95={row['score_p95']} "
            f"score_max={row['score_max']}"
        )


# ============================================================
# CLI
# ============================================================
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Тихий бенчмарк режимов random / relaxed / llm. "
            "Для каждого режима делает N независимых запусков, сохраняет scene.v1.json, "
            "evaluation.json, JSONL, CSV, summary.json и ссылку на лучший вариант."
        )
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

    p.add_argument("--room", default="__USE_CFG_DEFAULT__", help="Путь комнаты (.json room-spec)")
    p.add_argument("--prepared-info", default=None)
    p.add_argument("--future-root", default=None)

    p.add_argument("--run-dir", default=None, help="Корневая папка benchmark-run")
    p.add_argument("--keep-tmp", dest="keep_tmp", action="store_true", default=True)
    p.add_argument("--no-keep-tmp", dest="keep_tmp", action="store_false")

    p.add_argument(
        "--modes",
        default="random,relaxed,llm",
        help="Список режимов через запятую: random,relaxed,llm,m3dlayout_ar,m3dlayout_diffusion,infinigen_clean",
    )
    p.add_argument(
        "--count-per-mode",
        type=int,
        default=DEFAULT_COUNT_PER_MODE,
        help="Сколько запусков сделать на каждый режим",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Как часто печатать прогресс по режиму",
    )
    p.add_argument(
        "--quiet-console",
        action="store_true",
        help="Печатать в консоль только основные события и прогресс",
    )

    # Blender в этом бенчмарке намеренно не используем
    p.add_argument("--blender", default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--no-import-glb", action="store_true", help="Compat flag, unused in this benchmark")
    p.add_argument("--save-blend", default=None)
    p.add_argument("--render", default=None)

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

    return p


# ============================================================
# Main
# ============================================================
def parse_mode_labels(raw: str) -> list[str]:
    out = []
    seen = set()
    for part in raw.split(","):
        mode_label = part.strip().lower()
        if not mode_label:
            continue
        if mode_label not in MODE_ALIASES:
            raise ValueError(
                f"Неизвестный mode label: {mode_label}. "
                f"Допустимо: {', '.join(sorted(MODE_ALIASES.keys()))}"
            )
        if mode_label not in seen:
            out.append(mode_label)
            seen.add(mode_label)
    if not out:
        raise ValueError("Список modes пуст")
    return out


def make_benchmark_root(tmp_root: str, run_dir_arg: Optional[str]) -> tuple[Path, bool]:
    if run_dir_arg:
        p = Path(run_dir_arg).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p, True
    run_hash = secrets.token_urlsafe(7).replace("-", "").replace("_", "").lower()
    p = Path(tmp_root).expanduser().resolve() / f"mode_benchmark_{now_stamp()}_{run_hash}"
    p.mkdir(parents=True, exist_ok=True)
    return p, False


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
    room_path_obj = Path(room_path).expanduser().resolve()
    if not room_path_obj.is_file():
        raise FileNotFoundError(f"Файл комнаты не найден: {room_path_obj}")

    if not room_path.lower().endswith(".json"):
        raise RuntimeError("Этот бенчмарк поддерживает только room-spec .json")

    prompt_text = read_prompt_from_args(args)
    mode_labels = parse_mode_labels(args.modes)

    bench_root, explicit_run_dir = make_benchmark_root(cfg_runtime["TMP_ROOT"], args.run_dir)
    logger = RunLogger(bench_root, quiet_console=bool(args.quiet_console))

    manifest = {
        "created_at": iso_now(),
        "paths_config": str(cfg_path),
        "bench_root": str(bench_root.resolve()),
        "room": str(room_path_obj),
        "prompt": prompt_text,
        "prompt_style": args.prompt_style,
        "mode_labels": mode_labels,
        "mode_aliases": {k: {"placer": v[0], "mode": v[1]} for k, v in MODE_ALIASES.items()},
        "count_per_mode": int(args.count_per_mode),
        "progress_every": int(args.progress_every),
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "ollama_timeout": args.ollama_timeout,
        "ollama_temperature": args.ollama_temperature,
        "ollama_max_attempts": args.ollama_max_attempts,
    }
    write_json(bench_root / "benchmark_manifest.json", manifest)

    logger.log(f"🧭 paths-config: {cfg_path}", force=True)
    logger.log(f"📁 bench_root: {bench_root}", force=True)
    logger.log(f"🏠 room: {room_path_obj}", force=True)
    logger.log(f"📝 prompt: {prompt_text}", force=True)
    logger.log(f"🔁 modes: {mode_labels}", force=True)
    logger.log(f"🔢 count_per_mode: {args.count_per_mode}", force=True)

    created_dirs = [bench_root]

    try:
        for mode_label in mode_labels:
            placer, mode = MODE_ALIASES[mode_label]
            run_mode_benchmark(
                cfg_runtime=cfg_runtime,
                args=args,
                room_path=str(room_path_obj),
                prompt_text=prompt_text,
                bench_root=bench_root,
                mode_label=mode_label,
                placer=placer,
                mode=mode,
                prompt_style=args.prompt_style,
                logger=logger,
            )
            save_global_summary(bench_root, mode_labels)

        save_global_summary(bench_root, mode_labels)
        print_global_summary_console(bench_root, mode_labels)

    finally:
        if not args.keep_tmp and not explicit_run_dir:
            for p in created_dirs:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"🗑 Удалён bench_root: {p}")


if __name__ == "__main__":
    main()
