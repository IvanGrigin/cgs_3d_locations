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
        return "Нужно разместить следующие предметы: " + ", ".join(args.items)
    raise RuntimeError("Нужно передать либо positional items, либо --prompt, либо --prompt-file")


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
    runtime["DIFFUSCENE_REMOTE_SCRIPT"] = resolve_local_path(must_get(cfg, "local.scripts.diffuscene_remote"), cfg_base_dir)
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

    print("▶ Запуск Ollama LLM-расстановщика:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def execute_placer(
    cfg_runtime: dict[str, Any],
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    layout_mode: str,
    seed: int,
    out_path: Path,
    run_dir: Path,
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
        run_ollama_llm_placer(cfg_runtime, args, room_path, objects_path, layout_mode, out_path)
        return

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


def parse_csv_set(raw: str) -> set[str]:
    return {x.strip() for x in (raw or "").split(",") if x.strip()}


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
    if artifacts.scene_legacy and artifacts.scene_legacy.is_file():
        return artifacts.scene_legacy
    if artifacts.scene_v1 and artifacts.scene_v1.is_file():
        return artifacts.scene_v1
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

    out_dir = run_dir / "lego_post"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_placement_legacy = out_dir / f"placement_{layout_mode}_lego.json"
    out_placement_v1 = out_dir / f"placement_{layout_mode}_lego.v1.json"
    out_scene_legacy = out_dir / f"scene_{layout_mode}_lego.json"
    out_scene_v1 = out_dir / f"scene_{layout_mode}_lego.v1.json"

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
    ]
    ssh_run(args, " ".join(remote_cmd_parts))

    scp_download(args, remote_out_placement_legacy, out_placement_legacy)
    scp_download(args, remote_out_placement_v1, out_placement_v1)
    scp_download(args, remote_out_scene_legacy, out_scene_legacy)
    scp_download(args, remote_out_scene_v1, out_scene_v1)

    print(f"✅ LEGO-Net refined mode={layout_mode}, room_type={room_type}")

    return PlacementArtifacts(
        placement_legacy=out_placement_legacy,
        placement_v1=out_placement_v1,
        scene_v1=out_scene_v1,
        scene_legacy=out_scene_legacy if out_scene_legacy.is_file() else None,
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
    is_room_json = room_path.lower().endswith(".json")

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
    if args.no_import_glb or is_room_json:
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
    objects_path = run_choose_stage(
        args=args,
        cfg_runtime=cfg_runtime,
        room_path=room_path,
        prompt_text=prompt_text,
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
        "chooser_seed": chooser_seed,
        "placer": args.placer,
        "layout_mode": layout_mode,
        "run_dir": str(run_dir),
        "objects_legacy": str(objects_path.resolve()),
        "objects_v1": str(normalized_objects_path.resolve()),
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

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
        choices=["cube", "forest", "graph_stat", "diffusion", "layout_refiner", "diffuscene_remote", "ollama_llm"],
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
    p.add_argument("--ollama-timeout", type=int, default=None)
    p.add_argument("--ollama-temperature", type=float, default=None)
    p.add_argument("--ollama-max-attempts", type=int, default=None)

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