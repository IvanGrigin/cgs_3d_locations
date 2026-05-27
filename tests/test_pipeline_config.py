import argparse
from pathlib import Path
import pytest

from src.pipeline_config import (
    apply_config_defaults,
    build_runtime_paths,
    get_nested,
    load_yaml,
    make_mode_run_dir,
    must_get,
    normalize_prompt_for_object_choice,
    parse_modes,
    project_root_from_config,
    read_json,
    read_prompt_from_args,
    resolve_local_path,
    write_json,
)


def test_yaml_and_nested_helpers(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("a:\n  b:\n    c: 7\n", encoding="utf-8")
    cfg = load_yaml(cfg_path)
    assert get_nested(cfg, "a.b.c") == 7
    assert get_nested(cfg, "a.missing", default=2) == 2
    assert must_get(cfg, "a.b.c") == 7

    try:
        must_get(cfg, "a.b.d")
    except KeyError:
        assert True
    else:
        assert False, "must_get should raise"


def test_load_yaml_error_paths(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("123", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_yaml(bad_yaml)


def test_resolve_paths_and_room_root(tmp_path: Path):
    absolute = resolve_local_path("nested.txt", tmp_path)
    assert str(absolute).endswith("nested.txt")
    assert resolve_local_path(None, tmp_path) is None
    assert resolve_local_path(str(tmp_path / "abs.txt"), tmp_path) == str((tmp_path / "abs.txt").resolve())

    cfg = {"project": {"root": str(tmp_path / "custom_root")}}
    root = project_root_from_config(cfg, cfg_path=tmp_path / "cfg.yaml")
    assert root == (tmp_path / "custom_root").resolve()
    assert project_root_from_config({}, cfg_path=tmp_path / "config" / "paths.yaml") == tmp_path.resolve()

    payload = tmp_path / "payload.json"
    write_json(payload, {"ok": True})
    assert read_json(payload) == {"ok": True}


def test_read_prompt_from_args_variants(tmp_path: Path):
    args_prompt = argparse.Namespace(prompt="text", prompt_file=None, items=None)
    assert read_prompt_from_args(args_prompt) == "text"

    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("file text", encoding="utf-8")
    args_file = argparse.Namespace(prompt=None, prompt_file=str(prompt_file), items=None)
    assert read_prompt_from_args(args_file) == "file text"

    args_items = argparse.Namespace(prompt=None, prompt_file=None, items=["bed", "wardrobe"])
    result = read_prompt_from_args(args_items)
    assert "Need to place the following objects: bed, wardrobe" in result

    args_empty = argparse.Namespace(prompt=None, prompt_file=None, items=None)
    with pytest.raises(RuntimeError):
        read_prompt_from_args(args_empty)


def test_normalize_prompt_for_object_choice_defaults():
    prompt = "Небольшая спальня, нужен большой шкаф и дверь."
    normalized = normalize_prompt_for_object_choice(prompt, room_path="bedroom_001.json")
    assert "REQUIRED_ITEMS:" in normalized
    assert "- King-size Bed x1" in normalized
    assert "- Wardrobe x1" in normalized


def test_normalize_prompt_for_object_choice_various_rooms():
    assert "ROOM_TYPE: living" in normalize_prompt_for_object_choice(
        "Гостиная с диваном и шкафом", "anything.json"
    )
    assert "ROOM_TYPE: office" in normalize_prompt_for_object_choice(
        "Кабинет, нужен стол и кресло", "office_scene.json"
    )
    assert "ROOM_TYPE: kids" in normalize_prompt_for_object_choice(
        "Детская с кроватью и шкафом", "kids_room.json"
    )
    assert "ROOM_TYPE: bedroom" in normalize_prompt_for_object_choice(
        "Нужен шкаф", "room_bedroom_01.json"
    )
    assert "ROOM_TYPE: living" in normalize_prompt_for_object_choice(
        "Нужен стол, стол и лампа", "living_room.json"
    )
    raw = "minimal empty prompt"
    assert normalize_prompt_for_object_choice(raw, "unknown.json") == raw


def test_apply_config_defaults_and_modes(tmp_path: Path):
    cfg = {
        "project": {"root": str(tmp_path / "project")},
        "local": {
            "room": {"default_json": "rooms/base.json"},
            "data": {
                "prepared_info": "info/data.json",
                "future_root": "future",
            },
            "scripts": {
                "remote_runner_sh": "scripts/run.sh",
            },
            "blender": {"binary": "bin/blender"},
        },
        "defaults": {
            "max_attempts": 12,
            "placer": "cube",
            "ollama": {"url": "http://localhost:11434"},
        },
    }
    import yaml

    cfg_file = tmp_path / "cfg2.yaml"
    cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
    loaded = load_yaml(cfg_file)

    args = argparse.Namespace(
        room="__USE_CFG_DEFAULT__",
        prepared_info=None,
        future_root=None,
        remote_runner=None,
        blender=None,
        max_attempts=None,
        placer=None,
        ml_device=None,
        diffusion_steps=None,
        ollama_url=None,
        ollama_models=None,
        ollama_model=None,
        ollama_timeout=None,
        ollama_temperature=None,
        ollama_max_attempts=None,
        plan_model=None,
        plan_models=None,
        plan_temperature=None,
        plan_think=None,
        llm_think=None,
        critic_model=None,
        critic_models=None,
        critic_temperature=None,
        critic_think=None,
        max_scene_attempts=None,
        remote_host=None,
        remote_port=None,
        remote_user=None,
        remote_key=None,
        remote_conda_env=None,
        infinigen_src=None,
        remote_infinigen_src=None,
        ml_model=None,
        lego_repo=None,
        lego_python=None,
        lego_helper_script=None,
        lego_tmp_root=None,
        lego_checkpoint_bedroom=None,
        lego_checkpoint_livingroom=None,
        lego_modes=None,
        lego_generation_preset=None,
        lego_method=None,
        lego_outer_passes=None,
        lego_num_restarts=None,
        lego_init_pos_noise_std=None,
        lego_init_ang_noise_deg=None,
        lego_init_scene_mode=None,
        modes="",
    )
    apply_config_defaults(args, loaded, Path(cfg_file.parent))
    assert args.max_attempts == 12
    assert args.placer == "cube"
    assert args.room.endswith("rooms/base.json")


def test_make_mode_run_dir_and_parse_modes(tmp_path: Path):
    explicit = make_mode_run_dir(str(tmp_path), "base", str(tmp_path / "explicit"))
    assert explicit[1] is True
    assert explicit[0].name == "explicit"

    random_dir, random_created = make_mode_run_dir(str(tmp_path), "base", None)
    assert random_created is False
    assert random_dir.name.startswith("base_")

    args = argparse.Namespace(placer="cube", modes="infinigen_clean")
    cfg = {"modes_by_placer": {"cube": ["random", "relaxed"]}}
    assert parse_modes(args, cfg) == ["infinigen_clean"]
    args_empty = argparse.Namespace(placer="cube", modes="")
    assert parse_modes(args_empty, cfg) == ["random", "relaxed"]

    args_duplicate = argparse.Namespace(placer="cube", modes="random,random,relaxed")
    assert parse_modes(args_duplicate, cfg) == ["random", "relaxed"]

    args_empty_modes = argparse.Namespace(placer="cube", modes=",,,")
    with pytest.raises(RuntimeError):
        parse_modes(args_empty_modes, {})


def test_apply_config_defaults_extended_paths_and_remote_branches(tmp_path: Path):
    cfg = {
        "project": {"root": str(tmp_path / "project")},
        "local": {
            "room": {"default_json": "rooms/base.json"},
            "data": {
                "prepared_info": "info/data.json",
                "future_root": "future",
            },
            "scripts": {
                "remote_runner_sh": "scripts/run.sh",
                "chooser": "scripts/chooser.py",
                "cube": "scripts/cube.py",
                "blender_visualize": "scripts/blend_vis.py",
                "ml_placer": "scripts/ml.py",
                "layout_refiner_infer": "scripts/refine.py",
                "diffuscene_remote": "scripts/diff.py",
                "ollama_layout": "scripts/ollama.py",
                "normalize_scene_format": "scripts/normalize.py",
                "lego_postprocess": "scripts/lego_post.py",
            },
            "room": {"default_glb": "rooms/base.glb", "default_json": "rooms/base.json"},
            "input": {"objects_json": "objects.json"},
            "output": {"placement_json": "placement.json", "tmp_root": "tmp"},
            "blender": {"binary": "bin/blender"},
            "ml_models": {"m3dlayout_ar": "models/m3d.bin"},
            "infinigen": {"src": "infinigen/src"},
        },
        "defaults": {
            "max_attempts": 12,
            "placer": "m3dlayout_ar",
            "ml_device": "cuda",
            "diffusion_steps": 60,
            "ollama": {
                "url": "http://localhost:11434",
                "models": ["m1", "m2"],
                "model": "default-llm",
                "timeout": 99,
                "temperature": 0.2,
                "max_attempts": 4,
                "max_scene_attempts": 11,
            },
        },
        "remote": {
            "ssh": {"host": "10.0.0.1", "port": 2222, "user": "runner", "key": "keys/rsa"},
            "m3dlayout": {"conda_env": "m3d_env"},
            "infinigen": {"conda_env": "inf_env"},
            "lego_net": {
                "repo_root": "/opt/lego_repo",
                "python": "/opt/conda/bin/python",
                "helper_script": "/opt/lego_helper.py",
                "tmp_root": "/tmp/lego",
                "checkpoint_bedroom": "/check/bd",
                "checkpoint_livingroom": "/check/ld",
            },
        },
    }

    import yaml

    cfg_file = tmp_path / "cfg3.yaml"
    cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
    loaded = load_yaml(cfg_file)

    defaults = {name: None for name in [
        "room",
        "prepared_info",
        "future_root",
        "remote_runner",
        "blender",
        "max_attempts",
        "placer",
        "ml_device",
        "diffusion_steps",
        "ollama_url",
        "ollama_models",
        "ollama_model",
        "ollama_timeout",
        "ollama_temperature",
        "ollama_max_attempts",
        "plan_model",
        "plan_models",
        "plan_temperature",
        "plan_think",
        "llm_think",
        "critic_model",
        "critic_models",
        "critic_temperature",
        "critic_think",
        "max_scene_attempts",
        "remote_host",
        "remote_port",
        "remote_user",
        "remote_key",
        "remote_conda_env",
        "infinigen_src",
        "remote_infinigen_src",
        "ml_model",
        "lego_repo",
        "lego_python",
        "lego_helper_script",
        "lego_tmp_root",
        "lego_checkpoint_bedroom",
        "lego_checkpoint_livingroom",
        "lego_modes",
        "lego_generation_preset",
        "lego_method",
        "lego_outer_passes",
        "lego_num_restarts",
        "lego_init_pos_noise_std",
        "lego_init_ang_noise_deg",
        "lego_init_scene_mode",
    ]}
    defaults["room"] = "__USE_CFG_DEFAULT__"
    defaults["modes"] = ""
    args = argparse.Namespace(**defaults)

    apply_config_defaults(args, loaded, cfg_file.parent)

    assert args.room.endswith("rooms/base.json")
    assert args.ollama_models == ["m1", "m2"]
    assert args.plan_models == ["m1", "m2"]
    assert args.critic_models == ["m1", "m2"]
    assert args.critic_think == "low"
    assert args.remote_host == "10.0.0.1"
    assert args.remote_port == 2222
    assert args.remote_user == "runner"
    assert args.remote_key == str(Path("keys/rsa").expanduser())
    assert args.remote_conda_env == "m3d_env"
    assert args.infinigen_src.endswith("infinigen/src")
    assert args.lego_repo == "/opt/lego_repo"
    assert args.lego_python == "/opt/conda/bin/python"
    assert args.lego_helper_script == "/opt/lego_helper.py"


def test_pipeline_config_remaining_defaults_and_runtime_paths(tmp_path: Path):
    cfg = {
        "local": {
            "scripts": {
                "chooser": "scripts/chooser.py",
                "cube": "scripts/cube.py",
                "blender_visualize": "scripts/blender.py",
                "ml_placer": "scripts/ml.py",
                "layout_refiner_infer": "scripts/refine.py",
                "diffuscene_remote": "scripts/diff.py",
                "ollama_layout": "scripts/ollama.py",
                "normalize_scene_format": "scripts/normalize.py",
                "lego_postprocess": "scripts/lego_post.py",
            },
            "room": {"default_glb": "rooms/base.glb", "default_json": "rooms/base.json"},
            "input": {"objects_json": "input/objects.json"},
            "output": {"placement_json": "out/placement.json", "tmp_root": "tmp"},
            "ml_models": {"cube": "models/cube.pkl"},
        },
        "defaults": {
            "ollama": {
                "models": [" ", ""],
                "model": "",
            },
        },
        "remote": {
            "infinigen": {"conda_env": "inf", "src": "/workspace/infinigen/src"},
        },
    }

    runtime = build_runtime_paths(cfg, tmp_path)
    assert runtime["CHOOSER_SCRIPT"].endswith("scripts/chooser.py")
    assert runtime["M3DLAYOUT_SCRIPT"].endswith("src/Plasement/run_m3dlayout.py")
    assert runtime["INFINIGEN_CLEAN_SCRIPT"].endswith("src/Plasement/run_infinigen_clean.py")
    assert runtime["TMP_ROOT"].endswith("tmp")

    names = [
        "room",
        "prepared_info",
        "future_root",
        "remote_runner",
        "blender",
        "max_attempts",
        "placer",
        "ml_device",
        "diffusion_steps",
        "ollama_url",
        "ollama_models",
        "ollama_model",
        "ollama_timeout",
        "ollama_temperature",
        "ollama_max_attempts",
        "plan_model",
        "plan_models",
        "plan_temperature",
        "plan_think",
        "llm_think",
        "critic_model",
        "critic_models",
        "critic_temperature",
        "critic_think",
        "max_scene_attempts",
        "remote_host",
        "remote_port",
        "remote_user",
        "remote_key",
        "remote_conda_env",
        "infinigen_src",
        "remote_infinigen_src",
        "ml_model",
        "lego_repo",
        "lego_python",
        "lego_helper_script",
        "lego_tmp_root",
        "lego_checkpoint_bedroom",
        "lego_checkpoint_livingroom",
        "lego_modes",
        "lego_generation_preset",
        "lego_method",
        "lego_outer_passes",
        "lego_num_restarts",
        "lego_init_pos_noise_std",
        "lego_init_ang_noise_deg",
        "lego_init_scene_mode",
    ]
    values = {name: None for name in names}
    values["room"] = "explicit_room.json"
    args = argparse.Namespace(**values, modes="")
    args.placer = "infinigen_clean"
    args.ollama_models = []
    args.plan_models = []
    args.critic_models = []

    apply_config_defaults(args, cfg, tmp_path)

    assert args.room == "explicit_room.json"
    assert args.ollama_models == ["gpt-oss:20b"]
    assert args.ollama_model == "gpt-oss:20b"
    assert args.plan_models == ["gpt-oss:20b"]
    assert args.critic_models == ["gpt-oss:20b"]
    assert args.remote_conda_env == "inf"
    assert args.remote_infinigen_src == "/workspace/infinigen/src"

    fallback_modes = parse_modes(argparse.Namespace(placer="unknown", modes=""), {})
    assert fallback_modes == ["random", "relaxed"]
