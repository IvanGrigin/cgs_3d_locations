import argparse
import json
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src import pipeline_runners as pr
from src import pipeline_artifacts as pa


def test_semantic_helpers() -> None:
    assert pr._semantic_text("KingSizeBed") == "king size bed"
    assert pr._semantic_text("lamp_wall_fixture") == "lamp wall fixture"
    assert pr._semantic_group("KingSizeBed", "King Bed") == "bed"
    assert pr._semantic_group("Pendant", category="ceiling lamp", constraints={"mount_type": "ceiling"}) == "lamp_ceiling"


def test_size_and_asset_helpers() -> None:
    assert pr._extract_size_m({"size_m": [1, 2, 3]}) == [1.0, 2.0, 3.0]
    assert pr._extract_size_m({"size": [1, 2, 3]}) == [0.0, 0.0, 0.0]
    assert pr._extract_size_m({"min_size_mm": [1000, 2000, 3000], "max_size_mm": [3000, 4000, 5000]}) == [2.0, 3.0, 4.0]
    raw = {"asset_meta": {"size_x": 1, "size_y": 2, "size_z": 3}, "model_jid": "m1"}
    assert pr._extract_asset_block_from_selected_object(raw)["model_id"] == "m1"

    with_asset = {"asset": {"source": "x", "mesh_path": "/m.obj"}}
    asset_block = pr._extract_asset_block_from_selected_object(with_asset)
    assert asset_block["source"] == "x"


def test_semantic_group_mount_type_overrides() -> None:
    assert pr._semantic_group("lamp", category="Lamp", constraints={"mount_type": "ceiling"}) == "lamp_ceiling"
    assert pr._semantic_group("lamp", category="Lamp", constraints={"mount_type": "wall"}) == "lamp_wall"


def test_load_selected_and_generated_payloads(tmp_path: Path) -> None:
    objects_path = tmp_path / "objects.json"
    objects_path.write_text(json.dumps({"objects": [{"name": "bed", "category": "bed", "size_m": [1, 2, 3]}]}), encoding="utf-8")
    selected = pr._load_selected_objects(objects_path)
    assert selected[0]["group"] == "bed"

    objects_path.write_text(json.dumps({"items": [{"name": "chair", "size_m": [0.5, 0.5, 1.0]}]}), encoding="utf-8")
    selected = pr._load_selected_objects(objects_path)
    assert selected[0]["id"] == "sel_0000"

    bad_objects = tmp_path / "bad_objects.json"
    bad_objects.write_text(json.dumps({"broken": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        pr._load_selected_objects(bad_objects)

    placement_path = tmp_path / "placement.json"
    placement_path.write_text(json.dumps({"placements": [{"name": "desk", "size_m": [1, 2, 3]}]}), encoding="utf-8")
    raw, parsed = pr._load_generated_placements(placement_path)
    assert len(parsed) == 1
    assert raw["placements"][0]["size_m"] == [1, 2, 3]

    bad_placement = tmp_path / "bad_placement.json"
    bad_placement.write_text(json.dumps({"wrong": []}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        pr._load_generated_placements(bad_placement)


def test_geometric_matching_and_placeholders() -> None:
    assert pr._same_family("lamp_ceiling", "lamp_wall") is True
    assert pr._same_family("bed", "chair") is False

    assert pr._size_distance([1, 2, 4], [1, 2, 4]) == 0.0
    assert pr._size_distance([1, 2, 3], [2, 3, 4]) > 0.0

    selected = {"size_m": [1.0, 1.0, 1.0], "group": "bed"}
    generated = {"size_m": [1.0, 1.0, 1.0], "group": "desk"}
    assert pr._match_score(selected, generated) < 0
    generated["group"] = "bed"
    assert pr._match_score(selected, generated) > 900

    item = {
        "aabb": {"x_min": 0, "x_max": 2, "y_min": 2, "y_max": 4, "z_min": 0, "z_max": 1},
        "size_m": [0, 0, 0],
        "source": {},
    }
    sel = {
        "size_m": [1, 2, 3],
        "constraints": {"mount_type": "wall"},
        "name": "desk",
        "category": "desk",
        "color": [1, 0, 0],
        "source": {"placement_source": "initial"},
    }
    pr._apply_selected_geometry(item, sel)
    assert item["source"] == {}

    placeholder = pr._build_generated_placeholder_item({"name": "desk", "category": "desk"}, placer_name="llm", out_idx=7)
    assert placeholder["id"] == "obj_0007"
    assert "placeholder_reason" in placeholder["meta"]

    stripped = pr._strip_binding_annotations(placeholder)
    assert "selected_object_id" not in stripped["source"]


def test_rebind_generated_layout_to_selected_objects(tmp_path: Path) -> None:
    objects = {"objects": [{"id": "s1", "name": "desk", "category": "desk", "size_m": [1, 1, 1]}]}
    objects_path = tmp_path / "objects.json"
    objects_path.write_text(json.dumps(objects), encoding="utf-8")

    placement_path = tmp_path / "placement.json"
    placement_path.write_text(
        json.dumps(
            {
                "placements": [
                    {"name": "chair", "size_m": [0.6, 0.6, 0.6], "source": {}},
                    {"name": "sofa", "size_m": [2, 2, 1], "source": {}},
                ]
            }
        ),
        encoding="utf-8",
    )

    pr.rebind_generated_layout_to_selected_objects(placement_path, objects_path, placer_name="test")
    updated = json.loads(placement_path.read_text(encoding="utf-8"))
    assert updated["meta"]["selected_object_count"] == 1
    assert len(updated["placements"]) == 2

    with pytest.raises(FileNotFoundError):
        pr.rebind_generated_layout_to_selected_objects(tmp_path / "missing.json", objects_path, placer_name="test")


def test_run_choose_stage_and_seed(monkeypatch, tmp_path: Path) -> None:
    args = argparse.Namespace(
        prepared_info="prep",
        future_root="future",
        chooser_llm_provider="none",
        ollama_url="http://127.0.0.1:11434",
        ollama_model="m",
        ollama_timeout=12,
        ollama_temperature=0.4,
        ollama_max_attempts=3,
        ollama_models=["a", "b"],
    )
    room_path = tmp_path / "room.json"
    room_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "out"
    run_dir.mkdir()

    calls: list[list[str]] = []
    def fake_choose_run(cmd, check=True, **kwargs):
        calls.append(cmd)
        out_idx = cmd.index("--out") + 1
        Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[out_idx]).write_text(json.dumps({"existing": True}), encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(pr.subprocess, "run", fake_choose_run)
    out_json = pr.run_choose_stage(args, {"CHOOSER_SCRIPT": "/bin/true"}, str(room_path), "prompt", run_dir, 1)
    assert out_json == run_dir / "objects.json"
    assert any("--disable-llm" in row for row in calls[0])

    pr.patch_objects_seed(out_json, 42)
    assert json.loads(out_json.read_text(encoding="utf-8"))["seed"] == 42


def test_cube_placer_and_layout_refiner(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "LEGACY_OBJECTS_JSON": str(tmp_path / "legacy.json"),
        "LEGACY_PLACEMENT_JSON": str(tmp_path / "legacy_placement.json"),
        "CUBE_SCRIPT": "cube.py",
        "LAYOUT_REFINER_SCRIPT": "layout_refiner.py",
    }
    (tmp_path / "legacy.json").write_text("{}", encoding="utf-8")
    (tmp_path / "legacy_placement.json").write_text(json.dumps({"placements": []}), encoding="utf-8")
    room_path = tmp_path / "room.json"
    room_path.write_text("{}", encoding="utf-8")
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps({"objects": []}), encoding="utf-8")
    out = tmp_path / "out.json"
    monkeypatch.setattr(pr.subprocess, "run", lambda *_, **__: type("R", (), {"returncode": 0})())

    pr.run_cube_placer(cfg, str(room_path), objects, "random", out, 1)
    assert out.is_file()

    with pytest.raises(RuntimeError):
        pr.run_layout_refiner_placer(
            cfg,
            argparse.Namespace(
                ml_model="model.pth",
                ml_device="cpu",
                max_scene_attempts=1,
                remote_host="h",
                remote_port=22,
                remote_user="u",
                remote_key=None,
            ),
            str(tmp_path / "bad.txt"),
            objects,
            "random",
            1,
            out,
            tmp_path,
        )


def test_ml_and_diffuscene_ollama_placers(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "ML_PLACER_SCRIPT": "ml.py",
        "DIFFUSCENE_REMOTE_SCRIPT": "remote.py",
        "OLLAMA_LLM_SCRIPT": "ollama.py",
        "TMP_ROOT": str((tmp_path / "remote_root")),
    }
    room_path = tmp_path / "room.json"
    room_path.write_text("{}", encoding="utf-8")
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps({"objects": []}), encoding="utf-8")
    out = tmp_path / "out.json"
    out.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []
    monkeypatch.setattr(pr.subprocess, "run", lambda cmd, check=True, **kwargs: calls.append(cmd) or type("R", (), {"returncode": 0})())

    args = argparse.Namespace(
        ml_model="model.pth",
        ml_device="cpu",
        diffusion_steps=10,
        placer="diffusion",
        diff_model=None,
        max_scene_attempts=1,
        remote_host="h",
        remote_port=22,
        remote_user="u",
        remote_key=None,
        remote_runner="/bin/runner",
        remote_conda_env=None,
        remote_infinigen_src=None,
        infinigen_task=None,
        infinigen_configs=None,
        infinigen_src=None,
        infinigen_rebind_selected_objects=False,
        ollama_url="http://127.0.0.1:11434",
        ollama_model="m",
        ollama_timeout=10,
        ollama_temperature=0.4,
        ollama_max_attempts=3,
        plan_model=None,
        plan_models=None,
        critic_model=None,
        critic_models=None,
        plan_temperature=None,
        plan_think=None,
        critic_think=None,
        llm_think=None,
    )

    pr.run_ml_placer(cfg, args, str(room_path), objects, "random", 1, out)
    assert any("--model" in row for row in calls[0])

    calls.clear()
    pr.run_diffuscene_remote_placer(cfg, args, str(room_path), objects, out, tmp_path)
    assert out.is_file()

    calls.clear()
    pr.run_ollama_llm_placer(cfg, args, str(room_path), objects, "infinigen", out, "prompt")
    assert any("--design-brief" in row for row in calls[0])


def test_m3dlayout_clean(monkeypatch, tmp_path: Path) -> None:
    cfg = {"M3DLAYOUT_SCRIPT": "m3d.py"}
    room_path = tmp_path / "room.json"
    room_path.write_text("{}", encoding="utf-8")
    out = tmp_path / "out.json"

    args = argparse.Namespace(
        remote_host="h",
        remote_user="u",
        remote_port=22,
        remote_key=None,
        remote_conda_env=None,
    )
    monkeypatch.setattr(pr.subprocess, "run", lambda *_, **__: type("R", (), {"returncode": 0})())
    pr.run_m3dlayout_clean(cfg, args, str(room_path), None, "prompt", 1, out, model_type="autoregressive")

    args.remote_host = None
    with pytest.raises(RuntimeError):
        pr.run_m3dlayout_clean(cfg, args, str(room_path), None, "prompt", 1, out, model_type="autoregressive")


def test_infinigen_validation_and_runner(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    blend = run_dir / "infinigen_clean_scene.blend"
    blend.write_bytes(b"blend")
    placement = run_dir / "placement.json"
    placement.write_text(json.dumps({"placements": [{"id": "x"}]}), encoding="utf-8")
    (run_dir / "inventory_summary.json").write_text(json.dumps({"raw_real_object_count": 4, "real_object_count": 2}), encoding="utf-8")

    pr._validate_infinigen_clean_artifacts(out_path=placement, run_dir=run_dir)

    placement_empty = tmp_path / "empty.json"
    placement_empty.write_text(json.dumps({"placements": []}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        pr._validate_infinigen_clean_artifacts(out_path=placement_empty, run_dir=run_dir)

    cfg = {"INFINIGEN_CLEAN_SCRIPT": "inf.py"}
    room_path = tmp_path / "room.json"
    room_path.write_text("{}", encoding="utf-8")
    out = tmp_path / "placement.json"
    out.write_text(json.dumps({"placements": [{"id": "x"}]}), encoding="utf-8")

    args = argparse.Namespace(
        remote_host=None,
        remote_port=None,
        remote_user=None,
        remote_key=None,
        remote_conda_env=None,
        remote_infinigen_src=None,
        infinigen_task=None,
        infinigen_configs=None,
        infinigen_rebind_selected_objects=False,
        infinigen_src=None,
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(pr.subprocess, "run", lambda cmd, check=True: calls.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(pr, "_validate_infinigen_clean_artifacts", lambda out_path, run_dir: None)
    pr.run_infinigen_clean(cfg, args, str(room_path), None, 1, out, tmp_path)
    assert calls


def test_connection_and_utility_helpers() -> None:
    args = argparse.Namespace(remote_host="1.2.3.4", remote_port=2222, remote_user="u", remote_key="/tmp/key")
    assert pr.build_ssh_base(args) == ["ssh", "-p", "2222", "-i", "/tmp/key", "u@1.2.3.4"]
    assert pr.build_scp_base(args) == ["scp", "-P", "2222", "-i", "/tmp/key"]
    assert pr.parse_csv_set("a,b, c,, d") == {"a", "b", "c", "d"}
    assert pr.infer_lego_room_type("house/living_room_v2.json") == "livingroom"
    assert pr.infer_lego_room_type("random.path") == "bedroom"
    assert pr.require_objects_path(Path("x"), "cube") == Path("x")
    with pytest.raises(RuntimeError):
        pr.require_objects_path(None, "cube")


def test_lego_generation_params_and_run_cmd_wrappers(monkeypatch) -> None:
    class P:
        lego_generation_preset = "gen_medium"
        lego_method = None
        lego_outer_passes = None
        lego_num_restarts = None
        lego_init_pos_noise_std = None
        lego_init_ang_noise_deg = None
        lego_init_scene_mode = None

    resolved = pr.resolve_lego_generation_params(P())
    assert resolved["preset"] == "gen_medium"
    assert resolved["method"] == "grad_noise"
    with pytest.raises(RuntimeError):
        class Bad:
            lego_generation_preset = "bad"
            lego_method = None
            lego_outer_passes = None
            lego_num_restarts = None
            lego_init_pos_noise_std = None
            lego_init_ang_noise_deg = None
            lego_init_scene_mode = None

        pr.resolve_lego_generation_params(Bad())

    calls: list[list[str]] = []
    monkeypatch.setattr(pr.subprocess, "run", lambda cmd, check=True: calls.append(cmd) or type("R", (), {"returncode": 0})())
    pr.run_cmd(["echo", "x"])
    assert calls


def test_ssh_and_scp_wrappers(monkeypatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(pr.subprocess, "run", lambda cmd, check=True: captured.append(cmd) or type("R", (), {"returncode": 0})())
    args = argparse.Namespace(remote_host="h", remote_port=22, remote_user="u", remote_key=None)

    pr.ssh_run(args, "echo hi")
    pr.scp_upload(args, Path("/tmp/x"), "/remote/x")
    pr.scp_download(args, "/remote/x", Path("/tmp/y"))

    assert captured[0][0] == "ssh"
    assert captured[1][0] == "scp"
    assert captured[2][0] == "scp"


def test_execute_placer_branches(monkeypatch, tmp_path: Path) -> None:
    cfg = {
        "LEGACY_OBJECTS_JSON": str(tmp_path / "legacy.json"),
        "CUBE_SCRIPT": "cube.py",
        "ML_PLACER_SCRIPT": "ml.py",
        "LAYOUT_REFINER_SCRIPT": "layout_refiner.py",
        "DIFFUSCENE_REMOTE_SCRIPT": "remote.py",
        "OLLAMA_LLM_SCRIPT": "ollama.py",
        "INFINIGEN_CLEAN_SCRIPT": "inf.py",
    }
    room_path = tmp_path / "room.json"
    room_path.write_text("{}", encoding="utf-8")
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps({"objects": []}), encoding="utf-8")
    out = tmp_path / "out.json"

    args = argparse.Namespace(
        placer="cube",
        ml_model="model.pth",
        max_attempts=1,
        ml_device="cpu",
        diffusion_steps=10,
        max_scene_attempts=1,
        remote_host="h",
        remote_user="u",
        remote_port=22,
        remote_key=None,
        remote_runner="runner.sh",
        ollama_url="http://127.0.0.1:11434",
        ollama_model="m",
        ollama_timeout=10,
        ollama_temperature=0.4,
        ollama_max_attempts=3,
        max_passes=1,
        prompt_text="x",
    )

    called: list[str] = []
    monkeypatch.setattr(pr, "run_cube_placer", lambda *_, **__: called.append("cube"))
    pr.execute_placer(cfg, args, str(room_path), objects, "random", 1, out, tmp_path, "prompt")
    assert called == ["cube"]

    called.clear()
    args.placer = "layout_refiner"
    monkeypatch.setattr(pr, "run_layout_refiner_placer", lambda *_, **__: called.append("layout"))
    pr.execute_placer(cfg, args, str(room_path), objects, "random", 1, out, tmp_path, "prompt")
    assert called == ["layout"]

    called.clear()
    args.placer = "diffusion"
    monkeypatch.setattr(pr, "run_ml_placer", lambda *_, **__: called.append("ml"))
    pr.execute_placer(cfg, args, str(room_path), objects, "random", 1, out, tmp_path, "prompt")
    assert called == ["ml"]


def test_run_lego_generate_from_scratch_smoke(monkeypatch, tmp_path: Path) -> None:
    room_path = tmp_path / "room.json"
    room_path.write_text(json.dumps({"room": {"width_m": 4, "depth_m": 4}}), encoding="utf-8")
    objects_path = tmp_path / "objects.json"
    objects_path.write_text(json.dumps({"objects": [{"id": "a"}]}), encoding="utf-8")
    args = argparse.Namespace(
        lego_postprocess=True,
        lego_room_type="auto",
        lego_repo="/repo",
        lego_python="python",
        lego_helper_script="helper.py",
        lego_tmp_root=str(tmp_path / "remote"),
        lego_checkpoint_bedroom="/ckp/bed",
        lego_checkpoint_livingroom="/ckp/living",
        lego_generation_preset="gen_medium",
        lego_method=None,
        lego_outer_passes=None,
        lego_num_restarts=None,
        lego_init_pos_noise_std=None,
        lego_init_ang_noise_deg=None,
        lego_init_scene_mode=None,
        max_attempts=1,
        remote_host="h",
        remote_user="u",
        remote_port=22,
        remote_key=None,
        remote_runner="run.sh",
        lego_modes=None,
    )

    calls: list[str] = []
    cfg = {"NORMALIZE_JSON_SCRIPT": "normalize.py"}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pipeline_mod = importlib.import_module("pipeline_artifacts")

    class _CompatPath(str):
        def __new__(cls, value: object) -> "_CompatPath":
            return super().__new__(cls, str(value))

        @property
        def name(self) -> str:
            return Path(self).name

        def expanduser(self) -> "_CompatPath":
            return _CompatPath(str(Path(self).expanduser()))

    monkeypatch.setattr(pr, "Path", _CompatPath)
    monkeypatch.setattr(pr, "infer_lego_room_type", lambda *_args, **_kwargs: "bedroom")

    monkeypatch.setattr(pr, "read_json", lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(pr, "write_json", lambda path, data: Path(path).write_text(json.dumps(data), encoding="utf-8"))
    monkeypatch.setattr(pr, "room_area_m2", lambda room: 100.0)
    monkeypatch.setattr(pr, "total_objects_footprint_m2", lambda objects: 2.0)
    monkeypatch.setattr(pr, "sort_objects_for_generation", lambda objects: objects)
    monkeypatch.setattr(pr, "build_seed_scene_and_placement", lambda room, objects: ({"scene": "seed"}, {"placements": []}))
    monkeypatch.setattr(pr, "crop_last_object", lambda objects: objects)
    monkeypatch.setattr(pr, "ssh_run", lambda _args, remote_command: calls.append(remote_command))
    monkeypatch.setattr(pr, "scp_upload", lambda _args, local, remote: calls.append(f"up:{local}"))
    monkeypatch.setattr(
        pr,
        "scp_download",
        lambda _args, remote_path, local_path: local_path.write_text(json.dumps({"placements": []}), encoding="utf-8"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "normalize_json_artifact",
        lambda **kwargs: Path(kwargs["output_path"]).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        pr,
        "normalize_json_artifact",
        lambda **kwargs: Path(kwargs["output_path"]).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "build_normalized_scene_artifact",
        lambda **kwargs: Path(kwargs["output_path"]).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "merge_room_spec_and_placements",
        lambda *_args, **_kwargs: Path(_args[2]).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        pr,
        "copy_tree_contents",
        lambda *_args, **_kwargs: None,
    )

    artifacts = pr.run_lego_generate_from_scratch(cfg, args, str(room_path), objects_path, run_dir)
    assert artifacts.placement_legacy.is_file()
    assert artifacts.placement_v1.is_file()
    assert any("mkdir -p" in cmd for cmd in calls)


def test_pipeline_runner_remaining_helper_and_error_branches(monkeypatch, tmp_path: Path) -> None:
    assert pr._semantic_group("", "") == "unknown"
    assert pr._extract_size_m({"asset_meta": {"size_x": 1, "size_y": 2, "size_z": 3}}) == [1.0, 2.0, 3.0]
    assert pr._semantic_name_from_blend_object_name("BedFactory(123).spawn_asset") == "BedFactory"
    assert pr._same_family("desk", "side_table") is True
    assert pr._match_score({"group": "bed", "size_m": [1, 1, 1]}, {"group": "chair", "size_m": [1, 1, 1]}) < 0

    ceiling_item = {"position_m": [1, 2, 3], "rotation_deg": 90}
    pr._apply_selected_geometry(
        ceiling_item,
        {"size_m": [1, 2, 0.5], "constraints": {"mount_type": "ceiling"}},
    )
    assert ceiling_item["aabb"]["z_max"] == pytest.approx(3.25)

    wall_item = {"position_m": [1, 2, 3]}
    pr._apply_selected_geometry(
        wall_item,
        {"size_m": [1, 2, 0.5], "constraints": {"mount_type": "wall"}},
    )
    assert wall_item["position_m"][2] == 3

    selected_path = tmp_path / "objects.json"
    selected_path.write_text(json.dumps({"objects": [3, {"name": "bed", "category": "bed", "size_m": [1, 1, 1]}]}), encoding="utf-8")
    assert len(pr._load_selected_objects(selected_path)) == 1

    placement_path = tmp_path / "placement.json"
    placement_path.write_text(json.dumps({"items": [3, {"source": {"blend_object_name": "Chair(1)"}, "size_m": [1, 1, 1]}]}), encoding="utf-8")
    _, generated = pr._load_generated_placements(placement_path)
    assert generated[0]["name"] == "Chair"

    empty_selected = tmp_path / "empty_selected.json"
    empty_selected.write_text(json.dumps({"objects": []}), encoding="utf-8")
    pr.rebind_generated_layout_to_selected_objects(placement_path, None, placer_name="unit")
    pr.rebind_generated_layout_to_selected_objects(placement_path, empty_selected, placer_name="unit")

    selected_path.write_text(json.dumps({"objects": [{"name": "bed", "category": "bed", "size_m": [1, 1, 1]}]}), encoding="utf-8")
    empty_placement = tmp_path / "empty_placement.json"
    empty_placement.write_text(json.dumps({"placements": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="placement"):
        pr.rebind_generated_layout_to_selected_objects(empty_placement, selected_path, placer_name="unit")

    short_placement = tmp_path / "short_placement.json"
    selected_path.write_text(
        json.dumps(
            {
                "objects": [
                    {"name": "bed", "category": "bed", "size_m": [1, 1, 1]},
                    {"name": "desk", "category": "desk", "size_m": [1, 1, 1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    short_placement.write_text(json.dumps({"placements": [{"name": "bed", "size_m": [1, 1, 1]}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="меньше объектов"):
        pr.rebind_generated_layout_to_selected_objects(short_placement, selected_path, placer_name="unit")

    non_object = tmp_path / "non_object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="objects.json"):
        pr.patch_objects_seed(non_object, 1)


def test_pipeline_runner_command_branches_and_failures(monkeypatch, tmp_path: Path) -> None:
    room_json = tmp_path / "room.json"
    room_json.write_text("{}", encoding="utf-8")
    room_txt = tmp_path / "room.txt"
    room_txt.write_text("room", encoding="utf-8")
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps({"objects": [{"name": "bed", "size_m": [1, 1, 1]}]}), encoding="utf-8")
    out = tmp_path / "out.json"

    cfg = {
        "LEGACY_OBJECTS_JSON": str(tmp_path / "legacy.json"),
        "LEGACY_PLACEMENT_JSON": str(tmp_path / "missing_legacy.json"),
        "CUBE_SCRIPT": "cube.py",
        "LAYOUT_REFINER_SCRIPT": "layout_refiner.py",
        "DIFFUSCENE_REMOTE_SCRIPT": "diff.py",
        "OLLAMA_LLM_SCRIPT": "ollama.py",
        "M3DLAYOUT_SCRIPT": "m3d.py",
        "INFINIGEN_CLEAN_SCRIPT": "inf.py",
        "TMP_ROOT": str(tmp_path / "remote_artifacts_root"),
    }

    monkeypatch.setattr(pr, "sync_objects_to_legacy_input", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pr.subprocess, "run", lambda *_args, **_kwargs: type("R", (), {"returncode": 0})())
    with pytest.raises(RuntimeError, match="Cube placer"):
        pr.run_cube_placer(cfg, str(room_json), objects, "random", out, 3)

    args = argparse.Namespace(
        placer="forest",
        ml_model=None,
        ml_device="cpu",
        diffusion_steps=10,
        max_scene_attempts=2,
        remote_runner="runner.py",
        remote_host="h",
        remote_port=2222,
        remote_user="u",
        remote_key=str(tmp_path / "key"),
        remote_conda_env="env",
        ollama_url="http://localhost",
        ollama_model="model",
        ollama_timeout=11,
        ollama_temperature=0.2,
        ollama_max_attempts=4,
        ollama_models=["m1", "m2"],
        plan_model="plan",
        plan_models=["p1"],
        critic_model="critic",
        critic_models=["c1"],
        plan_temperature=0.1,
        critic_temperature=0.3,
        plan_think="low",
        critic_think="low",
        llm_think="none",
    )

    with pytest.raises(RuntimeError, match="room-spec"):
        pr.run_ml_placer(cfg, args, str(room_txt), objects, "mode", 1, out)
    with pytest.raises(RuntimeError, match="--ml-model"):
        pr.run_ml_placer(cfg, args, str(room_json), objects, "mode", 1, out)
    with pytest.raises(RuntimeError, match="layout_refiner"):
        pr.run_layout_refiner_placer(cfg, args, str(room_txt), objects, "mode", 1, out, tmp_path)

    args.ml_model = "model.pth"
    calls: list[list[str]] = []

    def fake_run(cmd, check=True, **kwargs):
        calls.append(cmd)
        if "--out" in cmd:
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"placements": [{"id": "x"}]}), encoding="utf-8")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    pr.run_layout_refiner_placer(cfg, args, str(room_json), objects, "warm", 2, out, tmp_path)
    assert (tmp_path / "layout_refiner_warm.log").is_file()

    out.unlink(missing_ok=True)
    pr.run_diffuscene_remote_placer(cfg, args, str(room_json), objects, out, tmp_path)
    assert "--remote-key" in calls[-1]

    out.unlink()
    monkeypatch.setattr(pr.subprocess, "run", lambda cmd, check=True, **kwargs: calls.append(cmd) or type("R", (), {"returncode": 0})())
    with pytest.raises(RuntimeError, match="DiffuScene"):
        pr.run_diffuscene_remote_placer(cfg, args, str(room_json), objects, out, tmp_path)

    with pytest.raises(RuntimeError, match="room-spec"):
        pr.run_ollama_llm_placer(cfg, args, str(room_txt), objects, "mode", out, "prompt")
    pr.run_ollama_llm_placer(cfg, args, str(room_json), objects, "mode", out, "prompt")
    assert "--ollama-models" in calls[-1]
    assert "--critic-temperature" in calls[-1]

    with pytest.raises(RuntimeError, match="m3dlayout"):
        pr.run_m3dlayout_clean({}, args, str(room_json), objects, "prompt", 1, out, "autoregressive")
    rebind_calls: list[tuple[Path, Path | None, str]] = []
    monkeypatch.setattr(pr, "rebind_generated_layout_to_selected_objects", lambda path, obj_path, *, placer_name: rebind_calls.append((path, obj_path, placer_name)))
    pr.run_m3dlayout_clean(cfg, args, str(room_json), objects, "prompt", 1, out, "diffusion")
    assert rebind_calls[-1][2] == "m3dlayout_diffusion"

    run_dir = tmp_path / "infinigen"
    run_dir.mkdir()
    (run_dir / "style_profile.json").write_text("{}", encoding="utf-8")
    inf_out = run_dir / "placement.json"
    inf_out.write_text(json.dumps({"placements": [{"id": "x"}]}), encoding="utf-8")
    (run_dir / "infinigen_clean_scene.blend").write_bytes(b"blend")
    (run_dir / "inventory_summary.json").write_text(json.dumps({"raw_real_object_count": 1, "real_object_count": 1}), encoding="utf-8")
    args.infinigen_src = "/inf/src"
    args.remote_infinigen_src = "/remote/inf"
    args.infinigen_task = "coarse"
    args.infinigen_configs = ["a.gin", "b.gin"]
    args.infinigen_rebind_selected_objects = True
    pr.run_infinigen_clean(cfg, args, str(room_json), objects, 5, inf_out, run_dir)
    assert "--infinigen-configs" in calls[-1]
    assert rebind_calls[-1][2] == "infinigen_clean"

    missing_blend_dir = tmp_path / "missing_blend"
    missing_blend_dir.mkdir()
    bad_out = missing_blend_dir / "placement.json"
    bad_out.write_text(json.dumps({"placements": [{"id": "x"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="blend"):
        pr._validate_infinigen_clean_artifacts(out_path=bad_out, run_dir=missing_blend_dir)

    empty_summary_dir = tmp_path / "empty_summary"
    empty_summary_dir.mkdir()
    (empty_summary_dir / "infinigen_clean_scene.blend").write_bytes(b"blend")
    empty_out = empty_summary_dir / "placement.json"
    empty_out.write_text(json.dumps({"placements": [{"id": "x"}]}), encoding="utf-8")
    (empty_summary_dir / "inventory_summary.json").write_text(json.dumps({"raw_real_object_count": 0}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory"):
        pr._validate_infinigen_clean_artifacts(out_path=empty_out, run_dir=empty_summary_dir)


def test_pipeline_runner_lego_execute_and_connection_error_edges(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="remote-host"):
        pr.build_ssh_base(argparse.Namespace(remote_host=None, remote_user="u", remote_port=None, remote_key=None))
    with pytest.raises(RuntimeError, match="remote-host"):
        pr.build_scp_base(argparse.Namespace(remote_host="h", remote_user=None, remote_port=None, remote_key=None))
    assert pr.infer_lego_room_type("rooms/bedroom_01.json") == "bedroom"

    class OverrideArgs:
        lego_generation_preset = "gen_medium"
        lego_method = "random"
        lego_outer_passes = 9
        lego_num_restarts = 8
        lego_init_pos_noise_std = 0.7
        lego_init_ang_noise_deg = 11
        lego_init_scene_mode = "custom"

    gen_cfg = pr.resolve_lego_generation_params(OverrideArgs())
    assert gen_cfg["method"] == "random"
    assert gen_cfg["outer_passes"] == 9
    assert gen_cfg["init_scene_mode"] == "custom"

    base = argparse.Namespace(
        lego_postprocess=False,
        lego_room_type="bedroom",
        lego_repo="/repo",
        lego_python="python",
        lego_helper_script="helper.py",
        lego_tmp_root="/tmp/lego",
        lego_checkpoint_bedroom="/ckpt",
        lego_checkpoint_livingroom="/ckpt_living",
        lego_generation_preset="gen_medium",
        lego_method=None,
        lego_outer_passes=None,
        lego_num_restarts=None,
        lego_init_pos_noise_std=None,
        lego_init_ang_noise_deg=None,
        lego_init_scene_mode=None,
        max_attempts=1,
        remote_host="h",
        remote_user="u",
        remote_port=22,
        remote_key=None,
    )
    room = tmp_path / "room.json"
    room.write_text(json.dumps({"room": {}}), encoding="utf-8")
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps({"objects": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lego-postprocess"):
        pr.run_lego_generate_from_scratch({}, base, str(room), objects, tmp_path)

    base.lego_postprocess = True
    base.lego_room_type = "kitchen"
    with pytest.raises(RuntimeError, match="room_type"):
        pr.run_lego_generate_from_scratch({}, base, str(room), objects, tmp_path)

    for attr, pattern in [
        ("lego_repo", "lego-repo"),
        ("lego_python", "lego-python"),
        ("lego_helper_script", "lego-helper-script"),
        ("lego_tmp_root", "lego-tmp-root"),
        ("lego_checkpoint_bedroom", "checkpoint"),
    ]:
        base.lego_room_type = "bedroom"
        old = getattr(base, attr)
        setattr(base, attr, None)
        with pytest.raises(RuntimeError, match=pattern):
            pr.run_lego_generate_from_scratch({}, base, str(room), objects, tmp_path)
        setattr(base, attr, old)

    cfg = {"NORMALIZE_JSON_SCRIPT": "normalize.py"}
    args = argparse.Namespace(
        placer="diffuscene_remote",
        ml_model="model",
        remote_host="h",
        remote_user="u",
        remote_port=22,
        remote_key=None,
        remote_runner="r",
        ml_device="cpu",
        diffusion_steps=10,
        max_scene_attempts=1,
        ollama_url="u",
        ollama_model="m",
        ollama_timeout=1,
        ollama_temperature=0.1,
        ollama_max_attempts=1,
    )
    called: list[str] = []
    monkeypatch.setattr(pr, "run_diffuscene_remote_placer", lambda *_args, **_kwargs: called.append("diff"))
    pr.execute_placer(cfg, args, str(room), objects, "mode", 1, tmp_path / "out.json", tmp_path, "prompt")
    assert called == ["diff"]

    args.placer = "ollama_llm"
    monkeypatch.setattr(pr, "run_ollama_llm_placer", lambda **_kwargs: called.append("ollama"))
    pr.execute_placer(cfg, args, str(room), objects, "mode", 1, tmp_path / "out.json", tmp_path, "prompt")
    assert called[-1] == "ollama"

    args.placer = "m3dlayout_ar"
    monkeypatch.setattr(pr, "run_m3dlayout_clean", lambda *_args, **_kwargs: called.append("m3d_ar"))
    pr.execute_placer(cfg, args, str(room), objects, "mode", 1, tmp_path / "out.json", tmp_path, "prompt")
    assert called[-1] == "m3d_ar"

    args.placer = "m3dlayout_diffusion"
    pr.execute_placer(cfg, args, str(room), objects, "mode", 1, tmp_path / "out.json", tmp_path, "prompt")
    assert called[-1] == "m3d_ar"

    args.placer = "infinigen_clean"
    monkeypatch.setattr(pr, "run_infinigen_clean", lambda *_args, **_kwargs: called.append("inf"))
    pr.execute_placer(cfg, args, str(room), objects, "mode", 1, tmp_path / "out.json", tmp_path, "prompt")
    assert called[-1] == "inf"

    original_specs = pr.PLACER_SPECS
    monkeypatch.setattr(pr, "PLACER_SPECS", {**original_specs, "fake_lego": {"requires_ml_model": False, "runner": "lego_gen"}})
    args.placer = "fake_lego"
    with pytest.raises(RuntimeError, match="lego_gen"):
        pr.execute_placer(cfg, args, str(room), objects, "mode", 1, tmp_path / "out.json", tmp_path, "prompt")

    monkeypatch.setattr(pr, "PLACER_SPECS", {**original_specs, "fake_unknown": {"requires_ml_model": False, "runner": "other"}})
    args.placer = "fake_unknown"
    with pytest.raises(RuntimeError, match="Неизвестный runner"):
        pr.execute_placer(cfg, args, str(room), objects, "mode", 1, tmp_path / "out.json", tmp_path, "prompt")
