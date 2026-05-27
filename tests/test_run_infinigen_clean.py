from __future__ import annotations

import argparse
import base64
import builtins
import importlib.util
from pathlib import Path
import json
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.Plasement import run_infinigen_clean as ric  # noqa: E402


def test_import_fallback_inventory_mapping_when_prompt_compiler_missing(monkeypatch) -> None:
    module_path = ROOT / "src" / "Plasement" / "run_infinigen_clean.py"
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.prompt_compiler.inventory_mapping":
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    module_name = "run_infinigen_clean_inventory_fallback_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.factory_to_semantic("BedFactory") == "Bed"
    assert module.factory_to_semantic("unknown") is None
    assert module.is_core_furniture_factory("BedFactory") is True
    assert module.is_core_furniture_factory("BookStackFactory") is False
    assert module.is_core_furniture_factory("") is False
    assert module.is_technical_factory_name("hoof_parent_temp.001") is True
    assert module.is_technical_factory_name("BezierCurve") is True
    assert module.is_technical_factory_name("BedFactory") is False


def test_looks_technical_name_and_clean_placement_payload() -> None:
    assert ric._looks_technical_name("cube.001") is True
    assert ric._looks_technical_name("cube") is True
    assert ric._looks_technical_name("some_real_object") is False

    data = {
        "placements": [
            {"id": "1", "name": "cube_001", "category": "Cube"},
            {"id": "2", "name": "BedFactory", "category": "Bed"},
        ]
    }
    cleaned = ric.clean_placement_payload(data)
    assert len(cleaned["placements"]) == 1
    assert cleaned["meta"]["filtered_technical_objects"] == 1


def test_build_inventory_from_placement_counts_core_objects() -> None:
    payload = {"placements": [{"id": "1", "name": "BedFactory", "category": "BedFactory", "size_m": [1, 2, 0.6], "position_m": [1, 2, 3]}]}
    report = ric.build_inventory_from_placement(payload)
    assert report["summary"]["raw_real_object_count"] == 1
    assert report["summary"]["real_object_count"] == 1
    assert report["items"][0]["factory_name"] == "BedFactory"
    assert report["summary"]["core_factory_counts"]["BedFactory"] == 1


def test_parse_solver_log_missing_and_with_markers(tmp_path: Path) -> None:
    missing = ric.parse_solver_log(tmp_path / "none.log")
    assert missing["exists"] is False
    assert missing["termination_status"] == "missing"

    log = tmp_path / "solver.log"
    log.write_text(
        "\n".join(
            [
                "info: solve_rooms",
                "info: populate_assets",
                "violations={'door_overlap': 2, 'window_overlap': 1}",
                "warning: unstable solver",
                "Error: sample fatal issue",
            ]
        ),
        encoding="utf-8",
    )
    parsed = ric.parse_solver_log(log)
    assert parsed["exists"] is True
    assert parsed["termination_status"] == "error"
    assert parsed["violations"]["door_overlap"] == 2
    assert any("solve_rooms" in marker for marker in parsed["stage_markers"])

    unknown = tmp_path / "unknown.log"
    unknown.write_text("solver started but did not finish", encoding="utf-8")
    parsed_unknown = ric.parse_solver_log(unknown)
    assert parsed_unknown["termination_status"] == "unknown"

    bad_violations = tmp_path / "bad_violations.log"
    bad_violations.write_text("violations={bad\nOK:", encoding="utf-8")
    assert ric.parse_solver_log(bad_violations)["termination_status"] == "success"


def test_write_inventory_artifacts_and_cleanup(tmp_path: Path) -> None:
    placement = tmp_path / "placement.json"
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    placement.write_text(
        json.dumps(
            {
                "placements": [
                    {"id": "1", "name": "cube_001", "category": "Cube", "size_m": [1, 1, 1]},
                    {"id": "2", "name": "LampFactory", "category": "Lighting", "size_m": [1, 1, 1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    inventory_path, summary_path = ric.write_inventory_artifacts(candidate_dir, placement)
    assert inventory_path.is_file()
    assert summary_path.is_file()
    assert json.loads(inventory_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["real_object_count"] == summary["core_factory_counts"]["LampFactory"]
    assert summary["core_factory_counts"]["LampFactory"] == 1


def test_remote_command_helpers_and_disk_space(monkeypatch, tmp_path: Path) -> None:
    args = argparse.Namespace(
        remote_host="host",
        remote_user="root",
        remote_port=32172,
        remote_key=str(tmp_path / "id_ed25519"),
        remote_conda_env="infinigen",
    )
    base = ric.build_ssh_base(args, allocate_tty=True)
    assert base[:2] == ["ssh", "-tt"]
    assert "-p" in base and "32172" in base
    assert base[-1] == "root@host"
    assert "bash -lc" in ric.wrap_remote("echo hi")
    assert "echo hi" in ric.wrap_remote_bash("echo hi")
    assert "conda activate infinigen" in "; ".join(ric.build_remote_preamble(args))

    calls: list[dict[str, object]] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        if kwargs.get("stdout") == subprocess.PIPE and kwargs.get("text"):
            return subprocess.CompletedProcess(cmd, 0, stdout="Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/root 1 0 999999999 0% /workspace\n", stderr="")
        if kwargs.get("stdout") == subprocess.PIPE:
            return subprocess.CompletedProcess(cmd, 0, stdout=base64.b64encode(b"payload"))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ric.subprocess, "run", fake_run)
    ric.ssh_run(args, "true")

    local = tmp_path / "local.txt"
    local.write_text("hello", encoding="utf-8")
    ric.ssh_upload_file(args, local, "/remote/local.txt")
    downloaded = tmp_path / "downloaded.txt"
    ric.ssh_download_file(args, "/remote/out.txt", downloaded)
    assert downloaded.read_bytes() == b"payload"

    captured = ric.ssh_capture(args, "df -Pk /workspace /workspace/tmp")
    assert "Available" in captured.stdout
    ric.ensure_remote_free_space(args, min_free_kb=1)

    def low_space(_args, _command):
        return subprocess.CompletedProcess([], 0, stdout="h\n/dev/root 1 0 10 0% /workspace\n", stderr="")

    monkeypatch.setattr(ric, "ssh_capture", low_space)
    with pytest.raises(RuntimeError, match="REMOTE_DISK_FULL"):
        ric.ensure_remote_free_space(args, min_free_kb=1024)

    assert any("base64 -d" in " ".join(row["cmd"]) for row in calls)

    timings = {"stages": "bad"}
    ric._remote_timing_entry(timings, stage="x", started=ric.datetime.now(), duration_sec=1.2345, status="ok")
    assert timings["stages"][0]["duration_sec"] == 1.234
    timings_path = tmp_path / "timings.json"
    ric._write_remote_timings(timings_path, {"stages": [{"duration_sec": "2.5"}]})
    assert json.loads(timings_path.read_text(encoding="utf-8"))["duration_sec"] == 2.5

    def malformed_space(_args, _command):
        return subprocess.CompletedProcess([], 0, stdout="h\nshort\n/dev/root a b bad /workspace\n", stderr="")

    monkeypatch.setattr(ric, "ssh_capture", malformed_space)
    ric.ensure_remote_free_space(args, min_free_kb=1024)


def test_seed_room_style_and_floorplan_helpers(monkeypatch, tmp_path: Path) -> None:
    assert ric.parse_seed_value("ff") == 255
    assert ric.normalize_seed(str(ric.MAX_NUMPY_SEED + 2)) == 1
    assert ric.normalize_seed_for_infinigen(0) == "1"

    infinigen_src = tmp_path / "infinigen" / "src"
    infinigen_src.mkdir(parents=True)
    monkeypatch.setenv("INFINIGEN_SRC", str(infinigen_src))
    assert ric.default_infinigen_src() == infinigen_src.resolve()

    assert ric.infer_room_semantic({"room": {"source_room_type": "studio apartment"}}) == "bedroom"
    assert ric.infer_room_semantic({"room": {"name": "Кухня"}}) == "kitchen"
    assert ric.infer_room_semantic_from_style_profile({"room_type": "living-room"}) == "living-room"
    assert ric.has_source_restroom_type({"room": {"source_room_type": "wc"}}) is True

    room = {
        "room": {
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}, {"x": 0, "y": 3}],
            "walls": [{"id": "w0", "from_vertex": 0, "to_vertex": 1}],
            "doors": [{"wall_id": "w0", "s": 1.0, "width": 0.8}],
            "windows": [{"wall_id": "w0", "s": 2.0, "width": 1.0}],
        }
    }
    poly = ric.infer_room_polygon(room)
    assert poly == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    assert ric.infer_walls(room, poly)["w0"] == ((0.0, 0.0), (4.0, 0.0))
    assert ric.place_segment_on_wall((0, 0), (4, 0), 1, 1) == ((1.0, 0.0), (2.0, 0.0))
    assert ric.place_segment_on_wall((0, 0), (0, 0), 1, 1) == ((0, 0), (0, 0))
    module_text = ric.build_custom_floorplan_module_text(room, "bedroom")
    assert "door_0" in module_text and "window_0" in module_text and "bedroom_0/0" in module_text

    with pytest.raises(RuntimeError, match="floor_polygon"):
        ric.infer_room_polygon({"room": {}})

    style_path = tmp_path / "style.json"
    style_path.write_text(json.dumps({"style_label": "minimal", "infinigen": {"monkeypatch_params": {"a": 1}, "overrides": ["x=1"]}}), encoding="utf-8")
    style = ric.load_style_profile(style_path)
    params, overrides, label = ric.style_profile_infinigen_patch(style)
    assert params == {"a": 1}
    assert overrides == ["x=1"]
    assert label == "minimal"

    bad_style = tmp_path / "bad_style.json"
    bad_style.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="JSON object"):
        ric.load_style_profile(bad_style)
    with pytest.raises(RuntimeError, match="must be an object"):
        ric.style_profile_infinigen_patch({"infinigen": [1]})
    with pytest.raises(RuntimeError, match="monkeypatch_params"):
        ric.style_profile_infinigen_patch({"infinigen": {"monkeypatch_params": ["bad"]}})
    with pytest.raises(RuntimeError, match="overrides"):
        ric.style_profile_infinigen_patch({"infinigen": {"overrides": {"bad": 1}}})
    assert "candidate-pool" in ric.styled_infinigen_runner_script_text()
    assert "def should_skip" in ric.blender_extract_script_text()

    assert ric.infer_room_semantic({"room": {"name": "formal dining"}}) == "dining-room"
    assert ric.infer_room_semantic({"room": {"name": "small wc"}}) == "restroom"
    assert ric.infer_room_semantic({"room": {"name": "ванная"}}) == "bathroom"
    assert ric.infer_room_semantic({"room": {"name": "hallway"}}) == "hallway"
    assert ric.infer_room_semantic({"room": {"name": "балкон"}}) == "balcony"
    assert ric.infer_room_semantic_from_style_profile({"room_type": "dining_room"}) == "dining-room"

    monkeypatch.delenv("INFINIGEN_SRC", raising=False)
    monkeypatch.setattr(ric.Path, "is_dir", lambda self: False)
    with pytest.raises(RuntimeError, match="infinigen/src"):
        ric.default_infinigen_src()

    monkeypatch.setattr(ric.Path, "is_dir", lambda self: str(self).endswith("/workspace/infinigen/src"))
    assert str(ric.default_infinigen_src()).endswith("/workspace/infinigen/src")
    assert ric.infer_room_semantic({"room": {"name": "plain storage"}}) == "bedroom"
    assert ric.infer_room_semantic_from_style_profile(None) is None


def test_generation_local_and_extract_are_mocked(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "infinigen_src"
    src.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_folder = tmp_path / "out_scene"
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ric.subprocess, "run", fake_run)
    ric.run_infinigen_generate(src, "mod", 42, out_folder, run_dir=run_dir, task="coarse", configs=["a.gin"])
    assert calls[-1][1:3] == ["-m", "infinigen_examples.generate_indoors"]
    assert 'Solver.floor_plan="infinigen_examples.configs_indoor.floor_plans.custom.mod.example"' in calls[-1]

    style_path = tmp_path / "style.json"
    style_path.write_text(json.dumps({"style_label": "x", "infinigen": {"overrides": ["A=B"]}}), encoding="utf-8")
    ric.run_infinigen_generate(
        src,
        "styled",
        "ff",
        out_folder,
        run_dir=run_dir,
        style_profile={"__source_path__": str(style_path), "style_label": "x", "infinigen": {"overrides": ["A=B"]}},
        log_path=tmp_path / "gen.log",
    )
    assert (run_dir / "_run_infinigen_with_style.py").is_file()
    assert calls[-1][1].endswith("_run_infinigen_with_style.py")

    with pytest.raises(RuntimeError, match="source path"):
        ric.run_infinigen_generate(src, "bad", 1, out_folder, run_dir=run_dir, style_profile={"infinigen": {"overrides": ["x"]}})

    ric.extract_placement_from_blend(tmp_path / "scene.blend", tmp_path / "placement.json", run_dir)
    assert (run_dir / "_extract_infinigen_blend.py").is_file()
    assert calls[-1][-2].endswith("scene.blend")

    plain_log = tmp_path / "plain.log"
    ric.run_infinigen_generate(src, "plain", 2, out_folder, run_dir=run_dir, task="", configs=None, log_path=plain_log)
    assert calls[-1][1:3] == ["-m", "infinigen_examples.generate_indoors"]
    assert plain_log.is_file()

    ric.run_infinigen_generate(
        src,
        "styled_no_log",
        3,
        out_folder,
        run_dir=run_dir,
        style_profile={"__source_path__": str(style_path), "style_label": "x", "infinigen": {"factory_whitelist": ["BedFactory"]}},
    )
    assert calls[-1][1].endswith("_run_infinigen_with_style.py")


def test_run_local_and_remote_use_file_and_ssh_mocks(monkeypatch, tmp_path: Path) -> None:
    room = tmp_path / "room.json"
    room.write_text(
        json.dumps({"room": {"type": "bedroom", "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 2}, {"x": 0, "y": 2}]}}),
        encoding="utf-8",
    )
    infinigen_src = tmp_path / "infinigen" / "src"
    custom_dir = infinigen_src / "infinigen_examples" / "configs_indoor" / "floor_plans" / "custom"
    custom_dir.mkdir(parents=True)
    run_dir = tmp_path / "local_run"
    out = tmp_path / "placement.json"

    def fake_generate(*, output_folder, **_kwargs):
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / "scene.blend").write_bytes(b"blend")

    def fake_extract(*, out_json, **_kwargs):
        out_json.write_text(json.dumps({"placements": [{"id": "bed", "name": "BedFactory", "category": "BedFactory", "size_m": [1, 2, 1]}]}), encoding="utf-8")

    monkeypatch.setattr(ric, "run_infinigen_generate", fake_generate)
    monkeypatch.setattr(ric, "extract_placement_from_blend", fake_extract)
    args = argparse.Namespace(
        room=str(room),
        seed="7",
        out=str(out),
        run_dir=str(run_dir),
        style_profile=None,
        infinigen_src=str(infinigen_src),
        infinigen_task="coarse",
        infinigen_configs=["singleroom.gin"],
    )
    ric.run_local(args)
    assert out.is_file()
    assert (run_dir / "infinigen_clean_scene.blend").read_bytes() == b"blend"
    assert (run_dir / "infinigen_clean_meta.json").is_file()
    assert not any(custom_dir.glob("_auto_fp_*.py"))

    remote_run = tmp_path / "remote_run"
    remote_out = tmp_path / "remote_placement.json"
    remote_calls: list[str] = []
    monkeypatch.setattr(ric.uuid, "uuid4", lambda: type("U", (), {"hex": "abc12345ffff"})())
    monkeypatch.setattr(ric, "ensure_remote_free_space", lambda _args: remote_calls.append("free"))
    monkeypatch.setattr(ric, "ssh_run", lambda _args, command: remote_calls.append(command))
    monkeypatch.setattr(ric, "ssh_upload_file", lambda _args, _local, remote_path: remote_calls.append(f"upload:{remote_path}"))

    def fake_download(_args, remote_path, local_path):
        if remote_path.endswith("placement.json"):
            local_path.write_text(json.dumps({"placements": []}), encoding="utf-8")
            return
        if remote_path.endswith("infinigen_clean_scene.blend"):
            local_path.write_bytes(b"blend")
            return
        raise subprocess.CalledProcessError(1, ["scp"])

    monkeypatch.setattr(ric, "ssh_download_file", fake_download)
    remote_args = argparse.Namespace(
        room=str(room),
        seed="7",
        out=str(remote_out),
        run_dir=str(remote_run),
        style_profile=None,
        remote_host="host",
        remote_user="root",
        remote_port=22,
        remote_key=None,
        remote_conda_env=None,
        remote_infinigen_src="/workspace/infinigen/src",
        infinigen_task="coarse",
        infinigen_configs=["singleroom.gin"],
    )
    ric.run_remote(remote_args)
    assert remote_out.is_file()
    assert (remote_run / "infinigen_clean_scene.blend").read_bytes() == b"blend"
    timings = json.loads((remote_run / "infinigen_remote_timings.json").read_text(encoding="utf-8"))
    assert any(row["stage"] == "download_remote_run_log" and row["status"] == "missing_or_failed" for row in timings["stages"])
    assert any("run_infinigen_clean.py" in call for call in remote_calls)


def test_run_local_missing_blend_and_compiled_policy_paths(monkeypatch, tmp_path: Path) -> None:
    room = tmp_path / "room.json"
    room.write_text(
        json.dumps({"room": {"type": "bedroom", "floor_polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}, {"x": 0, "y": 2}]}}),
        encoding="utf-8",
    )
    infinigen_src = tmp_path / "infinigen" / "src"
    (infinigen_src / "infinigen_examples" / "configs_indoor" / "floor_plans" / "custom").mkdir(parents=True)
    run_dir = tmp_path / "run_missing"
    args = argparse.Namespace(
        room=str(room),
        seed="7",
        out=str(tmp_path / "placement.json"),
        run_dir=str(run_dir),
        style_profile=None,
        infinigen_src=str(infinigen_src),
        infinigen_task="coarse",
        infinigen_configs=["singleroom.gin"],
    )

    def no_scene_generate(*, output_folder, **_kwargs):
        output_folder.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ric, "run_infinigen_generate", no_scene_generate)
    with pytest.raises(RuntimeError, match="scene.blend"):
        ric.run_local(args)
    assert not any((infinigen_src / "infinigen_examples" / "configs_indoor" / "floor_plans" / "custom").glob("_auto_fp_*.py"))

    compile_mod = types.ModuleType("src.prompt_compiler.compile_to_infinigen")
    compile_mod.build_room_json = lambda _policy: {"room": {"floor_polygon": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}]}}
    compile_mod.build_style_profile = lambda _policy: {"style_label": "test"}
    schemas_mod = types.ModuleType("src.prompt_compiler.schemas")
    schemas_mod.CompiledPolicy = types.SimpleNamespace(load=lambda path: {"loaded": str(path)})
    monkeypatch.setitem(sys.modules, "src.prompt_compiler.compile_to_infinigen", compile_mod)
    monkeypatch.setitem(sys.modules, "src.prompt_compiler.schemas", schemas_mod)
    calls = []
    monkeypatch.setattr(ric, "run_local", lambda ns: calls.append(("local", ns.room, ns.out)))
    result = ric.run_from_compiled_policy(tmp_path / "policy.json", tmp_path / "compiled_out", 11, infinigen_src=str(infinigen_src))
    assert calls and calls[0][0] == "local"
    assert Path(result["placement"]).name == "placement.json"

    calls.clear()
    monkeypatch.setattr(ric, "run_remote", lambda ns: calls.append(("remote", ns.remote_host, ns.remote_user)))
    ric.run_from_compiled_policy(tmp_path / "policy.json", tmp_path / "compiled_remote", 12, remote_host="host", remote_user="root")
    assert calls == [("remote", "host", "root")]
    batch = ric.run_screening_from_compiled_policy(tmp_path / "policy.json", tmp_path / "screening", [1, 2])
    assert len(batch) == 2


def test_run_local_style_profile_early_failure_and_remote_failure_timing(monkeypatch, tmp_path: Path) -> None:
    room = tmp_path / "room.json"
    room.write_text(
        json.dumps({"room": {"type": "bedroom", "floor_polygon": [{"x": 0, "y": 0}, {"x": 2, "y": 0}, {"x": 2, "y": 2}, {"x": 0, "y": 2}]}}),
        encoding="utf-8",
    )
    style = tmp_path / "style.json"
    style.write_text(json.dumps({"room_type": "bedroom", "infinigen": {"compiled_policy_path": "policy.json"}}), encoding="utf-8")
    infinigen_src = tmp_path / "infinigen" / "src"
    (infinigen_src / "infinigen_examples" / "configs_indoor" / "floor_plans" / "custom").mkdir(parents=True)
    run_dir = tmp_path / "early"
    out = tmp_path / "placement.json"

    def fake_generate(*, output_folder, **_kwargs):
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / "scene.blend").write_bytes(b"blend")

    def fake_extract(*, out_json, **_kwargs):
        out_json.write_text(json.dumps({"placements": [{"id": "lamp", "name": "LampFactory", "category": "LampFactory", "size_m": [1, 1, 1]}]}), encoding="utf-8")

    monkeypatch.setattr(ric, "run_infinigen_generate", fake_generate)
    monkeypatch.setattr(ric, "extract_placement_from_blend", fake_extract)
    args = argparse.Namespace(
        room=str(room),
        seed="7",
        out=str(out),
        run_dir=str(run_dir),
        style_profile=str(style),
        infinigen_src=str(infinigen_src),
        infinigen_task="coarse",
        infinigen_configs=["singleroom.gin"],
    )
    ric.run_local(args)
    assert json.loads((run_dir / "early_failure.json").read_text(encoding="utf-8"))["reason"] == "missing_required_bed_early"

    remote_run = tmp_path / "remote_fail"
    remote_args = argparse.Namespace(
        room=str(room),
        seed="7",
        out=str(tmp_path / "remote.json"),
        run_dir=str(remote_run),
        style_profile=str(style),
        remote_host="host",
        remote_user="root",
        remote_port=22,
        remote_key=None,
        remote_conda_env=None,
        remote_infinigen_src="/workspace/infinigen/src",
        infinigen_task="coarse",
        infinigen_configs=[],
    )
    monkeypatch.setattr(ric, "ensure_remote_free_space", lambda _args: None)
    monkeypatch.setattr(ric, "ssh_run", lambda *_args: (_ for _ in ()).throw(RuntimeError("ssh failed")))
    with pytest.raises(RuntimeError, match="ssh failed"):
        ric.run_remote(remote_args)
    timings = json.loads((remote_run / "infinigen_remote_timings.json").read_text(encoding="utf-8"))
    assert timings["stages"][-1]["status"] == "failed"


def test_main_compiled_policy_and_required_arg_paths(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(ric, "run_from_compiled_policy", lambda **kwargs: calls.append(("single", kwargs)) or {})
    monkeypatch.setattr(ric, "run_screening_from_compiled_policy", lambda **kwargs: calls.append(("screen", kwargs)) or [])

    monkeypatch.setattr(sys, "argv", ["run", "--compiled-policy", str(tmp_path / "policy.json"), "--run-dir", str(tmp_path / "out"), "--seed", "3"])
    ric.main()
    assert calls[-1][0] == "single"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--compiled-policy", str(tmp_path / "policy.json"), "--screening-seeds", "4,5", "--screening-base-dir", str(tmp_path / "screen")],
    )
    ric.main()
    assert calls[-1][0] == "screen"

    monkeypatch.setattr(sys, "argv", ["run", "--compiled-policy", str(tmp_path / "policy.json")])
    with pytest.raises(RuntimeError, match="--run-dir or --out"):
        ric.main()

    monkeypatch.setattr(sys, "argv", ["run"])
    with pytest.raises(RuntimeError, match="--room, --out and --run-dir"):
        ric.main()

    monkeypatch.setattr(ric, "run_remote", lambda ns: calls.append(("remote-main", ns.remote_host)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--room",
            str(tmp_path / "room.json"),
            "--out",
            str(tmp_path / "placement.json"),
            "--run-dir",
            str(tmp_path / "run"),
            "--remote-host",
            "host",
            "--remote-user",
            "root",
        ],
    )
    ric.main()
    assert calls[-1] == ("remote-main", "host")
