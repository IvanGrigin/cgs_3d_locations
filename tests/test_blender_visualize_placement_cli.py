from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "Plasement" / "BlenderVisualizePlacement.py"


def _load_module():
    name = "BlenderVisualizePlacement_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_find_executable_reference_blend_and_cli_command(tmp_path, monkeypatch, capsys):
    mod = _load_module()
    fake_blender = tmp_path / "Blender"
    fake_blender.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_blender.chmod(0o755)
    assert mod.find_executable([str(fake_blender)]) == str(fake_blender)
    monkeypatch.setattr(mod.shutil, "which", lambda value: "/usr/bin/blender" if value == "blender" else None)
    assert mod.find_executable(["blender"]) == "/usr/bin/blender"
    with pytest.raises(FileNotFoundError):
        mod.find_executable([None, str(tmp_path / "missing")])

    scene_blend = tmp_path / "ref.blend"
    scene_blend.write_bytes(b"blend")
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps({"meta": {"placement_meta": {"scene_blend": str(scene_blend)}}}), encoding="utf-8")
    assert mod.infer_reference_blend(scene_json) == str(scene_blend.resolve())

    fallback_json = tmp_path / "fallback.json"
    fallback_json.write_text("{}", encoding="utf-8")
    fallback_blend = tmp_path / "infinigen_clean_scene.blend"
    fallback_blend.write_bytes(b"blend")
    assert mod.infer_reference_blend(fallback_json) == str(fallback_blend.resolve())
    assert mod.infer_reference_blend(tmp_path / "bad.json") is None

    commands = []
    monkeypatch.setattr(mod, "find_executable", lambda candidates: str(fake_blender))
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, check: commands.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "BlenderVisualizePlacement.py",
            "--json",
            str(scene_json),
            "--background",
            "--overlay-bbox-only",
            "--bbox-fallback",
            "--highlight-item-ids",
            "a,b",
            "--hide-room-shell",
            "--render-layer",
            "kitchen",
            "--force-tint",
            "--save-blend",
            str(tmp_path / "out.blend"),
            "--build-report",
            str(tmp_path / "report.json"),
            "--render",
            str(tmp_path / "render.png"),
            "--turntable-render-dir",
            str(tmp_path / "frames"),
            "--turntable-frames",
            "3",
            "--turntable-frame-index",
            "1",
            "--turntable-elevation-deg",
            "25",
            "--no-pack-assets",
            "--verbose",
        ],
    )
    mod.main()
    cmd = commands[0]
    assert cmd[0] == str(fake_blender)
    assert os.path.abspath(scene_blend) in cmd
    assert "-b" in cmd
    assert "--overlay-bbox-only" in cmd
    assert "--no-bbox-fallback" not in cmd
    assert "--render-layer" in cmd and "kitchen" in cmd
    assert "--turntable-frame-index" in cmd and "1" in cmd
    assert "RUN:" in capsys.readouterr().out
