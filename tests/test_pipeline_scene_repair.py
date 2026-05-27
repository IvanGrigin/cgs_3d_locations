import argparse
import json
from pathlib import Path

from src.pipeline_scene_repair import (
    add_scene_repair_arguments,
    _repair_script_path,
    _looks_like_scene_v1,
    _repair_summary,
    maybe_repair_scene_json,
    scene_repair_enabled,
)


def test_add_scene_repair_arguments():
    parser = argparse.ArgumentParser()
    add_scene_repair_arguments(parser)
    args = parser.parse_args([])
    assert args.scene_repair_max_passes == 1
    assert args.scene_repair_selector_topk == 3


def test_scene_repair_switch_and_schema_checks():
    parser = argparse.ArgumentParser()
    add_scene_repair_arguments(parser)
    args = parser.parse_args([])
    assert scene_repair_enabled(args) is False
    args.scene_repair_model = "model.ckpt"
    assert scene_repair_enabled(args) is True

    good = Path("/tmp/nonexistent")
    assert _looks_like_scene_v1(good) is False
    assert _repair_script_path().name == "apply_repair_proposal_v1.py"


def test_repair_summary_counts():
    report = {
        "initial_bad_indices": [1, 2],
        "final_bad_indices": [3],
        "passes": [
            {"accepted": [1, 2]},
            {"accepted": [3]},
            {"accepted": []},
        ],
    }
    out = _repair_summary(report)
    assert out["initial_bad_count"] == 2
    assert out["final_bad_count"] == 1
    assert out["pass_count"] == 3
    assert out["accepted_move_count"] == 3


def test_maybe_repair_scene_json_disabled_and_schema_guard(tmp_path: Path) -> None:
    class Args:
        scene_repair_model = None
        scene_repair_selector_model = None
        scene_repair_device = "auto"
        scene_repair_max_passes = 1
        scene_repair_candidate_limit = 4
        scene_repair_selector_topk = 3
        scene_repair_selector_candidate_limit = 6
        scene_repair_selector_global_fallback_k = 3

    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    args = Args()
    out, info = maybe_repair_scene_json(args=args, scene_json_path=scene_path, run_dir=tmp_path, tag="base")
    assert out == scene_path
    assert info is None

    bad_scene = tmp_path / "bad.json"
    bad_scene.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    args.scene_repair_model = "model.ckpt"
    out, info = maybe_repair_scene_json(args=args, scene_json_path=bad_scene, run_dir=tmp_path, tag="base")
    assert out == bad_scene
    assert info["skipped_reason"] == "unsupported_schema"

    missing = tmp_path / "missing.json"
    try:
        maybe_repair_scene_json(args=args, scene_json_path=missing, run_dir=tmp_path, tag="base")
    except RuntimeError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing repair input should raise")


def test_maybe_repair_scene_json_success(monkeypatch, tmp_path: Path) -> None:
    class Args:
        scene_repair_model = "model.ckpt"
        scene_repair_selector_model = None
        scene_repair_device = "auto"
        scene_repair_max_passes = 1
        scene_repair_candidate_limit = 4
        scene_repair_selector_topk = 3
        scene_repair_selector_candidate_limit = 6
        scene_repair_selector_global_fallback_k = 3

    scene = {"schema": "scene.v1", "room": {}}
    scene_path = tmp_path / "scene.v1.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_run(cmd, check=True):
        # emulate external script: write report.json
        report_idx = cmd.index("--report-json")
        report_path = Path(cmd[report_idx + 1])
        report_path.write_text(
            json.dumps(
                {
                    "initial_bad_indices": [],
                    "final_bad_indices": [],
                    "passes": [{"accepted": []}],
                }
            ),
            encoding="utf-8",
        )
        return None

    def fake_script_path():
        return tmp_path / "repair.py"

    monkeypatch.setattr("src.pipeline_scene_repair.subprocess.run", fake_run)
    monkeypatch.setattr("src.pipeline_scene_repair._repair_script_path", fake_script_path)

    out, info = maybe_repair_scene_json(args=Args(), scene_json_path=scene_path, run_dir=run_dir, tag="base")
    assert out.name == "scene_repaired.base.v1.json"
    assert info["summary"]["initial_bad_count"] == 0
    assert info["summary"]["pass_count"] == 1


def test_maybe_repair_scene_json_selector_args(monkeypatch, tmp_path: Path) -> None:
    class Args:
        scene_repair_model = "model.ckpt"
        scene_repair_selector_model = "selector.ckpt"
        scene_repair_device = "cpu"
        scene_repair_max_passes = 2
        scene_repair_candidate_limit = 5
        scene_repair_selector_topk = 4
        scene_repair_selector_candidate_limit = 7
        scene_repair_selector_global_fallback_k = 8

    scene_path = tmp_path / "scene.v1.json"
    scene_path.write_text(json.dumps({"schema": "scene.v1", "room": {}}), encoding="utf-8")
    seen = {}

    def fake_run(cmd, check=True):
        seen["cmd"] = cmd
        report_path = Path(cmd[cmd.index("--report-json") + 1])
        report_path.write_text(json.dumps({"passes": []}), encoding="utf-8")

    monkeypatch.setattr("src.pipeline_scene_repair.subprocess.run", fake_run)
    monkeypatch.setattr("src.pipeline_scene_repair._repair_script_path", lambda: tmp_path / "repair.py")

    _, info = maybe_repair_scene_json(args=Args(), scene_json_path=scene_path, run_dir=tmp_path, tag="selector")
    assert "--selector-model" in seen["cmd"]
    assert "--selector-global-fallback-k" in seen["cmd"]
    assert info["selector_model"].endswith("selector.ckpt")
