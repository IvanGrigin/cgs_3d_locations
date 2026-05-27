from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest

pytest.skip("legacy test for archived module src.tools.run_semantic_room_planner", allow_module_level=True)

from src.tools import run_semantic_room_planner as tool  # noqa: E402


def test_read_prompt_prefers_direct_prompt_and_file(tmp_path: Path) -> None:
    data = {"prompt": "from_room"}
    args = tool.argparse.Namespace(prompt="direct", prompt_file=None)
    assert tool._read_prompt(args, data) == "direct"

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("from_file", encoding="utf-8")
    args = tool.argparse.Namespace(prompt=None, prompt_file=str(prompt_file))
    assert tool._read_prompt(args, data) == "from_file"

    args = tool.argparse.Namespace(prompt=None, prompt_file=None)
    assert tool._read_prompt(args, data) == "from_room"


def test_main_returns_zero_with_stubbed_runner(tmp_path: Path, monkeypatch, capsys) -> None:
    room_path = tmp_path / "room.json"
    room_path.write_text(json.dumps({"room": {"id": "r1"}}), encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        tool,
        "run_semantic_room_planner",
        lambda **kwargs: {
            "status": "success",
            "hard_errors": [],
            "warnings": ["ok"],
            "out_dir": str(out_dir),
            "final_room_scene_plan": str(out_dir / "final_room_scene_plan.json"),
            "scene_v1": str(out_dir / "scene.semantic.v1.json"),
            "placement_v1": str(out_dir / "placement.semantic.v1.json"),
            "validation_score": 0.99,
        },
    )

    monkeypatch.setattr("sys.argv", [
        "run_semantic_room_planner.py",
        "--input-json",
        str(room_path),
        "--out-dir",
        str(out_dir),
    ])

    assert tool.main() == 0
    captured = capsys.readouterr().out
    assert "final status: success" in captured
