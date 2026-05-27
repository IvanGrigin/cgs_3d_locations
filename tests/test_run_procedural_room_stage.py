from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest

pytest.skip("legacy test for archived module src.tools.run_procedural_room_stage", allow_module_level=True)

import src.tools.run_procedural_room_stage as rp
from src.pipeline import procedural_rooms


def test_build_scene_from_room_and_batch_jobs() -> None:
    scene = rp.build_scene_from_room({"room": {"id": "r1"}})
    assert scene["schema"] == "scene.v1"
    assert scene["room"]["id"] == "r1"

    payload = [
        {"id": "a", "room": {"id": "r1"}},
        {"room": {"id": "r2"}},
        123,
    ]
    assert rp._batch_jobs(payload) == [{"id": "a", "room": {"id": "r1"}}, {"room": {"id": "r2"}}]


def test_job_scene_path_with_embedded_room(tmp_path: Path) -> None:
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    room_json = tmp_path / "room.json"
    room_json.write_text(json.dumps({"room": {"id": "r1"}}), encoding="utf-8")
    job = {"room_json": str(room_json)}
    scene_path = rp._job_scene_path(job, out_dir)
    assert scene_path.read_text(encoding="utf-8").find("\"scene.v1\"") != -1
    assert scene_path.name == "input_scene_from_room.v1.json"


def test_run_batch_with_procedural_stub(tmp_path: Path) -> None:
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "policy": "always",
                "density": "very_high",
                "seed": 21,
                "jobs": [
                    {"id": "j1", "room": {"id": "r1"}, "prompt": "kitchen"},
                    {"id": "j2", "room": {"id": "r2"}, "prompt": "bedroom", "tag": "bed"},
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    args = type(
        "Args",
        (),
        {
            "batch_file": str(batch),
            "out_dir": str(out_dir),
            "policy": "always",
            "density": "very_high",
            "replace_existing": False,
            "seed": 5,
            "prompt": "",
            "tag": "standalone",
        },
    )()

    original = rp.apply_procedural_room_stage

    def fake_apply_procedural_room_stage(
        scene_json_path,
        out_dir,
        prompt,
        policy,
        density,
        replace_existing,
        seed,
        tag,
    ):
        output_scene = out_dir / f"{tag}_scene.v1.json"
        output_placement = out_dir / f"{tag}_placement.v1.json"
        output_scene.write_text(json.dumps({"tag": tag}), encoding="utf-8")
        output_placement.write_text(json.dumps({"tag": tag}), encoding="utf-8")
        return {
            "room_type": "kitchen",
            "generated_count": 1,
            "output_scene_json": str(output_scene),
            "output_placement_json": str(output_placement),
            "validation": {"accessibility_ok": True},
            "schema": "procedural_room_stage/v1",
            "report_json": str(out_dir / f"{tag}.json"),
            "room_id": tag,
        }

    rp.apply_procedural_room_stage = fake_apply_procedural_room_stage
    try:
        summary = rp.run_batch(args)
    finally:
        rp.apply_procedural_room_stage = original

    assert summary["job_count"] == 2
    assert summary["summary"][0]["generated_count"] == 1
    assert summary["summary"][1]["id"] == "j2"
    assert (tmp_path / "out" / "procedural_room_batch_report.json").is_file()
