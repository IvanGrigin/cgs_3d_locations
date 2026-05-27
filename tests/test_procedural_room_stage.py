from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import types

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.pipeline import procedural_room_stage as prs  # noqa: E402
from src.pipeline_config import PlacementArtifacts


def test_split_types_and_report_manifest_update(tmp_path: Path) -> None:
    assert prs._split_types("  Bedroom, living_room, ,toilet ") == {"bedroom", "living_room", "toilet"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    report = {"replacement_count": 1}
    prs._write_report_to_manifest(manifest, report)
    assert manifest.read_text(encoding="utf-8").find("\"procedural_room_stage\"") != -1

    missing = tmp_path / "missing.json"
    prs._write_report_to_manifest(missing, report)
    assert not missing.exists()

    broken = tmp_path / "broken.json"
    broken.write_text("{bad", encoding="utf-8")
    prs._write_report_to_manifest(broken, report)
    assert broken.read_text(encoding="utf-8") == "{bad"


def test_add_procedural_room_arguments_and_manifest_write_error(monkeypatch, tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    prs.add_procedural_room_arguments(parser)
    args = parser.parse_args(["--procedural-rooms", "always", "--procedural-room-types", "bedroom", "--procedural-density", "high", "--procedural-replace-existing", "--procedural-seed", "5"])
    assert args.procedural_rooms == "always"
    assert args.procedural_replace_existing is True

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")

    real_open = Path.open

    def flaky_open(self, *args, **kwargs):
        if self == manifest and args and args[0] == "w":
            raise OSError("cannot write")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    prs._write_report_to_manifest(manifest, {"stage": "ignored"})
    assert json.loads(manifest.read_text(encoding="utf-8")) == {}


def test_maybe_apply_procedural_room_stage_disabled_if_never(tmp_path: Path) -> None:
    place_v1 = tmp_path / "placement.v1.json"
    place_v1.write_text("{}", encoding="utf-8")
    artifacts = PlacementArtifacts(
        placement_legacy=tmp_path / "placement_legacy.json",
        placement_v1=place_v1,
        scene_v1=tmp_path / "scene.v1.json",
        scene_legacy=tmp_path / "scene_legacy.json",
    )
    for f in [artifacts.scene_v1, artifacts.scene_legacy]:
        f.write_text("{}", encoding="utf-8")

    class Args:
        procedural_rooms = "never"

    class Args2:
        procedural_rooms = "auto"
        procedural_room_types = "bedroom"
        procedural_density = "very_high"
        procedural_replace_existing = False
        procedural_seed = 11

    updated = prs.maybe_apply_procedural_room_stage(
        args=Args(),
        artifacts=artifacts,
        run_dir=tmp_path,
        prompt_text="bedroom",
        manifest_path=None,
        tag="base",
    )
    assert updated is artifacts

    # Keep the callable for the auto path behind a monkeypatch; function exists
    # and should return new artifacts when called.
    args = types.SimpleNamespace(
        procedural_rooms="auto",
        procedural_room_types="bedroom",
        procedural_density="very_high",
        procedural_replace_existing=False,
        procedural_seed=123,
    )

    def fake_apply(
        *,
        artifacts,
        run_dir,
        prompt,
        policy,
        density,
        replace_existing,
        seed,
        tag,
        enabled_room_types,
    ):
        return artifacts, {"replacement_count": 0}

    import src.pipeline.procedural_room_stage as proc_mod
    import pytest

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(proc_mod, "apply_procedural_room_stage_to_artifacts", fake_apply)
        updated2 = prs.maybe_apply_procedural_room_stage(
            args=args,
            artifacts=artifacts,
            run_dir=tmp_path,
            prompt_text="bedroom",
            manifest_path=tmp_path / "manifest.json",
            tag="base",
        )
        assert isinstance(updated2, PlacementArtifacts) or updated2 is artifacts
    finally:
        monkeypatch.undo()
