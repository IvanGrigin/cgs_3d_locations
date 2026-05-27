import json
from pathlib import Path
from types import SimpleNamespace

from src import pipeline_artifacts as pa


def test_json_helpers_roundtrip(tmp_path: Path) -> None:
    payload = {"x": 1}
    path = tmp_path / "data" / "x.json"
    pa.write_json(path, payload)
    assert pa.read_json(path) == payload


def test_blender_output_paths_and_flags(tmp_path: Path) -> None:
    args = SimpleNamespace(
        save_blend=None,
        render=None,
        blender=None,
        keep_blend=False,
    )

    blend, render = pa.blender_outputs_for_mode(args, tmp_path, "infinigen", variant_suffix="x1")
    assert blend.endswith("scene_infinigen_x1.blend")
    assert render.endswith("render_infinigen_x1.png")

    args.save_blend = str((tmp_path / "custom").resolve())
    args.render = str((tmp_path / "img").resolve())
    blend, render = pa.blender_outputs_for_mode(args, tmp_path, "infinigen", variant_suffix="")
    assert blend.endswith("custom")
    assert render.endswith("img")
    assert pa._should_keep_blend(args) is True
    assert pa._blender_output_mode(SimpleNamespace(blender_output="both")) == "both"
    assert pa._blender_output_mode(SimpleNamespace(blender_output="bad")) == "render"


def test_sync_and_merge_room_spec_and_placements(tmp_path: Path) -> None:
    room = {"room": {"id": "r1", "area_m2": 12.0}, "meta": {"kind": "room"}}
    placement = {"placements": [{"id": "obj"}], "items": [{"id": "fallback"}]}

    room_path = tmp_path / "room.json"
    place_path = tmp_path / "placement.json"
    out_path = tmp_path / "scene.json"
    room_path.write_text(json.dumps(room), encoding="utf-8")
    place_path.write_text(json.dumps(placement), encoding="utf-8")
    pa.merge_room_spec_and_placements(str(room_path), str(place_path), str(out_path))

    scene = pa.read_json(out_path)
    assert scene["meta"] == room["meta"]
    assert scene["placements"] == placement["placements"]
    assert scene["room"]["id"] == "r1"


def test_merge_copies_fallback_items_when_placements_missing(tmp_path: Path) -> None:
    room = {"room": {"id": "r1"}}
    placement = {"items": [{"id": "fallback"}]}
    room_path = tmp_path / "room.json"
    place_path = tmp_path / "placement.json"
    out_path = tmp_path / "scene.json"
    room_path.write_text(json.dumps(room), encoding="utf-8")
    place_path.write_text(json.dumps(placement), encoding="utf-8")
    pa.merge_room_spec_and_placements(str(room_path), str(place_path), str(out_path))
    scene = pa.read_json(out_path)
    assert scene["placements"] == placement["items"]


def test_sync_objects_to_legacy_input(tmp_path: Path) -> None:
    src = tmp_path / "objects.json"
    src.write_text(json.dumps({"items": []}), encoding="utf-8")
    dst = tmp_path / "legacy" / "objects.json"
    pa.sync_objects_to_legacy_input(src, str(dst))
    assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_copy_tree_contents_recursive(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("a", encoding="utf-8")
    sub = src / "dir"
    sub.mkdir()
    (sub / "b.txt").write_text("b", encoding="utf-8")
    dst = tmp_path / "dst"
    pa.copy_tree_contents(src, dst)
    assert (dst / "a.txt").read_text(encoding="utf-8") == "a"
    assert (dst / "dir" / "b.txt").read_text(encoding="utf-8") == "b"


def test_normalize_and_build_scene_artifacts_use_subprocess_commands(monkeypatch, tmp_path: Path) -> None:
    called: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs) -> None:
        called.append(cmd)

    monkeypatch.setattr(pa.subprocess, "run", fake_run)
    args = {"NORMALIZE_JSON_SCRIPT": "normalize.py"}
    pa.normalize_json_artifact(
        cfg_runtime=args,
        input_path=tmp_path / "in.json",
        output_path=tmp_path / "out.json",
        target="placement",
    )
    pa.build_normalized_scene_artifact(
        cfg_runtime=args,
        room_path=str((tmp_path / "room.json").resolve()),
        placement_path=tmp_path / "in.json",
        output_path=tmp_path / "scene.json",
    )
    assert any("--target" in row and "placement" in row for row in called)
    assert any("--target" in row and "scene" in row for row in called)


def test_build_scene_artifacts_and_choose_scene(monkeypatch, tmp_path: Path) -> None:
    room_json = tmp_path / "room.json"
    room_json.write_text(json.dumps({"room": {"id": "a"}}), encoding="utf-8")
    placement_out = tmp_path / "placement.json"
    placement_out.write_text(json.dumps({"placements": []}), encoding="utf-8")

    args = {
        "NORMALIZE_JSON_SCRIPT": "normalize.py",
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_normalize_json_artifact(cfg_runtime, input_path, output_path, target):
        output_path.write_text(
            json.dumps({"ok": target, "input": str(input_path)}),
            encoding="utf-8",
        )

    def fake_merge(room_json_path, placement_json_path, out_json_path):
        Path(out_json_path).write_text(
            json.dumps({"merged": True, "room": room_json_path, "placement": placement_json_path}),
            encoding="utf-8",
        )

    monkeypatch.setattr(pa, "normalize_json_artifact", fake_normalize_json_artifact)
    def fake_build_normalized_scene_artifact(cfg_runtime, room_path, placement_path, output_path):
        Path(output_path).write_text(
            json.dumps({"ok": "scene", "room": str(room_path), "placement": str(placement_path)}),
            encoding="utf-8",
        )

    monkeypatch.setattr(pa, "build_normalized_scene_artifact", fake_build_normalized_scene_artifact)
    monkeypatch.setattr(pa, "merge_room_spec_and_placements", fake_merge)

    artifacts = pa.build_scene_artifacts(
        cfg_runtime=args,
        room_path=str(room_json),
        run_dir=run_dir,
        layout_mode="infinigen",
        placement_out=placement_out,
        variant_suffix="v1",
    )
    assert artifacts.placement_v1.name == "placement_v1.v1.json"
    assert artifacts.scene_v1 is not None
    assert artifacts.scene_legacy is not None
    assert (run_dir / "placement_v1.v1.json").is_file()
    assert (run_dir / "scene_v1.v1.json").is_file()
    assert (run_dir / "scene_infinigen_v1.json").is_file()

    selected = pa.choose_scene_for_render(artifacts)
    assert selected == artifacts.scene_v1

    artifacts_no_v1 = SimpleNamespace(
        placement_legacy=placement_out,
        placement_v1=Path("/no"),
        scene_v1=None,
        scene_legacy=tmp_path / "legacy_scene.json",
    )
    artifacts_no_v1.scene_legacy.write_text(json.dumps({"scene": "legacy"}), encoding="utf-8")
    selected = pa.choose_scene_for_render(artifacts_no_v1)
    assert selected == artifacts_no_v1.scene_legacy


def test_run_blender_for_mode_keeps_and_renders(monkeypatch, tmp_path: Path) -> None:
    cfg_runtime = {"BLENDER_VIS_SCRIPT": "blender_vis.py"}
    args = SimpleNamespace(
        blender=None,
        headless=False,
        keep_blend=False,
        no_bbox_fallback=False,
        save_blend=None,
        render=None,
        blender_output="both",
        blender_gif_width=256,
        blender_gif_height=256,
        blender_gif_samples=4,
        blender_gif_frames=4,
        blender_gif_fps=10,
        blender_gif_elevation=10.0,
    )
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps({}), encoding="utf-8")

    run_cmds: list[tuple[list[str], dict]] = []

    def fake_run(cmd: list[str], **kwargs):
        run_cmds.append((cmd, kwargs))
        if "--save-blend" in cmd:
            i = cmd.index("--save-blend")
            Path(cmd[i + 1]).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pa.subprocess, "run", fake_run)

    def fake_render(*, cfg_runtime, args, blend_path, frame_dir, gif_path, frame_count, elevation_deg, fps) -> None:
        assert frame_count == 4
        assert elevation_deg == 10.0
        assert fps == 10

    monkeypatch.setattr(pa, "_render_gif_from_blend_isolated", fake_render)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = pa.run_blender_for_mode(
        cfg_runtime=cfg_runtime,
        args=args,
        room_path="",
        run_dir=run_dir,
        layout_mode="infinigen",
        scene_json_path=scene_json,
        variant_suffix="a",
    )
    assert result["blender_output"] == "both"
    assert run_cmds


def test_run_blender_for_mode_invalid_scene_raises(tmp_path: Path) -> None:
    cfg_runtime = {"BLENDER_VIS_SCRIPT": "blender_vis.py"}
    args = SimpleNamespace(
        blender=None,
        headless=False,
        keep_blend=False,
        no_bbox_fallback=False,
        save_blend=None,
        render=None,
        blender_output="render",
        blender_gif_width=256,
        blender_gif_height=256,
        blender_gif_samples=4,
        blender_gif_frames=4,
        blender_gif_fps=10,
        blender_gif_elevation=10.0,
    )
    import pytest

    with pytest.raises(RuntimeError):
        pa.run_blender_for_mode(
            cfg_runtime=cfg_runtime,
            args=args,
            room_path="",
            run_dir=tmp_path,
            layout_mode="infinigen",
            scene_json_path=tmp_path / "missing.json",
        )


def test_merge_room_spec_falls_back_when_placements_is_not_list(tmp_path: Path) -> None:
    room = {"room": {"id": "r1", "meta": {"kind": "room"}}}
    placement = {"placements": {"bad": "value"}, "items": [{"id": "fallback"}]}

    room_path = tmp_path / "room.json"
    place_path = tmp_path / "placement.json"
    out_path = tmp_path / "scene.json"
    room_path.write_text(json.dumps(room), encoding="utf-8")
    place_path.write_text(json.dumps(placement), encoding="utf-8")
    pa.merge_room_spec_and_placements(str(room_path), str(place_path), str(out_path))

    scene = pa.read_json(out_path)
    assert scene["placements"] == []


def test_copy_tree_contents_noop_for_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    target = tmp_path / "target"
    pa.copy_tree_contents(file_path, target)
    assert not target.exists()


def test_render_gif_from_frames_and_isolated_blend(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(pa.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pa.subprocess, "run", fake_run)

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    pa._render_gif_from_frames(frame_dir=frame_dir, out_gif=frame_dir / "out.gif", fps=12)
    assert any("palettegen=stats_mode=diff" in str(item["cmd"]) for item in calls)

    commands: list[tuple] = []
    saved_blend = tmp_path / "room.blend"
    saved_blend.write_text("blend", encoding="utf-8")
    frame_dir_out = tmp_path / "iso_frames"
    args = SimpleNamespace(
        blender=None,
        blender_gif_width=640,
        blender_gif_height=480,
        blender_gif_samples=2,
        blender_gif_frames=5,
        blender_gif_elevation=22,
        blender_gif_fps=15,
        keep_blender_gif_frames=True,
    )
    blender_gif = None

    def fake_run_isolated(cmd, **kwargs):
        commands.append((tuple(cmd), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pa.subprocess, "run", fake_run_isolated)
    pa._render_gif_from_blend_isolated(
        cfg_runtime={"BLENDER_VIS_SCRIPT": "x", "NORMALIZE_JSON_SCRIPT": "y"},
        args=args,
        blend_path=saved_blend,
        frame_dir=frame_dir_out,
        gif_path=tmp_path / "out.gif",
        frame_count=5,
        elevation_deg=22.0,
        fps=15,
    )
    assert commands
    cmd, _ = commands[0]
    assert "--keep-frames" in cmd


def test_run_blender_for_mode_adds_extra_flags_and_cleans_frames(tmp_path: Path, monkeypatch) -> None:
    cfg_runtime = {"BLENDER_VIS_SCRIPT": "blender_vis.py"}
    args = SimpleNamespace(
        blender="blender-bin",
        headless=True,
        keep_blend=False,
        no_bbox_fallback=True,
        save_blend=str((tmp_path / "scene.blend").resolve()),
        render=str((tmp_path / "render.png").resolve()),
        blender_output="both",
        blender_gif_width=256,
        blender_gif_height=256,
        blender_gif_samples=4,
        blender_gif_frames=4,
        blender_gif_fps=10,
        blender_gif_elevation=10.0,
        keep_blender_gif_frames=False,
    )

    scene_json = tmp_path / "scene.json"
    scene_json.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    frame_dir = run_dir / "_frames_render_infinigen"

    def fake_run(cmd, **kwargs):
        if "--save-blend" in cmd:
            out_idx = cmd.index("--save-blend") + 1
            Path(cmd[out_idx]).write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pa.subprocess, "run", fake_run)
    monkeypatch.setattr(pa, "_render_gif_from_blend_isolated", lambda **_: None)
    pa.run_blender_for_mode(
        cfg_runtime=cfg_runtime,
        args=args,
        room_path="",
        run_dir=run_dir,
        layout_mode="infinigen",
        scene_json_path=scene_json,
        variant_suffix="",
    )

    assert not (run_dir / "scene.infinigen.blend").is_file()
    assert not frame_dir.exists() or len(list(frame_dir.glob("*"))) == 0
