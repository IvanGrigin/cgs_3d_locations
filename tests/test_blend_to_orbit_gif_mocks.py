from __future__ import annotations

import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.tools import blend_to_orbit_gif as orbit


class FakeVector:
    def __init__(self, values):
        vals = list(values)
        self.x = float(vals[0])
        self.y = float(vals[1])
        self.z = float(vals[2])

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __sub__(self, other):
        return FakeVector((self.x - other.x, self.y - other.y, self.z - other.z))

    def __add__(self, other):
        return FakeVector((self.x + other.x, self.y + other.y, self.z + other.z))


class IdentityMatrix:
    def __matmul__(self, other):
        return FakeVector(other)


class FakeConstraints(list):
    def new(self, type):
        constraint = types.SimpleNamespace(type=type, target=None, track_axis="", up_axis="")
        self.append(constraint)
        return constraint


class FakeObject:
    def __init__(self, name, *, obj_type="MESH", hide_render=False, bounds=None):
        self.name = name
        self.type = obj_type
        self.hide_render = hide_render
        self.hide_viewport = False
        self.children = []
        self.matrix_world = IdentityMatrix()
        self.bound_box = bounds or [
            (x, y, z)
            for x in (0.0, 1.0)
            for y in (0.0, 1.0)
            for z in (0.0, 1.0)
        ]
        self.data = types.SimpleNamespace(angle=math.radians(50.0), materials=[])
        self.constraints = FakeConstraints()
        self.location = FakeVector((0.0, 0.0, 0.0))


class NamedStore(list):
    def get(self, name):
        for item in self:
            if getattr(item, "name", None) == name:
                return item
        return None

    def new(self, name, data=None):
        obj = FakeObject(name, obj_type="CAMERA" if data is not None else "EMPTY")
        obj.data = data
        self.append(obj)
        return obj

    def remove(self, item):
        if item in self:
            super().remove(item)


class FakeMaterials(NamedStore):
    def new(self, name):
        bsdf = types.SimpleNamespace(
            inputs={
                "Base Color": types.SimpleNamespace(default_value=None),
                "Roughness": types.SimpleNamespace(default_value=None),
            }
        )
        nodes = NamedStore([types.SimpleNamespace(bl_idname="ShaderNodeTexImage"), bsdf])
        nodes.get = lambda key: bsdf if key == "Principled BSDF" else None
        mat = types.SimpleNamespace(
            name=name,
            diffuse_color=(0.0, 0.0, 0.0, 1.0),
            use_nodes=False,
            node_tree=types.SimpleNamespace(nodes=nodes),
        )
        self.append(mat)
        return mat


def _fake_bpy(tmp_path):
    objects = NamedStore(
        [
            FakeObject("Room_Wall_w0"),
            FakeObject("chair"),
            FakeObject("huge_outlier", bounds=[(200.0, 0.0, 0.0), (201.0, 1.0, 1.0)]),
        ]
    )
    collection_objects = types.SimpleNamespace(link=lambda obj: objects.append(obj))
    scene = types.SimpleNamespace(
        objects=objects,
        collection=types.SimpleNamespace(objects=collection_objects),
        camera=None,
        render=types.SimpleNamespace(
            engine="CYCLES",
            image_settings=types.SimpleNamespace(file_format=""),
            resolution_x=0,
            resolution_y=0,
            resolution_percentage=0,
            filepath="",
        ),
        cycles=types.SimpleNamespace(samples=0, use_denoising=False),
        eevee=types.SimpleNamespace(taa_render_samples=0),
        display=types.SimpleNamespace(
            shading=types.SimpleNamespace(light="", color_type="", show_xray=True, show_shadows=False, show_cavity=False)
        ),
    )
    rendered = []
    return types.SimpleNamespace(
        app=types.SimpleNamespace(driver_namespace={"orbit_gif_args": {
            "frames_dir": str(tmp_path / "frames"),
            "width": 320,
            "height": 240,
            "samples": 2,
            "margin": 1.1,
            "distance_scale": 1.0,
            "yaw_step": 180,
            "elevations_deg": [30],
            "frame_indices": None,
            "hide_room_shell": True,
            "hide_outliers": True,
            "clay": True,
            "no_textures": False,
            "workbench_materials": True,
        }}),
        context=types.SimpleNamespace(scene=scene, view_layer=types.SimpleNamespace(update=lambda: None)),
        data=types.SimpleNamespace(
            objects=objects,
            cameras=types.SimpleNamespace(new=lambda name: types.SimpleNamespace(name=name, angle=math.radians(50.0))),
            materials=FakeMaterials(),
            images=NamedStore([types.SimpleNamespace(name="img")]),
        ),
        ops=types.SimpleNamespace(render=types.SimpleNamespace(render=lambda write_still=True: rendered.append(scene.render.filepath))),
        _rendered=rendered,
    )


def test_embedded_blender_helper_runs_against_fake_bpy(tmp_path, monkeypatch):
    fake_bpy = _fake_bpy(tmp_path)
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=FakeVector))

    module_path = Path(orbit.__file__).resolve()
    padded_helper = "\n" * 15 + orbit.BLENDER_HELPER_CODE
    namespace: dict[str, object] = {}
    exec(compile(padded_helper, str(module_path), "exec"), namespace)

    assert len(fake_bpy._rendered) == 2
    assert fake_bpy.context.scene.camera is not None
    assert fake_bpy.data.objects.get("Room_Wall_w0").hide_render is True
    assert fake_bpy.data.objects.get("huge_outlier").hide_render is True


def test_orbit_cli_and_outer_helpers_use_mocks(tmp_path, monkeypatch):
    args = orbit.build_cli().parse_args(["--blend", str(tmp_path / "scene.blend"), "--elevations", "0, 45", "--yaw-step", "90"])
    assert args.yaw_step == 90
    assert orbit.parse_elevations("0, 30,") == [0.0, 30.0]
    with pytest.raises(RuntimeError, match="хотя бы один"):
        orbit.parse_elevations(" , ")

    blender = tmp_path / "Blender"
    blender.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(orbit.shutil, "which", lambda name: None)
    assert orbit.resolve_blender_binary(str(blender)) == str(blender.resolve())

    calls = []
    monkeypatch.setattr(orbit.subprocess, "run", lambda cmd, check=True: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    blend = tmp_path / "scene.blend"
    blend.write_text("blend", encoding="utf-8")
    frames = tmp_path / "frames"
    orbit.run_blender_render(
        str(blender),
        blend,
        frames,
        width=100,
        height=80,
        samples=1,
        yaw_step=180,
        elevations_deg=[30],
        margin=1.0,
        distance_scale=1.0,
        frame_indices=[0],
        hide_room_shell=True,
        hide_outliers=True,
        clay=False,
        no_textures=True,
        workbench_materials=True,
    )
    assert calls[0][0] == str(blender)

    calls.clear()
    orbit.render_frames_isolated(
        str(blender),
        blend,
        frames,
        width=100,
        height=80,
        samples=1,
        yaw_step=180,
        elevations_deg=[30],
        margin=1.0,
        distance_scale=1.0,
        hide_room_shell=False,
        hide_outliers=False,
        clay=False,
        no_textures=False,
        workbench_materials=False,
        frame_count=2,
    )
    assert len(calls) == 2

    frames.mkdir(exist_ok=True)
    for i in range(2):
        (frames / f"frame_{i:03d}.png").write_bytes(b"png")
    monkeypatch.setattr(orbit.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    orbit.build_gif_from_frames(frames, tmp_path / "out.gif", duration_ms=500)
    assert len(calls) == 4

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="Не найдено кадров"):
        orbit.build_gif_from_frames(empty, tmp_path / "missing.gif", duration_ms=500)


def test_pillow_gif_fallback_and_binary_error_paths(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    for idx in range(3):
        (frames / f"frame_{idx:03d}.png").write_bytes(b"png")

    monkeypatch.setattr(orbit.shutil, "which", lambda name: None)

    saved = []
    closed = []

    class FakeImage:
        def __init__(self, path):
            self.path = str(path)

        def convert(self, mode):
            assert mode == "RGBA"
            return self

        def save(self, gif_path, **kwargs):
            saved.append((Path(gif_path), kwargs))
            Path(gif_path).write_bytes(b"gif")

        def close(self):
            closed.append(self.path)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()
            return False

    fake_image_module = types.SimpleNamespace(open=lambda path: FakeImage(path))
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_image_module
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    out_gif = tmp_path / "out.gif"
    orbit.build_gif_from_frames(frames, out_gif, duration_ms=250)

    assert out_gif.read_bytes() == b"gif"
    assert saved[0][0] == out_gif
    assert saved[0][1]["duration"] == 250
    assert len(closed) >= 3

    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="ffmpeg или Pillow"):
        orbit.build_gif_from_frames(frames, tmp_path / "no_pillow.gif", duration_ms=250)

    monkeypatch.setattr(orbit.shutil, "which", lambda name: None)
    monkeypatch.setattr(orbit.Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="Не найден Blender"):
        orbit.resolve_blender_binary(None)


def test_main_nonisolated_and_isolated_paths_are_mocked(tmp_path, monkeypatch, capsys):
    blend = tmp_path / "scene.blend"
    blend.write_text("blend", encoding="utf-8")
    existing_frames = tmp_path / "existing_frames"
    existing_frames.mkdir()
    gif = tmp_path / "scene.gif"
    calls = {"run": 0, "isolated": 0, "gif": 0, "rmtree": []}

    monkeypatch.setattr(orbit, "resolve_blender_binary", lambda arg: "/mock/Blender")
    monkeypatch.setattr(
        orbit,
        "run_blender_render",
        lambda **kwargs: calls.__setitem__("run", calls["run"] + 1)
        or kwargs["frames_dir"].mkdir(parents=True, exist_ok=True)
        or (kwargs["frames_dir"] / "frame_000.png").write_bytes(b"png"),
    )
    monkeypatch.setattr(
        orbit,
        "render_frames_isolated",
        lambda **kwargs: calls.__setitem__("isolated", calls["isolated"] + 1)
        or kwargs["frames_dir"].mkdir(parents=True, exist_ok=True)
        or (kwargs["frames_dir"] / "frame_000.png").write_bytes(b"png"),
    )
    monkeypatch.setattr(
        orbit,
        "build_gif_from_frames",
        lambda frames_dir, gif_path, duration_ms: calls.__setitem__("gif", calls["gif"] + 1)
        or Path(gif_path).write_bytes(b"gif"),
    )
    monkeypatch.setattr(
        orbit.shutil,
        "rmtree",
        lambda path, ignore_errors=False: calls["rmtree"].append((Path(path), ignore_errors)),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orbit",
            "--blend",
            str(blend),
            "--frames-dir",
            str(existing_frames),
            "--gif",
            str(gif),
            "--yaw-step",
            "180",
            "--elevations",
            "0",
            "--hide-room-shell",
            "--hide-outliers",
            "--no-textures",
            "--workbench-materials",
        ],
    )
    orbit.main()

    assert calls["run"] == 1
    assert calls["gif"] == 1
    assert calls["rmtree"][0][0] == existing_frames
    assert calls["rmtree"][-1] == (existing_frames, True)
    assert "получится 2 кадров" in capsys.readouterr().out

    isolated_frames = tmp_path / "isolated_frames"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orbit",
            "--blend",
            str(blend),
            "--frames-dir",
            str(isolated_frames),
            "--gif",
            str(gif),
            "--yaw-step",
            "180",
            "--elevations",
            "0",
            "--isolated-frames",
            "--keep-frames",
        ],
    )
    orbit.main()

    assert calls["isolated"] == 1
    assert calls["rmtree"][-1][0] != isolated_frames

    monkeypatch.setattr(sys, "argv", ["orbit", "--blend", str(tmp_path / "missing.blend")])
    with pytest.raises(RuntimeError, match="Не найден .blend"):
        orbit.main()

    monkeypatch.setattr(sys, "argv", ["orbit", "--blend", str(blend), "--yaw-step", "1000000", "--elevations", "0"])
    with pytest.raises(RuntimeError, match="Некорректное число"):
        orbit.main()
