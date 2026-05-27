from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src.tools import blender_supplier_asset as bsa


class FakeVector:
    def __init__(self, values):
        if isinstance(values, FakeVector):
            self.x = values.x
            self.y = values.y
            self.z = values.z
            return
        vals = list(values)
        self.x = float(vals[0])
        self.y = float(vals[1])
        self.z = float(vals[2])

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


class IdentityMatrix:
    def __matmul__(self, other):
        return FakeVector(other)


class FakeObject:
    def __init__(self, name, *, object_type="MESH", bounds=None):
        self.name = name
        self.type = object_type
        self.matrix_world = IdentityMatrix()
        self.bound_box = bounds or [
            (x, y, z)
            for x in (0.0, 2.0)
            for y in (0.0, 1.0)
            for z in (0.0, 1.0)
        ]
        self.scale = (1.0, 1.0, 1.0)
        self.location = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.data = types.SimpleNamespace(materials=[])

    def select_set(self, _value):
        return None


class RemovableList(list):
    def remove(self, item, *args, **kwargs):
        if item in self:
            super().remove(item)

    def unlink(self, item):
        self.remove(item)


class FakeCollection:
    def __init__(self, name, children=None):
        self.name = name
        self.children = RemovableList(children or [])


class FakeNodes(list):
    def get(self, name):
        if name == "Principled BSDF":
            return self[0]
        return None

    def new(self, type):
        node = types.SimpleNamespace(
            type=type,
            outputs={"Color": object(), "Generated": object(), "Vector": object()},
            inputs={"Base Color": types.SimpleNamespace(default_value=None), "Vector": object()},
            image=None,
        )
        self.append(node)
        return node


class FakeMaterials(RemovableList):
    def new(self, name):
        bsdf = types.SimpleNamespace(inputs={"Base Color": types.SimpleNamespace(default_value=None)})
        mat = types.SimpleNamespace(
            name=name,
            users=1,
            use_nodes=False,
            node_tree=types.SimpleNamespace(nodes=FakeNodes([bsdf]), links=types.SimpleNamespace(new=lambda *_: None)),
        )
        self.append(mat)
        return mat


def fake_bpy(tmp_path):
    calls = []
    mesh = FakeObject("Mesh")
    mat_collection = FakeCollection("mat_helpers")
    root_collection = FakeCollection("root", [mat_collection])
    objects = RemovableList([mesh, FakeObject("mat_preview"), FakeObject("Camera", object_type="CAMERA")])
    active_holder = types.SimpleNamespace(active=None)

    def cube_add(size, location):
        obj = FakeObject("Cube")
        obj.location = types.SimpleNamespace(x=location[0], y=location[1], z=location[2])
        objects.append(obj)
        bpy.context.active_object = obj

    bpy = types.SimpleNamespace()
    bpy.data = types.SimpleNamespace(
        meshes=RemovableList([types.SimpleNamespace(users=0)]),
        materials=FakeMaterials([types.SimpleNamespace(users=0)]),
        images=RemovableList([types.SimpleNamespace(users=0)]),
        objects=objects,
        collections=RemovableList([root_collection, mat_collection]),
    )
    bpy.context = types.SimpleNamespace(
        active_object=mesh,
        scene=types.SimpleNamespace(
            objects=objects,
            collection=types.SimpleNamespace(children=RemovableList([mat_collection])),
            unit_settings=types.SimpleNamespace(system="", scale_length=1.0),
        ),
        view_layer=types.SimpleNamespace(objects=active_holder),
    )
    bpy.ops = types.SimpleNamespace(
        object=types.SimpleNamespace(
            select_all=lambda **kwargs: calls.append(("select_all", kwargs)),
            delete=lambda **kwargs: calls.append(("delete", kwargs)),
            transform_apply=lambda **kwargs: calls.append(("transform_apply", kwargs)),
        ),
        wm=types.SimpleNamespace(
            open_mainfile=lambda filepath: calls.append(("open_blend", filepath)),
            obj_import=lambda filepath: calls.append(("obj_import", filepath)),
        ),
        import_scene=types.SimpleNamespace(
            fbx=lambda filepath: calls.append(("fbx_import", filepath)),
            gltf=lambda filepath: calls.append(("gltf_import", filepath)),
        ),
        mesh=types.SimpleNamespace(primitive_cube_add=cube_add),
        export_scene=types.SimpleNamespace(
            gltf=lambda **kwargs: calls.append(("export_glb", kwargs)),
            fbx=lambda **kwargs: calls.append(("export_fbx", kwargs)),
        ),
    )
    bpy.data.images.load = lambda path: types.SimpleNamespace(filepath=path)
    bpy._calls = calls
    return bpy


def test_blender_supplier_asset_helpers_use_fake_bpy(tmp_path, monkeypatch):
    bpy = fake_bpy(tmp_path)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=FakeVector))

    args = bsa._parse_args(["--mode", "proxy", "--width-m", "1"])
    assert args.mode == "proxy"

    bsa._clear_scene(bpy)
    assert not bpy.data.meshes

    obj_file = tmp_path / "model.obj"
    obj_file.write_text("o cube", encoding="utf-8")
    bsa._import_input(bpy, obj_file)
    assert ("obj_import", str(obj_file)) in bpy._calls
    bsa._import_input(bpy, tmp_path / "asset.fbx")
    bsa._import_input(bpy, tmp_path / "asset.glb")
    bsa._import_input(bpy, tmp_path / "asset.blend")
    with pytest.raises(RuntimeError, match="Unsupported input format"):
        bsa._import_input(bpy, tmp_path / "asset.max")

    bsa._remove_helper_material_objects(bpy)
    assert all(not obj.name.lower().startswith("mat_") for obj in bpy.context.scene.objects)

    bbox = bsa._compute_scene_bbox(bpy)
    assert bbox == ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    bsa._normalize_import_scale(bpy, width_m=4.0, depth_m=2.0, height_m=2.0)
    assert bpy.context.view_layer.objects.active is not None

    texture = tmp_path / "tex.jpg"
    texture.write_bytes(b"img")
    bsa._build_textured_proxy(bpy, 1.2, 0.8, 0.5, texture)
    assert bpy.context.active_object.name == "SupplierProxy"
    assert bpy.context.active_object.data.materials

    out_glb = tmp_path / "out" / "asset.glb"
    out_fbx = tmp_path / "out" / "asset.fbx"
    bsa._export_outputs(bpy, out_glb, out_fbx)
    assert any(call[0] == "export_glb" for call in bpy._calls)
    assert any(call[0] == "export_fbx" for call in bpy._calls)


def test_blender_supplier_asset_main_modes_are_mocked(tmp_path, monkeypatch):
    bpy = fake_bpy(tmp_path)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=FakeVector))

    source = tmp_path / "source.obj"
    source.write_text("o cube", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "blender_supplier_asset.py",
            "--",
            "--mode",
            "convert",
            "--input",
            str(source),
            "--width-m",
            "1",
            "--depth-m",
            "1",
            "--height-m",
            "1",
            "--out-glb",
            str(tmp_path / "converted.glb"),
        ],
    )
    bsa.main()
    assert bpy.context.scene.unit_settings.system == "METRIC"
    assert any(call[0] == "export_glb" for call in bpy._calls)

    monkeypatch.setattr(sys, "argv", ["blender_supplier_asset.py", "--", "--mode", "sanitize", "--out-glb", str(tmp_path / "bad.glb")])
    with pytest.raises(RuntimeError, match="--input is required"):
        bsa.main()
