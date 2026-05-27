from __future__ import annotations

import importlib
import json
import math
import sys
import types

import pytest


class FakeInput:
    def __init__(self):
        self.default_value = None


class FakeBsdf:
    def __init__(self):
        self.inputs = {
            "Base Color": FakeInput(),
            "Metallic": FakeInput(),
            "Roughness": FakeInput(),
            "Alpha": FakeInput(),
            "Transmission Weight": FakeInput(),
            "Transmission": FakeInput(),
        }


class FakeNodes:
    def __init__(self):
        self.bsdf = FakeBsdf()

    def get(self, name: str):
        return self.bsdf if name == "Principled BSDF" else None


class FakeMaterial:
    def __init__(self, name: str):
        self.name = name
        self.diffuse_color = None
        self.use_nodes = False
        self.node_tree = types.SimpleNamespace(nodes=FakeNodes())
        self.blend_method = None
        self.use_screen_refraction = False


class FakeMaterials(dict):
    def new(self, name: str):
        mat = FakeMaterial(name)
        self[name] = mat
        return mat


class FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self.linked = []
        self.objects = types.SimpleNamespace(link=self.link, unlink=self.unlink)

    def link(self, obj):
        self.linked.append(obj)
        if self not in obj.users_collection:
            obj.users_collection.append(self)

    def unlink(self, obj):
        if obj in self.linked:
            self.linked.remove(obj)
        if self in obj.users_collection:
            obj.users_collection.remove(self)


class FakeCollections(dict):
    def new(self, name: str):
        collection = FakeCollection(name)
        self[name] = collection
        return collection


class FakeModifiers:
    def __init__(self):
        self.created = []

    def new(self, name: str, kind: str):
        modifier = types.SimpleNamespace(name=name, type=kind)
        self.created.append(modifier)
        return modifier


class FakeObject(dict):
    def __init__(self, name: str = "obj"):
        super().__init__()
        self.name = name
        self.rotation_euler = [0.0, 0.0, 0.0]
        self.data = types.SimpleNamespace(materials=[])
        self.modifiers = FakeModifiers()
        self.users_collection = []
        self.location = (0.0, 0.0, 0.0)
        self.dimensions = (1.0, 1.0, 1.0)
        self.scale = (1.0, 1.0, 1.0)

    def __hash__(self):
        return id(self)

    def select_set(self, _selected: bool):
        return None


@pytest.fixture()
def factory_module(monkeypatch):
    module_name = "src.Plasement.procedural_object_factory_blender"
    sys.modules.pop(module_name, None)

    linked_collections = []
    fake_objects = []

    def add_object(kind: str, location=(0.0, 0.0, 0.0), **kwargs):
        obj = FakeObject(kind)
        obj.location = location
        obj.data = types.SimpleNamespace(materials=[])
        fake_objects.append(obj)
        fake_bpy.data.objects.append(obj)
        fake_bpy.context.object = obj
        return obj

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(
            materials=FakeMaterials(),
            collections=FakeCollections(),
            objects=[],
        ),
        context=types.SimpleNamespace(
            object=None,
            scene=types.SimpleNamespace(
                collection=types.SimpleNamespace(
                    children=types.SimpleNamespace(link=lambda col: linked_collections.append(col))
                ),
                render=types.SimpleNamespace(resolution_x=0, resolution_y=0),
                camera=None,
            ),
            view_layer=types.SimpleNamespace(objects=types.SimpleNamespace(active=None)),
        ),
        ops=types.SimpleNamespace(
            object=types.SimpleNamespace(
                select_all=lambda action=None: None,
                delete=lambda: None,
                transform_apply=lambda **_kwargs: None,
                shade_smooth=lambda: None,
                text_add=lambda **kwargs: add_object("Text", kwargs.get("location", (0.0, 0.0, 0.0))),
                light_add=lambda **kwargs: add_object("Light", kwargs.get("location", (0.0, 0.0, 0.0))),
                camera_add=lambda **kwargs: add_object("Camera", kwargs.get("location", (0.0, 0.0, 0.0))),
            ),
            mesh=types.SimpleNamespace(
                primitive_cube_add=lambda **kwargs: add_object("Cube", kwargs.get("location", (0.0, 0.0, 0.0))),
                primitive_cylinder_add=lambda **kwargs: add_object("Cylinder", kwargs.get("location", (0.0, 0.0, 0.0))),
                primitive_uv_sphere_add=lambda **kwargs: add_object("Sphere", kwargs.get("location", (0.0, 0.0, 0.0))),
                primitive_cone_add=lambda **kwargs: add_object("Cone", kwargs.get("location", (0.0, 0.0, 0.0))),
            ),
            wm=types.SimpleNamespace(save_as_mainfile=lambda **_kwargs: None),
        ),
        types=types.SimpleNamespace(Material=FakeMaterial, Collection=FakeCollection, Object=FakeObject),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=lambda values: values))

    module = importlib.import_module(module_name)
    module._MAT_CACHE.clear()
    module._FACTORY_CACHE = None
    module._TEST_FAKE_BPY = fake_bpy
    module._TEST_LINKED_COLLECTIONS = linked_collections
    module._TEST_FAKE_OBJECTS = fake_objects
    yield module
    sys.modules.pop(module_name, None)


def test_registry_color_parsing_and_factory_metadata(factory_module, monkeypatch):
    factory = factory_module

    assert factory.SUBCLASS_TO_BASE["queen_bed"] == "bed"
    assert "office_chair" in factory.TAXONOMY["chair"]
    assert factory.sanitize_name("шкаф №1 / test!") == "шкаф__1___test"
    assert factory.parse_color("#336699") == pytest.approx((0x33 / 255, 0x66 / 255, 0x99 / 255, 1.0))
    assert factory.parse_color("серый") == factory.COLOR_TABLE["gray"]
    assert factory.parse_color(None, fallback="wood") == factory.COLOR_TABLE["wood"]
    assert factory.lighten((0.5, 0.5, 0.5, 1.0), 0.2) == pytest.approx((0.6, 0.6, 0.6, 1.0))
    assert factory.darken((0.5, 0.5, 0.5, 1.0), 0.2) == pytest.approx((0.4, 0.4, 0.4, 1.0))

    monkeypatch.setattr(sys, "argv", ["script.py", "--", "--subclass", "chair"])
    assert factory.script_argv() == ["--subclass", "chair"]

    builder = object.__new__(factory.ProceduralObjectFactory)
    assert builder.infer_base_type("compact_microwave") == "microwave"
    assert builder.infer_base_type("unknown_custom") == "generic"
    assert builder.default_material("sofa", "straight_sofa") == "fabric"
    assert builder.default_material("refrigerator", "built_in_refrigerator") == "metal"
    dims = builder.dimensions("desk", factory.BuildSpec(subclass="compact_desk", width=1.2, depth=0.6, height=0.75))
    assert dims == pytest.approx((0.9, 0.6, 0.75))
    large = builder.dimensions("bed", factory.BuildSpec(subclass="king_bed"))
    assert large[0] > factory.DEFAULT_DIMS["bed"][0]


def test_material_collection_and_object_tag_helpers(factory_module):
    factory = factory_module
    fake_bpy = factory._TEST_FAKE_BPY

    metal = factory.make_material("steel material", (0.4, 0.4, 0.4, 1.0), "metal")
    cached = factory.make_material("steel material", (0.4, 0.4, 0.4, 1.0), "metal")
    assert cached is metal
    bsdf = metal.node_tree.nodes.bsdf
    assert bsdf.inputs["Metallic"].default_value == 0.8
    assert bsdf.inputs["Roughness"].default_value == 0.28

    glass = factory.make_material("glass material", (0.7, 0.9, 1.0, 0.3), "glass")
    assert glass.blend_method == "BLEND"
    assert glass.node_tree.nodes.bsdf.inputs["Alpha"].default_value == 0.3

    palette = factory.make_palette("oak", "wood")
    assert palette.base.name == "m_base"
    assert palette.black.diffuse_color == factory.COLOR_TABLE["black"]

    collection = factory.get_collection("unit_collection")
    assert collection.name == "unit_collection"
    assert collection in factory._TEST_LINKED_COLLECTIONS
    assert factory.get_collection("unit_collection") is collection

    obj = FakeObject("rotating")
    mat = FakeMaterial("manual")
    factory.apply_material(obj, mat)
    assert obj.data.materials == [mat]
    factory.rotate_z(obj, 90)
    factory.rotate_x(obj, 45)
    assert obj.rotation_euler[2] == pytest.approx(math.pi / 2)
    assert obj.rotation_euler[0] == pytest.approx(math.pi / 4)
    factory.add_bevel(obj, amount=0.02, segments=3)
    assert [modifier.type for modifier in obj.modifiers.created] == ["BEVEL", "WEIGHTED_NORMAL"]

    spec = factory.BuildSpec(subclass="office_chair")
    factory.tag_object(obj, spec, "chair", "seat")
    assert obj["taxonomy_schema"] == "interior_object_taxonomy/v2"
    assert obj["taxonomy_subclass"] == "office_chair"
    assert obj["taxonomy_base_type"] == "chair"
    assert obj["procedural_part"] == "seat"

    old = FakeObject("old")
    new = FakeObject("new")
    fake_bpy.data.objects[:] = [old, new]
    tagged = factory.tag_created({old}, spec, "chair")
    assert tagged == [new]
    assert new["taxonomy_base_type"] == "chair"


def test_primitive_parts_helpers_create_and_link_fake_meshes(factory_module):
    factory = factory_module
    col = factory.get_collection("primitive_unit")
    palette = factory.make_palette("oak", "wood")
    parts = factory.Parts(col, palette)

    cube = factory.cube_obj("Cube/Name", (1, 2, 3), (0.4, 0.5, 0.6), palette.base, col)
    cyl = factory.cylinder_obj("Cylinder", (0, 0, 0.5), 0.1, 1.0, palette.metal, col, bevel=0.01)
    sphere = factory.sphere_obj("Sphere", (0, 0, 1), (0.2, 0.3, 0.4), palette.fabric, col)
    cone = factory.cone_obj("Cone", (0, 0, 1), 0.2, 0.1, 0.5, palette.base, col)

    assert cube.name == "Cube_Name"
    assert cube.dimensions == (0.4, 0.5, 0.6)
    assert cyl.data.materials == [palette.metal]
    assert sphere.scale == (0.2, 0.3, 0.4)
    assert cone in col.linked

    parts.four_legs("unit", (0, 0, 0), 1.0, 1.0, 0.5)
    parts.drawer_faces("drawer", (0, 0, 0), 1.0, 0.5, 0.8, drawers=2)
    parts.door_pair("door", (0, 0, 0), 1.0, 0.5, 1.2)
    parts.burners("cooktop", (0, 0, 0), 1.0, 0.8, 0.1, count=5)
    parts.screen("screen", (0, 0, 0), 1.0, 0.05, 0.6)
    assert len(factory._TEST_FAKE_BPY.data.objects) >= 20


def test_factory_builds_every_registered_subclass_with_fake_blender(factory_module):
    factory = factory_module
    pairs = factory.all_subclasses()
    assert pairs

    for index, (base_type, subclass) in enumerate(pairs):
        created = factory.build_taxonomy_object(
            subclass,
            base_type=base_type,
            color="#8a7662",
            loc=(float(index % 9), float(index // 9), 0.0),
            collection_name="all_registered_subclasses",
        )
        assert created, (base_type, subclass)
        assert all(obj["taxonomy_schema"] == "interior_object_taxonomy/v2" for obj in created)
        assert all(obj["taxonomy_subclass"] == subclass for obj in created)
        assert all(obj["taxonomy_base_type"] == base_type for obj in created)


def test_catalog_inference_dynamic_builder_cli_and_json_helpers(factory_module, tmp_path):
    factory = factory_module

    assert factory.infer_material_from_catalog_item({"title": "metal chrome lamp"}) == "metal"
    assert factory.infer_material_from_catalog_item({"description": "бархатная ткань"}) == "fabric"
    assert factory.infer_material_from_catalog_item({"description": "oak wood mdf"}) == "wood"
    assert factory.infer_material_from_catalog_item({"description": "фарфор"}) == "ceramic"
    assert factory.infer_material_from_catalog_item({"description": "unknown"}) == "fabric"

    assert factory.infer_subclass_from_catalog_item({"title": "угловой диван"}) == "straight_sofa"
    assert factory.infer_subclass_from_catalog_item({"title": "подвес бра"}) == "single_pendant_lamp"
    assert factory.infer_subclass_from_catalog_item({"title": "not matched"}, fallback="fallback_subclass") == "fallback_subclass"

    item = {
        "title": "Oak desk",
        "category_norm": "desk",
        "width_cm": 120,
        "depth_cm": 60,
        "height_cm": 75,
        "image_color_features": {"colors": {"top5": [{"hex": "#445566"}]}},
        "materials": "oak wood",
    }
    created = factory.build_from_catalog_item(item, collection_name="catalog_unit")
    assert created
    assert created[0]["taxonomy_base_type"] == "desk"
    assert created[0]["taxonomy_subclass"] == "writing_desk"

    dynamic_builder = factory._make_subclass_builder("office_chair", "chair")
    dynamic_created = dynamic_builder(collection_name="dynamic_unit")
    assert dynamic_created[0]["taxonomy_subclass"] == "office_chair"
    assert getattr(factory, "build_office_chair")(collection_name="dynamic_unit_2")[0]["taxonomy_base_type"] == "chair"

    parser = factory.build_cli()
    args = parser.parse_args(["--subclass", "office_chair", "--width", "0.5", "--build-all"])
    assert args.subclass == "office_chair"
    assert args.width == 0.5
    assert args.build_all is True

    path = tmp_path / "data.json"
    factory.write_json(path, {"items": [{"title": "chair"}]})
    assert factory.read_json(path) == {"items": [{"title": "chair"}]}


def test_factory_error_edges_preview_and_cli_main_paths(factory_module, tmp_path, monkeypatch, capsys):
    factory = factory_module
    fake_bpy = factory._TEST_FAKE_BPY

    assert factory.parse_color((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4)

    old_collection = types.SimpleNamespace(objects=types.SimpleNamespace(unlink=lambda _obj: (_ for _ in ()).throw(RuntimeError("unlink"))))
    obj = FakeObject("linked")
    obj.users_collection.append(old_collection)
    target_col = factory.get_collection("link_target")
    assert factory.link_to_collection(obj, target_col) is obj
    assert obj in target_col.linked

    class BadSelect(FakeObject):
        def select_set(self, _selected: bool):
            raise RuntimeError("select failed")

    bad_select = BadSelect("bad")
    assert factory.shade_smooth(bad_select) is bad_select
    assert factory.add_bevel(FakeObject("flat"), amount=0) is not None

    bad_mod = FakeObject("bad_mod")
    bad_mod.modifiers = types.SimpleNamespace(new=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("modifier")))
    assert factory.add_bevel(bad_mod, amount=0.1) is bad_mod

    created = factory.build_taxonomy_object("unmapped_custom_proxy", collection_name="generic_build")
    assert created[0]["taxonomy_base_type"] == "generic"
    wet = factory.build_taxonomy_object("wet_room_floor_drain", base_type="shower", collection_name="wet_room")
    assert any("wet_room_drain" in obj.name for obj in wet)
    assert factory.infer_material_from_catalog_item({"description": "leather upholstery"}) == "leather"
    assert factory.infer_material_from_catalog_item({"description": "clear glass crystal"}) == "glass"

    label_col = factory.get_collection("labels")
    factory.add_label("Label text", (0, 0, 0), label_col)
    assert any(obj.name.startswith("label_") for obj in label_col.linked)
    original_text_add = fake_bpy.ops.object.text_add
    fake_bpy.ops.object.text_add = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("text"))
    factory.add_label("broken", (0, 0, 0), label_col)
    fake_bpy.ops.object.text_add = original_text_add

    factory.setup_preview(count=3, grid_cols=2, sx=1.0, sy=1.2)
    assert fake_bpy.context.scene.render.resolution_x == 1800
    assert fake_bpy.context.scene.camera.name == "preview_camera"

    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"items": [{"title": "Oak desk", "category_norm": "desk"}]}), encoding="utf-8")
    report = tmp_path / "report.json"
    out_blend = tmp_path / "catalog.blend"
    monkeypatch.setattr(sys, "argv", ["script.py", "--", "--catalog-json", str(catalog), "--catalog-limit", "2", "--setup-preview", "--grid-cols", "1", "--out-blend", str(out_blend), "--report-json", str(report), "--clear-scene"])
    factory.main()
    assert json.loads(report.read_text(encoding="utf-8"))["items"][0]["mode"] == "catalog"

    build_all_report = tmp_path / "build_all.json"
    monkeypatch.setattr(sys, "argv", ["script.py", "--", "--build-all", "--limit", "1", "--out-blend", str(tmp_path / "all.blend"), "--report-json", str(build_all_report)])
    factory.main()
    assert json.loads(build_all_report.read_text(encoding="utf-8"))["items"][0]["mode"] == "taxonomy"

    single_report = tmp_path / "single.json"
    monkeypatch.setattr(sys, "argv", ["script.py", "--", "--subclass", "office_chair", "--setup-preview", "--out-blend", str(tmp_path / "single.blend"), "--report-json", str(single_report)])
    factory.main()
    assert json.loads(single_report.read_text(encoding="utf-8"))["items"][0]["mode"] == "single"
    assert "items = 1" in capsys.readouterr().out

    bad_catalog = tmp_path / "bad_catalog.json"
    bad_catalog.write_text(json.dumps({"items": {"bad": True}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["script.py", "--", "--catalog-json", str(bad_catalog), "--out-blend", str(tmp_path / "bad.blend")])
    with pytest.raises(RuntimeError, match="catalog JSON"):
        factory.main()
