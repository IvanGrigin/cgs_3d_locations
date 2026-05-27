from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

from src.suppliers.kitchen import kitchen_blender_builder as kitchen


class FakeVector:
    def __init__(self, values):
        if isinstance(values, FakeVector):
            self.x = values.x
            self.y = values.y
            self.z = values.z
            return
        vals = list(values)
        self.x = float(vals[0]) if len(vals) > 0 else 0.0
        self.y = float(vals[1]) if len(vals) > 1 else 0.0
        self.z = float(vals[2]) if len(vals) > 2 else 0.0

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def __neg__(self):
        return FakeVector((-self.x, -self.y, -self.z))

    def __add__(self, other):
        other = FakeVector(other)
        return FakeVector((self.x + other.x, self.y + other.y, self.z + other.z))

    def __sub__(self, other):
        other = FakeVector(other)
        return FakeVector((self.x - other.x, self.y - other.y, self.z - other.z))

    def __mul__(self, value):
        return FakeVector((self.x * float(value), self.y * float(value), self.z * float(value)))

    __rmul__ = __mul__

    def __iadd__(self, other):
        other = FakeVector(other)
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def copy(self):
        return FakeVector(self)


class IdentityMatrix:
    def __matmul__(self, other):
        if isinstance(other, IdentityMatrix):
            return self
        return FakeVector(other)

    def copy(self):
        return self

    def inverted(self):
        return self


class FakeMatrix:
    @staticmethod
    def Identity(_size: int):
        return IdentityMatrix()

    @staticmethod
    def Translation(_vector):
        return IdentityMatrix()

    @staticmethod
    def Rotation(_angle, _size, _axis):
        return IdentityMatrix()

    @staticmethod
    def Diagonal(_values):
        return IdentityMatrix()


class FakeVertex:
    def __init__(self, co):
        self.co = FakeVector(co)


class FakeMesh:
    def __init__(self, vertices):
        self.vertices = [FakeVertex(vertex) for vertex in vertices]
        self.materials = []

    def copy(self):
        return FakeMesh([(v.co.x, v.co.y, v.co.z) for v in self.vertices])

    def transform(self, _matrix):
        return None

    def update(self):
        return None


class FakeModifier:
    def __init__(self, name: str, modifier_type: str):
        self.name = name
        self.type = modifier_type
        self.operation = ""
        self.object = None
        self.solver = ""


class FakeModifiers(list):
    def new(self, name: str, type: str):
        modifier = FakeModifier(name, type)
        self.append(modifier)
        return modifier

    def remove(self, modifier):
        if modifier in self:
            super().remove(modifier)


class FakeObject(dict):
    def __init__(
        self,
        name: str,
        vertices=((0.0, 0.0, 0.0),),
        *,
        object_type: str = "MESH",
        dimensions: tuple[float, float, float] | None = None,
        parent=None,
    ):
        super().__init__()
        self.name = name
        self.type = object_type
        self.data = FakeMesh(vertices) if object_type == "MESH" else None
        self.parent = parent
        self.children = []
        self.location = FakeVector((0.0, 0.0, 0.0))
        self.rotation_euler = [0.0, 0.0, 0.0]
        self.scale = (1.0, 1.0, 1.0)
        self.matrix_world = IdentityMatrix()
        self.material_slots = []
        self.users_collection = []
        self.modifiers = FakeModifiers()
        if dimensions is None:
            xs = [float(v[0]) for v in vertices]
            ys = [float(v[1]) for v in vertices]
            zs = [float(v[2]) for v in vertices]
            dimensions = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        self.dimensions = FakeVector(dimensions)
        self.bound_box = self._make_bound_box(vertices)

    @staticmethod
    def _make_bound_box(vertices):
        xs = [float(v[0]) for v in vertices]
        ys = [float(v[1]) for v in vertices]
        zs = [float(v[2]) for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        return [
            (x, y, z)
            for x in (min_x, max_x)
            for y in (min_y, max_y)
            for z in (min_z, max_z)
        ]

    def copy(self):
        verts = [(v.co.x, v.co.y, v.co.z) for v in self.data.vertices] if self.data else [(0.0, 0.0, 0.0)]
        return FakeObject(self.name, verts, object_type=self.type, parent=self.parent)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def select_set(self, _selected: bool):
        return None


class FakeMaterial(dict):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.diffuse_color = (0.8, 0.8, 0.8, 1.0)
        self.use_nodes = False
        self.node_tree = None


class FakeMaterials(dict):
    def new(self, name: str):
        material = FakeMaterial(name)
        self[name] = material
        return material


def make_box(name: str, mins: tuple[float, float, float], maxs: tuple[float, float, float]) -> FakeObject:
    min_x, min_y, min_z = mins
    max_x, max_y, max_z = maxs
    vertices = [
        (x, y, z)
        for x in (min_x, max_x)
        for y in (min_y, max_y)
        for z in (min_z, max_z)
    ]
    return FakeObject(name, vertices)


@pytest.fixture(autouse=True)
def fake_mathutils(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "mathutils",
        types.SimpleNamespace(Vector=FakeVector, Matrix=FakeMatrix),
    )


def test_material_texture_helpers_use_fake_bpy(tmp_path, monkeypatch):
    texture = tmp_path / "texture.jpg"
    texture.write_bytes(b"fake image")
    monkeypatch.chdir(tmp_path)

    assert kitchen._visual_color(None) == (0.8, 0.8, 0.8, 1.0)
    assert kitchen._visual_color({"chosen_material": {"visual": {"base_colors": ["black"]}}}) == (0.03, 0.03, 0.03, 1.0)
    assert kitchen._visual_color({"visual": {"base_colors": ["light_wood"]}}) == (0.68, 0.52, 0.34, 1.0)
    assert kitchen._visual_color({"visual": {"pattern": "marble"}}) == (0.68, 0.68, 0.66, 1.0)
    assert kitchen._resolve_texture_path({"chosen_material": {"local_image": "texture.jpg"}}) == texture.resolve()
    assert kitchen._resolve_texture_path({"chosen_material": {"local_image": "missing.jpg"}}) is None

    updates = []
    fake_bpy = types.SimpleNamespace(
        path=types.SimpleNamespace(abspath=lambda raw: str(Path(raw))),
        data=types.SimpleNamespace(materials=FakeMaterials()),
        context=types.SimpleNamespace(view_layer=types.SimpleNamespace(update=lambda: updates.append("updated"))),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)

    missing = types.SimpleNamespace(filepath=str(tmp_path / "missing.png"), packed_file=None)
    packed = types.SimpleNamespace(filepath=str(tmp_path / "missing.png"), packed_file=object())
    existing = types.SimpleNamespace(filepath=str(texture), packed_file=None)
    assert kitchen._image_path_missing(missing) is True
    assert kitchen._image_path_missing(packed) is False
    assert kitchen._image_path_missing(existing) is False

    broken_node = types.SimpleNamespace(type="TEX_IMAGE", image=missing)
    broken_material = types.SimpleNamespace(
        diffuse_color=(0.8, 0.1, 0.8, 1.0),
        node_tree=types.SimpleNamespace(nodes=[broken_node]),
    )
    assert kitchen._material_looks_magenta_missing(broken_material) is True
    assert kitchen._material_has_missing_texture(broken_material) is True

    slot = types.SimpleNamespace(material=broken_material)
    obj = FakeObject("broken_material_object", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
    obj.material_slots = [slot]
    assert kitchen._replace_missing_texture_materials([obj]) == 1
    assert slot.material.name == "kitchen_missing_texture_neutral_silver"
    assert updates == ["updated"]


def test_geometry_bbox_polygon_and_opening_helpers():
    obj = make_box("countertop", (0.0, 0.0, 0.0), (2.0, 1.0, 0.2))

    assert kitchen._convex_hull_xy([(0, 0), (1, 0), (0.5, 0.5), (1, 1), (0, 1)]) == [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
    ]
    simplified = kitchen._simplify_polygon_xy([(0, 0), (0.001, 0), (1, 0), (1, 1), (0, 1)], min_edge_m=0.01)
    assert (0.001, 0) not in simplified

    assert kitchen._bbox_world([obj]) == ((0.0, 0.0, 0.0), (2.0, 1.0, 0.2))
    assert kitchen._objects_fit_within_size([obj], (2.0, 1.0, 0.2)) is True
    assert kitchen._objects_fit_within_size([obj], (1.0, 1.0, 0.2), tolerance=1.0) is False
    assert kitchen._mesh_xy_bbox_below_z([obj], 0.01) == ((0.0, 0.0), (2.0, 1.0))
    assert kitchen._mesh_xy_bbox_between_z([obj], 0.19, 0.21) == ((0.0, 0.0), (2.0, 1.0))

    point_cloud = FakeObject(
        "dense_points",
        [(float(i), float(i % 5), 0.5) for i in range(10)],
    )
    inner = kitchen._mesh_xy_inner_bbox_between_z([point_cloud], 0.4, 0.6)
    assert inner == ((1.0, 0.0), (7.0, 3.0))

    hull = kitchen._mesh_outer_hull_xy_from_objects([obj], sample_z_min=0.0, sample_z_max=0.2, inset_m=0.0)
    assert len(hull) == 4
    polygon = kitchen._mesh_outer_polygon_xy_from_objects([obj], sample_z_min=0.0, sample_z_max=0.2, inset_m=0.0)
    assert len(polygon) >= 3

    opening = kitchen._real_bbox_opening_from_objects([obj], inset_x=0.1, inset_y=0.2)
    assert opening == ((1.0, 0.5), (1.8, 0.6))

    scaled = kitchen._scale_polygon_xy([(0, 0), (1, 0), (1, 1), (0, 1)], 0.1)
    for actual, expected in zip(scaled, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]):
        assert actual == pytest.approx(expected)
    fitted = kitchen._fit_polygon_xy_to_opening([(0, 0), (10, 0), (10, 10), (0, 10)], (5, 5), (2, 2))
    xs = [point[0] for point in fitted]
    ys = [point[1] for point in fitted]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((3.98, 6.02, 3.98, 6.02))


def test_assets_filtering_orientation_and_faucet_helpers(monkeypatch):
    chosen = {
        "asset_local_path": "/tmp/a.glb",
        "unique_key": "asset-a",
        "title": "Sink with mixer faucet",
        "asset_format": "fbx",
    }
    duplicate = {**chosen}
    second = {"asset_local_path": "/tmp/b.glb", "unique_key": "asset-b", "name": "plain sink"}
    assembly = {
        "appliance_bindings": {
            "appliances": {
                "sink": {"chosen_asset": chosen, "top_candidates": [duplicate, second]},
            }
        }
    }

    assert kitchen._appliance_asset(assembly, "sink") == chosen
    assert kitchen._appliance_asset_candidates(assembly, "sink") == [chosen, second]
    assert kitchen._sink_asset_includes_faucet(chosen) is True
    assert kitchen._asset_title(second) == "plain sink"

    deleted = []
    monkeypatch.setattr(kitchen, "_delete_objects", lambda objects: deleted.extend(objects))
    sink_small = make_box("sink_small", (0.0, 0.0, 0.0), (0.4, 0.4, 0.1))
    sink_large = make_box("sink_large", (0.0, 0.0, 0.0), (0.8, 0.5, 0.2))
    cube = make_box("Cube001", (0.0, 0.0, 0.0), (0.3, 0.3, 0.3))
    chosen_objects = kitchen._filter_imported_appliance_objects("sink", [sink_small, sink_large, cube], chosen)
    assert chosen_objects == [sink_large]
    assert deleted == [sink_small, cube]

    root = FakeObject("kitchen_appliance_asset_root_import", [(0.0, 0.0, 0.0)], object_type="EMPTY")
    child = make_box("child", (0.0, 0.0, 0.0), (0.2, 0.2, 0.2))
    child.parent = root
    loose = make_box("loose", (0.0, 0.0, 0.0), (0.2, 0.2, 0.2))
    assert kitchen._translation_roots([child, loose]) == [root, loose]

    kitchen._rotate_imported_roots_z([child], math.pi / 4.0)
    assert root.rotation_euler[2] == pytest.approx(math.pi / 4.0)
    kitchen._orient_countertop_appliance_front([loose], "y")
    assert loose.rotation_euler[2] == pytest.approx(-math.pi / 2.0)
    kitchen._apply_asset_import_orientation(
        {"blender_import": {"rotation_z_deg_by_layout": {"x": 90}}},
        [loose],
        "x",
    )
    assert loose.rotation_euler[2] == pytest.approx(0.0)
    assert kitchen._asset_rotation_z_deg({"blender_import": {"rotation_z_deg_by_layout": {"y": -45}}}, "y") == -45.0

    base = make_box("faucet_base", (-0.03, -0.03, 0.0), (0.03, 0.03, 0.04))
    neck = make_box("faucet_neck", (0.18, -0.01, 0.2), (0.22, 0.01, 0.9))
    assert kitchen._faucet_base_anchor_xy([base, neck], (0.0, 0.0)) == pytest.approx((0.0, 0.0))
    assert kitchen._faucet_lowest_mount_xy([base, neck]) == pytest.approx((0.0, 0.0))
    direction = kitchen._faucet_direction_xy([base, neck], (0.0, 0.0))
    assert direction is not None
    assert direction[0] > 0.95
    assert abs(direction[1]) < 0.05
    assert kitchen._signed_angle_between_xy((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)


def test_appliance_filter_material_sanitizing_and_dense_outer_polygon(monkeypatch):
    deleted = []
    monkeypatch.setattr(kitchen, "_delete_objects", lambda objects: deleted.extend(objects))

    preferred_high = make_box("Faucet_part_13", (-0.05, -0.05, 8.8), (0.05, 0.05, 9.3))
    preferred_low = make_box("Faucet_part_14", (-0.05, -0.05, 0.0), (0.05, 0.05, 0.4))
    non_preferred = make_box("Faucet_part_2", (-0.05, -0.05, 8.8), (0.05, 0.05, 9.3))
    cube = make_box("Cube_helper", (0.0, 0.0, 0.0), (0.2, 0.2, 0.2))
    faucet_kept = kitchen._filter_imported_appliance_objects("faucet", [preferred_high, preferred_low, non_preferred, cube])
    assert faucet_kept == [preferred_high]
    assert preferred_low in deleted and non_preferred in deleted and cube in deleted

    mats: dict[str, FakeMaterial] = {}

    def fake_material(name, color, texture_path=None):
        del color, texture_path
        mats.setdefault(name, FakeMaterial(name))
        return mats[name]

    monkeypatch.setattr(kitchen, "_get_or_create_material", fake_material)
    sink = make_box("sink", (0, 0, 0), (1, 1, 0.2))
    cooktop_glass = make_box("glass", (0, 0, 0), (1, 1, 0.05))
    cooktop_trim = make_box("box_trim", (0, 0, 0), (1, 1, 0.05))
    fridge_display = make_box("display_shape", (0, 0, 0), (1, 1, 2))
    fridge_handle = make_box("cylinder_handle", (0, 0, 0), (1, 1, 2))
    fridge_body = make_box("door_panel", (0, 0, 0), (1, 1, 2))

    kitchen._sanitize_imported_appliance_materials("sink", [sink, FakeObject("empty", object_type="EMPTY")])
    assert sink.data.materials[0].name == "kitchen_sink_asset_dark_pvd"
    kitchen._sanitize_imported_appliance_materials("cooktop", [cooktop_glass, cooktop_trim])
    assert cooktop_glass.data.materials[0].name == "kitchen_cooktop_asset_black_glass"
    assert cooktop_trim.data.materials[0].name == "kitchen_cooktop_asset_dark_trim"
    kitchen._sanitize_imported_appliance_materials("fridge", [fridge_display, fridge_handle, fridge_body])
    assert fridge_display.data.materials[0].name == "kitchen_fridge_asset_dark_display"
    assert fridge_handle.data.materials[0].name == "kitchen_fridge_asset_warm_gray_trim"
    assert fridge_body.data.materials[0].name == "kitchen_fridge_asset_satin_white"

    dense = FakeObject(
        "round_sink_rim",
        [
            (math.cos(i * math.tau / 32.0), math.sin(i * math.tau / 32.0), 0.5)
            for i in range(32)
        ],
    )
    polygon = kitchen._mesh_outer_polygon_xy_from_objects([dense], sample_z_min=0.4, sample_z_max=0.6, inset_m=0.02, radial_bins=32)
    assert len(polygon) >= 8
    assert max(x for x, _y in polygon) < 1.0


def test_surface_and_cooktop_cutout_helpers():
    assert kitchen._surface_point(10, 20, 1, 2, 0.5, 0.25, "x") == (10.5, 22.25)
    assert kitchen._surface_point(10, 20, 1, 2, 0.5, 0.25, "y") == (11.25, 20.5)
    assert kitchen._surface_point_local(10, 20, 1, 2, 0.5, 0.25, "x") == (11.5, 22.25)
    assert kitchen._surface_point_local(10, 20, 1, 2, 0.5, 0.25, "y") == (11.25, 22.5)

    assembly = {
        "countertop_segments": [
            {
                "orientation": "x",
                "x_m": 1.0,
                "y_m": 2.0,
                "cutouts": [
                    {
                        "type": "cooktop",
                        "module_id": "cooktop_1",
                        "x_m": 0.1,
                        "y_m": 0.2,
                        "width_m": 0.6,
                        "depth_m": 0.4,
                    }
                ],
            }
        ]
    }
    assert kitchen._find_cooktop_cutout_center(assembly, "cooktop_1", 10.0, 20.0) == (
        11.4,
        22.4,
        0.6,
        0.4,
        "x",
    )
    assert kitchen._find_cooktop_cutout_center(assembly, "missing", 10.0, 20.0) is None
    assert kitchen._find_cooktop_cutout_center(assembly, None, 10.0, 20.0) is None


def test_blender_primitive_material_import_and_cutout_paths(tmp_path, monkeypatch):
    fake_bpy, objects, _scene_collection = _install_fake_bpy_for_primitives(monkeypatch)
    collection = FakeCollection("Kitchen")
    texture = tmp_path / "wood.jpg"
    texture.write_bytes(b"img")

    mat = kitchen._get_or_create_material("wood", (0.4, 0.3, 0.2, 1.0), texture)
    assert mat["basisrf_texture_path"] == str(texture)
    assert kitchen._get_or_create_material("wood", (1, 1, 1, 1)) is mat
    emission = kitchen._get_or_create_emission_material("led", (1, 1, 1, 1), 2.5)
    assert emission.use_nodes is True

    cube = kitchen._create_box("cabinet_box", (1, 2, 0.5), (2, 1, 1), mat, collection)
    cyl = kitchen._create_cylinder("leg", (0, 0, 0.5), 0.05, 1.0, mat, collection, vertices=12)
    torus = kitchen._create_torus("handle", (0, 0, 1), 0.2, 0.02, mat, collection)
    assert [cube.name, cyl.name, torus.name] == ["cabinet_box", "leg", "handle"]
    assert cube in collection.objects and cyl in collection.objects and torus in collection.objects

    target = make_box("counter", (0, 0, 0), (2, 1, 0.1))
    assert kitchen._apply_rectangular_cutout(target, "sink", (1, 0.5, 0.05), (0.5, 0.4, 0.2), collection)
    assert kitchen._apply_mesh_objects_cutout(target, "mesh_sink", [make_box("sink_mesh", (0, 0, 0), (0.4, 0.4, 0.2))], collection)
    assert kitchen._apply_polygon_cutout(target, "poly_sink", [(0, 0), (1, 0), (1, 1), (0, 1)], cutter_z_min=-0.1, cutter_z_max=0.2, collection=collection)
    assert kitchen._apply_polygon_cutout(target, "bad", [(0, 0), (1, 0)], cutter_z_min=0, cutter_z_max=1, collection=collection) is False

    existing = tmp_path / "asset.glb"
    existing.write_bytes(b"glb")
    imported = kitchen._import_asset_objects(str(existing), collection)
    assert imported and imported[0].name == "asset"
    assert kitchen._import_asset_objects(str(tmp_path / "missing.glb"), collection) == []

    created_collection = kitchen._collection("NewKitchen")
    assert fake_bpy.data.collections["NewKitchen"] is created_collection
    assert objects


def test_fit_transform_and_baked_mesh_paths(monkeypatch):
    _fake_bpy, _objects, _scene_collection = _install_fake_bpy_for_primitives(monkeypatch)
    obj = make_box("appliance", (0, 0, 0), (1, 1, 1))
    child = make_box("child", (2, 0, 0), (3, 1, 1))
    child.parent = obj

    object_group = [obj, child]
    assert kitchen._fit_objects_to_box(object_group, (2, 2, 0.5), (1, 1, 1), margin=0.9)
    assert any(getattr(o, "name", "") == "kitchen_appliance_asset_root" for o in object_group)

    footprint = make_box("footprint", (0, 0, 0), (2, 1, 0.5))
    assert kitchen._fit_objects_to_footprint([footprint], (1, 1), (1.5, 0.8), bottom_z=0.2)
    top = make_box("top", (0, 0, 0), (2, 1, 0.5))
    assert kitchen._fit_objects_to_footprint_top([top], (1, 1), (1.5, 0.8), top_z=1.0)

    baked_a = make_box("baked_a", (0, 0, 0), (1, 1, 1))
    baked_b = make_box("baked_b", (10, 0, 0), (11, 1, 1))
    assert kitchen._fit_mesh_objects_to_box_baked([baked_a, baked_b], (0, 0, 0.5), (1, 1, 1), compact_disconnected=True)
    kitchen._rotate_baked_mesh_objects_around_point_z([baked_a], (0.5, 0.5), 90)
    kitchen._translate_baked_mesh_objects_xy([baked_a], 0.1, 0.2)
    kitchen._translate_baked_mesh_objects_z([baked_a], 0.3)
    assert kitchen._snap_baked_mesh_objects_bottom_to_z([baked_a], 0.0) is True
    assert kitchen._snap_baked_mesh_objects_bottom_to_z([], 0.0) is False


def test_remaining_low_level_edge_branches(monkeypatch, tmp_path, capsys):
    assert kitchen._image_path_missing(None) is False
    assert kitchen._image_path_missing(types.SimpleNamespace(filepath="", packed_file=None)) is False
    assert kitchen._material_looks_magenta_missing(types.SimpleNamespace(diffuse_color=None)) is False
    assert kitchen._material_has_missing_texture(types.SimpleNamespace(node_tree=None)) is False

    fake_bpy, _objects, scene_collection = _install_fake_bpy_for_primitives(monkeypatch)
    neutral = FakeMaterial("neutral")
    monkeypatch.setattr(kitchen, "_get_or_create_material", lambda *_args, **_kwargs: neutral)
    clean_mat = types.SimpleNamespace(diffuse_color=(0.2, 0.2, 0.2, 1.0), node_tree=types.SimpleNamespace(nodes=[]))
    obj = FakeObject("mat_edges")
    obj.material_slots = [types.SimpleNamespace(material=None), types.SimpleNamespace(material=clean_mat)]
    assert kitchen._replace_missing_texture_materials([FakeObject("empty", object_type="EMPTY"), obj]) == 0

    dead_bpy = types.SimpleNamespace(
        ops=types.SimpleNamespace(
            mesh=types.SimpleNamespace(
                primitive_cube_add=lambda **_kwargs: None,
                primitive_cylinder_add=lambda **_kwargs: None,
                primitive_torus_add=lambda **_kwargs: None,
            )
        ),
        context=types.SimpleNamespace(object=None, view_layer=types.SimpleNamespace(objects=types.SimpleNamespace(active=None))),
    )
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: dead_bpy)
    with pytest.raises(RuntimeError, match="cube object"):
        kitchen._create_box("bad_box", (0, 0, 0), (1, 1, 1))
    with pytest.raises(RuntimeError, match="cylinder object"):
        kitchen._create_cylinder("bad_cyl", (0, 0, 0), 1, 1)
    with pytest.raises(RuntimeError, match="torus object"):
        kitchen._create_torus("bad_torus", (0, 0, 0), 1, 0.1)

    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    assert kitchen._apply_mesh_objects_cutout(FakeObject("target"), "empty", [FakeObject("empty", object_type="EMPTY")]) is False

    class RaisingModifiers(FakeModifiers):
        def remove(self, _modifier):
            raise RuntimeError("remove failed")

    target = make_box("target", (0, 0, 0), (1, 1, 1))
    target.modifiers = RaisingModifiers()
    fake_bpy.ops.object.modifier_apply = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("apply failed"))
    source = make_box("source", (0, 0, 0), (0.5, 0.5, 0.5))
    assert kitchen._apply_mesh_objects_cutout(target, "bad_mesh", [source], collection=None) is False
    assert "failed to apply mesh cutout bad_mesh" in capsys.readouterr().out
    assert kitchen._apply_polygon_cutout(target, "bad_poly", [(0, 0), (1, 0), (1, 1)], cutter_z_min=0, cutter_z_max=1, collection=None) is False

    monkeypatch.setattr(kitchen, "_apply_polygon_cutout", lambda *args, **kwargs: ("poly", args, kwargs))
    assert kitchen._apply_mesh_footprint_cutout(source, "foot", [source], sample_z_min=0, sample_z_max=1, cutter_z_min=-1, cutter_z_max=1) [0] == "poly"

    assert kitchen._convex_hull_xy([(0, 0), (1, 1)]) == [(0, 0), (1, 1)]
    assert kitchen._simplify_polygon_xy([(0, 0), (0.001, 0), (0.002, 0), (0.003, 0)], min_edge_m=0.01) == [(0, 0)]
    assert kitchen._mesh_outer_polygon_xy_from_objects([], sample_z_min=0, sample_z_max=1) == []
    centered_points = FakeObject("centered", [(0, 0, 0.5)] * 16)
    assert kitchen._mesh_outer_polygon_xy_from_objects([centered_points], sample_z_min=0, sample_z_max=1) == []
    monkeypatch.setattr(kitchen, "_bbox_world", lambda _objects: None)
    triangle = FakeObject("triangle", [(0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5), (0.2, 0.2, 0.5)])
    assert kitchen._mesh_outer_hull_xy_from_objects([triangle], sample_z_min=0, sample_z_max=1, inset_m=0.0)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=FakeVector, Matrix=FakeMatrix))

    real_delete_objects = kitchen._delete_objects
    deleted: list[str] = []
    monkeypatch.setattr(kitchen, "_delete_objects", lambda objects: deleted.extend(obj.name for obj in objects))
    sink_a = make_box("sink_a", (0, 0, 0), (0.3, 0.3, 0.1))
    sink_b = make_box("sink_b", (0.4, 0, 0), (0.7, 0.3, 0.1))
    assert kitchen._filter_imported_appliance_objects("sink", [sink_a, sink_b], {"asset_format": "obj"}) == [sink_a, sink_b]
    assert deleted == []
    assert kitchen._filter_imported_appliance_objects("unknown", [sink_a]) == [sink_a]

    fake_bpy, objects, _scene_collection = _install_fake_bpy_for_primitives(monkeypatch)
    existing_collection = FakeCollection("Existing")
    fake_bpy.data.collections["Existing"] = existing_collection
    assert kitchen._collection("Existing") is existing_collection

    obj_path = tmp_path / "asset.obj"
    obj_path.write_bytes(b"obj")
    fake_bpy.ops.wm = types.SimpleNamespace()
    imported_obj = kitchen._import_asset_objects(str(obj_path), scene_collection)
    assert imported_obj and imported_obj[0].name == "asset"
    bad_path = tmp_path / "bad.bad"
    bad_path.write_bytes(b"bad")
    assert kitchen._import_asset_objects(str(bad_path), scene_collection) == []
    fake_bpy.ops.import_scene.gltf = lambda filepath: (_ for _ in ()).throw(RuntimeError("import failed"))
    glb_path = tmp_path / "bad.glb"
    glb_path.write_bytes(b"glb")
    assert kitchen._import_asset_objects(str(glb_path), scene_collection) == []

    class NameAwareObjects(FakeObjectStore):
        def __contains__(self, value):
            if isinstance(value, str):
                return any(obj.name == value for obj in self)
            return super().__contains__(value)

    name_objects = NameAwareObjects()
    fake_bpy.data.objects = name_objects
    parent = FakeObject("kitchen_appliance_asset_root_parent", object_type="EMPTY")
    child = FakeObject("child", [(0, 0, 0), (1, 1, 1)], parent=parent)
    parent.children = [child]
    name_objects.extend([parent, child])
    real_delete_objects([child])
    assert child not in name_objects
    assert parent not in name_objects

    assert kitchen._bbox_world([FakeObject("empty", object_type="EMPTY")]) is None
    assert kitchen._objects_fit_within_size([make_box("flat", (0, 0, 0), (2, 2, 2))], (0, 2, 2)) is True
    assert kitchen._snap_objects_bottom_to_z([], 0.0) is False
    assert kitchen._mesh_xy_bbox_below_z([FakeObject("empty", object_type="EMPTY")], 1.0) is None
    high_vertices_low_bbox = make_box("bbox_low", (0, 0, 0), (1, 1, 1))
    high_vertices_low_bbox.data.vertices = [FakeVertex((0.2, 0.2, 2.0))]
    assert kitchen._mesh_xy_bbox_below_z([high_vertices_low_bbox], 0.5) == ((0.0, 0.0), (1.0, 1.0))
    assert kitchen._mesh_xy_bbox_between_z([FakeObject("empty", object_type="EMPTY")], 0, 1) is None
    assert kitchen._mesh_xy_bbox_between_z([high_vertices_low_bbox], 0, 0.5) == ((0.0, 0.0), (1.0, 1.0))
    assert kitchen._mesh_xy_inner_bbox_between_z([high_vertices_low_bbox], 0, 0.5) == ((0.0, 0.0), (1.0, 1.0))

    nested_parent = FakeObject("root", object_type="EMPTY")
    nested_child = FakeObject("nested", parent=nested_parent)
    assert kitchen._translation_roots([nested_child, nested_parent]) == [nested_parent]
    assert kitchen._fit_objects_to_box([], (0, 0, 0), (1, 1, 1)) is False
    assert kitchen._fit_mesh_objects_to_box_baked([FakeObject("empty", object_type="EMPTY")], (0, 0, 0), (1, 1, 1)) is False
    assert kitchen._fit_objects_to_footprint([], (0, 0), (1, 1), 0) is False
    assert kitchen._fit_objects_to_footprint_top([], (0, 0), (1, 1), 1) is False

    wall_obj = FakeObject("wall")
    kitchen._orient_wall_appliance_front([wall_obj], "y")
    assert wall_obj.rotation_euler[2] == pytest.approx(-math.pi / 2.0)
    kitchen._apply_asset_import_orientation({}, [wall_obj], "x")
    assert kitchen._asset_rotation_z_deg({}, "x") == 0.0
    kitchen._rotate_baked_mesh_objects_around_point_z([wall_obj], (0, 0), 0)
    kitchen._translate_baked_mesh_objects_xy([wall_obj], 0, 0)
    kitchen._translate_baked_mesh_objects_z([wall_obj], 0)

    empty_mesh = FakeObject("empty_mesh")
    empty_mesh.data = FakeMesh([])
    empty_mesh.bound_box = []
    assert kitchen._faucet_lowest_mount_xy([empty_mesh]) is None
    assert kitchen._faucet_direction_xy([], (0, 0)) is None
    symmetric = FakeObject("symmetric", [(-1, 0, 1), (1, 0, 1)])
    assert kitchen._faucet_direction_xy([symmetric], (0, 0)) is None
    assert kitchen._scale_polygon_xy([], 0.1) == []
    assert kitchen._fit_polygon_xy_to_opening([(0, 0), (1, 0)], (2, 2), (1, 1)) == [(0, 0), (1, 0)]


class FakeLinkedObjects(list):
    def link(self, obj):
        self.append(obj)
        if hasattr(obj, "users_collection"):
            obj.users_collection.append(getattr(self, "owner", self))

    def unlink(self, obj):
        if obj in self:
            self.remove(obj)


class FakeCollection:
    def __init__(self, name: str = "Kitchen"):
        self.name = name
        self.objects = FakeLinkedObjects()
        self.objects.owner = self


class FakeObjectStore(list):
    def new(self, name, data):
        obj = FakeObject(name, object_type=("EMPTY" if data is None else "MESH"))
        obj.data = data
        self.append(obj)
        return obj

    def remove(self, obj, do_unlink=True):
        if obj in self:
            super().remove(obj)


class FakeMeshStore:
    def new(self, name):
        mesh = FakeMesh([(0.0, 0.0, 0.0)])
        mesh.name = name

        def from_pydata(vertices, _edges, _faces):
            mesh.vertices = [FakeVertex(vertex) for vertex in vertices]

        mesh.from_pydata = from_pydata
        return mesh


class FakeCollectionStore(dict):
    def new(self, name):
        collection = FakeCollection(name)
        self[name] = collection
        return collection


def _install_fake_bpy_for_primitives(monkeypatch):
    objects = FakeObjectStore()
    scene_collection = FakeCollection("Scene")
    active = types.SimpleNamespace(active=None)
    context = types.SimpleNamespace(
        object=None,
        scene=types.SimpleNamespace(collection=types.SimpleNamespace(objects=scene_collection.objects, children=types.SimpleNamespace(link=lambda _c: None))),
        view_layer=types.SimpleNamespace(objects=active, update=lambda: None),
    )

    def set_created(obj):
        objects.append(obj)
        context.object = obj
        context.view_layer.objects.active = obj
        scene_collection.objects.link(obj)
        return obj

    def cube_add(size=1.0, location=(0, 0, 0)):
        half = size / 2.0
        obj = make_box("Cube", (location[0] - half, location[1] - half, location[2] - half), (location[0] + half, location[1] + half, location[2] + half))
        return set_created(obj)

    def cylinder_add(vertices=32, radius=0.5, depth=1.0, location=(0, 0, 0)):
        del vertices
        obj = make_box("Cylinder", (location[0] - radius, location[1] - radius, location[2] - depth / 2), (location[0] + radius, location[1] + radius, location[2] + depth / 2))
        return set_created(obj)

    def torus_add(major_radius=1.0, minor_radius=0.1, major_segments=72, minor_segments=8, location=(0, 0, 0)):
        del major_segments, minor_segments
        r = major_radius + minor_radius
        obj = make_box("Torus", (location[0] - r, location[1] - r, location[2] - minor_radius), (location[0] + r, location[1] + r, location[2] + minor_radius))
        return set_created(obj)

    class Nodes(dict):
        def __init__(self):
            bsdf = types.SimpleNamespace(
                inputs={
                    "Base Color": types.SimpleNamespace(default_value=None),
                    "Roughness": types.SimpleNamespace(default_value=None),
                }
            )
            super().__init__({"Principled BSDF": bsdf})

        def clear(self):
            super().clear()

        def new(self, type):
            node = types.SimpleNamespace(
                type=type,
                name=type,
                inputs={
                    "Color": types.SimpleNamespace(default_value=None),
                    "Strength": types.SimpleNamespace(default_value=None),
                    "Surface": types.SimpleNamespace(default_value=None),
                },
                outputs={
                    "Color": types.SimpleNamespace(),
                    "Emission": types.SimpleNamespace(),
                },
            )
            self[type] = node
            return node

    class Materials(FakeMaterials):
        def new(self, name):
            mat = super().new(name)
            mat.node_tree = types.SimpleNamespace(nodes=Nodes(), links=types.SimpleNamespace(new=lambda *_args: None))
            return mat

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(
            objects=objects,
            meshes=FakeMeshStore(),
            collections=FakeCollectionStore(),
            materials=Materials(),
            images=types.SimpleNamespace(load=lambda path, check_existing=True: types.SimpleNamespace(filepath=path)),
        ),
        context=context,
        ops=types.SimpleNamespace(
            mesh=types.SimpleNamespace(
                primitive_cube_add=cube_add,
                primitive_cylinder_add=cylinder_add,
                primitive_torus_add=torus_add,
            ),
            object=types.SimpleNamespace(
                transform_apply=lambda **_kwargs: None,
                modifier_apply=lambda **_kwargs: None,
            ),
            import_scene=types.SimpleNamespace(
                fbx=lambda filepath: set_created(make_box(Path(filepath).stem, (0, 0, 0), (1, 1, 1))),
                obj=lambda filepath: set_created(make_box(Path(filepath).stem, (0, 0, 0), (1, 1, 1))),
                gltf=lambda filepath: set_created(make_box(Path(filepath).stem, (0, 0, 0), (1, 1, 1))),
            ),
            wm=types.SimpleNamespace(obj_import=lambda filepath: set_created(make_box(Path(filepath).stem, (0, 0, 0), (1, 1, 1)))),
        ),
        path=types.SimpleNamespace(abspath=lambda raw: str(raw)),
    )
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    return fake_bpy, objects, scene_collection


def _fake_box_object(
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material=None,
    collection=None,
) -> FakeObject:
    cx, cy, cz = center
    sx, sy, sz = size
    obj = make_box(
        name,
        (cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0),
        (cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0),
    )
    obj.location = FakeVector(center)
    obj.dimensions = FakeVector(size)
    if material is not None and obj.data is not None:
        obj.data.materials.append(material)
    if collection is not None:
        collection.objects.link(obj)
    return obj


def test_build_kitchen_assembly_exercises_major_branches(monkeypatch):
    collection = FakeCollection()
    created_names: list[str] = []

    def fake_create_box(name, center, size, material=None, collection_arg=None):
        obj = _fake_box_object(name, center, size, material, collection_arg or collection)
        created_names.append(obj.name)
        return obj

    def fake_create_oriented_box(name, origin_x, origin_y, width, depth, height, z, orientation, material, collection_arg):
        if orientation == "y":
            center = (origin_x + depth / 2.0, origin_y + width / 2.0, z + height / 2.0)
            size = (depth, width, height)
        else:
            center = (origin_x + width / 2.0, origin_y + depth / 2.0, z + height / 2.0)
            size = (width, depth, height)
        return fake_create_box(name, center, size, material, collection_arg)

    def fake_create_cylinder(name, center, radius, depth, material=None, collection_arg=None, vertices=32):
        del vertices
        return fake_create_box(name, center, (radius * 2.0, radius * 2.0, depth), material, collection_arg)

    def fake_create_torus(name, center, major_radius, minor_radius, material=None, collection_arg=None):
        return fake_create_box(
            name,
            center,
            (major_radius * 2.0, major_radius * 2.0, minor_radius * 2.0),
            material,
            collection_arg,
        )

    def fake_imported(_assembly, role, name, center, size, fallback_mat, collection_arg, **_kwargs):
        obj = fake_create_box(f"{name}_{role}_asset", center, size, fallback_mat, collection_arg)
        obj["kitchen_appliance_role"] = role
        return [obj]

    def fake_imported_decor(_assembly, role, name, center, size, _bottom_z, fallback_mat, collection_arg, **_kwargs):
        return fake_imported(_assembly, role, name, center, size, fallback_mat, collection_arg)

    class FakeLights(dict):
        def new(self, name, light_type):
            light = types.SimpleNamespace(name=name, type=light_type, energy=0.0, size=0.0)
            self[name] = light
            return light

    class FakeObjects:
        def new(self, name, data):
            obj = FakeObject(name, object_type="LIGHT")
            obj.data = data
            return obj

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(lights=FakeLights(), objects=FakeObjects()),
        context=types.SimpleNamespace(view_layer=types.SimpleNamespace(update=lambda: None)),
    )
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    monkeypatch.setattr(kitchen, "_collection", lambda _name: collection)
    monkeypatch.setattr(kitchen, "_get_or_create_material", lambda name, *_args, **_kwargs: FakeMaterial(name))
    monkeypatch.setattr(kitchen, "_get_or_create_emission_material", lambda name, *_args, **_kwargs: FakeMaterial(name))
    monkeypatch.setattr(kitchen, "_create_box", fake_create_box)
    monkeypatch.setattr(kitchen, "_create_oriented_box", fake_create_oriented_box)
    monkeypatch.setattr(kitchen, "_create_cylinder", fake_create_cylinder)
    monkeypatch.setattr(kitchen, "_create_torus", fake_create_torus)
    monkeypatch.setattr(kitchen, "_apply_rectangular_cutout", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_create_or_import_appliance", fake_imported)
    monkeypatch.setattr(kitchen, "_create_or_import_countertop_decor_asset", fake_imported_decor)

    assembly = {
        "position": [1.0, 2.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "base_modules": [
            {"id": "fridge", "type": "fridge_slot", "x_m": 0.0, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.7, "height_m": 1.8, "z_m": 0.0},
            {"id": "dishwasher", "type": "dishwasher_slot", "x_m": 0.7, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1},
            {"id": "washer", "type": "washing_machine_slot", "x_m": 1.4, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "orientation": "y"},
            {"id": "drawers", "x_m": 2.1, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "facade_layout": "three_drawers"},
            {"id": "doors", "x_m": 2.8, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "facade_layout": "two_doors", "orientation": "y"},
            {"id": "ovenbase", "x_m": 3.5, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "facade_layout": "oven_front"},
            {"id": "plain", "x_m": 4.2, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1},
            {"id": "open", "x_m": 4.9, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "has_facade": False},
        ],
        "countertop_segments": [
            {
                "id": "counter",
                "x_m": 2.0,
                "y_m": 0.0,
                "width_m": 3.2,
                "depth_m": 0.62,
                "thickness_m": 0.04,
                "z_m": 0.82,
                "cutouts": [
                    {"type": "sink", "module_id": "drawers", "x_m": 0.15, "y_m": 0.12, "width_m": 0.48, "depth_m": 0.36},
                    {"type": "entry_handwash", "x_m": 0.82, "y_m": 0.12, "width_m": 0.38, "depth_m": 0.30},
                    {"type": "cooktop", "module_id": "cooktop_1", "x_m": 1.35, "y_m": 0.10, "width_m": 0.58, "depth_m": 0.42},
                ],
            }
        ],
        "backsplash_segments": [
            {"id": "backsplash_x", "x_m": 2.0, "y_m": 0.0, "width_m": 2.0, "height_m": 0.55},
            {"id": "backsplash_y", "x_m": 0.0, "y_m": 0.0, "width_m": 1.2, "height_m": 0.55, "orientation": "y"},
        ],
        "upper_modules": [
            {"id": "hoodcab", "type": "hood_cabinet", "x_m": 3.0, "y_m": 0.0, "width_m": 0.8, "depth_m": 0.35, "height_m": 0.7, "above_base_module_id": "cooktop_1"},
            {"id": "mw_shelf", "type": "microwave_open_shelf", "x_m": 3.9, "y_m": 0.0, "width_m": 0.7, "depth_m": 0.34, "height_m": 0.45, "orientation": "y"},
            {"id": "upper", "x_m": 4.7, "y_m": 0.0, "width_m": 0.7, "depth_m": 0.34, "height_m": 0.65},
        ],
        "decor_items": [
            {"id": "board", "type": "cutting_board", "x_m": 2.2, "y_m": 0.2, "z_m": 0.86},
            {"id": "micro", "type": "microwave", "x_m": 2.5, "y_m": 0.2, "z_m": 0.86, "placement": "countertop"},
            {"id": "small", "type": "small_kitchen_appliance", "x_m": 2.8, "y_m": 0.2, "z_m": 0.86, "orientation": "y"},
            {"id": "vase", "type": "flowers_vase", "x_m": 3.1, "y_m": 0.2, "z_m": 0.86},
            {"id": "oil", "type": "oil_bottles_decor", "x_m": 3.4, "y_m": 0.2, "z_m": 0.86},
            {"id": "set", "type": "decorative_kitchen_set", "x_m": 3.7, "y_m": 0.2, "z_m": 0.86},
            {"id": "ignored", "type": "unknown", "x_m": 4.0, "y_m": 0.2, "z_m": 0.86},
        ],
    }

    created = kitchen.build_kitchen_assembly_in_blender(assembly)

    assert len(created) >= 60
    assert "fridge_display" in created_names
    assert "counter_sink_visible_insert_drain" in created_names
    assert "counter_cooktop_visible_flush_burner_4" in created_names
    assert "hoodcab_hood_chimney" in created_names
    assert any(obj.get("kitchen_under_cabinet_lighting") for obj in created)


def test_build_kitchen_assembly_import_asset_orientation_variants(monkeypatch):
    collection = FakeCollection()
    created_names: list[str] = []

    def fake_create_box(name, center, size, material=None, collection_arg=None):
        obj = _fake_box_object(name, center, size, material, collection_arg or collection)
        created_names.append(obj.name)
        return obj

    def fake_create_oriented_box(name, origin_x, origin_y, width, depth, height, z, orientation, material, collection_arg):
        if orientation == "y":
            center = (origin_x + depth / 2.0, origin_y + width / 2.0, z + height / 2.0)
            size = (depth, width, height)
        else:
            center = (origin_x + width / 2.0, origin_y + depth / 2.0, z + height / 2.0)
            size = (width, depth, height)
        return fake_create_box(name, center, size, material, collection_arg)

    def fake_imported(assembly, role, name, center, size, fallback_mat, collection_arg, **_kwargs):
        del assembly
        obj = fake_create_box(f"{name}_{role}_asset", center, size, fallback_mat, collection_arg)
        obj["kitchen_appliance_role"] = role
        return [obj]

    class FakeObjects:
        def new(self, name, data):
            obj = FakeObject(name, object_type="LIGHT" if data is not None else "EMPTY")
            obj.data = data
            return obj

    class FakeLights(dict):
        def new(self, name, light_type):
            light = types.SimpleNamespace(name=name, type=light_type, energy=0.0, size=0.0)
            self[name] = light
            return light

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(objects=FakeObjects(), lights=FakeLights()),
        context=types.SimpleNamespace(view_layer=types.SimpleNamespace(update=lambda: None)),
    )
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    monkeypatch.setattr(kitchen, "_collection", lambda _name: collection)
    monkeypatch.setattr(kitchen, "_get_or_create_material", lambda name, *_args, **_kwargs: FakeMaterial(name))
    monkeypatch.setattr(kitchen, "_get_or_create_emission_material", lambda name, *_args, **_kwargs: FakeMaterial(name))
    monkeypatch.setattr(kitchen, "_create_box", fake_create_box)
    monkeypatch.setattr(kitchen, "_create_oriented_box", fake_create_oriented_box)
    monkeypatch.setattr(kitchen, "_create_cylinder", lambda name, center, radius, depth, material=None, collection_arg=None, vertices=32: fake_create_box(name, center, (radius * 2.0, radius * 2.0, depth), material, collection_arg))
    monkeypatch.setattr(kitchen, "_create_torus", lambda name, center, major_radius, minor_radius, material=None, collection_arg=None: fake_create_box(name, center, (major_radius * 2.0, major_radius * 2.0, minor_radius * 2.0), material, collection_arg))
    monkeypatch.setattr(kitchen, "_apply_rectangular_cutout", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_apply_polygon_cutout", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_mesh_outer_hull_xy_from_objects", lambda *_args, **_kwargs: [(0.2, 0.2), (0.6, 0.2), (0.6, 0.5), (0.2, 0.5)])
    monkeypatch.setattr(kitchen, "_real_bbox_opening_from_objects", lambda _objects: ((0.4, 0.35), (0.35, 0.25)))
    monkeypatch.setattr(kitchen, "_create_polygon_sink_backing_basin", lambda name, polygon, z, material, collection_arg: [fake_create_box(name, (0.4, 0.35, z - 0.07), (0.3, 0.2, 0.14), material, collection_arg)])
    monkeypatch.setattr(kitchen, "_create_sink_backing_basin", lambda name, center, size, z, material, collection_arg: [fake_create_box(name, (center[0], center[1], z - 0.07), (size[0], size[1], 0.14), material, collection_arg)])
    monkeypatch.setattr(kitchen, "_create_or_import_appliance", fake_imported)
    monkeypatch.setattr(kitchen, "_sink_asset_includes_faucet", lambda _asset: False)

    assembly = {
        "position": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "appliance_bindings": {
            "appliances": {
                "fridge": {"chosen_asset": {"asset_local_path": "/fridge.glb"}},
                "sink": {"chosen_asset": {"asset_local_path": "/sink.glb", "title": "black pvd sink"}},
                "faucet": {"chosen_asset": {"asset_local_path": "/faucet.glb"}},
                "cooktop": {"chosen_asset": {"asset_local_path": "/cooktop.glb"}},
                "hood": {"chosen_asset": {"asset_local_path": "/hood.glb"}},
            }
        },
        "base_modules": [
            {"id": "fridge_asset", "type": "fridge_slot", "x_m": 0.0, "y_m": 0.0, "width_m": 0.7, "depth_m": 0.65, "height_m": 1.8, "z_m": 0.0, "orientation": "y"},
            {"id": "drawers_y", "x_m": 1.0, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "orientation": "y", "facade_layout": "three_drawers"},
            {"id": "doors_x", "x_m": 1.7, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "facade_layout": "two_doors"},
            {"id": "plain_y", "x_m": 2.4, "y_m": 0.0, "width_m": 0.6, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1, "orientation": "y"},
        ],
        "countertop_segments": [
            {
                "id": "counter_y",
                "x_m": 1.0,
                "y_m": 0.0,
                "width_m": 1.8,
                "depth_m": 0.62,
                "thickness_m": 0.04,
                "z_m": 0.82,
                "orientation": "y",
                "cutouts": [
                    {"type": "sink", "module_id": "drawers_y", "x_m": 0.10, "y_m": 0.10, "width_m": 0.50, "depth_m": 0.35},
                    {"type": "cooktop", "module_id": "cooktop_y", "x_m": 0.80, "y_m": 0.10, "width_m": 0.50, "depth_m": 0.36},
                ],
            }
        ],
        "upper_modules": [
            {"id": "hood_y", "type": "hood_cabinet", "x_m": 1.0, "y_m": 0.0, "width_m": 0.7, "depth_m": 0.35, "height_m": 0.65, "z_m": 1.45, "orientation": "y"},
            {"id": "mw_x", "type": "microwave_open_shelf", "x_m": 1.8, "y_m": 0.0, "width_m": 0.7, "depth_m": 0.34, "height_m": 0.45},
            {"id": "upper_y", "x_m": 2.6, "y_m": 0.0, "width_m": 0.7, "depth_m": 0.34, "height_m": 0.65, "orientation": "y"},
        ],
    }

    created = kitchen.build_kitchen_assembly_in_blender(assembly)

    assert any(obj.get("kitchen_appliance_role") == "fridge" for obj in created)
    assert any(obj.get("kitchen_appliance_role") == "sink" for obj in created)
    assert any(obj.get("kitchen_appliance_role") == "faucet" for obj in created)
    assert any(obj.get("kitchen_appliance_role") == "cooktop" for obj in created)
    assert any(obj.get("kitchen_appliance_role") == "hood" for obj in created)
    assert "drawers_y_drawer_1" in created_names
    assert "doors_x_door_1" in created_names
    assert "plain_y_facade" in created_names
    assert "mw_x_back" in created_names
    assert "upper_y_facade" in created_names


def test_create_or_import_appliance_success_reject_and_role_branches(monkeypatch):
    assets = [
        {"asset_local_path": "/bad.glb", "unique_key": "bad", "title": "bad asset"},
        {"asset_local_path": "/good.glb", "unique_key": "good", "title": "good asset"},
    ]
    assembly = {"appliance_bindings": {"appliances": {"microwave": {"top_candidates": assets}}}}
    imported_by_path: dict[str, list[FakeObject]] = {}
    deleted: list[str] = []
    fit_calls = {"box": 0}

    def fake_import(path, collection=None):
        obj = make_box(Path(path).stem, (0, 0, 0), (0.2, 0.2, 0.2))
        imported_by_path[path] = [obj]
        return [obj]

    def fake_fit_box(objects, _center, _size, margin=1.0):
        del margin
        fit_calls["box"] += 1
        return bool(objects and objects[0].name == "good")

    monkeypatch.setattr(kitchen, "_import_asset_objects", fake_import)
    monkeypatch.setattr(kitchen, "_filter_imported_appliance_objects", lambda _role, objects, _asset: objects)
    monkeypatch.setattr(kitchen, "_fit_objects_to_box", fake_fit_box)
    monkeypatch.setattr(kitchen, "_objects_fit_within_size", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_delete_objects", lambda objects: deleted.extend(obj.name for obj in objects))
    monkeypatch.setattr(kitchen, "_sanitize_imported_appliance_materials", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kitchen, "_replace_missing_texture_materials", lambda _objects: 0)
    monkeypatch.setattr(kitchen, "_create_box", lambda name, center, size, material=None, collection=None: _fake_box_object(name, center, size, material, collection))

    result = kitchen._create_or_import_appliance(
        assembly,
        "microwave",
        "micro",
        (1, 2, 3),
        (0.4, 0.3, 0.2),
        FakeMaterial("fallback"),
        FakeCollection(),
    )

    assert deleted == ["bad"]
    assert result[0].name == "micro_good"
    assert result[0]["supplier_unique_key"] == "good"

    for role, fit_name in [("sink", "_fit_objects_to_footprint_top"), ("cooktop", "_fit_objects_to_footprint_top")]:
        monkeypatch.setattr(kitchen, "_appliance_asset_candidates", lambda _assembly, _role: [assets[1]])
        monkeypatch.setattr(kitchen, fit_name, lambda *_args, **_kwargs: True)
        result = kitchen._create_or_import_appliance(
            {},
            role,
            role,
            (1, 2, 3),
            (0.4, 0.3, 0.2),
            FakeMaterial("fallback"),
            FakeCollection(),
        )
        assert result[0]["kitchen_appliance_role"] == role

    rotations: list[tuple[tuple[float, float], float]] = []
    translations: list[tuple[float, float]] = []
    monkeypatch.setattr(kitchen, "_appliance_asset_candidates", lambda _assembly, _role: [assets[1]])
    monkeypatch.setattr(kitchen, "_fit_mesh_objects_to_box_baked", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_faucet_lowest_mount_xy", lambda _objects: (0.8, 2.0))
    monkeypatch.setattr(kitchen, "_faucet_base_anchor_xy", lambda _objects, _fallback: (0.8, 2.0))
    monkeypatch.setattr(kitchen, "_translate_baked_mesh_objects_xy", lambda _objects, dx, dy: translations.append((dx, dy)))
    monkeypatch.setattr(kitchen, "_rotate_baked_mesh_objects_around_point_z", lambda _objects, point, angle: rotations.append((point, angle)))
    monkeypatch.setattr(kitchen, "_faucet_direction_xy", lambda _objects, _anchor: (1.0, 0.0))
    faucet = kitchen._create_or_import_appliance(
        {},
        "faucet",
        "faucet",
        (1, 2, 3),
        (0.4, 0.3, 0.2),
        FakeMaterial("fallback"),
        FakeCollection(),
        aim_xy=(1.0, 3.0),
    )
    assert faucet[0]["kitchen_appliance_role"] == "faucet"
    assert translations
    assert rotations


def test_countertop_decor_import_fallback_polygon_basin_and_rotation(monkeypatch):
    asset = {"asset_local_path": "/decor.glb", "unique_key": "decor", "title": "Decor"}
    assembly = {"appliance_bindings": {"appliances": {"flowers_vase": {"chosen_asset": asset}}}}
    collection = FakeCollection()
    monkeypatch.setattr(kitchen, "_import_asset_objects", lambda *_args, **_kwargs: [make_box("decor", (0, 0, 0), (0.2, 0.2, 0.2))])
    monkeypatch.setattr(kitchen, "_filter_imported_appliance_objects", lambda _role, objects, _asset: objects)
    monkeypatch.setattr(kitchen, "_apply_asset_import_orientation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kitchen, "_fit_mesh_objects_to_box_baked", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_snap_baked_mesh_objects_bottom_to_z", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_snap_objects_bottom_to_z", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_replace_missing_texture_materials", lambda _objects: 0)
    monkeypatch.setattr(kitchen, "_create_box", lambda name, center, size, material=None, collection=None: _fake_box_object(name, center, size, material, collection))

    imported = kitchen._create_or_import_countertop_decor_asset(
        assembly,
        "flowers_vase",
        "vase",
        (1, 2, 3),
        (0.3, 0.3, 0.4),
        2.8,
        FakeMaterial("fallback"),
        collection,
    )
    assert imported[0]["supplier_unique_key"] == "decor"

    fallback = kitchen._create_or_import_countertop_decor_asset(
        {},
        "flowers_vase",
        "vase_fallback",
        (1, 2, 3),
        (0.3, 0.3, 0.4),
        2.8,
        FakeMaterial("fallback"),
        collection,
    )
    assert fallback[0].name == "vase_fallback"

    class FakeMeshData:
        def __init__(self, name):
            self.name = name
            self.vertices = []
            self.faces = []
            self.materials = []

        def from_pydata(self, vertices, _edges, faces):
            self.vertices = vertices
            self.faces = faces

        def update(self):
            return None

    class FakeMeshes:
        def new(self, name):
            return FakeMeshData(name)

    class FakeObjects:
        def new(self, name, mesh):
            obj = FakeObject(name)
            obj.data = mesh
            return obj

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(meshes=FakeMeshes(), objects=FakeObjects()),
        context=types.SimpleNamespace(scene=types.SimpleNamespace(collection=FakeCollection())),
    )
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    basin = kitchen._create_polygon_sink_backing_basin(
        "basin",
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        0.9,
        FakeMaterial("basin"),
        collection,
    )
    assert basin[0].name == "basin"
    assert basin[0]["kitchen_appliance_role"] == "sink"

    rot_obj = FakeObject("rot")
    rot_obj.location = FakeVector((2.0, 0.0, 0.0))
    rot_obj.rotation_euler = types.SimpleNamespace(z=0.0)
    kitchen._apply_assembly_rotation([rot_obj], {"rotation": [0.0, 0.0, 90.0]}, (0.0, 0.0, 0.0))
    assert rot_obj.location.x == pytest.approx(0.0, abs=1e-8)
    assert rot_obj.location.y == pytest.approx(2.0)
    assert rot_obj.rotation_euler.z == pytest.approx(math.radians(90.0))


def test_low_level_kitchen_error_and_placeholder_edges(monkeypatch, tmp_path):
    fake_bpy, objects, _scene_collection = _install_fake_bpy_for_primitives(monkeypatch)
    collection = FakeCollection()

    assert kitchen._visual_color({"visual": {"base_colors": ["gray"]}}) == (0.45, 0.45, 0.45, 1.0)
    assert kitchen._visual_color({"visual": {"base_colors": ["white"]}}) == (0.92, 0.90, 0.86, 1.0)
    assert kitchen._visual_color({"visual": {"base_colors": ["beige"]}}) == (0.72, 0.63, 0.50, 1.0)
    assert kitchen._visual_color({"visual": {"base_colors": ["dark_wood"]}}) == (0.25, 0.14, 0.08, 1.0)
    assert kitchen._visual_color({"visual": {"base_colors": ["green"]}}) == (0.75, 0.75, 0.72, 1.0)

    mat = kitchen._get_or_create_material("cached", (1, 1, 1, 1))
    assert kitchen._get_or_create_material("cached", (0, 0, 0, 1)) is mat
    assert kitchen._resolve_texture_path({"chosen_material": {"local_image": ""}}) is None
    assert kitchen._material_looks_magenta_missing(types.SimpleNamespace(diffuse_color="bad")) is False
    assert kitchen._material_has_missing_texture(types.SimpleNamespace(node_tree=object())) is False

    target = make_box("target", (0, 0, 0), (1, 1, 0.2))
    fake_bpy.ops.object.modifier_apply = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("apply failed"))
    assert kitchen._apply_rectangular_cutout(target, "bad_rect", (0.5, 0.5, 0.1), (0.2, 0.2, 0.4), collection) is False
    assert kitchen._apply_mesh_objects_cutout(target, "bad_mesh", [make_box("source", (0, 0, 0), (0.4, 0.4, 0.2))], collection) is False
    assert kitchen._apply_polygon_cutout(
        target,
        "bad_poly",
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        cutter_z_min=-0.1,
        cutter_z_max=0.2,
        collection=collection,
    ) is False

    class ContainsNameObjectStore(FakeObjectStore):
        def __contains__(self, value):
            if isinstance(value, str):
                return any(getattr(obj, "name", None) == value for obj in self)
            return super().__contains__(value)

    parent = FakeObject("kitchen_appliance_asset_root_old", object_type="EMPTY")
    child = FakeObject("child", parent=parent)
    loose = FakeObject("loose")
    fake_bpy.data.objects = ContainsNameObjectStore([parent, child, loose])
    kitchen._delete_objects([child, loose])
    assert child not in fake_bpy.data.objects
    assert loose not in fake_bpy.data.objects
    fake_bpy.data.objects = objects

    assert kitchen._mesh_outer_hull_xy_from_objects([], sample_z_min=0, sample_z_max=1) == []
    sparse_points = FakeObject("sparse", [(0.0, 0.0, 0.5), (1.0, 1.0, 0.5)])
    hull = kitchen._mesh_outer_hull_xy_from_objects([sparse_points], sample_z_min=0, sample_z_max=1, inset_m=0.0)
    assert len(hull) == 4
    assert kitchen._mesh_outer_polygon_xy_from_objects([], sample_z_min=0, sample_z_max=1) == []

    created = []

    def fake_box(name, center, size, material=None, collection_arg=None):
        obj = _fake_box_object(name, center, size, material, collection_arg or collection)
        created.append(obj.name)
        return obj

    def fake_cylinder(name, center, radius, depth, material=None, collection_arg=None, vertices=32):
        del vertices
        return fake_box(name, center, (radius * 2.0, radius * 2.0, depth), material, collection_arg)

    monkeypatch.setattr(kitchen, "_create_box", fake_box)
    monkeypatch.setattr(kitchen, "_create_cylinder", fake_cylinder)
    assert len(kitchen._create_sink_backing_basin("basin", (0.5, 0.5), (0.5, 0.4), 0.9, FakeMaterial("m"), collection)) == 5
    assert kitchen._create_polygon_sink_backing_basin("bad_basin", [(0, 0), (1, 0)], 0.9, FakeMaterial("m"), collection) == []
    assert len(kitchen._create_faucet_placeholder("faucet_y", (1, 2, 0.8), "y", FakeMaterial("metal"), collection)) == 3
    assert len(kitchen._create_dishwasher_placeholder("dish", (1, 1, 0.4), (0.6, 0.55, 0.8), FakeMaterial("body"), FakeMaterial("detail"), collection)) == 4
    assert len(kitchen._create_integrated_appliance_front("front_y", (1, 1, 0.5), (0.6, 0.5, 0.8), "y", FakeMaterial("facade"), FakeMaterial("handle"), collection)) == 3
    assert len(kitchen._create_fridge_placeholder("fridge_y", (1, 1, 1.0), (0.7, 0.65, 1.8), "y", FakeMaterial("body"), FakeMaterial("handle"), collection)) == 5
    assert len(kitchen._create_hood_placeholder("hood_y", 0, 0, 0.8, 0.4, 1.5, "y", FakeMaterial("body"), FakeMaterial("dark"), collection)) == 3
    assert "hood_y_chimney" in created

    fbx = tmp_path / "asset.fbx"
    obj = tmp_path / "asset.obj"
    bad = tmp_path / "asset.bad"
    for path in (fbx, obj, bad):
        path.write_bytes(b"x")
    assert kitchen._import_asset_objects(str(fbx), collection)
    assert kitchen._import_asset_objects(str(obj), collection)
    assert kitchen._import_asset_objects(str(bad), collection) == []
    fake_bpy.ops.import_scene.fbx = lambda filepath: (_ for _ in ()).throw(RuntimeError("fbx failed"))
    assert kitchen._import_asset_objects(str(fbx), collection) == []


def test_kitchen_remaining_unit_edge_branches(monkeypatch):
    monkeypatch.delitem(sys.modules, "bpy", raising=False)
    with pytest.raises(RuntimeError, match="inside Blender Python"):
        kitchen._require_bpy()

    fake_bpy, _objects, _scene_collection = _install_fake_bpy_for_primitives(monkeypatch)
    mat = kitchen._get_or_create_emission_material("cached_emission", (1, 0, 0, 1), 3.0)
    assert kitchen._get_or_create_emission_material("cached_emission", (0, 1, 0, 1), 1.0) is mat

    fake_bpy.path.abspath = lambda _raw: (_ for _ in ()).throw(RuntimeError("bad path"))
    assert kitchen._image_path_missing(types.SimpleNamespace(filepath="bad.png", packed_file=None)) is False
    assert kitchen._image_path_missing(types.SimpleNamespace(packed_file=None)) is False
    assert kitchen._material_has_missing_texture(types.SimpleNamespace(node_tree=None)) is False
    assert kitchen._replace_missing_texture_materials([types.SimpleNamespace(material_slots=[]), types.SimpleNamespace(material_slots=[types.SimpleNamespace(material=None)])]) == 0

    assert kitchen._convex_hull_xy([(1, 1), (1, 1)]) == [(1, 1)]
    assert kitchen._simplify_polygon_xy([(0, 0), (1, 0), (1, 1)]) == [(0, 0), (1, 0), (1, 1)]
    assert kitchen._simplify_polygon_xy([(0, 0), (1, 0), (1, 1), (0.001, 0.001)], min_edge_m=0.01) == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]

    empty_mesh = FakeObject("empty_mesh")
    empty_mesh.data.vertices = []
    empty_mesh.bound_box = []
    assert kitchen._bbox_world([types.SimpleNamespace(type="EMPTY", data=None)]) is None
    assert kitchen._objects_fit_within_size([types.SimpleNamespace(type="EMPTY", data=None)], (1, 1, 1)) is False
    assert kitchen._mesh_xy_bbox_below_z([empty_mesh], 0.1) is None
    assert kitchen._mesh_xy_bbox_between_z([empty_mesh], 0.0, 1.0) is None
    assert kitchen._mesh_xy_inner_bbox_between_z([empty_mesh], 0.0, 1.0) is None
    assert kitchen._mesh_outer_hull_xy_from_objects([empty_mesh], sample_z_min=0, sample_z_max=1) == []
    assert kitchen._real_bbox_opening_from_objects([]) is None

    created = []
    monkeypatch.setattr(
        kitchen,
        "_create_box",
        lambda name, center, size, material=None, collection=None: created.append((name, center, size)) or FakeObject(name),
    )
    kitchen._create_oriented_box("orient_y", 1, 2, 3, 0.5, 0.8, 0.1, "y")
    kitchen._create_oriented_box("orient_x", 1, 2, 3, 0.5, 0.8, 0.1, "x")
    assert created[0][1] == pytest.approx((1.25, 3.5, 0.5))
    assert created[1][1] == pytest.approx((2.5, 2.25, 0.5))
    assert kitchen._surface_point(1, 2, 0.3, 0.4, 0.5, 0.6, "y") == pytest.approx((1.9, 2.5))
    assert kitchen._surface_point(1, 2, 0.3, 0.4, 0.5, 0.6, "x") == pytest.approx((1.5, 3.0))
    assert kitchen._surface_point_local(1, 2, 0.3, 0.4, 0.5, 0.6, "y") == pytest.approx((1.9, 2.9))
    assert kitchen._surface_point_local(1, 2, 0.3, 0.4, 0.5, 0.6, "x") == pytest.approx((1.8, 3.0))

    assert kitchen._faucet_base_anchor_xy([], (0, 0)) is None
    assert kitchen._faucet_base_anchor_xy([empty_mesh], (0, 0)) is None
    assert kitchen._faucet_lowest_mount_xy([]) is None
    assert kitchen._faucet_direction_xy([], (0, 0)) is None
    centered = make_box("centered", (-0.01, -0.01, 0.0), (0.01, 0.01, 1.0))
    assert kitchen._faucet_direction_xy([centered], (0, 0)) is None


def test_kitchen_blender_builder_remaining_branch_edges(monkeypatch, capsys):
    fake_bpy, _objects, _scene_collection = _install_fake_bpy_for_primitives(monkeypatch)
    collection = FakeCollection()

    original_remove = fake_bpy.data.objects.remove
    fake_bpy.data.objects.remove = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("remove failed"))
    assert kitchen._apply_mesh_objects_cutout(
        make_box("target", (0, 0, 0), (1, 1, 0.2)),
        "remove_error",
        [make_box("source", (0.2, 0.2, 0), (0.4, 0.4, 0.2))],
        collection,
    ) is True
    fake_bpy.data.objects.remove = original_remove

    faucet_named_badly = make_box("faucet_without_numeric_suffix", (-0.02, -0.02, 9.0), (0.02, 0.02, 9.2))
    assert kitchen._filter_imported_appliance_objects("faucet", [faucet_named_badly]) == [faucet_named_badly]

    dense = FakeObject("dense", [(float(i), float(i % 4), 0.5) for i in range(12)])
    assert kitchen._mesh_xy_inner_bbox_between_z(
        [FakeObject("empty", object_type="EMPTY"), dense],
        0.4,
        0.6,
    ) is not None

    a = make_box("a", (0, 0, 0), (1, 1, 1))
    b = make_box("b", (1, 0, 0), (2, 1, 1))
    a.parent = b
    b.parent = a
    assert kitchen._fit_objects_to_box([a, b], (0, 0, 0), (1, 1, 1)) is True

    with monkeypatch.context() as m:
        bbox_values = iter([((0, 0, 0), (1, 1, 1)), None])
        m.setattr(kitchen, "_bbox_world", lambda _objects: next(bbox_values))
        assert kitchen._fit_objects_to_box([make_box("box", (0, 0, 0), (1, 1, 1))], (0, 0, 0), (1, 1, 1)) is False

    with monkeypatch.context() as m:
        m.setattr(kitchen, "_bbox_world", lambda _objects: None)
        assert kitchen._fit_mesh_objects_to_box_baked([make_box("mesh", (0, 0, 0), (1, 1, 1))], (0, 0, 0), (1, 1, 1)) is False

    low_sparse = FakeObject("low_sparse", [(0, 0, 0), (1, 1, 1)])
    assert kitchen._faucet_lowest_mount_xy([low_sparse]) == pytest.approx((0.5, 0.5))
    direction = kitchen._faucet_direction_xy([FakeObject("empty", object_type="EMPTY"), make_box("neck", (0, 0, 0), (0.2, 0, 1))], (0, 0))
    assert direction is not None

    mats: dict[str, FakeMaterial] = {}
    monkeypatch.setattr(kitchen, "_get_or_create_material", lambda name, *_args, **_kwargs: mats.setdefault(name, FakeMaterial(name)))
    empty = FakeObject("empty", object_type="EMPTY")
    kitchen._sanitize_imported_appliance_materials("cooktop", [empty, make_box("cooktop", (0, 0, 0), (1, 1, 0.1))])
    kitchen._sanitize_imported_appliance_materials("fridge", [empty, make_box("fridge", (0, 0, 0), (1, 1, 2))])

    asset = {"asset_local_path": "/candidate.glb", "unique_key": "cand", "title": "candidate"}
    monkeypatch.setattr(kitchen, "_appliance_asset_candidates", lambda _assembly, _role: [asset])
    monkeypatch.setattr(kitchen, "_import_asset_objects", lambda *_args, **_kwargs: [make_box("imported", (0, 0, 0), (0.2, 0.2, 0.2))])
    monkeypatch.setattr(kitchen, "_filter_imported_appliance_objects", lambda _role, objects, _asset: objects)
    monkeypatch.setattr(kitchen, "_fit_objects_to_box", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_fit_mesh_objects_to_box_baked", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(kitchen, "_objects_fit_within_size", lambda *_args, **_kwargs: False)
    deleted: list[str] = []
    monkeypatch.setattr(kitchen, "_delete_objects", lambda objects: deleted.extend(obj.name for obj in objects))
    monkeypatch.setattr(kitchen, "_create_box", lambda name, center, size, material=None, collection=None: _fake_box_object(name, center, size, material, collection))
    fallback = kitchen._create_or_import_appliance(
        {},
        "hood",
        "hood",
        (1, 2, 3),
        (0.4, 0.3, 0.2),
        FakeMaterial("fallback"),
        collection,
    )
    assert fallback[0].name == "hood"
    assert deleted == ["imported"]
    assert "appliance asset rejected after fit" in capsys.readouterr().out

    rotations: list[str] = []
    monkeypatch.setattr(kitchen, "_objects_fit_within_size", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_apply_asset_import_orientation", lambda *_args, **_kwargs: rotations.append("rotated"))
    faucet_generic = kitchen._create_or_import_appliance(
        {},
        "faucet",
        "faucet_generic",
        (1, 2, 3),
        (0.4, 0.3, 0.2),
        FakeMaterial("fallback"),
        collection,
    )
    assert faucet_generic[0]["kitchen_appliance_role"] == "faucet"
    assert rotations

    monkeypatch.setattr(kitchen, "_appliance_asset_candidates", lambda _assembly, _role: [])
    assert kitchen._create_or_import_appliance({}, "microwave", "missing_micro", (0, 0, 0), (1, 1, 1), FakeMaterial("m"), collection)[0].name == "missing_micro"

    monkeypatch.setattr(kitchen, "_appliance_asset", lambda _assembly, _role: asset)
    monkeypatch.setattr(kitchen, "_fit_mesh_objects_to_box_baked", lambda *_args, **_kwargs: False)
    deleted.clear()
    decor_fallback = kitchen._create_or_import_countertop_decor_asset(
        {},
        "flowers_vase",
        "vase",
        (0, 0, 1),
        (0.2, 0.2, 0.4),
        0.8,
        FakeMaterial("m"),
        collection,
    )
    assert decor_fallback[0].name == "vase"
    assert deleted


def test_kitchen_decor_lighting_and_basin_fallback_edges(monkeypatch):
    collection = FakeCollection()
    _install_fake_bpy_for_primitives(monkeypatch)
    created: list[tuple[str, tuple[float, float, float]]] = []

    def fake_box(name, center, size, material=None, collection_arg=None):
        del size, material
        created.append((name, center))
        return _fake_box_object(name, center, (0.1, 0.1, 0.1), collection=collection_arg or collection)

    monkeypatch.setattr(kitchen, "_create_box", fake_box)
    monkeypatch.setattr(kitchen, "_create_or_import_appliance", lambda _assembly, role, name, center, size, *_args, **_kwargs: [_fake_box_object(f"{name}_{role}", center, size)])
    monkeypatch.setattr(kitchen, "_create_or_import_countertop_decor_asset", lambda _assembly, role, name, center, size, *_args, **_kwargs: [_fake_box_object(f"{name}_{role}", center, size)])

    assert kitchen._create_kitchen_decor_item({}, {"id": "board_y", "type": "cutting_board", "orientation": "y"}, 0, 0, 0, FakeMaterial("m"), FakeMaterial("a"), collection)
    assert kitchen._create_kitchen_decor_item(
        {},
        {"id": "micro_shelf", "type": "microwave", "orientation": "y", "placement": "upper_open_shelf", "shelf_width_m": 0.36, "shelf_depth_m": 0.26},
        0,
        0,
        0,
        FakeMaterial("m"),
        FakeMaterial("a"),
        collection,
    )
    assert kitchen._create_kitchen_decor_item({}, {"id": "oil_y", "type": "oil_bottles_decor", "orientation": "y"}, 0, 0, 0, FakeMaterial("m"), FakeMaterial("a"), collection)
    assert kitchen._create_kitchen_decor_item({}, {"type": "unknown"}, 0, 0, 0, FakeMaterial("m"), FakeMaterial("a"), collection) == []
    assert any(name == "board_y" for name, _center in created)

    class RaisingAreaLight:
        def __init__(self):
            self.energy = 0.0
            self.shape = None
            self.size = 0.0

        @property
        def size_y(self):
            return 0.0

        @size_y.setter
        def size_y(self, _value):
            raise RuntimeError("rectangle unsupported")

    class FakeLights(dict):
        def new(self, name, light_type):
            light = RaisingAreaLight()
            light.name = name
            light.type = light_type
            self[name] = light
            return light

    class FakeObjects:
        def new(self, name, data):
            obj = FakeObject(name, object_type="LIGHT")
            obj.data = data
            return obj

    fake_bpy = types.SimpleNamespace(data=types.SimpleNamespace(lights=FakeLights(), objects=FakeObjects()))
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    lights = kitchen._create_under_cabinet_lighting({"id": "upper", "orientation": "y", "width_m": 0.5}, 0, 0, 0, FakeMaterial("led"), collection)
    assert lights[-1].data.size >= 0.46

    class Meshes:
        def new(self, name):
            mesh = FakeMesh([])
            mesh.name = name
            mesh.from_pydata = lambda vertices, _edges, _faces: setattr(mesh, "vertices", [FakeVertex(vertex) for vertex in vertices])
            return mesh

    class Objects:
        def new(self, name, mesh):
            obj = FakeObject(name)
            obj.data = mesh
            return obj

    scene_collection = FakeCollection("Scene")
    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(meshes=Meshes(), objects=Objects()),
        context=types.SimpleNamespace(scene=types.SimpleNamespace(collection=types.SimpleNamespace(objects=scene_collection.objects))),
    )
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    with monkeypatch.context() as m:
        m.setattr(kitchen, "_scale_polygon_xy", lambda _polygon, _inset: [])
        basin = kitchen._create_polygon_sink_backing_basin("scene_basin", [(0, 0), (1, 0), (1, 1)], 0.9, FakeMaterial("m"), None)
    assert basin[0] in scene_collection.objects


def test_build_kitchen_sink_real_opening_and_faucet_fallback_edges(monkeypatch):
    collection = FakeCollection()
    created_names: list[str] = []

    def fake_create_box(name, center, size, material=None, collection_arg=None):
        obj = _fake_box_object(name, center, size, material, collection_arg or collection)
        created_names.append(name)
        return obj

    def fake_create_oriented_box(name, origin_x, origin_y, width, depth, height, z, orientation, material, collection_arg):
        center = (origin_x + width / 2.0, origin_y + depth / 2.0, z + height / 2.0)
        size = (width, depth, height)
        if orientation == "y":
            center = (origin_x + depth / 2.0, origin_y + width / 2.0, z + height / 2.0)
            size = (depth, width, height)
        return fake_create_box(name, center, size, material, collection_arg)

    def fake_imported(_assembly, role, name, center, size, fallback_mat, collection_arg, **_kwargs):
        obj = fake_create_box(f"{name}_{role}_asset", center, size, fallback_mat, collection_arg)
        if role == "sink":
            obj["kitchen_appliance_role"] = "sink"
        if role not in {"faucet"}:
            obj["kitchen_appliance_role"] = role
        return [obj]

    class FakeObjects:
        def new(self, name, data):
            obj = FakeObject(name, object_type="LIGHT" if data is not None else "EMPTY")
            obj.data = data
            return obj

    class FakeLights(dict):
        def new(self, name, light_type):
            light = types.SimpleNamespace(name=name, type=light_type, energy=0.0, size=0.0, size_y=0.0)
            self[name] = light
            return light

    fake_bpy = types.SimpleNamespace(
        data=types.SimpleNamespace(objects=FakeObjects(), lights=FakeLights()),
        context=types.SimpleNamespace(view_layer=types.SimpleNamespace(update=lambda: None)),
    )
    monkeypatch.setattr(kitchen, "_require_bpy", lambda: fake_bpy)
    monkeypatch.setattr(kitchen, "_collection", lambda _name: collection)
    monkeypatch.setattr(kitchen, "_get_or_create_material", lambda name, *_args, **_kwargs: FakeMaterial(name))
    monkeypatch.setattr(kitchen, "_get_or_create_emission_material", lambda name, *_args, **_kwargs: FakeMaterial(name))
    monkeypatch.setattr(kitchen, "_create_box", fake_create_box)
    monkeypatch.setattr(kitchen, "_create_oriented_box", fake_create_oriented_box)
    monkeypatch.setattr(kitchen, "_create_cylinder", lambda name, center, radius, depth, material=None, collection_arg=None, vertices=32: fake_create_box(name, center, (radius * 2.0, radius * 2.0, depth), material, collection_arg))
    monkeypatch.setattr(kitchen, "_create_torus", lambda name, center, major_radius, minor_radius, material=None, collection_arg=None: fake_create_box(name, center, (major_radius * 2.0, major_radius * 2.0, minor_radius * 2.0), material, collection_arg))
    monkeypatch.setattr(kitchen, "_apply_rectangular_cutout", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(kitchen, "_mesh_outer_hull_xy_from_objects", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(kitchen, "_real_bbox_opening_from_objects", lambda _objects: ((0.45, 0.32), (0.30, 0.22)))
    monkeypatch.setattr(kitchen, "_create_sink_backing_basin", lambda name, center, size, z, material, collection_arg: [fake_create_box(name, (center[0], center[1], z - 0.07), (size[0], size[1], 0.14), material, collection_arg)])
    monkeypatch.setattr(kitchen, "_create_or_import_appliance", fake_imported)
    monkeypatch.setattr(kitchen, "_sink_asset_includes_faucet", lambda _asset: False)

    assembly = {
        "appliance_bindings": {
            "appliances": {
                "sink": {"chosen_asset": {"asset_local_path": "/sink.glb", "title": "white ceramic sink"}},
                "faucet": {"chosen_asset": {"asset_local_path": "/faucet.glb", "title": "faucet"}},
            }
        },
        "base_modules": [
            {"id": "sink_base", "x_m": 0.0, "y_m": 0.0, "width_m": 0.7, "depth_m": 0.56, "height_m": 0.72, "z_m": 0.1}
        ],
        "countertop_segments": [
            {
                "id": "counter",
                "x_m": 0.0,
                "y_m": 0.0,
                "width_m": 0.9,
                "depth_m": 0.62,
                "thickness_m": 0.04,
                "z_m": 0.82,
                "cutouts": [{"type": "sink", "module_id": "sink_base", "x_m": 0.18, "y_m": 0.12, "width_m": 0.45, "depth_m": 0.32}],
            }
        ],
    }
    kitchen.build_kitchen_assembly_in_blender(assembly)
    assert "counter_sink_fbx_backing_basin" in created_names
    assert "counter_sink_faucet_stem" in created_names

    monkeypatch.setattr(kitchen, "_sink_asset_includes_faucet", lambda _asset: True)
    created_names.clear()
    kitchen.build_kitchen_assembly_in_blender(assembly)
    assert "counter_sink_faucet_stem" not in created_names
