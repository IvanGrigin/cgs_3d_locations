from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest


class FakeVector:
    def __init__(self, values=(0.0, 0.0, 0.0)):
        vals = list(values) + [0.0, 0.0, 0.0]
        self.x = float(vals[0])
        self.y = float(vals[1])
        self.z = float(vals[2])

    def __add__(self, other):
        return FakeVector((self.x + other.x, self.y + other.y, self.z + other.z))

    def __sub__(self, other):
        return FakeVector((self.x - other.x, self.y - other.y, self.z - other.z))

    def __mul__(self, scalar):
        return FakeVector((self.x * float(scalar), self.y * float(scalar), self.z * float(scalar)))

    __rmul__ = __mul__

    def __eq__(self, other):
        return (
            isinstance(other, FakeVector)
            and abs(self.x - other.x) < 1e-9
            and abs(self.y - other.y) < 1e-9
            and abs(self.z - other.z) < 1e-9
        )

    @property
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def copy(self):
        return FakeVector((self.x, self.y, self.z))

    def normalize(self):
        length = self.length
        if length > 1e-12:
            self.x /= length
            self.y /= length
            self.z /= length

    def normalized(self):
        copied = self.copy()
        copied.normalize()
        return copied

    def to_track_quat(self, *_args):
        return types.SimpleNamespace(to_euler=lambda: ("tracked", self.x, self.y, self.z))


class FakeMatrix:
    def __init__(self, offset=(0.0, 0.0, 0.0)):
        self.translation = FakeVector(offset)

    def __matmul__(self, value):
        return FakeVector((value.x + self.translation.x, value.y + self.translation.y, value.z + self.translation.z))

    def to_3x3(self):
        return self

    def inverted(self):
        return self

    def transposed(self):
        return self


class FakeMaterials(list):
    def clear(self):
        del self[:]


class FakeObject(dict):
    def __init__(self, name, *, obj_type="MESH", bounds=None, offset=(0, 0, 0), parent=None):
        super().__init__()
        self.name = name
        self.type = obj_type
        self.hide_render = False
        self._hide_viewport = False
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)
        self.bound_box = bounds or [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
        self.matrix_world = FakeMatrix(offset)
        self.rotation_euler = types.SimpleNamespace(z=0.0)
        self.data = types.SimpleNamespace(materials=FakeMaterials(), name=f"{name}_mesh")
        self.location = FakeVector(offset)

    def hide_get(self):
        return self._hide_viewport

    def hide_set(self, value):
        self._hide_viewport = bool(value)


class ObjectStore(list):
    def get(self, name, default=None):
        for obj in self:
            if obj.name == name:
                return obj
        return default

    def __contains__(self, value):
        if isinstance(value, str):
            return self.get(value) is not None
        return list.__contains__(self, value)

    def __getitem__(self, value):
        if isinstance(value, str):
            obj = self.get(value)
            if obj is None:
                raise KeyError(value)
            return obj
        return list.__getitem__(self, value)

    def new(self, name, data):
        obj_type = "CAMERA" if "Camera" in name else "MESH"
        obj = FakeObject(name, obj_type=obj_type)
        obj.data = data
        self.append(obj)
        return obj


class MaterialStore(dict):
    def new(self, name):
        mat = types.SimpleNamespace(name=name, diffuse_color=None)
        self[name] = mat
        return mat


@pytest.fixture()
def topview(monkeypatch):
    objects = ObjectStore()
    scene = types.SimpleNamespace(
        objects=objects,
        collection=types.SimpleNamespace(objects=types.SimpleNamespace(link=lambda obj: None)),
        render=types.SimpleNamespace(
            engine="",
            resolution_x=0,
            resolution_y=0,
            resolution_percentage=0,
            filepath="",
            image_settings=types.SimpleNamespace(file_format=""),
            film_transparent=False,
        ),
        display=types.SimpleNamespace(shading=types.SimpleNamespace()),
        eevee=types.SimpleNamespace(),
        view_settings=types.SimpleNamespace(),
        camera=None,
    )
    fake_bpy = types.SimpleNamespace(
        types=types.SimpleNamespace(Object=object, Mesh=object, Material=object),
        context=types.SimpleNamespace(scene=scene, object=None),
        data=types.SimpleNamespace(
            objects=objects,
            cameras=types.SimpleNamespace(new=lambda name: types.SimpleNamespace(name=name, type="PERSP", lens=0, clip_start=0, clip_end=0)),
            materials=MaterialStore(),
            curves=types.SimpleNamespace(new=lambda name, curve_type: types.SimpleNamespace(name=name, type=curve_type, body="", align_x="", align_y="", size=0, materials=FakeMaterials())),
        ),
        ops=types.SimpleNamespace(
            mesh=types.SimpleNamespace(primitive_plane_add=lambda **kwargs: setattr(fake_bpy.context, "object", FakeObject("Plane", bounds=[(0, 0, 0)]))),
            render=types.SimpleNamespace(render=lambda write_still=True: None, opengl=lambda write_still=True, view_context=False: None),
            wm=types.SimpleNamespace(save_as_mainfile=lambda filepath: None),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=FakeVector))
    monkeypatch.setitem(sys.modules, "bmesh", types.SimpleNamespace(new=lambda: None, ops=types.SimpleNamespace()))
    module_path = Path(__file__).resolve().parents[1] / "src" / "tools" / "render_saved_blend_top_view.py"
    module_name = "render_saved_blend_top_view_pure_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_visibility_wall_and_side_helpers(topview):
    visible = FakeObject("chair", offset=(1, 2, 0))
    hidden = FakeObject("hidden", offset=(10, 10, 0))
    hidden.hide_render = True
    bbox = FakeObject("debug_bbox", offset=(20, 20, 0))
    ceiling = FakeObject("ceiling_panel", bounds=[(0, 0, 2.8), (4, 0, 2.8), (4, 4, 2.8), (0, 4, 2.8), (0, 0, 3), (4, 0, 3), (4, 4, 3), (0, 4, 3)])
    exterior = FakeObject("room.exterior")
    topview.bpy.context.scene.objects[:] = [visible, hidden, bbox, ceiling, exterior]
    topview.bpy.data.objects[:] = topview.bpy.context.scene.objects

    bb_min, bb_max = topview._visible_mesh_bounds()
    assert bb_min == FakeVector((0, 0, 0))
    assert bb_max.z == 3.0
    assert topview._hide_ceiling_caps() == 1
    assert ceiling.hide_render
    assert topview._hide_exterior_shell_objects() == 1
    assert exterior.hide_render

    wall = FakeObject("room_0/0.wall")
    floor = FakeObject("room_floor")
    art = FakeObject("wall_art")
    assert topview._is_wall_render_candidate(wall)
    assert not topview._is_wall_render_candidate(floor)
    assert not topview._is_wall_render_candidate(art)

    topview.bpy.context.scene.objects[:] = [wall]
    wall_state = topview._capture_wall_state()
    wall.hide_render = True
    wall.hide_set(True)
    topview._restore_wall_state(wall_state)
    assert not wall.hide_render and not wall.hide_get()

    preview_wall = FakeObject("preview_wall_w0_panel")
    topview.bpy.context.scene.objects[:] = [preview_wall]
    assert topview._hide_preview_wall_ids({"w0"}) == 1
    assert preview_wall.hide_render

    center = FakeVector((2, 2, 0))
    assert topview._side_key_for_point(FakeVector((4, 2, 0)), center, 2, 2) == "x_pos"
    assert topview._nearest_wall_side_keys(FakeVector((5, -1, 0)), center) == {"x_pos", "y_neg"}
    assert topview._point_is_on_hidden_side(FakeVector((4, 2, 0)), center, 2, 2, {"x_pos"})

    topview.bpy.context.scene.objects[:] = [FakeObject("debug_bbox", offset=(10, 10, 10))]
    assert topview._visible_mesh_bounds() == (FakeVector((-3.0, -3.0, 0.0)), FakeVector((3.0, 3.0, 3.0)))
    empty_bounds = FakeObject("empty_bounds", bounds=[])
    shell = FakeObject("living-room/0", bounds=[(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0), (0, 0, 2.5), (5, 0, 2.5), (5, 5, 2.5), (0, 5, 2.5)])
    topview.bpy.context.scene.objects[:] = [empty_bounds, shell]
    assert topview._hide_ceiling_caps() == 1
    assert shell.hide_render
    assert topview._side_key_for_point(FakeVector((2, 4, 0)), center, 2, 2) == "y_pos"


def test_hide_nearest_walls_camera_and_bounds(topview):
    wall = FakeObject("room_wall_xpos", bounds=[(3.8, 1, 0), (4, 1, 0), (4, 3, 0), (3.8, 3, 0), (3.8, 1, 2), (4, 1, 2), (4, 3, 2), (3.8, 3, 2)])
    chair = FakeObject("chair")
    topview.bpy.context.scene.objects[:] = [wall, chair]
    hidden = topview._hide_nearest_room_walls(FakeVector((6, 6, 2)), FakeVector((0, 0, 0)), FakeVector((4, 4, 3)))
    assert hidden == 1
    assert wall.hide_render

    target = FakeObject("Camera", obj_type="CAMERA")
    target.location = FakeVector((1, 1, 1))
    topview._look_at(target, FakeVector((2, 1, 1)))
    assert target.rotation_euler[0] == "tracked"

    orbit = topview._relative_xy_point(FakeVector((0, 0, 2)), xy_span=10, azimuth_deg=0, radius_mult=0.5)
    assert orbit == FakeVector((5, 0, 2))
    assert topview._norm_name("Chair.001 / Стул!") == "chair 001 стул"

    root = FakeObject("root", obj_type="EMPTY")
    child = FakeObject("child", parent=root)
    assert topview._descendant_meshes(root) == [child]
    bounds = topview._bounds_for_roots([root])
    assert bounds[0] == FakeVector((0, 0, 0))
    assert bounds[1] == FakeVector((1, 1, 1))
    direct_mesh_bounds = topview._bounds_for_roots([FakeObject("direct_mesh")])
    assert direct_mesh_bounds[1] == FakeVector((1, 1, 1))

    empty_root = FakeObject("empty_root", obj_type="EMPTY", offset=(5, 6, 7))
    empty_root.children = []
    loc_bounds = topview._bounds_for_roots([empty_root])
    assert loc_bounds == (FakeVector((5, 6, 7)), FakeVector((5, 6, 7)))
    assert topview._bounds_for_roots([]) is None


def test_bmesh_cap_and_wall_face_deletion_paths(topview, monkeypatch):
    class FakeMeshData:
        def __init__(self, name="mesh"):
            self.name = name
            self.materials = FakeMaterials()
            self.copied = False
            self.updated = False

        def copy(self):
            copied = FakeMeshData(self.name + "_copy")
            copied.copied = True
            return copied

        def update(self):
            self.updated = True

    class FaceList(list):
        def ensure_lookup_table(self):
            return None

    class FakeFace:
        def __init__(self, center, normal=(0, 0, 1), verts=None, area=1.0):
            self._center = FakeVector(center)
            self.normal = FakeVector(normal)
            self.verts = [types.SimpleNamespace(co=FakeVector(v)) for v in (verts or [(0, 0, center[2]), (1, 0, center[2]), (1, 1, center[2]), (0, 1, center[2])])]
            self.area = area

        def calc_center_median(self):
            return self._center

    class FakeBM:
        def __init__(self, faces):
            self.faces = FaceList(faces)
            self.deleted = []

        def from_mesh(self, _mesh):
            return None

        def to_mesh(self, mesh):
            mesh.updated = True

        def free(self):
            return None

    deleted_batches = []

    def install_faces(faces):
        bm = FakeBM(faces)
        monkeypatch.setattr(topview.bmesh, "new", lambda: bm)
        monkeypatch.setattr(
            topview.bmesh,
            "ops",
            types.SimpleNamespace(delete=lambda bm_arg, geom, context: deleted_batches.append((bm_arg, list(geom), context))),
        )
        return bm

    top_face = FakeFace((0.5, 0.5, 3.0), verts=[(0, 0, 3), (4, 0, 3), (4, 4, 3), (0, 4, 3)])
    side_face = FakeFace((0.5, 0.5, 1.0), normal=(1, 0, 0), verts=[(0, 0, 0), (0, 4, 0), (0, 4, 2), (0, 0, 2)])
    partial_cap = FakeObject("large_cap", bounds=[(0, 0, 2.8), (4, 0, 2.8), (4, 4, 3), (0, 4, 3)])
    partial_cap.data = FakeMeshData("cap_mesh")
    skipped_floor = FakeObject("floor_mesh", bounds=[(0, 0, 3), (4, 0, 3), (4, 4, 3), (0, 4, 3)])
    topview.bpy.context.scene.objects[:] = [skipped_floor, partial_cap]
    install_faces([top_face, side_face])
    assert topview._delete_large_top_cap_faces(FakeVector((0, 0, 0)), FakeVector((4, 4, 3))) == 1
    assert partial_cap.data.name == "cap_mesh_copy"
    assert deleted_batches[-1][2] == "FACES"

    full_cap = FakeObject("full_cap", bounds=[(0, 0, 2.8), (4, 0, 2.8), (4, 4, 3), (0, 4, 3)])
    full_cap.data = FakeMeshData("full")
    topview.bpy.context.scene.objects[:] = [full_cap]
    install_faces([top_face])
    assert topview._delete_large_top_cap_faces(FakeVector((0, 0, 0)), FakeVector((4, 4, 3))) == 1
    assert full_cap.hide_render is True and full_cap.hide_get() is True

    partial_wall = FakeObject(
        "room_wall_partial",
        bounds=[(-2, -3, 0), (6, -3, 0), (6, 3, 0), (-2, 3, 0), (-2, -3, 2), (6, -3, 2), (6, 3, 2), (-2, 3, 2)],
    )
    partial_wall.data = FakeMeshData("wall_mesh")
    wall_delete_face = FakeFace((3.8, 2.0, 1.0), normal=(1, 0, 0), verts=[(3.8, 1, 0.5), (3.8, 3, 0.5), (3.8, 3, 2), (3.8, 1, 2)])
    wall_keep_face = FakeFace((0.5, 0.5, 1.0), normal=(1, 0, 0), verts=[(0, 0, 0.5), (0, 1, 0.5), (0, 1, 2), (0, 0, 2)])
    topview.bpy.context.scene.objects[:] = [partial_wall]
    install_faces([wall_delete_face, wall_keep_face])
    hidden = topview._hide_nearest_room_walls(FakeVector((6, 6, 2)), FakeVector((0, 0, 0)), FakeVector((4, 4, 3)))
    assert hidden == 1
    assert partial_wall.data.name == "wall_mesh_copy"

    full_wall = FakeObject(
        "room_wall_full",
        bounds=[(-2, -3, 0), (6, -3, 0), (6, 3, 0), (-2, 3, 0), (-2, -3, 2), (6, -3, 2), (6, 3, 2), (-2, 3, 2)],
    )
    full_wall.data = FakeMeshData("full_wall")
    topview.bpy.context.scene.objects[:] = [full_wall]
    install_faces([wall_delete_face])
    assert topview._hide_nearest_room_walls(FakeVector((6, 6, 2)), FakeVector((0, 0, 0)), FakeVector((4, 4, 3))) == 1
    assert full_wall.hide_render is True


def test_target_label_real_creation_paths(topview):
    root = FakeObject("target", offset=(1, 1, 0))
    scene_center = FakeVector((0, 0, 0))
    assert topview._add_target_label("T1", [root], scene_center)
    assert topview.bpy.context.object.name == "CGS_Topview_Label_BG_T1"
    assert "CGS_Topview_Label_Text" in topview.bpy.data.materials

    centered = FakeObject("centered", offset=(0, 0, 0))
    assert topview._add_target_label("chair", [centered], FakeVector((0.5, 0.5, 0)))
    assert topview.bpy.context.object.name == "CGS_Topview_Label_BG_chair"

    assert topview._add_target_label("missing", [], scene_center) is False


def test_scene_ref_orientation_highlight_and_label_reports(topview, tmp_path, monkeypatch):
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps({"items": [{"id": "chair1"}]}), encoding="utf-8")
    root = FakeObject("chair1")
    child = FakeObject("chair_mesh", parent=root)
    prop_obj = FakeObject("prop_holder")
    prop_obj["cgs_item_id"] = "chair1"
    named = FakeObject("Soft Chair")
    topview.bpy.context.scene.objects[:] = [root, child, prop_obj, named]
    topview.bpy.data.objects[:] = topview.bpy.context.scene.objects

    ref = types.SimpleNamespace(object_id="chair1", name="Soft Chair", yaw_deg=90.0)
    monkeypatch.setattr(topview, "collect_scene_objects", lambda data, max_objects=10000: [ref])
    monkeypatch.setattr(topview, "filter_target_objects", lambda refs, scope, include_armchairs: refs)

    roots = topview._objects_for_scene_ref(ref)
    assert root in roots
    assert named in roots

    report_path = tmp_path / "orientation_report.json"
    report = topview._apply_scene_orientations(scene_json, {"chair1"}, "chairs", False, report_path)
    assert len(report["applied"]) >= 1
    assert root.rotation_euler.z == pytest.approx(math.radians(90.0))
    assert root["cgs_topview_vlm_object_id"] == "chair1"
    assert json.loads(report_path.read_text(encoding="utf-8"))["applied"]

    highlighted = topview._highlight_scene_targets(scene_json, {"chair1"}, "chairs", False, highlight_style="material")
    assert highlighted >= 1
    assert child.data.materials

    monkeypatch.setattr(topview, "_add_target_label", lambda label, roots, scene_center: True)
    labeled = topview._label_scene_refs(scene_json, {"chair1"}, {"chair1": "T1"})
    assert labeled == 1


def test_orientation_and_highlight_skip_paths(topview, tmp_path, monkeypatch):
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps({"items": []}), encoding="utf-8")
    monkeypatch.setattr(topview, "collect_scene_objects", None)
    monkeypatch.setattr(topview, "filter_target_objects", None)
    report_path = tmp_path / "skip.json"
    report = topview._apply_scene_orientations(scene_json, {"missing"}, "chairs", False, report_path)
    assert report["skipped"][0]["reason"] == "topview_vlm_orientation_repair_import_failed"
    assert topview._highlight_scene_targets(scene_json, {"missing"}, "chairs", False) == 0
    assert topview._label_scene_refs(scene_json, {"missing"}, {"missing": "T1"}) == 0

    monkeypatch.setattr(topview, "collect_scene_objects", lambda data, max_objects=10000: [types.SimpleNamespace(object_id="x", name="X", yaw_deg=None)])
    monkeypatch.setattr(topview, "filter_target_objects", lambda refs, scope, include_armchairs: refs)
    report = topview._apply_scene_orientations(scene_json, {"x"}, "chairs", False, None)
    assert report["skipped"][0]["reason"] == "missing_yaw"

    monkeypatch.setattr(topview, "collect_scene_objects", lambda data, max_objects=10000: [types.SimpleNamespace(object_id="missing_obj", name="Missing", yaw_deg=45)])
    monkeypatch.setattr(topview, "filter_target_objects", lambda refs, scope, include_armchairs: refs)
    report_path = tmp_path / "missing_object_report.json"
    report = topview._apply_scene_orientations(scene_json, {"missing_obj"}, "chairs", False, report_path)
    assert report["skipped"][0]["reason"] == "blend_object_not_found"
    assert json.loads(report_path.read_text(encoding="utf-8"))["skipped"]

    root = FakeObject("target")
    topview.bpy.context.scene.objects[:] = [root]
    topview.bpy.data.objects[:] = topview.bpy.context.scene.objects
    monkeypatch.setattr(topview, "collect_scene_objects", lambda data, max_objects=10000: [types.SimpleNamespace(object_id="target", name="target", yaw_deg=0)])
    monkeypatch.setattr(topview, "filter_target_objects", lambda refs, scope, include_armchairs: refs)
    assert topview._highlight_scene_targets(scene_json, {"target"}, "all", False, highlight_style="label_only") == 1
    assert topview._highlight_scene_targets(scene_json, {"target"}, "all", False, highlight_style="none") == 1
    with pytest.raises(ValueError, match="Unsupported highlight_style"):
        topview._highlight_scene_targets(scene_json, {"target"}, "all", False, highlight_style="bad")
    assert topview._label_scene_refs(scene_json, {"missing_root"}, {"missing_root": "T2"}) == 0


def test_main_renders_view_specs_with_all_cli_hooks(topview, tmp_path, monkeypatch, capsys):
    scene_json = tmp_path / "scene.json"
    scene_json.write_text(json.dumps({"items": [{"id": "chair1"}]}), encoding="utf-8")
    label_map = tmp_path / "labels.json"
    label_map.write_text(json.dumps({"T1": "chair1"}), encoding="utf-8")
    view_specs = tmp_path / "views.json"
    view_specs.write_text(
        json.dumps(
            [
                {"name": "orbit", "out": str(tmp_path / "orbit.png"), "azimuth_deg": -90, "elevation_deg": 82, "radius_mult": 0.8, "lens": 35, "hide_nearest_walls": True},
                {
                    "name": "interior",
                    "out": str(tmp_path / "interior.png"),
                    "camera_mode": "interior",
                    "camera_z": 1.4,
                    "target_z": 1.1,
                    "camera_radius_mult": 0.4,
                    "look_radius_mult": 0.1,
                    "look_azimuth_deg": 45,
                },
                "bad-spec",
            ]
        ),
        encoding="utf-8",
    )
    save_blend = tmp_path / "inspection.blend"

    mesh = FakeObject("chair1", offset=(1, 1, 0))
    ceiling = FakeObject("ceiling_cap", bounds=[(0, 0, 2.8), (4, 0, 2.8), (4, 4, 2.8), (0, 4, 2.8), (0, 0, 3.0), (4, 0, 3.0), (4, 4, 3.0), (0, 4, 3.0)])
    exterior = FakeObject("room_exterior_shell")
    preview_wall = FakeObject("preview_wall_w0_panel")
    topview.bpy.context.scene.objects[:] = [mesh, ceiling, exterior, preview_wall]
    topview.bpy.data.objects[:] = topview.bpy.context.scene.objects

    calls = {"orient": 0, "highlight": 0, "label": 0, "nearest": 0, "renders": 0}
    monkeypatch.setattr(topview, "_apply_scene_orientations", lambda *args, **kwargs: calls.__setitem__("orient", calls["orient"] + 1) or {"applied": [{"id": "chair1"}]})
    monkeypatch.setattr(topview, "_highlight_scene_targets", lambda *args, **kwargs: calls.__setitem__("highlight", calls["highlight"] + 1) or 1)
    monkeypatch.setattr(topview, "_label_scene_refs", lambda *args, **kwargs: calls.__setitem__("label", calls["label"] + 1) or 1)
    monkeypatch.setattr(topview, "_delete_large_top_cap_faces", lambda *args, **kwargs: 0)
    monkeypatch.setattr(topview, "_hide_nearest_room_walls", lambda *args, **kwargs: calls.__setitem__("nearest", calls["nearest"] + 1) or 2)
    monkeypatch.setattr(topview, "_look_at", lambda obj, target: setattr(obj, "rotation_euler", ("look", target.x, target.y, target.z)))
    monkeypatch.setattr(topview.bpy.ops.render, "opengl", lambda write_still=True, view_context=False: calls.__setitem__("renders", calls["renders"] + 1))
    monkeypatch.setattr(topview.bpy.ops.wm, "save_as_mainfile", lambda filepath: Path(filepath).write_text("blend", encoding="utf-8"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "blender",
            "--",
            "--out",
            str(tmp_path / "default.png"),
            "--scene-json",
            str(scene_json),
            "--target-ids",
            "chair1",
            "--label-ids",
            "chair1",
            "--target-label-map",
            str(label_map),
            "--target-scope",
            "all",
            "--include-armchairs",
            "--apply-scene-orientations",
            "--highlight-targets",
            "--highlight-style",
            "label_only",
            "--orientation-report",
            str(tmp_path / "orientation.json"),
            "--hide-exterior",
            "--hide-wall-ids",
            "w0",
            "--transparent-background",
            "--render-engine",
            "workbench",
            "--view-specs-json",
            str(view_specs),
            "--save-blend",
            str(save_blend),
        ],
    )

    topview.main()
    out = capsys.readouterr().out
    assert calls == {"orient": 1, "highlight": 1, "label": 1, "nearest": 1, "renders": 2}
    assert "Applied scene orientations: 1" in out
    assert "Hidden exterior shell objects:" in out
    assert "Hidden explicit preview wall objects: 1" in out
    assert save_blend.read_text(encoding="utf-8") == "blend"
    assert topview.bpy.context.scene.render.engine == "BLENDER_WORKBENCH"
    assert topview.bpy.context.scene.render.film_transparent is True


def test_main_rejects_non_list_view_specs(topview, tmp_path, monkeypatch):
    specs = tmp_path / "bad_views.json"
    specs.write_text(json.dumps({"out": "x.png"}), encoding="utf-8")
    monkeypatch.setattr(topview, "_delete_large_top_cap_faces", lambda *args, **kwargs: 0)
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--out", str(tmp_path / "out.png"), "--view-specs-json", str(specs)])
    with pytest.raises(ValueError, match="JSON list"):
        topview.main()
