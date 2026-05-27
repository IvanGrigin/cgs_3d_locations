import importlib.util, math, sys, types
from pathlib import Path

import pytest

ns = types.SimpleNamespace


class FakeVector:
    def __init__(self, values=(0.0, 0.0, 0.0)):
        vals = list(values) + [0.0, 0.0, 0.0]
        self.x, self.y, self.z = map(float, vals[:3])

    def __iter__(self):
        yield from (self.x, self.y, self.z)

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

    def __setitem__(self, index, value):
        if index not in (0, 1, 2):
            raise IndexError(index)
        setattr(self, ("x", "y", "z")[index], float(value))

    def __eq__(self, other):
        return isinstance(other, FakeVector) and all(abs(a - b) < 1e-9 for a, b in zip(self, other))

    def _op(self, other, fn):
        return FakeVector((fn(self.x, other.x), fn(self.y, other.y), fn(self.z, other.z)))

    def __add__(self, other):
        return self._op(other, lambda a, b: a + b)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __sub__(self, other):
        return self._op(other, lambda a, b: a - b)

    def __neg__(self):
        return FakeVector((-self.x, -self.y, -self.z))

    def __mul__(self, scalar):
        scalar = float(scalar)
        return FakeVector((self.x * scalar, self.y * scalar, self.z * scalar))

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        scalar = float(scalar)
        return FakeVector((self.x / scalar, self.y / scalar, self.z / scalar))

    def __itruediv__(self, scalar):
        self.x /= float(scalar)
        self.y /= float(scalar)
        self.z /= float(scalar)
        return self

    @property
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        if self.length > 1e-12:
            self /= self.length

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def to_2d(self):
        return FakeVector((self.x, self.y, 0.0))

    def copy(self):
        return FakeVector((self.x, self.y, self.z))

    def to_track_quat(self, _track, _up):
        return ns(to_euler=lambda: FakeVector((0.0, 0.0, 0.0)))


class FakeMatrix:
    def __matmul__(self, value): return value

    def to_3x3(self): return self

    def inverted(self): return self

    def copy(self): return self


class FakeObjects(list):
    def __contains__(self, value):
        return self.get(value) is not None if isinstance(value, str) else super().__contains__(value)

    def __getitem__(self, value):
        if not isinstance(value, str):
            return super().__getitem__(value)
        obj = self.get(value)
        if obj is None:
            raise KeyError(value)
        return obj

    def get(self, name, default=None):
        return next((obj for obj in self if getattr(obj, "name", None) == name), default)

    def remove(self, obj, do_unlink=True):
        if obj in self:
            super().remove(obj)


class FakeObject(dict):
    def __init__(self, name, *, obj_type="MESH", parent=None, collections=None):
        super().__init__()
        self.name = name
        self.type = obj_type
        self.parent = parent
        self.children = []
        self.users_collection = collections or []
        self.hide_render = False
        self.hide_viewport = False
        self.location = FakeVector()
        self.scale = FakeVector((1.0, 1.0, 1.0))
        self.rotation_euler = FakeVector()
        self.matrix_world = FakeMatrix()
        self.matrix_parent_inverse = FakeMatrix()
        self.modifiers = ns(new=lambda **_kwargs: ns())
        self.data = ns(name=f"{name}_mesh", materials=[])
        if parent is not None:
            parent.children.append(self)

    def __setattr__(self, name, value):
        if name in {"location", "scale", "rotation_euler"} and isinstance(value, (tuple, list)):
            value = FakeVector(value)
        super().__setattr__(name, value)

    def hide_get(self):
        return bool(self.hide_viewport)

    def hide_set(self, value):
        self.hide_viewport = bool(value)

    def as_pointer(self):
        return id(self)

    def copy(self):
        copied = FakeObject(f"{self.name}_copy", obj_type=self.type, collections=list(self.users_collection))
        copied.data = self.data.copy() if hasattr(self.data, "copy") else self.data
        for attr in ("location", "scale", "rotation_euler", "matrix_world", "matrix_parent_inverse"):
            value = getattr(self, attr)
            setattr(copied, attr, value.copy() if hasattr(value, "copy") else value)
        copied.update(dict(self))
        return copied

    def evaluated_get(self, _depsgraph):
        return self

    def to_mesh(self):
        return self.data

    def to_mesh_clear(self):
        return None

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class FakeUVLayers(list):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.active = None

    def new(self, name="UVMap"):
        self.active = ns(name=name, data=[ns(uv=(0.0, 0.0)) for _ in self.mesh.loops])
        self.append(self.active)
        return self.active


class FakeMesh:
    def __init__(self, name):
        self.name = name
        self.vertices = []
        self.polygons = []
        self.loops = []
        self.uv_layers = FakeUVLayers(self)
        self.materials = []

    def from_pydata(self, verts, _edges, faces):
        self.vertices = [ns(co=FakeVector(v)) for v in verts]
        self.loops = []
        self.polygons = []
        for face in faces:
            loop_indices = []
            for vertex_index in face:
                loop_indices.append(len(self.loops))
                self.loops.append(ns(vertex_index=vertex_index))
            self.polygons.append(ns(vertices=tuple(face), loop_indices=loop_indices, use_smooth=False, area=1.0))

    def update(self):
        return None

    def flip_normals(self):
        return None

    def copy(self):
        copied = FakeMesh(self.name)
        copied.vertices = [ns(co=v.co.copy()) for v in self.vertices]
        copied.polygons = list(self.polygons)
        copied.loops = list(self.loops)
        copied.materials = list(self.materials)
        return copied


class FakeCollectionObjects(list):
    def __init__(self, collection):
        super().__init__()
        self.collection = collection

    def link(self, obj):
        if obj not in self:
            self.append(obj)
        if self.collection not in obj.users_collection:
            obj.users_collection.append(self.collection)

    def unlink(self, obj):
        if obj in self:
            self.remove(obj)
        if self.collection in obj.users_collection:
            obj.users_collection.remove(self.collection)


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.objects = FakeCollectionObjects(self)


class FakeObjectStore(FakeObjects):
    def new(self, name, data):
        obj = FakeObject(name, obj_type=("EMPTY" if data is None else "MESH"))
        obj.data = data if data is not None else ns(name=f"{name}_empty_data", materials=[])
        self.append(obj)
        return obj


class FakeMeshStore(list):
    def new(self, name):
        mesh = FakeMesh(name)
        self.append(mesh)
        return mesh


class FakeNodeSocket:
    def __init__(self, name="", node=None):
        self.name = name
        self.node = node
        self.default_value = None
        self.is_linked = False
        self.links = []


class FakeNode:
    INPUTS = "Vector|Base Color|Roughness|Specular|Metallic|Normal|Alpha|Height|Emission|Emission Color|Emission Strength|Surface|Displacement|Scale|Color|Color1|Color2|Fac|Mortar|Brick Width|Row Height|Mortar Size|Mortar Smooth|Strength".split("|")
    OUTPUTS = "BSDF|Generated|UV|Vector|Color|Alpha|Normal|Displacement|Background|Emission".split("|")

    def __init__(self, node_type):
        type_map = {"ShaderNodeBsdfPrincipled": "BSDF_PRINCIPLED", "ShaderNodeOutputMaterial": "OUTPUT_MATERIAL", "ShaderNodeTexImage": "TEX_IMAGE", "ShaderNodeNormalMap": "NORMAL_MAP"}
        self.type = type_map.get(node_type, node_type)
        self.name = node_type
        self.label = ""
        self.location = (0, 0)
        self.inputs = {name: FakeNodeSocket(name, self) for name in self.INPUTS}
        self.outputs = {name: FakeNodeSocket(name, self) for name in self.OUTPUTS}
        self.image = None

    def as_pointer(self):
        return id(self)


class FakeNodes(list):
    def clear(self):
        del self[:]

    def new(self, node_type):
        node = FakeNode(node_type)
        self.append(node)
        return node


class FakeLinks:
    def __init__(self):
        self.links = []

    def new(self, from_socket, to_socket):
        link = ns(from_socket=from_socket, to_socket=to_socket, from_node=getattr(from_socket, "node", None), to_node=getattr(to_socket, "node", None))
        self.links.append(link)
        to_socket.is_linked = True
        to_socket.links.append(link)
        return link

    def __iter__(self):
        return iter(self.links)

    def remove(self, link):
        if link in self.links:
            self.links.remove(link)
        to_socket = getattr(link, "to_socket", None)
        if to_socket is not None and link in getattr(to_socket, "links", []):
            to_socket.links.remove(link)
            to_socket.is_linked = bool(to_socket.links)


class FakeMaterialStore:
    def __init__(self):
        self.created = {}

    def get(self, name):
        return self.created.get(name)

    def new(self, name):
        self.created[name] = ns(name=name, diffuse_color=(1.0, 1.0, 1.0, 1.0), use_nodes=False, node_tree=ns(nodes=FakeNodes(), links=FakeLinks()))
        return self.created[name]


@pytest.fixture()
def builder(monkeypatch):
    class SceneWithAttrs(types.SimpleNamespace):
        def __init__(self):
            super().__init__(
                collection=ns(children=ns(link=lambda _coll: None), objects=ns(link=lambda _obj: None)),
                unit_settings=ns(system="", scale_length=1.0),
                render=ns(engine="", resolution_percentage=100, filepath=""),
                cycles=ns(),
                world=None,
                camera=None,
            )
            self._store = {}

        def __getitem__(self, key): return self._store[key]

        def __setitem__(self, key, value): self._store[key] = value

        def get(self, key, default=None): return self._store.get(key, default)

    fake_bpy = ns(
        types=ns(Object=object, Collection=object, Material=object, Image=object, Mesh=object, Scene=object, Node=object),
        data=ns(
            objects=FakeObjects(),
            meshes=[],
            images=[],
            materials=ns(get=lambda _name: None, new=lambda name: ns(name=name)),
            collections=ns(get=lambda _name: None, new=lambda name: ns(name=name)),
            worlds=ns(new=lambda name: ns(name=name)),
        ),
        context=ns(
            scene=SceneWithAttrs(),
            view_layer=ns(update=lambda: None, objects=ns(active=None)),
            window_manager=ns(windows=[]),
            evaluated_depsgraph_get=lambda: object(),
            selected_objects=[],
        ),
        ops=ns(
            wm=ns(read_factory_settings=lambda use_empty=True: None),
            file=ns(make_paths_relative=lambda: None, pack_all=lambda: None),
            object=ns(select_all=lambda action="DESELECT": None),
        ),
        app=ns(build_options=set()),
        path=ns(abspath=lambda p: str(Path(p).expanduser())),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bmesh", ns())
    monkeypatch.setitem(sys.modules, "mathutils", ns(Vector=FakeVector))

    module_name = "blender_scene_builder_pure_test"
    module_path = Path(__file__).resolve().parents[2] / "src" / "Plasement" / "blender_scene_builder.py"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
