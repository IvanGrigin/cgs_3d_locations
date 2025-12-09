from pygltflib import GLTF2
import numpy as np


class Room:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        # НАША ЦЕЛЕВАЯ СИСТЕМА:
        # X, Y – плоскость пола, Z – высота
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def depth(self):
        return self.y_max - self.y_min

    @property
    def height(self):
        return self.z_max - self.z_min


def load_room_from_glb(path: str) -> Room:
    """
    Читает room.glb, достаёт вершины POSITION и строит bounding box.

    В GLB (по факту):
        X_raw = 6 м  — длина
        Y_raw = 2.8 м — высота
        Z_raw = 4 м  — ширина

    Хотим получить комнату в системе:
        X = длина   = X_raw
        Y = ширина  = Z_raw
        Z = высота  = Y_raw
    """
    print("Чтение GLB:", path)

    gltf = GLTF2().load(path)
    binary_blob = gltf.binary_blob()

    vertices_all = []

    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            attrs = prim.attributes
            if not hasattr(attrs, "POSITION") or attrs.POSITION is None:
                continue

            accessor_index = attrs.POSITION
            accessor = gltf.accessors[accessor_index]
            buffer_view = gltf.bufferViews[accessor.bufferView]

            byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
            byte_length = accessor.count * 3 * 4  # 3 * float32

            raw = binary_blob[byte_offset: byte_offset + byte_length]
            data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)
            vertices_all.append(data)

    if not vertices_all:
        raise RuntimeError("В GLB не найдено ни одной вершины POSITION.")

    verts = np.vstack(vertices_all)

    x_min_raw, y_min_raw, z_min_raw = verts.min(axis=0)
    x_max_raw, y_max_raw, z_max_raw = verts.max(axis=0)

    print("Сырые границы из GLB:")
    print(f"X_raw: {x_min_raw:.3f} .. {x_max_raw:.3f}")
    print(f"Y_raw: {y_min_raw:.3f} .. {y_max_raw:.3f}")
    print(f"Z_raw: {z_min_raw:.3f} .. {z_max_raw:.3f}")

    # ЖЁСТКИЙ МАППИНГ:
    #   X = X_raw
    #   Y = Z_raw
    #   Z = Y_raw
    x_min = x_min_raw
    x_max = x_max_raw

    y_min = z_min_raw
    y_max = z_max_raw

    z_min = y_min_raw
    z_max = y_max_raw

    print("Интерпретированные границы комнаты:")
    print(f"X: {x_min:.3f} .. {x_max:.3f}  (длина, должно быть ~0..6)")
    print(f"Y: {y_min:.3f} .. {y_max:.3f}  (ширина, должно быть ~0..4)")
    print(f"Z: {z_min:.3f} .. {z_max:.3f}  (высота, должно быть ~0..2.8)")

    return Room(x_min, x_max, y_min, y_max, z_min, z_max)