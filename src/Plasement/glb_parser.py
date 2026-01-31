# src/Plasement/glb_parser.py
# -*- coding: utf-8 -*-

from pygltflib import GLTF2
import numpy as np


class Room:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        # Целевая система координат движка расстановки:
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


# ---------- служебные функции для «внутреннего» бокса ----------

def _cluster_axis(values: np.ndarray, tol: float = 1e-5):
    """
    Группируем значения координаты плоскостей по оси в кластеры.
    В типовой «комнате-коробке» получаем ~4 кластера:
      [outer_min, inner_min, inner_max, outer_max]
    Возвращаем отсортированный список центров кластеров.
    """
    vals = np.sort(values.copy())
    clusters = []
    if vals.size == 0:
        return clusters

    cur = [vals[0]]
    for v in vals[1:]:
        if abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            clusters.append((np.mean(cur), len(cur)))
            cur = [v]
    clusters.append((np.mean(cur), len(cur)))

    # только центры, отсортированы по возрастанию
    return [c[0] for c in clusters]


def _pick_inner_bounds(axis_vals: np.ndarray, tol: float = 1e-5):
    """
    Выбор внутренних границ по одной оси.
    Стандарт: берём второй и предпоследний кластеры.
    Fallback’и, если кластеров меньше четырёх.
    """
    cl = _cluster_axis(axis_vals, tol=tol)
    n = len(cl)

    if n >= 4:
        inner_min, inner_max = cl[1], cl[-2]
        outer_min, outer_max = cl[0], cl[-1]
    elif n == 3:
        # Толщина, вероятно, только с одной стороны по этой оси.
        # Берём средний как одну из внутренних граней и «противоположный» край как вторую.
        # Стараться выбрать внутреннюю пару ближе к центру диапазона.
        mids = cl[1]
        # Выбираем противоположный по расстоянию от среднего
        if abs(cl[0] - mids) < abs(cl[2] - mids):
            inner_min, inner_max = cl[1], cl[2]
            outer_min, outer_max = cl[0], cl[2]
        else:
            inner_min, inner_max = cl[0], cl[1]
            outer_min, outer_max = cl[0], cl[2]
    elif n == 2:
        # Стен как «толстых» нет (две параллельные плоскости).
        inner_min, inner_max = cl[0], cl[1]
        outer_min, outer_max = cl[0], cl[1]
    elif n == 1:
        # Дегенерат — используем один уровень как обе границы.
        inner_min = inner_max = outer_min = outer_max = cl[0]
    else:
        raise RuntimeError("Не удалось выделить кластеры по оси.")

    return inner_min, inner_max, outer_min, outer_max


def load_room_from_glb(path: str) -> Room:
    """
    Читает room.glb, достаёт вершины POSITION и строит ВНУТРЕННИЙ bounding box.

    По glTF→Blender маппингу:
        Blender.X = glTF.X
        Blender.Y = glTF.Z
        Blender.Z = glTF.Y

    Мы работаем в той же системе: X, Y — пол, Z — высота.
    """
    print("Чтение GLB:", path)

    gltf = GLTF2().load(path)
    binary_blob = gltf.binary_blob()

    vertices_all = []

    # собираем все позиции
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

    verts_raw = np.vstack(vertices_all)

    # Сырые границы glTF (X_raw,Y_raw,Z_raw)
    x_min_raw, y_min_raw, z_min_raw = verts_raw.min(axis=0)
    x_max_raw, y_max_raw, z_max_raw = verts_raw.max(axis=0)

    print("Сырые границы из GLB (сырой glTF-бокс):")
    print(f"X_raw: {x_min_raw:.3f} .. {x_max_raw:.3f}")
    print(f"Y_raw: {y_min_raw:.3f} .. {y_max_raw:.3f}")
    print(f"Z_raw: {z_min_raw:.3f} .. {z_max_raw:.3f}")

    # Перекладываем оси в нашу систему (как это делает Blender-импортёр):
    # X := X_raw, Y := Z_raw, Z := Y_raw
    X = verts_raw[:, 0]
    Y = verts_raw[:, 2]
    Z = verts_raw[:, 1]

    # Находим внутренние границы для каждой оси
    x_in_min, x_in_max, x_out_min, x_out_max = _pick_inner_bounds(X, tol=1e-4)
    y_in_min, y_in_max, y_out_min, y_out_max = _pick_inner_bounds(Y, tol=1e-4)
    z_in_min, z_in_max, z_out_min, z_out_max = _pick_inner_bounds(Z, tol=1e-4)

    # Оценка толщин (по двум сторонам — может отличаться)
    thick_x_left  = x_in_min - x_out_min
    thick_x_right = x_out_max - x_in_max
    thick_y_front = y_in_min - y_out_min
    thick_y_back  = y_out_max - y_in_max
    thick_z_floor = z_in_min - z_out_min
    thick_z_ceil  = z_out_max - z_in_max

    print("ИНТЕРПРЕТАЦИЯ (внутренний бокс комнаты):")
    print(f"X: {x_in_min:.3f} .. {x_in_max:.3f}  (длина)")
    print(f"Y: {y_in_min:.3f} .. {y_in_max:.3f}  (ширина)")
    print(f"Z: {z_in_min:.3f} .. {z_in_max:.3f}  (высота)")
    print("Оценка толщин (м): "
          f"walls X[L/R]={thick_x_left:.3f}/{thick_x_right:.3f}, "
          f"walls Y[F/B]={thick_y_front:.3f}/{thick_y_back:.3f}, "
          f"floor/ceiling={thick_z_floor:.3f}/{thick_z_ceil:.3f}")

    # Возвращаем внутренний объём комнаты:
    return Room(
        x_min=x_in_min, x_max=x_in_max,
        y_min=y_in_min, y_max=y_in_max,
        z_min=z_in_min, z_max=z_in_max,
    )