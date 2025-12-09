import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from glb_parser import load_room_from_glb, Room
from pathfinding_astar import find_path_to_object, path_to_band

DEFAULT_GLB = "src/data/input/room.glb"
DEFAULT_JSON = "src/data/output/placement_result.json"


# ---------- геометрия коробок ----------

def box_vertices(aabb: Dict[str, float]) -> List[List[float]]:
    x_min, x_max = aabb["x_min"], aabb["x_max"]
    y_min, y_max = aabb["y_min"], aabb["y_max"]
    z_min, z_max = aabb["z_min"], aabb["z_max"]

    return [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]


def draw_box(
    ax,
    aabb: Dict[str, float],
    color=(0.6, 0.6, 0.6),
    alpha: float = 0.3,
    wire: bool = False,
):
    v = box_vertices(aabb)

    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[1], v[2], v[6], v[5]],
        [v[2], v[3], v[7], v[6]],
        [v[3], v[0], v[4], v[7]],
    ]

    if not wire:
        poly = Poly3DCollection(
            faces,
            facecolors=[color],
            edgecolors="k",
            alpha=alpha,
        )
        ax.add_collection3d(poly)
    else:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            ax.plot(
                [v[i][0], v[j][0]],
                [v[i][1], v[j][1]],
                [v[i][2], v[j][2]],
                color="black",
            )


def set_equal_3d_scale(ax, x_min, x_max, y_min, y_max, z_min, z_max):
    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    cz = (z_min + z_max) / 2

    half = max_range / 2

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


# ---------- визуализация полосы пути ----------

def draw_path_band_from_polys(
    ax,
    band_polys: List[List[Tuple[float, float]]],
    z: float,
    color: str = "#ff00ff",
    alpha: float = 0.85,
):
    """
    band_polys — список четырёхугольников в 2D ([(x,y), ...]),
    здесь поднимаем их на уровень z и рисуем как 3D-полосы.
    """
    faces_3d = []
    for quad in band_polys:
        if len(quad) != 4:
            continue
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = quad
        faces_3d.append([
            [x1, y1, z],
            [x2, y2, z],
            [x3, y3, z],
            [x4, y4, z],
        ])

    if not faces_3d:
        return

    poly = Poly3DCollection(
        faces_3d,
        facecolors=color,
        edgecolors="none",
        alpha=alpha,
    )
    ax.add_collection3d(poly)


# ---------- MAIN ----------

def main():
    print("=== Визуализация комнаты, объектов и проходов (A*) ===")

    glb_path = input(f"Файл комнаты (.glb) [{DEFAULT_GLB}]: ").strip() or DEFAULT_GLB
    json_path = input(f"Файл расстановки (.json) [{DEFAULT_JSON}]: ").strip() or DEFAULT_JSON

    # просто для логов границ
    load_room_from_glb(glb_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    room = data["room"]
    items = data["items"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Комната
    draw_box(ax, room, wire=True)

    # Объекты
    for obj in items:
        draw_box(ax, obj["aabb"], color=obj.get("color", [0.7, 0.7, 0.7]), alpha=0.4)
        cx, cy = obj["center"][0], obj["center"][1]
        cz = obj["aabb"]["z_max"]
        ax.text(cx, cy, cz + 0.05, obj["name"], fontsize=9)

    floor_z = room["z_min"] + 0.02

    # Пути ко всем объектам
    for obj in items:
        path_world = find_path_to_object(room, items, obj)

        if path_world is None:
            print(f"⚠️ Нет пути к объекту: {obj['name']}")
            continue

        band_polys = path_to_band(path_world)  # ширина = ширине человека
        draw_path_band_from_polys(
            ax,
            band_polys,
            z=floor_z,
            color="#ff00ff",
            alpha=0.85,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_equal_3d_scale(
        ax,
        room["x_min"], room["x_max"],
        room["y_min"], room["y_max"],
        room["z_min"], room["z_max"],
    )

    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()