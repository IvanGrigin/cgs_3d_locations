import json
import random
import math
import os
from typing import List, Tuple

from glb_parser import load_room_from_glb, Room
from pathfinding_astar import find_path_to_object


DEFAULT_GLB = "src/data/input/room.glb"
DEFAULT_JSON = "src/data/input/objects.json"
OUTPUT_JSON = "src/data/output/placement_result.json"


# ============================================================
# МОДЕЛИ
# ============================================================

class Item:
    def __init__(self, name, min_size, max_size, color, extra):
        self.name = name

        # случайный размер из диапазона (мм → м)
        self.sx = random.uniform(min_size[0], max_size[0]) / 1000.0
        self.sy = random.uniform(min_size[1], max_size[1]) / 1000.0
        self.sz = random.uniform(min_size[2], max_size[2]) / 1000.0

        self.color = color
        self.extra = extra or {}


class PlacedItem:
    def __init__(
        self,
        item: Item,
        center: Tuple[float, float, float],
        rotation_deg: float,
        wall_contact_side: str | None = None,
    ):
        self.item = item
        self.cx, self.cy, self.cz = center
        self.rotation = rotation_deg  # в градусах (0..360)
        self.wall_contact_side = wall_contact_side  # "front/back/left/right" или None

        # реальный размер AABB в XY после поворота
        self.rx, self.ry = rotated_size(item.sx, item.sy, rotation_deg)

    # ---------- геометрия ----------

    def aabb(self):
        return {
            "x_min": self.cx - self.rx / 2,
            "x_max": self.cx + self.rx / 2,
            "y_min": self.cy - self.ry / 2,
            "y_max": self.cy + self.ry / 2,
            "z_min": self.cz - self.item.sz / 2,
            "z_max": self.cz + self.item.sz / 2,
        }

    def forward_vector(self) -> Tuple[float, float, float]:
        """
        Направление "вперёд" предмета в мировых координатах.
        По соглашению: локальная ось +Y.
        """
        a = math.radians(self.rotation)
        dx = math.cos(a)
        dy = math.sin(a)
        return dx, dy, 0.0

    def local_sides(self) -> dict:
        """
        Центры каждой стороны (front/back/left/right/top/bottom)
        в мировых координатах.
        """
        rx, ry = self.rx, self.ry
        cx, cy, cz = self.cx, self.cy, self.cz
        a = math.radians(self.rotation)

        dx = math.cos(a)  # "вперёд"
        dy = math.sin(a)

        return {
            "front": (cx + dx * ry / 2, cy + dy * ry / 2, cz),
            "back":  (cx - dx * ry / 2, cy - dy * ry / 2, cz),
            "right": (cx + dy * rx / 2, cy - dx * rx / 2, cz),
            "left":  (cx - dy * rx / 2, cy + dx * rx / 2, cz),
            "top":   (cx, cy, cz + self.item.sz / 2),
            "bottom": (cx, cy, cz - self.item.sz / 2),
        }

    def is_side_touching_wall(self, side: str, room: Room, epsilon: float = 0.02) -> bool:
        """
        Проверка, что указанная сторона действительно у стенки комнаты.
        Пока по AABB.
        """
        box = self.aabb()

        if side == "front":
            return abs(box["y_min"] - room.y_min) < epsilon
        if side == "back":
            return abs(box["y_max"] - room.y_max) < epsilon
        if side == "left":
            return abs(box["x_min"] - room.x_min) < epsilon
        if side == "right":
            return abs(box["x_max"] - room.x_max) < epsilon

        return False


# ============================================================
# ГЕОМЕТРИЯ
# ============================================================

def rotated_size(sx: float, sy: float, angle_deg: float) -> Tuple[float, float]:
    """
    Корректный AABB в плоскости XY при произвольном повороте.
    """
    a = math.radians(angle_deg)
    cos_a = abs(math.cos(a))
    sin_a = abs(math.sin(a))

    rx = sx * cos_a + sy * sin_a
    ry = sx * sin_a + sy * cos_a
    return rx, ry


def aabb_intersect(a, b) -> bool:
    return not (
        a["x_max"] <= b["x_min"] or
        a["x_min"] >= b["x_max"] or
        a["y_max"] <= b["y_min"] or
        a["y_min"] >= b["y_max"] or
        a["z_max"] <= b["z_min"] or
        a["z_min"] >= b["z_max"]
    )


def inside_room(aabb, room: Room) -> bool:
    return (
        aabb["x_min"] >= room.x_min and
        aabb["x_max"] <= room.x_max and
        aabb["y_min"] >= room.y_min and
        aabb["y_max"] <= room.y_max and
        aabb["z_min"] >= room.z_min and
        aabb["z_max"] <= room.z_max
    )


# ============================================================
# СЛУЧАЙНОЕ ПОЛОЖЕНИЕ
# ============================================================

def random_center(room: Room, rx: float, ry: float, sz: float) -> Tuple[float, float, float]:
    """
    Случайный центр внутри комнаты (по XY и Z, до поправок on_floor/under_ceiling/mount_height).
    """
    return (
        random.uniform(room.x_min + rx / 2, room.x_max - rx / 2),
        random.uniform(room.y_min + ry / 2, room.y_max - ry / 2),
        random.uniform(room.z_min + sz / 2, room.z_max - sz / 2),
    )


# ============================================================
# РАССТАНОВКА С ПОВОРОТАМИ И КОНСТРЕЙНТАМИ
# ============================================================

def place_all(room: Room, items: List[Item]) -> List[PlacedItem]:
    """
    Рандомная расстановка с учётом:
      - поворота (шаг 30°),
      - mount_height_m (настенные/висячие),
      - under_ceiling,
      - "по умолчанию всё на полу".
    """

    for global_try in range(60):
        placed: List[PlacedItem] = []
        failed = False

        for item in items:
            success = False

            for _ in range(800):
                # поворот каждые 30°
                rotation = random.choice(list(range(0, 360, 30)))
                rx, ry = rotated_size(item.sx, item.sy, rotation)

                # базовый центр
                cx, cy, cz = random_center(room, rx, ry, item.sz)

                extra = item.extra
                mount_height = extra.get("mount_height_m")

                # -------- ВЫСОТА (ГЛАВНОЕ ИЗМЕНЕНИЕ) --------
                if mount_height is not None:
                    # висячие / настенные — высота центра над полом
                    cz = room.z_min + mount_height
                elif extra.get("under_ceiling"):
                    cz = room.z_max - item.sz / 2
                else:
                    # ВСЁ ОСТАЛЬНОЕ СТАВИМ НА ПОЛ
                    cz = room.z_min + item.sz / 2

                wall_contact_side = None

                # -------- ПРИЖАТИЕ К СТЕНЕ --------
                if extra.get("touch_wall"):
                    allowed_sides = extra.get(
                        "touch_wall_sides",
                        ["front", "back", "left", "right"]
                    )
                    wall_contact_side = random.choice(allowed_sides)
                    margin = 0.01

                    if wall_contact_side == "back":
                        cy = room.y_max - ry / 2 - margin
                    elif wall_contact_side == "front":
                        cy = room.y_min + ry / 2 + margin
                    elif wall_contact_side == "left":
                        cx = room.x_min + rx / 2 + margin
                    elif wall_contact_side == "right":
                        cx = room.x_max - rx / 2 - margin

                candidate = PlacedItem(
                    item,
                    (cx, cy, cz),
                    rotation_deg=rotation,
                    wall_contact_side=wall_contact_side,
                )
                box = candidate.aabb()

                if not inside_room(box, room):
                    continue

                if any(aabb_intersect(box, other.aabb()) for other in placed):
                    continue

                if extra.get("touch_wall") and wall_contact_side is not None:
                    if not candidate.is_side_touching_wall(wall_contact_side, room):
                        continue

                placed.append(candidate)
                success = True
                break

            if not success:
                print(f"⚠️ Не влез: {item.name}")
                failed = True
                break

        if not failed:
            return placed

    raise RuntimeError("❌ Не удалось расставить предметы")


# ============================================================
# ПРОВЕРКА ДОСТУПА ЧЕЛОВЕКА (A*)
# ============================================================

def check_human_access_astar(room: Room, placed: List[PlacedItem]) -> bool:
    """
    Проверяем, что человек может подойти к предметам, для которых
    constraints.human_approach = True, и которые НЕ висят:
      - не under_ceiling
      - нет mount_height_m
    """

    room_dict = vars(room)

    items_dicts = [
        {
            "name": p.item.name,
            "aabb": p.aabb(),
            "extra": p.item.extra,
        }
        for p in placed
    ]

    for p, obj in zip(placed, items_dicts):
        extra = p.item.extra

        if not extra.get("human_approach", False):
            continue

        if extra.get("under_ceiling") or extra.get("mount_height_m") is not None:
            # люстры / настенные светильники и т.п. — не проверяем
            continue

        allowed_sides = ["front", "back", "left", "right"]
        if "free_side_named" in extra:
            allowed_sides = [extra["free_side_named"]["side"]]

        box = obj["aabb"]
        offset = 0.6

        side_targets = {
            "front": ((box["x_min"] + box["x_max"]) / 2, box["y_min"] - offset),
            "back":  ((box["x_min"] + box["x_max"]) / 2, box["y_max"] + offset),
            "left":  (box["x_min"] - offset, (box["y_min"] + box["y_max"]) / 2),
            "right": (box["x_max"] + offset, (box["y_min"] + box["y_max"]) / 2),
        }

        path_found = False

        for side in allowed_sides:
            tx, ty = side_targets[side]

            path = find_path_to_object(
                room_dict,
                items_dicts,
                {
                    "name": p.item.name,
                    "aabb": box,
                    "target_override": (tx, ty),
                },
            )

            if path is not None:
                path_found = True
                break

        if not path_found:
            print(f"❌ Нет подхода к объекту: {p.item.name}")
            return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== РАССТАНОВКА ОБЪЕКТОВ (ВСЁ НА ПОЛУ ПО УМОЛЧАНИЮ) ===")

    glb_path = input(f"GLB комнаты [{DEFAULT_GLB}]: ").strip() or DEFAULT_GLB
    json_path = input(f"JSON объектов [{DEFAULT_JSON}]: ").strip() or DEFAULT_JSON

    room = load_room_from_glb(glb_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [
        Item(
            obj["name"],
            obj["min_size_mm"],
            obj["max_size_mm"],
            obj.get("color", [1, 1, 1]),
            obj.get("constraints", {}),
        )
        for obj in data["items"]
    ]

    placed = place_all(room, items)

    if not check_human_access_astar(room, placed):
        raise RuntimeError("❌ ЧЕЛОВЕК НЕ МОЖЕТ ПОДОЙТИ КО ВСЕМ НУЖНЫМ ОБЪЕКТАМ")

    result = {
        "room": vars(room),
        "items": [],
    }

    for p in placed:
        fx, fy, fz = p.forward_vector()
        result["items"].append({
            "name": p.item.name,
            "center": [p.cx, p.cy, p.cz],
            "size": [p.item.sx, p.item.sy, p.item.sz],
            "rotation": p.rotation,
            "aabb": p.aabb(),
            "color": p.item.color,
            "forward": [fx, fy, fz],
            "wall_contact_side": p.wall_contact_side,
        })

    os.makedirs("src/data/output", exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n✅ ГОТОВО! placement_result.json создан\n")
    for item in result["items"]:
        print(
            item["name"],
            "→ центр", item["center"],
            "rot:", item["rotation"],
            "wall_side:", item["wall_contact_side"],
        )


if __name__ == "__main__":
    main()