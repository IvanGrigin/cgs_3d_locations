import heapq
import math
from typing import Dict, List, Tuple, Optional


# ============================================================
# НАСТРОЙКИ ЧЕЛОВЕКА
# ============================================================

HUMAN_SIZE = (1.0, 1.0, 1.8)   # ширина X, ширина Y, высота
GRID_STEP = 0.1               # шаг сетки


# ============================================================
# ПОСТРОЕНИЕ ПРОХОДИМОЙ 2D СЕТКИ (XY)
# ============================================================

def build_walk_grid(
    room: Dict[str, float],
    items: List[Dict],
    human_size=HUMAN_SIZE,
    step=GRID_STEP,
):
    """
    Строит бинарную 2D-сетку:
    True  = человек ПОЛНОСТЬЮ помещается
    False = заблокировано мебелью
    """

    human_sx, human_sy, _ = human_size

    x_min, x_max = room["x_min"], room["x_max"]
    y_min, y_max = room["y_min"], room["y_max"]

    nx = int((x_max - x_min) / step) + 1
    ny = int((y_max - y_min) / step) + 1

    def in_bounds(gx, gy):
        return 0 <= gx < nx and 0 <= gy < ny

    def world_to_grid(x, y) -> Tuple[int, int]:
        gx = int((x - x_min) / step)
        gy = int((y - y_min) / step)
        return gx, gy

    def grid_to_world_center(gx, gy) -> Tuple[float, float]:
        x = x_min + (gx + 0.5) * step
        y = y_min + (gy + 0.5) * step
        return x, y

    # изначально всё проходимо
    grid = [[True for _ in range(ny)] for _ in range(nx)]

    # блокируем области под мебель + радиус человека
    for obj in items:
        box = obj["aabb"]

        bx_min = box["x_min"] - human_sx / 2
        bx_max = box["x_max"] + human_sx / 2
        by_min = box["y_min"] - human_sy / 2
        by_max = box["y_max"] + human_sy / 2

        gx_min, gy_min = world_to_grid(bx_min, by_min)
        gx_max, gy_max = world_to_grid(bx_max, by_max)

        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                if in_bounds(gx, gy):
                    grid[gx][gy] = False

    return grid, world_to_grid, grid_to_world_center, in_bounds


# ============================================================
# A* АЛГОРИТМ ПОИСКА ПУТИ
# ============================================================

def astar_path(
    grid: List[List[bool]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    in_bounds
) -> Optional[List[Tuple[int, int]]]:

    if not in_bounds(*start) or not in_bounds(*goal):
        return None
    if not grid[start[0]][start[1]] or not grid[goal[0]][goal[1]]:
        return None

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # восстановление пути
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        x, y = current

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if not in_bounds(nx, ny):
                continue
            if not grid[nx][ny]:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None


# ============================================================
# ПОИСК ПУТИ ОТ СТЕНЫ К ОБЪЕКТУ
# ============================================================

def find_path_to_object(
    room: Dict[str, float],
    items: List[Dict],
    obj: Dict,
):
    """
    Возвращает путь (в мировых координатах) шириной ровно человека.
    Старт всегда от НИЖНЕЙ СТЕНЫ.
    """

    grid, world_to_grid, grid_to_world, in_bounds = build_walk_grid(room, items)

    # ===== СТАРТ ОТ СТЕНЫ =====
    start_world = (
        (room["x_min"] + room["x_max"]) / 2,
        room["y_min"] + HUMAN_SIZE[1] / 2
    )
    start_cell = world_to_grid(*start_world)

    # ===== ЦЕЛИ ПОДХОДА К ОБЪЕКТУ =====
    box = obj["aabb"]
    offset = HUMAN_SIZE[1] / 2 + 0.05

    targets_world = [
        ((box["x_min"] + box["x_max"]) / 2, box["y_min"] - offset),
        ((box["x_min"] + box["x_max"]) / 2, box["y_max"] + offset),
        (box["x_min"] - offset, (box["y_min"] + box["y_max"]) / 2),
        (box["x_max"] + offset, (box["y_min"] + box["y_max"]) / 2),
    ]

    for tx, ty in targets_world:
        gx, gy = world_to_grid(tx, ty)
        if not in_bounds(gx, gy):
            continue
        if not grid[gx][gy]:
            continue

        cell_path = astar_path(grid, start_cell, (gx, gy), in_bounds)
        if cell_path:
            return [grid_to_world(px, py) for px, py in cell_path]

    return None


# ============================================================
# ПРЕОБРАЗОВАНИЕ В ПОЛОСУ ШИРИНОЙ ЧЕЛОВЕКА
# ============================================================

def path_to_band(path_world: List[Tuple[float, float]], width=HUMAN_SIZE[0]):
    """
    Преобразует линию пути в набор четырёхугольников (полоса ширины человека)
    """
    half = width / 2
    polys = []

    for (x1, y1), (x2, y2) in zip(path_world[:-1], path_world[1:]):
        dx = x2 - x1
        dy = y2 - y1
        L = math.hypot(dx, dy)
        if L == 0:
            continue

        nx = -dy / L
        ny = dx / L

        p1 = (x1 + nx * half, y1 + ny * half)
        p2 = (x1 - nx * half, y1 - ny * half)
        p3 = (x2 - nx * half, y2 - ny * half)
        p4 = (x2 + nx * half, y2 + ny * half)

        polys.append([p1, p2, p3, p4])

    return polys