#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Разметка комнат в SVG по линиям стен.
1. Читает план (src/data/plan.svg)
2. Извлекает линии стен
3. Продлевает линии до пересечений
4. Находит замкнутые области (комнаты)
5. Сохраняет PNG с раскрашенными комнатами
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import polygonize, unary_union
from svgpathtools import svg2paths
from itertools import combinations


def svg_lines_to_segments(svg_path):
    """Извлекает все линейные сегменты из SVG"""
    paths, _ = svg2paths(svg_path)
    segments = []
    for path in paths:
        for seg in path:
            # svgpathtools segment → LineString
            start = (seg.start.real, seg.start.imag)
            end = (seg.end.real, seg.end.imag)
            segments.append(LineString([start, end]))
    return segments


def extend_lines_to_intersections(segments, tol=50.0):
    """Продлевает линии до ближайших пересечений"""
    extended = []
    for line in segments:
        x1, y1, x2, y2 = *line.coords[0], *line.coords[-1]
        v = np.array([x2 - x1, y2 - y1])
        v = v / np.linalg.norm(v)
        # продлеваем в обе стороны
        new_line = LineString([
            (x1 - v[0] * tol, y1 - v[1] * tol),
            (x2 + v[0] * tol, y2 + v[1] * tol)
        ])
        extended.append(new_line)
    return extended


def find_polygons(segments):
    """Находит замкнутые области (полигоны) из набора линий"""
    merged = unary_union(segments)
    polygons = list(polygonize(merged))
    return polygons


def draw_colored_plan(polygons, output_path, figsize=(8, 8)):
    """Сохраняет PNG с раскрашенными полигонами"""
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.axis("off")

    for poly in polygons:
        color = (
            random.random(),
            random.random(),
            random.random(),
            0.5
        )
        x, y = poly.exterior.xy
        plt.fill(x, y, color=color, linewidth=1.5, edgecolor='white')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def main():
    svg_path = os.path.join("src", "data", "input", "plan.svg")
    output_path = os.path.join("src","data", "output", "plan_colored.png")
    os.makedirs(os.path.join("src","data", "output"), exist_ok=True)

    print(f"[INFO] Чтение {svg_path}")
    segments = svg_lines_to_segments(svg_path)
    print(f"[INFO] Найдено {len(segments)} сегментов")

    extended = extend_lines_to_intersections(segments)
    polygons = find_polygons(extended)

    print(f"[INFO] Найдено {len(polygons)} возможных комнат")

    if not polygons:
        print("[WARN] Комнаты не найдены (проверь замкнутость контуров или масштаб SVG).")
        return

    draw_colored_plan(polygons, output_path)
    print(f"[OK] План с раскрашенными комнатами сохранён → {output_path}")


if __name__ == "__main__":
    main()