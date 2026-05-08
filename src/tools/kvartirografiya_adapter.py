#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET


DEFAULT_INPUT_DIR = Path("data/input/Тестовые_генерации_Квартирография_рф_планировки_2026_05_06")
DEFAULT_OUTPUT_DIR = Path("data/output/kvartirografiya_rooms")
DEFAULT_CEILING_HEIGHT = 2.8
DXF_STRUCTURAL_LAYERS = {
    "Room Walls",
    "Inner Walls",
    "Outer Walls",
    "One-Bedroom Apartment",
    "Two-Bedroom Apartment",
    "Three-Bedroom Apartment",
    "Studio Apartment",
}

ROOM_TYPE_MAP = {
    "BEDROOM": "bedroom",
    "KITCHEN": "kitchen",
    "STUDIO": "living_room",
    "LIVING_ROOM": "living_room",
    "ROOM": "living_room",
    "HALL": "hallway",
    "CORRIDOR": "hallway",
    "BATHROOM": "bathroom",
    "JOINT_BATHROOM": "bathroom",
    "TOILET": "bathroom",
}

DEFAULT_INCLUDE_ROOM_TYPES = tuple(ROOM_TYPE_MAP.keys())

XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass(frozen=True)
class FloorInput:
    project_id: str
    floor: int
    project_dir: Path
    geojson_path: Path
    dxf_path: Path | None
    floor_xlsx_path: Path | None
    project_xlsx_path: Path | None
    floor_jpg_path: Path | None
    overview_jpg_path: Path | None


def parse_floor_from_name(path: Path) -> int | None:
    match = re.search(r"(?:floor_|dxfFile_\d+_|xlsxFile_\d+_)(-?\d+)", path.stem)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)(?:й|$)", path.stem)
    if match:
        return int(match.group(1))
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_docx_text(path: Path, limit: int = 3000) -> str:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def read_xlsx_preview(path: Path, max_rows: int = 40) -> dict[str, Any]:
    with ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", XLSX_NS):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", XLSX_NS)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        sheets: list[dict[str, Any]] = []
        for sheet in workbook.findall("a:sheets/a:sheet", XLSX_NS):
            name = sheet.attrib.get("name", "Sheet")
            rid = sheet.attrib[f"{{{XLSX_NS['r']}}}id"]
            target = rel_targets[rid].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            ws = ET.fromstring(zf.read(target))
            rows: list[list[str]] = []
            for row in ws.findall("a:sheetData/a:row", XLSX_NS)[:max_rows]:
                values: list[str] = []
                for cell in row.findall("a:c", XLSX_NS):
                    value_node = cell.find("a:v", XLSX_NS)
                    value = "" if value_node is None or value_node.text is None else value_node.text
                    if cell.attrib.get("t") == "s" and value:
                        value = shared[int(value)]
                    values.append(value)
                rows.append(values)
            sheets.append({"name": name, "rows": rows})
        return {"path": str(path), "sheets": sheets}


def discover_floors(input_dir: Path) -> list[FloorInput]:
    floors: list[FloorInput] = []
    if any(input_dir.glob("project_*_floor_*.geojson")):
        project_dirs = [input_dir]
    else:
        project_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    for project_dir in project_dirs:
        project_id = project_dir.name
        project_xlsx = project_dir / f"xlsxFile_{project_id}.xlsx"
        overview_jpg = next(project_dir.glob("*общий план*.jpg"), None)
        for geojson_path in sorted(project_dir.glob("project_*_floor_*.geojson")):
            floor = parse_floor_from_name(geojson_path)
            if floor is None:
                continue
            dxf_path = project_dir / f"dxfFile_{project_id}_{floor}.dxf"
            floor_xlsx = project_dir / f"xlsxFile_{project_id}_{floor}.xlsx"
            floor_jpg = next(project_dir.glob(f"{project_id}_{floor}*.jpg"), None)
            floors.append(
                FloorInput(
                    project_id=project_id,
                    floor=floor,
                    project_dir=project_dir,
                    geojson_path=geojson_path,
                    dxf_path=dxf_path if dxf_path.is_file() else None,
                    floor_xlsx_path=floor_xlsx if floor_xlsx.is_file() else None,
                    project_xlsx_path=project_xlsx if project_xlsx.is_file() else None,
                    floor_jpg_path=floor_jpg,
                    overview_jpg_path=overview_jpg,
                )
            )
    return floors


def discover_response_block_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file() and input_dir.name.endswith("_response_blocks.json"):
        return [input_dir]
    return sorted(input_dir.rglob("*_response_blocks.json"))


def _geometry_type_counts(value: Any, counts: dict[str, int]) -> None:
    if isinstance(value, dict):
        geometry_type = value.get("type")
        if isinstance(geometry_type, str) and "coordinates" in value:
            counts[geometry_type] = counts.get(geometry_type, 0) + 1
        for child in value.values():
            _geometry_type_counts(child, counts)
    elif isinstance(value, list):
        for child in value:
            _geometry_type_counts(child, counts)


def _floor_plan_counts(section: dict[str, Any]) -> dict[str, int]:
    floor_plans = section.get("floorPlans") or []
    apartments = 0
    service_areas = 0
    rooms = 0
    if isinstance(floor_plans, list):
        for floor_plan in floor_plans:
            if not isinstance(floor_plan, dict):
                continue
            plan_apartments = floor_plan.get("apartments") or []
            plan_service_areas = floor_plan.get("serviceAreas") or []
            if isinstance(plan_apartments, list):
                apartments += len(plan_apartments)
                for apartment in plan_apartments:
                    if isinstance(apartment, dict):
                        generated_rooms = apartment.get("generatedRooms") or apartment.get("rooms") or []
                        if isinstance(generated_rooms, list):
                            rooms += len(generated_rooms)
            if isinstance(plan_service_areas, list):
                service_areas += len(plan_service_areas)
    return {
        "floor_plans": len(floor_plans) if isinstance(floor_plans, list) else 0,
        "apartments": apartments,
        "rooms": rooms,
        "service_areas": service_areas,
    }


def convert_response_blocks(path: Path, out_dir: Path) -> dict[str, Any]:
    payload = read_json(path)
    data = payload.get("data") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        data = {}
    project_id = str(data.get("id") or path.stem.replace("_response_blocks", ""))
    results = data.get("results") or []
    if not isinstance(results, list):
        results = []

    project_out = out_dir / "_response_blocks" / project_id
    variants: list[dict[str, Any]] = []
    geometry_type_counts: dict[str, int] = {}
    total_blocks = 0
    total_sections = 0
    total_floor_plans = 0
    total_apartments = 0
    total_rooms = 0
    total_service_areas = 0

    for result_index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue
        blocks = result.get("blocks") or []
        if not isinstance(blocks, list):
            blocks = []
        variant_dir = project_out / f"variant_{result_index:02d}"
        blocks_out: list[dict[str, Any]] = []
        for block_index, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                continue
            sections = block.get("sections") or []
            if not isinstance(sections, list):
                sections = []
            section_rows: list[dict[str, Any]] = []
            for section_index, section in enumerate(sections, start=1):
                if not isinstance(section, dict):
                    continue
                counts = _floor_plan_counts(section)
                total_floor_plans += counts["floor_plans"]
                total_apartments += counts["apartments"]
                total_rooms += counts["rooms"]
                total_service_areas += counts["service_areas"]
                section_rows.append(
                    {
                        "section_index": section_index,
                        "floors": section.get("floors"),
                        "height": section.get("height"),
                        **counts,
                    }
                )
            block_path = variant_dir / "blocks" / f"block_{block_index:04d}.json"
            write_json(block_path, block)
            blocks_out.append(
                {
                    "block_index": block_index,
                    "block_json": str(block_path),
                    "sections_count": len(sections),
                    "sections": section_rows,
                    "underground_floors": block.get("undergroundFloors"),
                    "balconies": block.get("balconies"),
                }
            )
            total_sections += len(sections)
        _geometry_type_counts(result, geometry_type_counts)
        total_blocks += len(blocks)
        variants.append(
            {
                "variant_index": result_index,
                "blocks_count": len(blocks),
                "blocks": blocks_out,
                "economy_parameters": result.get("economyParameters"),
                "needs_parking": result.get("needsParking"),
            }
        )

    manifest = {
        "source_json": str(path),
        "project_id": project_id,
        "state": data.get("state"),
        "messages": data.get("messages") or [],
        "results_count": len(results),
        "blocks_count": total_blocks,
        "sections_count": total_sections,
        "floor_plans_count": total_floor_plans,
        "apartments_count": total_apartments,
        "rooms_count": total_rooms,
        "service_areas_count": total_service_areas,
        "geometry_type_counts": geometry_type_counts,
        "variants": variants,
        "notes": (
            "This response_blocks export contains building block/section geometry. "
            "Apartment room bundles can be generated only when floorPlans.apartments contains rooms/generatedRooms."
        ),
    }
    manifest_path = project_out / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_json"] = str(manifest_path)
    return manifest


def iter_polygon_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    if geometry.get("type") == "Polygon":
        rings = geometry.get("coordinates") or []
        return [rings[0]] if rings else []
    if geometry.get("type") == "MultiPolygon":
        out = []
        for polygon in geometry.get("coordinates") or []:
            if polygon:
                out.append(polygon[0])
        return out
    return []


def feature_centroid_lonlat(feature: dict[str, Any]) -> tuple[float, float] | None:
    rings = iter_polygon_rings(feature.get("geometry") or {})
    if not rings:
        return None
    pts = rings[0]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def collect_origin(features: list[dict[str, Any]]) -> tuple[float, float]:
    points: list[tuple[float, float]] = []
    for feature in features:
        for ring in iter_polygon_rings(feature.get("geometry") or {}):
            points.extend((float(p[0]), float(p[1])) for p in ring)
    if not points:
        raise ValueError("GeoJSON does not contain polygon coordinates")
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def lonlat_to_local(point: list[float], origin: tuple[float, float]) -> tuple[float, float]:
    lon0, lat0 = origin
    lon, lat = float(point[0]), float(point[1])
    meters_per_lon = 111_320.0 * math.cos(math.radians(lat0))
    meters_per_lat = 110_540.0
    return (lon - lon0) * meters_per_lon, (lat - lat0) * meters_per_lat


def parse_dxf_line_entities(path: Path, layers: set[str] | None = None) -> list[dict[str, Any]]:
    lines = path.read_text(errors="ignore").splitlines()
    entities: list[dict[str, Any]] = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].strip() == "0" and lines[i + 1].strip() == "LINE":
            i += 2
            pairs: dict[str, list[str]] = {}
            while i < len(lines) - 1 and lines[i].strip() != "0":
                pairs.setdefault(lines[i].strip(), []).append(lines[i + 1].strip())
                i += 2
            layer = pairs.get("8", [""])[0]
            if layers is not None and layer not in layers:
                continue
            try:
                entities.append(
                    {
                        "layer": layer,
                        "start": (float(pairs["10"][0]) / 1000.0, float(pairs["20"][0]) / 1000.0),
                        "end": (float(pairs["11"][0]) / 1000.0, float(pairs["21"][0]) / 1000.0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        else:
            i += 1
    return entities


def _geojson_reference_points(features: list[dict[str, Any]], origin: tuple[float, float]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for feature in features:
        if (feature.get("properties") or {}).get("type") not in {"apartment", "room", "block", "section"}:
            continue
        for ring in iter_polygon_rings(feature.get("geometry") or {}):
            points.extend(lonlat_to_local(point, origin) for point in ring)
    return points


def estimate_dxf_to_local_transform(
    *,
    dxf_path: Path | None,
    features: list[dict[str, Any]],
    origin: tuple[float, float],
) -> dict[str, Any] | None:
    if dxf_path is None or not dxf_path.is_file():
        return None
    try:
        import numpy as np
    except Exception:
        return None

    lines = parse_dxf_line_entities(dxf_path, DXF_STRUCTURAL_LAYERS)
    src_points = [point for line in lines for point in (line["start"], line["end"])]
    dst_points = _geojson_reference_points(features, origin)
    if len(src_points) < 8 or len(dst_points) < 8:
        return None

    src = np.array(src_points, dtype=float)
    dst = np.array(dst_points, dtype=float)

    def trim(points: Any) -> Any:
        if len(points) < 20:
            return points
        qx = np.percentile(points[:, 0], [1, 99])
        qy = np.percentile(points[:, 1], [1, 99])
        mask = (points[:, 0] >= qx[0]) & (points[:, 0] <= qx[1]) & (points[:, 1] >= qy[0]) & (points[:, 1] <= qy[1])
        trimmed = points[mask]
        return trimmed if len(trimmed) >= 8 else points

    src = trim(src)
    dst = trim(dst)

    def pca(points: Any) -> tuple[Any, Any, Any]:
        center = points.mean(axis=0)
        centered = points - center
        cov = centered.T @ centered / max(1, len(centered))
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        std = np.sqrt(np.maximum(vals, 1e-12))
        return center, vecs, std

    src_center, src_basis, src_std = pca(src)
    dst_center, dst_basis, dst_std = pca(dst)
    best: tuple[float, Any, Any, Any] | None = None
    sample_step = max(1, len(src) // 500)
    sample = src[::sample_step]
    for swap in (False, True):
        src_basis2 = src_basis[:, [1, 0]] if swap else src_basis
        src_std2 = src_std[[1, 0]] if swap else src_std
        for sign_x in (-1, 1):
            for sign_y in (-1, 1):
                matrix = dst_basis @ np.diag([sign_x, sign_y]) @ np.diag(dst_std / src_std2) @ src_basis2.T
                pred = (sample - src_center) @ matrix.T + dst_center
                distances: list[float] = []
                for point in pred:
                    d2 = np.sum((dst - point) ** 2, axis=1)
                    distances.append(float(np.sqrt(d2.min())))
                score = float(np.median(distances))
                if best is None or score < best[0]:
                    best = (score, matrix, src_center, dst_center)
    if best is None:
        return None
    score, matrix, src_center, dst_center = best
    return {
        "matrix": matrix.tolist(),
        "src_center": src_center.tolist(),
        "dst_center": dst_center.tolist(),
        "median_reference_error_m": round(score, 4),
        "source": "pca_structural_layers",
    }


def apply_dxf_transform(point: tuple[float, float], transform: dict[str, Any]) -> tuple[float, float]:
    matrix = transform["matrix"]
    src_center = transform["src_center"]
    dst_center = transform["dst_center"]
    x = point[0] - float(src_center[0])
    y = point[1] - float(src_center[1])
    return (
        x * float(matrix[0][0]) + y * float(matrix[0][1]) + float(dst_center[0]),
        x * float(matrix[1][0]) + y * float(matrix[1][1]) + float(dst_center[1]),
    )


def signed_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(signed_area(points))


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)


def closest_point_on_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return a
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return ax + dx * t, ay + dy * t


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polygon_edges(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(points) < 2:
        return []
    return [(points[idx], points[(idx + 1) % len(points)]) for idx in range(len(points))]


def closest_point_on_polygon_boundary(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[tuple[float, float], float]:
    best_point = polygon[0] if polygon else point
    best_dist = float("inf")
    for a, b in polygon_edges(polygon):
        candidate = closest_point_on_segment(point, a, b)
        dist = point_distance(point, candidate)
        if dist < best_dist:
            best_point = candidate
            best_dist = dist
    return best_point, best_dist


def segment_on_nearest_boundary(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
    width: float,
) -> list[list[float]]:
    best_edge: tuple[tuple[float, float], tuple[float, float]] | None = None
    best_dist = float("inf")
    for a, b in polygon_edges(polygon):
        candidate = closest_point_on_segment(point, a, b)
        dist = point_distance(point, candidate)
        if dist < best_dist:
            best_edge = (a, b)
            best_dist = dist
    if best_edge is None:
        x, y = point
        half = max(float(width), 0.1) / 2.0
        return [[round(x - half, 4), round(y, 4)], [round(x + half, 4), round(y, 4)]]
    (ax, ay), (bx, by) = best_edge
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = dx / length, dy / length
    half = max(float(width), 0.1) / 2.0
    x, y = point
    return [
        [round(x - ux * half, 4), round(y - uy * half, 4)],
        [round(x + ux * half, 4), round(y + uy * half, 4)],
    ]


def closest_points_between_polygons(
    a_poly: list[tuple[float, float]],
    b_poly: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    best_a = a_poly[0]
    best_b = b_poly[0]
    best_dist = float("inf")
    for point in a_poly:
        for edge_a, edge_b in polygon_edges(b_poly):
            candidate = closest_point_on_segment(point, edge_a, edge_b)
            dist = point_distance(point, candidate)
            if dist < best_dist:
                best_a, best_b, best_dist = point, candidate, dist
    for point in b_poly:
        for edge_a, edge_b in polygon_edges(a_poly):
            candidate = closest_point_on_segment(point, edge_a, edge_b)
            dist = point_distance(point, candidate)
            if dist < best_dist:
                best_a, best_b, best_dist = candidate, point, dist
    return best_a, best_b, best_dist


def _segment_shared_part(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    tol: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    ab_len = point_distance(a, b)
    cd_len = point_distance(c, d)
    if ab_len < 1e-6 or cd_len < 1e-6:
        return None
    dist_c, tc = _point_segment_distance_and_t(c, a, b)
    dist_d, td = _point_segment_distance_and_t(d, a, b)
    dist_a, _ = _point_segment_distance_and_t(a, c, d)
    dist_b, _ = _point_segment_distance_and_t(b, c, d)
    if not ((dist_c < tol or dist_d < tol) and (dist_a < tol or dist_b < tol)):
        return None
    lo = max(0.0, min(tc, td))
    hi = min(1.0, max(tc, td))
    if hi <= lo:
        return None
    ux = (b[0] - a[0]) / ab_len
    uy = (b[1] - a[1]) / ab_len
    p0 = (a[0] + ux * lo * ab_len, a[1] + uy * lo * ab_len)
    p1 = (a[0] + ux * hi * ab_len, a[1] + uy * hi * ab_len)
    return p0, p1, point_distance(p0, p1)


def _point_segment_distance_and_t(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return point_distance(point, a), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    closest = (ax + dx * t, ay + dy * t)
    return point_distance(point, closest), t


def best_shared_boundary_segment(
    a_poly: list[tuple[float, float]],
    b_poly: list[tuple[float, float]],
    tol: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    best: tuple[tuple[float, float], tuple[float, float], float] | None = None
    for a, b in polygon_edges(a_poly):
        for c, d in polygon_edges(b_poly):
            shared = _segment_shared_part(a, b, c, d, tol=tol)
            if shared is not None and (best is None or shared[2] > best[2]):
                best = shared
    return best


def segment_length(line: list[list[float]]) -> float:
    if len(line) < 2:
        return 0.0
    return point_distance((line[0][0], line[0][1]), (line[-1][0], line[-1][1]))


def distance_point_to_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return point_distance(point, closest_point_on_segment(point, a, b))


def distance_point_to_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> float:
    if point_in_polygon(point, polygon):
        return 0.0
    _, dist = closest_point_on_polygon_boundary(point, polygon)
    return dist


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def normalize_polygon(points: list[tuple[float, float]]) -> list[dict[str, float]]:
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    normalized = [(x - min_x, y - min_y) for x, y in points]
    if signed_area(normalized) < 0:
        normalized = list(reversed(normalized))
    rounded = [{"x": round(x, 4), "y": round(y, 4)} for x, y in normalized]
    deduped: list[dict[str, float]] = []
    for point in rounded:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def _dominant_edge_angle(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    best_a, best_b = points[0], points[1]
    best_len = -1.0
    for a, b in polygon_edges(points):
        length = point_distance(a, b)
        if length > best_len:
            best_len = length
            best_a, best_b = a, b
    return math.atan2(best_b[1] - best_a[1], best_b[0] - best_a[0])


def _rotate_point(point: tuple[float, float], angle_rad: float) -> tuple[float, float]:
    x, y = point
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return x * c - y * s, x * s + y * c


def make_oriented_room_frame(points: list[tuple[float, float]]) -> dict[str, Any]:
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    angle = _dominant_edge_angle(points)
    origin = points[0] if points else (0.0, 0.0)
    rotated = [
        _rotate_point((x - origin[0], y - origin[1]), -angle)
        for x, y in points
    ]
    min_x = min((x for x, _ in rotated), default=0.0)
    min_y = min((y for _, y in rotated), default=0.0)
    return {
        "origin_xy": [float(origin[0]), float(origin[1])],
        "rotation_rad": float(angle),
        "rotation_deg": float(math.degrees(angle)),
        "offset_xy": [float(-min_x), float(-min_y)],
    }


def apply_oriented_room_frame(point: tuple[float, float], frame: dict[str, Any]) -> tuple[float, float]:
    origin = frame.get("origin_xy") or [0.0, 0.0]
    offset = frame.get("offset_xy") or [0.0, 0.0]
    angle = float(frame.get("rotation_rad") or 0.0)
    rotated = _rotate_point((point[0] - float(origin[0]), point[1] - float(origin[1])), -angle)
    return rotated[0] + float(offset[0]), rotated[1] + float(offset[1])


def normalize_polygon_oriented(points: list[tuple[float, float]], frame: dict[str, Any]) -> list[dict[str, float]]:
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    normalized = [apply_oriented_room_frame(point, frame) for point in points]
    if signed_area(normalized) < 0:
        normalized = list(reversed(normalized))
    rounded = [{"x": round(x, 4), "y": round(y, 4)} for x, y in normalized]
    deduped: list[dict[str, float]] = []
    for point in rounded:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    if len(deduped) > 1 and deduped[0] == deduped[-1]:
        deduped.pop()
    return deduped


def transform_line_with_room_frame(line_xy: Any, frame: dict[str, Any]) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(line_xy, list):
        return out
    for point in line_xy:
        if isinstance(point, list) and len(point) >= 2:
            x, y = apply_oriented_room_frame((float(point[0]), float(point[1])), frame)
            out.append([round(x, 4), round(y, 4)])
    return out


def make_walls(count: int) -> list[dict[str, Any]]:
    return [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % count} for i in range(count)]


def bounds(points: list[dict[str, float]]) -> tuple[float, float, float, float]:
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def local_offset(points: list[tuple[float, float]]) -> tuple[float, float]:
    return min(x for x, _ in points), min(y for _, y in points)


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "room"


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _geometry_coordinates(geometry: Any) -> Any:
    if isinstance(geometry, dict):
        return geometry.get("coordinates")
    return None


def _geometry_point_to_local(
    geometry: Any,
    *,
    origin: tuple[float, float],
    coordinate_offset: tuple[float, float],
) -> tuple[float, float] | None:
    coords = _geometry_coordinates(geometry)
    if not (isinstance(coords, list) and len(coords) >= 2):
        return None
    point = lonlat_to_local([float(coords[0]), float(coords[1])], origin)
    return point[0] - coordinate_offset[0], point[1] - coordinate_offset[1]


def _geometry_line_to_local(
    geometry: Any,
    *,
    origin: tuple[float, float],
    coordinate_offset: tuple[float, float],
) -> list[list[float]]:
    coords = _geometry_coordinates(geometry)
    if not isinstance(coords, list):
        return []
    points: list[list[float]] = []
    for coord in coords:
        if isinstance(coord, list) and len(coord) >= 2:
            point = lonlat_to_local([float(coord[0]), float(coord[1])], origin)
            points.append([round(point[0] - coordinate_offset[0], 4), round(point[1] - coordinate_offset[1], 4)])
    return points


def _room_distance_to_local_point(room: dict[str, Any], point_xy: tuple[float, float], offset: tuple[float, float]) -> float:
    polygon = room.get("_global_polygon") or []
    global_point = (point_xy[0] + offset[0], point_xy[1] + offset[1])
    _, dist = closest_point_on_polygon_boundary(global_point, polygon)
    return dist


def _nearest_room_id(
    child_rooms: list[dict[str, Any]],
    point_xy: tuple[float, float],
    offset: tuple[float, float],
) -> str | None:
    if not child_rooms:
        return None
    room = min(child_rooms, key=lambda row: _room_distance_to_local_point(row, point_xy, offset))
    return str(room["room_id"])


def _nearest_two_room_ids(
    child_rooms: list[dict[str, Any]],
    point_xy: tuple[float, float],
    offset: tuple[float, float],
) -> tuple[str | None, str | None]:
    ranked = sorted(child_rooms, key=lambda row: _room_distance_to_local_point(row, point_xy, offset))
    first = str(ranked[0]["room_id"]) if ranked else None
    second = str(ranked[1]["room_id"]) if len(ranked) > 1 else None
    return first, second


def _linear_object_center(
    linear_object: dict[str, Any],
    *,
    origin: tuple[float, float],
    coordinate_offset: tuple[float, float],
) -> tuple[float, float] | None:
    point = _geometry_point_to_local(linear_object.get("point"), origin=origin, coordinate_offset=coordinate_offset)
    if point is not None:
        return point
    line = _geometry_line_to_local(linear_object.get("line"), origin=origin, coordinate_offset=coordinate_offset)
    if line:
        return sum(p[0] for p in line) / len(line), sum(p[1] for p in line) / len(line)
    return None


def _linear_object_width(linear_object: dict[str, Any], default: float) -> float:
    try:
        return float(linear_object.get("width"))
    except (TypeError, ValueError):
        return default


def _real_door_candidates(props: dict[str, Any], *names: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in names:
        for item in _as_list(props.get(name)):
            if isinstance(item, dict):
                out.append(item)
    return out


def _assign_door_ids_to_child_rooms(doors: list[dict[str, Any]], child_rooms: list[dict[str, Any]], key: str) -> None:
    door_ids_by_room: dict[str, list[str]] = {str(room["room_id"]): [] for room in child_rooms}
    for door in doors:
        if door.get("kind") == "entrance":
            to_room = door.get("to")
            if to_room in door_ids_by_room:
                door_ids_by_room[str(to_room)].append(str(door["id"]))
        else:
            for endpoint in ("from", "to"):
                room_id = door.get(endpoint)
                if room_id in door_ids_by_room:
                    door_ids_by_room[str(room_id)].append(str(door["id"]))
    for room in child_rooms:
        room[key] = door_ids_by_room.get(str(room["room_id"]), [])


def _wall_opening_from_center(
    *,
    opening_id: str,
    polygon: list[dict[str, float]],
    center_xy: list[float],
    width_m: float,
    kind: str,
    connects_to: str | None = None,
    line_xy: list[list[float]] | None = None,
) -> dict[str, Any] | None:
    if len(polygon) < 2 or len(center_xy) < 2:
        return None
    center = (float(center_xy[0]), float(center_xy[1]))
    edges = []
    for idx in range(len(polygon)):
        a = (float(polygon[idx]["x"]), float(polygon[idx]["y"]))
        b = (float(polygon[(idx + 1) % len(polygon)]["x"]), float(polygon[(idx + 1) % len(polygon)]["y"]))
        closest = closest_point_on_segment(center, a, b)
        edges.append((point_distance(center, closest), idx, a, b, closest))
    _, wall_idx, a, b, closest = min(edges, key=lambda row: row[0])
    wall_len = max(point_distance(a, b), 1e-9)
    wall_id = f"w{wall_idx}"

    if line_xy and len(line_xy) >= 2:
        p0 = (float(line_xy[0][0]), float(line_xy[0][1]))
        p1 = (float(line_xy[-1][0]), float(line_xy[-1][1]))
        seg_len = point_distance(p0, p1)
        if seg_len > 1e-6:
            width_m = seg_len
            segment = {"x1": round(p0[0], 4), "y1": round(p0[1], 4), "x2": round(p1[0], 4), "y2": round(p1[1], 4)}
            s = point_distance(a, closest_point_on_segment(p0, a, b))
        else:
            line_xy = None
    if not line_xy or len(line_xy) < 2:
        ux = (b[0] - a[0]) / wall_len
        uy = (b[1] - a[1]) / wall_len
        half = max(float(width_m), 0.1) / 2.0
        s_center = max(0.0, min(wall_len, point_distance(a, closest)))
        s0 = max(0.0, s_center - half)
        s1 = min(wall_len, s_center + half)
        p0 = (a[0] + ux * s0, a[1] + uy * s0)
        p1 = (a[0] + ux * s1, a[1] + uy * s1)
        segment = {"x1": round(p0[0], 4), "y1": round(p0[1], 4), "x2": round(p1[0], 4), "y2": round(p1[1], 4)}
        s = s0

    out = {
        "id": opening_id,
        "wall_id": wall_id,
        "s": round(float(s), 4),
        "width": round(float(width_m), 4),
        "z0": 0.9 if kind == "window" else 0.0,
        "height": 1.2 if kind == "window" else 2.05,
        "segment": segment,
    }
    if kind == "window":
        out["glazing"] = "double"
    else:
        out["swing"] = {"hinge": "unknown", "direction": "unknown"}
        out["connects_to"] = connects_to or "unknown"
    return out


def _translate_line(line_xy: Any, dx: float, dy: float) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(line_xy, list):
        return out
    for point in line_xy:
        if isinstance(point, list) and len(point) >= 2:
            out.append([round(float(point[0]) + dx, 4), round(float(point[1]) + dy, 4)])
    return out


def build_apartment_openings(
    *,
    apartment_polygon_local: list[dict[str, float]],
    door_graph: dict[str, Any],
    window_graph: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doors: list[dict[str, Any]] = []
    for door in door_graph.get("doors") or []:
        if door.get("kind") != "entrance":
            continue
        opening = _wall_opening_from_center(
            opening_id=str(door.get("id") or f"door_{len(doors)}"),
            polygon=apartment_polygon_local,
            center_xy=door.get("center_xy") or [],
            width_m=float(door.get("width_m") or 0.9),
            kind="door",
            connects_to=str(door.get("to") or "unknown"),
            line_xy=door.get("line_xy"),
        )
        if opening:
            doors.append(opening)

    windows: list[dict[str, Any]] = []
    for window in window_graph.get("windows") or []:
        opening = _wall_opening_from_center(
            opening_id=str(window.get("id") or f"win_{len(windows)}"),
            polygon=apartment_polygon_local,
            center_xy=window.get("center_xy") or [],
            width_m=float(window.get("width_m") or 1.2),
            kind="window",
            line_xy=window.get("line_xy"),
        )
        if opening:
            windows.append(opening)
    return doors, windows


def build_child_room_openings(
    *,
    room: dict[str, Any],
    door_graph: dict[str, Any],
    window_graph: dict[str, Any],
    apartment_offset: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    room_id = str(room["room_id"])
    room_frame = room.get("_room_frame") or make_oriented_room_frame(room.get("_global_polygon") or [])
    polygon = room.get("_local_polygon") or normalize_polygon_oriented(room.get("_global_polygon") or [], room_frame)

    doors: list[dict[str, Any]] = []
    for door in door_graph.get("doors") or []:
        if room_id not in {str(door.get("from")), str(door.get("to"))}:
            continue
        center = door.get("center_xy") or []
        if not (isinstance(center, list) and len(center) >= 2):
            continue
        if door.get("kind") == "entrance":
            connects_to = "outside"
        else:
            other = door.get("to") if str(door.get("from")) == room_id else door.get("from")
            connects_to = str(other or "unknown")
        global_center = (float(center[0]) + float(apartment_offset[0]), float(center[1]) + float(apartment_offset[1]))
        local_center = apply_oriented_room_frame(global_center, room_frame)
        line_global = []
        for point in door.get("line_xy") or []:
            if isinstance(point, list) and len(point) >= 2:
                line_global.append([float(point[0]) + float(apartment_offset[0]), float(point[1]) + float(apartment_offset[1])])
        opening = _wall_opening_from_center(
            opening_id=str(door.get("id") or f"door_{len(doors)}"),
            polygon=polygon,
            center_xy=[round(local_center[0], 4), round(local_center[1], 4)],
            width_m=float(door.get("width_m") or 0.8),
            kind="door",
            connects_to=connects_to,
            line_xy=transform_line_with_room_frame(line_global, room_frame),
        )
        if opening:
            doors.append(opening)

    windows: list[dict[str, Any]] = []
    for window in window_graph.get("windows") or []:
        if str(window.get("room_id")) != room_id:
            continue
        center = window.get("center_xy") or []
        if not (isinstance(center, list) and len(center) >= 2):
            continue
        global_center = (float(center[0]) + float(apartment_offset[0]), float(center[1]) + float(apartment_offset[1]))
        local_center = apply_oriented_room_frame(global_center, room_frame)
        line_global = []
        for point in window.get("line_xy") or []:
            if isinstance(point, list) and len(point) >= 2:
                line_global.append([float(point[0]) + float(apartment_offset[0]), float(point[1]) + float(apartment_offset[1])])
        opening = _wall_opening_from_center(
            opening_id=str(window.get("id") or f"win_{len(windows)}"),
            polygon=polygon,
            center_xy=[round(local_center[0], 4), round(local_center[1], 4)],
            width_m=float(window.get("width_m") or 1.2),
            kind="window",
            line_xy=transform_line_with_room_frame(line_global, room_frame),
        )
        if opening:
            windows.append(opening)
    return doors, windows


def build_real_door_graph(
    *,
    apartment_feature: dict[str, Any],
    child_rooms: list[dict[str, Any]],
    origin: tuple[float, float],
    coordinate_offset: tuple[float, float],
) -> dict[str, Any] | None:
    props = apartment_feature.get("properties") or {}
    doors: list[dict[str, Any]] = []

    for index, item in enumerate(_real_door_candidates(props, "entranceDoor", "entranceDoors", "entrance_door"), start=1):
        center = _linear_object_center(item, origin=origin, coordinate_offset=coordinate_offset)
        if center is None:
            continue
        to_room = _nearest_room_id(child_rooms, center, coordinate_offset)
        line = _geometry_line_to_local(item.get("line"), origin=origin, coordinate_offset=coordinate_offset)
        doors.append(
            {
                "id": f"door_real_entrance_{index:03d}",
                "kind": "entrance",
                "from": "outside",
                "to": to_room,
                "center_xy": [round(center[0], 4), round(center[1], 4)],
                "line_xy": line,
                "width_m": _linear_object_width(item, 0.9),
                "source": "real_generated_linear_object",
                "raw": item,
            }
        )

    for index, item in enumerate(_real_door_candidates(props, "interiorDoors", "interiorDoor", "doors"), start=1):
        center = _linear_object_center(item, origin=origin, coordinate_offset=coordinate_offset)
        if center is None:
            continue
        from_room, to_room = _nearest_two_room_ids(child_rooms, center, coordinate_offset)
        line = _geometry_line_to_local(item.get("line"), origin=origin, coordinate_offset=coordinate_offset)
        doors.append(
            {
                "id": f"door_real_internal_{index:03d}",
                "kind": "internal",
                "from": from_room,
                "to": to_room,
                "center_xy": [round(center[0], 4), round(center[1], 4)],
                "line_xy": line,
                "width_m": _linear_object_width(item, 0.8),
                "source": "real_generated_linear_object",
                "raw": item,
            }
        )

    room_door_index = 1
    for room in child_rooms:
        room_props = room.get("_feature_properties") or {}
        for item in _real_door_candidates(room_props, "doors", "door"):
            center = _linear_object_center(item, origin=origin, coordinate_offset=coordinate_offset)
            if center is None:
                continue
            from_room = str(room["room_id"])
            nearest_a, nearest_b = _nearest_two_room_ids(child_rooms, center, coordinate_offset)
            to_room = nearest_b if nearest_a == from_room else nearest_a
            line = _geometry_line_to_local(item.get("line"), origin=origin, coordinate_offset=coordinate_offset)
            doors.append(
                {
                    "id": f"door_real_room_{room_door_index:03d}",
                    "kind": "internal",
                    "from": from_room,
                    "to": to_room,
                    "center_xy": [round(center[0], 4), round(center[1], 4)],
                    "line_xy": line,
                    "width_m": _linear_object_width(item, 0.8),
                    "source": "real_generated_room_door",
                    "raw": item,
                }
            )
            room_door_index += 1

    if not doors:
        return None

    _assign_door_ids_to_child_rooms(doors, child_rooms, "real_door_ids")
    linked_rooms = set()
    for door in doors:
        for endpoint in ("from", "to"):
            room_id = door.get(endpoint)
            if isinstance(room_id, str) and room_id != "outside":
                linked_rooms.add(room_id)

    entrance = next((door for door in doors if door.get("kind") == "entrance"), None)
    return {
        "entrance_room_id": entrance.get("to") if entrance else None,
        "doors": doors,
        "real_doors_count": len(doors),
        "internal_doors_count": sum(1 for door in doors if door.get("kind") == "internal"),
        "is_connected": len(linked_rooms) == len(child_rooms) if child_rooms else True,
        "coordinate_system": "apartment_local_xy",
        "strategy": "real_generated_linear_objects",
    }


def _combine_real_with_synthetic_completion(
    real_graph: dict[str, Any] | None,
    synthetic_graph: dict[str, Any],
    child_rooms: list[dict[str, Any]],
) -> dict[str, Any]:
    if real_graph is None:
        _assign_door_ids_to_child_rooms(synthetic_graph.get("doors") or [], child_rooms, "door_ids")
        out = dict(synthetic_graph)
        out["source"] = "synthetic"
        return out

    real_doors = list(real_graph.get("doors") or [])
    if real_graph.get("is_connected"):
        _assign_door_ids_to_child_rooms(real_doors, child_rooms, "door_ids")
        out = dict(real_graph)
        out["source"] = "real"
        return out

    real_pairs = {
        (door.get("kind"), door.get("from"), door.get("to"))
        for door in real_doors
    }
    completion_doors: list[dict[str, Any]] = []
    for door in synthetic_graph.get("doors") or []:
        pair = (door.get("kind"), door.get("from"), door.get("to"))
        reverse_pair = (door.get("kind"), door.get("to"), door.get("from"))
        if pair in real_pairs or reverse_pair in real_pairs:
            continue
        completed = dict(door)
        completed["id"] = f"door_synthetic_completion_{len(completion_doors) + 1:03d}"
        completed["source"] = "synthetic_connectivity_completion"
        completion_doors.append(completed)

    doors = real_doors + completion_doors
    _assign_door_ids_to_child_rooms(doors, child_rooms, "door_ids")
    return {
        "entrance_room_id": real_graph.get("entrance_room_id") or synthetic_graph.get("entrance_room_id"),
        "doors": doors,
        "real_doors_count": len(real_doors),
        "synthetic_completion_doors_count": len(completion_doors),
        "internal_doors_count": sum(1 for door in doors if door.get("kind") == "internal"),
        "is_connected": True,
        "coordinate_system": "apartment_local_xy",
        "source": "real_with_synthetic_completion",
        "strategy": "real_generated_linear_objects_then_minimum_connectivity_completion",
    }


def extract_dxf_windows(
    dxf_path: Path | None,
    transform: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if dxf_path is None or transform is None:
        return []
    lines = parse_dxf_line_entities(dxf_path, {"Windows"})
    if not lines:
        return []

    parent = list(range(len(lines)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    endpoint_owner: dict[tuple[int, int], int] = {}
    for idx, line in enumerate(lines):
        for point in (line["start"], line["end"]):
            key = (round(point[0] * 1000), round(point[1] * 1000))
            if key in endpoint_owner:
                union(idx, endpoint_owner[key])
            else:
                endpoint_owner[key] = idx

    groups: dict[int, list[dict[str, Any]]] = {}
    for idx, line in enumerate(lines):
        groups.setdefault(find(idx), []).append(line)

    windows: list[dict[str, Any]] = []
    for group_index, group_lines in enumerate(groups.values(), start=1):
        transformed_lines: list[list[list[float]]] = []
        transformed_points: list[tuple[float, float]] = []
        for line in group_lines:
            a = apply_dxf_transform(line["start"], transform)
            b = apply_dxf_transform(line["end"], transform)
            transformed_lines.append([[round(a[0], 4), round(a[1], 4)], [round(b[0], 4), round(b[1], 4)]])
            transformed_points.extend([a, b])
        if not transformed_points:
            continue
        center = (
            sum(x for x, _ in transformed_points) / len(transformed_points),
            sum(y for _, y in transformed_points) / len(transformed_points),
        )
        longest = max(transformed_lines, key=segment_length)
        width = segment_length(longest)
        # Skip legend samples and degenerate marks.
        if width < 0.15 or width > 8.0:
            continue
        windows.append(
            {
                "id": f"window_dxf_{group_index:04d}",
                "kind": "window",
                "center_xy_global": [round(center[0], 4), round(center[1], 4)],
                "line_xy_global": longest,
                "width_m": round(width, 4),
                "source": "dxf_windows_layer",
                "dxf_line_count": len(group_lines),
            }
        )
    return windows


def _response_block_floor_elevation(payload: dict[str, Any], floor: int) -> float | None:
    elevations: set[float] = set()
    data = payload.get("data") if isinstance(payload, dict) else {}
    results = data.get("results") if isinstance(data, dict) else []
    if not isinstance(results, list):
        return None
    for result in results:
        if not isinstance(result, dict):
            continue
        for point in result.get("insolationPoints") or []:
            if not isinstance(point, dict) or point.get("elevation") is None:
                continue
            try:
                elevations.add(float(point["elevation"]))
            except (TypeError, ValueError):
                continue
    ordered = sorted(elevations)
    if 1 <= int(floor) <= len(ordered):
        return ordered[int(floor) - 1]
    return None


def _score_insolation_points_for_floor(
    points: list[tuple[float, float]],
    apartment_polygons: list[list[tuple[float, float]]],
) -> tuple[int, float]:
    if not points or not apartment_polygons:
        return 0, float("inf")
    close_count = 0
    total_dist = 0.0
    for point in points:
        dist = min(distance_point_to_polygon(point, polygon) for polygon in apartment_polygons)
        total_dist += dist
        if dist <= 2.0:
            close_count += 1
    return close_count, total_dist / max(1, len(points))


def extract_response_block_windows(
    *,
    response_blocks_path: Path | None,
    floor: int,
    origin: tuple[float, float],
    apartment_polygons: list[list[tuple[float, float]]],
    default_width_m: float = 1.0,
) -> list[dict[str, Any]]:
    if response_blocks_path is None or not response_blocks_path.is_file():
        return []
    payload = read_json(response_blocks_path)
    target_elevation = _response_block_floor_elevation(payload, floor)
    if target_elevation is None:
        return []
    data = payload.get("data") if isinstance(payload, dict) else {}
    results = data.get("results") if isinstance(data, dict) else []
    if not isinstance(results, list):
        return []

    best_result_index = None
    best_points: list[dict[str, Any]] = []
    best_score = (-1, float("inf"))
    for result_index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue
        raw_points = []
        local_points = []
        for point in result.get("insolationPoints") or []:
            if not isinstance(point, dict):
                continue
            try:
                elevation = float(point.get("elevation"))
                lon = float(point.get("x"))
                lat = float(point.get("y"))
            except (TypeError, ValueError):
                continue
            if abs(elevation - target_elevation) > 1e-6:
                continue
            local = lonlat_to_local([lon, lat], origin)
            raw_points.append(point)
            local_points.append(local)
        score = _score_insolation_points_for_floor(local_points, apartment_polygons)
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best_points = raw_points
            best_result_index = result_index

    windows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, point in enumerate(best_points, start=1):
        try:
            center = lonlat_to_local([float(point["x"]), float(point["y"])], origin)
        except (KeyError, TypeError, ValueError):
            continue
        key = (round(center[0] * 1000), round(center[1] * 1000))
        if key in seen:
            continue
        seen.add(key)
        windows.append(
            {
                "id": f"window_insolation_floor_{floor:02d}_{index:04d}",
                "kind": "window",
                "center_xy_global": [round(center[0], 4), round(center[1], 4)],
                "line_xy_global": [],
                "width_m": default_width_m,
                "source": "response_blocks_insolation_points",
                "response_blocks_json": str(response_blocks_path),
                "response_result_index": best_result_index,
                "elevation": target_elevation,
                "insolated": bool(point.get("insolated")),
                "result_close_points_count": best_score[0],
                "result_mean_boundary_distance_m": round(best_score[1], 4) if math.isfinite(best_score[1]) else None,
            }
        )
    return windows


def assign_windows_to_apartment(
    *,
    windows: list[dict[str, Any]],
    apartment_index: int,
    nearest_apartment_by_window_id: dict[str, int],
    apartment_polygon: list[tuple[float, float]],
    child_rooms: list[dict[str, Any]],
    coordinate_offset: tuple[float, float],
    max_boundary_distance_m: float = 1.2,
) -> dict[str, Any]:
    assigned: list[dict[str, Any]] = []
    for window in windows:
        center_global_raw = window.get("center_xy_global")
        if not (isinstance(center_global_raw, list) and len(center_global_raw) == 2):
            continue
        center_global = (float(center_global_raw[0]), float(center_global_raw[1]))
        if nearest_apartment_by_window_id.get(str(window["id"])) != apartment_index:
            continue
        apt_dist = distance_point_to_polygon(center_global, apartment_polygon)
        if apt_dist > max_boundary_distance_m:
            continue
        nearest_room = None
        nearest_dist = float("inf")
        for room in child_rooms:
            polygon = room.get("_global_polygon") or []
            dist = distance_point_to_polygon(center_global, polygon)
            if dist < nearest_dist:
                nearest_room = room
                nearest_dist = dist
        center_local = [round(center_global[0] - coordinate_offset[0], 4), round(center_global[1] - coordinate_offset[1], 4)]
        line_global = window.get("line_xy_global") or []
        if not line_global:
            boundary_polygon = nearest_room.get("_global_polygon") if nearest_room else apartment_polygon
            if not boundary_polygon:
                boundary_polygon = apartment_polygon
            line_global = segment_on_nearest_boundary(center_global, boundary_polygon, float(window.get("width_m") or 1.0))
        line_local = []
        for point in line_global:
            if isinstance(point, list) and len(point) >= 2:
                line_local.append([round(float(point[0]) - coordinate_offset[0], 4), round(float(point[1]) - coordinate_offset[1], 4)])
        row = {
            "id": window["id"],
            "kind": "window",
            "room_id": str(nearest_room["room_id"]) if nearest_room else None,
            "center_xy": center_local,
            "line_xy": line_local,
            "width_m": window["width_m"],
            "source": window["source"],
            "apartment_boundary_distance_m": round(apt_dist, 4),
            "room_boundary_distance_m": round(nearest_dist, 4) if nearest_room else None,
        }
        assigned.append(row)

    window_ids_by_room: dict[str, list[str]] = {str(room["room_id"]): [] for room in child_rooms}
    for window in assigned:
        room_id = window.get("room_id")
        if room_id in window_ids_by_room:
            window_ids_by_room[str(room_id)].append(str(window["id"]))
    for room in child_rooms:
        room["window_ids"] = window_ids_by_room.get(str(room["room_id"]), [])

    sources = sorted({str(window.get("source") or "") for window in assigned if window.get("source")})
    return {
        "source": "mixed" if len(sources) > 1 else (sources[0] if sources else "none"),
        "sources": sources,
        "coordinate_system": "apartment_local_xy",
        "windows_count": len(assigned),
        "windows": assigned,
    }


def build_nearest_apartment_by_window_id(
    windows: list[dict[str, Any]],
    apartment_polygons: list[list[tuple[float, float]]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    if not apartment_polygons:
        return out
    for window in windows:
        center_global_raw = window.get("center_xy_global")
        if not (isinstance(center_global_raw, list) and len(center_global_raw) == 2):
            continue
        center_global = (float(center_global_raw[0]), float(center_global_raw[1]))
        nearest_index = min(
            range(len(apartment_polygons)),
            key=lambda idx: distance_point_to_polygon(center_global, apartment_polygons[idx]),
        )
        out[str(window["id"])] = nearest_index
    return out


def build_synthetic_door_graph(
    *,
    apartment_polygon: list[tuple[float, float]],
    child_rooms: list[dict[str, Any]],
    coordinate_offset: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    if not child_rooms:
        return {
            "entrance_room_id": None,
            "doors": [],
            "internal_doors_count": 0,
            "notes": "No exported child rooms were assigned to this apartment.",
        }

    for room in child_rooms:
        polygon = room.get("_global_polygon") or []
        room["_centroid_xy"] = polygon_centroid(polygon)
        _, boundary_dist = closest_point_on_polygon_boundary(room["_centroid_xy"], apartment_polygon)
        room["_boundary_distance_m"] = boundary_dist

    preferred = [
        room for room in child_rooms
        if str(room.get("source_room_type") or "").upper() in {"HALL", "CORRIDOR"}
    ]
    entrance_room = min(preferred or child_rooms, key=lambda room: float(room.get("_boundary_distance_m") or 0.0))
    entrance_room_id = str(entrance_room["room_id"])
    entrance_center, entrance_boundary_dist = closest_point_on_polygon_boundary(
        entrance_room["_centroid_xy"],
        apartment_polygon,
    )

    offset_x, offset_y = coordinate_offset

    def _local_xy(point: tuple[float, float]) -> list[float]:
        return [round(point[0] - offset_x, 4), round(point[1] - offset_y, 4)]

    doors: list[dict[str, Any]] = [
        {
            "id": "door_entrance",
            "kind": "entrance",
            "from": "outside",
            "to": entrance_room_id,
            "center_xy": _local_xy(entrance_center),
            "width_m": 0.9,
            "source": "synthetic_min_connectivity",
            "boundary_distance_m": round(entrance_boundary_dist, 4),
        }
    ]

    room_by_id = {str(room["room_id"]): room for room in child_rooms}
    connected = {entrance_room_id}
    pending = set(room_by_id) - connected
    door_index = 1
    while pending:
        best: tuple[tuple[float, float], str, str, tuple[float, float], tuple[float, float], float, float] | None = None
        for from_id in connected:
            from_poly = room_by_id[from_id].get("_global_polygon") or []
            for to_id in pending:
                to_poly = room_by_id[to_id].get("_global_polygon") or []
                shared = best_shared_boundary_segment(from_poly, to_poly)
                point_a, point_b, dist = closest_points_between_polygons(from_poly, to_poly)
                if shared is not None and shared[2] >= 0.65:
                    shared_a, shared_b, shared_len = shared
                    midpoint = ((shared_a[0] + shared_b[0]) / 2.0, (shared_a[1] + shared_b[1]) / 2.0)
                    score = (0.0, -shared_len)
                    candidate = (score, from_id, to_id, midpoint, midpoint, dist, shared_len)
                else:
                    shared_len = shared[2] if shared is not None else 0.0
                    score = (1.0, dist)
                    candidate = (score, from_id, to_id, point_a, point_b, dist, shared_len)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            break
        _, from_id, to_id, point_a, point_b, dist, shared_len = best
        center = ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)
        doors.append(
            {
                "id": f"door_internal_{door_index:03d}",
                "kind": "internal",
                "from": from_id,
                "to": to_id,
                "center_xy": _local_xy(center),
                "width_m": 0.8,
                "source": "synthetic_min_connectivity",
                "room_gap_m": round(dist, 4),
                "shared_boundary_m": round(shared_len, 4),
                "placement_confidence": "high" if shared_len >= 0.65 else "low",
            }
        )
        connected.add(to_id)
        pending.remove(to_id)
        door_index += 1

    door_ids_by_room: dict[str, list[str]] = {room_id: [] for room_id in room_by_id}
    for door in doors:
        if door["kind"] == "entrance":
            door_ids_by_room[str(door["to"])].append(str(door["id"]))
        else:
            door_ids_by_room[str(door["from"])].append(str(door["id"]))
            door_ids_by_room[str(door["to"])].append(str(door["id"]))
    for room in child_rooms:
        room["synthetic_door_ids"] = door_ids_by_room.get(str(room["room_id"]), [])

    return {
        "entrance_room_id": entrance_room_id,
        "doors": doors,
        "internal_doors_count": max(0, len(doors) - 1),
        "minimum_internal_doors_for_connectivity": max(0, len(child_rooms) - 1),
        "is_connected": len(connected) == len(child_rooms),
        "coordinate_system": "apartment_local_xy",
        "strategy": "prim_minimum_spanning_tree_from_entrance_room",
    }


def build_room_spec(
    *,
    project_id: str,
    floor: int,
    room_index: int,
    room_feature: dict[str, Any],
    local_points: list[tuple[float, float]],
    apartment_feature: dict[str, Any] | None,
    source_geojson: Path,
) -> dict[str, Any]:
    props = room_feature.get("properties") or {}
    room_type_raw = str(props.get("roomType") or props.get("type") or "ROOM").upper()
    room_type = ROOM_TYPE_MAP.get(room_type_raw, "living_room")
    room_frame = make_oriented_room_frame(local_points)
    polygon = normalize_polygon_oriented(local_points, room_frame)
    x0, y0, x1, y1 = bounds(polygon)
    area = float(props.get("area") or polygon_area([(p["x"], p["y"]) for p in polygon]))
    apt_props = apartment_feature.get("properties") if apartment_feature else {}
    apartment_id = None
    if apartment_feature is not None:
        apartment_id = apartment_feature.get("_adapter_id")

    room_id = sanitize_id(f"kv_{project_id}_floor_{floor}_{room_type_raw.lower()}_{room_index:04d}")
    return {
        "version": "1.0",
        "units": "m",
        "coordinate_system": {"floor_plane": "XY", "up": "Z", "right_handed": True},
        "room": {
            "id": room_id,
            "name": room_id,
            "type": room_type,
            "room_type": room_type,
            "source_room_type": room_type_raw,
            "area_m2": round(area, 4),
            "width_m": round(x1 - x0, 4),
            "depth_m": round(y1 - y0, 4),
            "ceiling_height": DEFAULT_CEILING_HEIGHT,
            "ceiling_height_m": DEFAULT_CEILING_HEIGHT,
            "floor_polygon": polygon,
            "floor_polygon_xz": [{"x": p["x"], "z": p["y"]} for p in polygon],
            "walls": make_walls(len(polygon)),
            "doors": [],
            "windows": [],
            "openings": [],
            "meta": {
                "source": "kvartirografiya_geojson",
                "source_geojson": str(source_geojson),
                "project_id": project_id,
                "floor": floor,
                "feature_properties": props,
                "apartment_id": apartment_id,
                "apartment_properties": apt_props or None,
                "coordinate_frame": {
                    "kind": "oriented_room_local_xy",
                    "source": "dominant_wall_aligned",
                    **room_frame,
                },
                "notes": "Openings are not present in the provided flat GeoJSON export; room geometry is projected from WGS84 to local meters.",
            },
        },
    }


def build_apartment_spec(
    *,
    project_id: str,
    floor: int,
    apartment_index: int,
    apartment_feature: dict[str, Any],
    local_points: list[tuple[float, float]],
    child_rooms: list[dict[str, Any]],
    door_graph: dict[str, Any],
    window_graph: dict[str, Any],
    doors: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    synthetic_door_graph: dict[str, Any],
    real_door_graph: dict[str, Any] | None,
    source_geojson: Path,
) -> dict[str, Any]:
    props = apartment_feature.get("properties") or {}
    polygon = normalize_polygon(local_points)
    x0, y0, x1, y1 = bounds(polygon)
    area = float(props.get("totalArea") or polygon_area([(p["x"], p["y"]) for p in polygon]))
    apartment_id = str(apartment_feature.get("_adapter_id") or f"apt_{apartment_index:04d}")
    room_id = sanitize_id(f"kv_{project_id}_floor_{floor}_{apartment_id}")
    return {
        "version": "1.0",
        "units": "m",
        "coordinate_system": {"floor_plane": "XY", "up": "Z", "right_handed": True},
        "room": {
            "id": room_id,
            "name": room_id,
            "type": "apartment",
            "room_type": "apartment",
            "source_room_type": "APARTMENT",
            "area_m2": round(area, 4),
            "width_m": round(x1 - x0, 4),
            "depth_m": round(y1 - y0, 4),
            "ceiling_height": DEFAULT_CEILING_HEIGHT,
            "ceiling_height_m": DEFAULT_CEILING_HEIGHT,
            "floor_polygon": polygon,
            "floor_polygon_xz": [{"x": p["x"], "z": p["y"]} for p in polygon],
            "walls": make_walls(len(polygon)),
            "doors": doors,
            "windows": windows,
            "openings": [],
            "meta": {
                "source": "kvartirografiya_geojson",
                "source_geojson": str(source_geojson),
                "project_id": project_id,
                "floor": floor,
                "apartment_id": apartment_id,
                "feature_properties": props,
                "child_rooms": child_rooms,
                "door_graph": door_graph,
                "window_graph": window_graph,
                "synthetic_door_graph": synthetic_door_graph,
                "real_door_graph": real_door_graph,
                "notes": "Whole-apartment export uses the apartment boundary as the generation contour; internal rooms are preserved as metadata.",
            },
        },
    }


def convert_floor(floor_input: FloorInput, out_dir: Path, include_room_types: set[str]) -> dict[str, Any]:
    data = read_json(floor_input.geojson_path)
    features = data.get("features") or []
    origin = collect_origin(features)
    dxf_transform = estimate_dxf_to_local_transform(
        dxf_path=floor_input.dxf_path,
        features=features,
        origin=origin,
    )
    dxf_windows = extract_dxf_windows(floor_input.dxf_path, dxf_transform)

    apartments: list[dict[str, Any]] = []
    apartment_polygons: list[list[tuple[float, float]]] = []
    rooms: list[tuple[dict[str, Any], list[tuple[float, float]]]] = []
    type_counts: dict[str, int] = {}

    for feature in features:
        props = feature.get("properties") or {}
        ftype = str(props.get("type") or "")
        rings = iter_polygon_rings(feature.get("geometry") or {})
        if not rings:
            continue
        local = [lonlat_to_local(p, origin) for p in rings[0]]
        if len(local) > 1 and local[0] == local[-1]:
            local = local[:-1]
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
        if ftype == "apartment":
            feature["_adapter_id"] = f"apt_{len(apartments) + 1:04d}"
            apartments.append(feature)
            apartment_polygons.append(local)
        elif ftype == "room":
            room_type = str(props.get("roomType") or "").upper()
            if room_type in include_room_types:
                rooms.append((feature, local))

    floor_out = out_dir / floor_input.project_id / f"floor_{floor_input.floor}"
    rooms_out = floor_out / "rooms"
    room_rows: list[dict[str, Any]] = []
    child_rooms_by_apartment: dict[str, list[dict[str, Any]]] = {}
    for idx, (room_feature, local_points) in enumerate(rooms, start=1):
        center = (
            sum(x for x, _ in local_points) / len(local_points),
            sum(y for _, y in local_points) / len(local_points),
        )
        apartment_feature = None
        for candidate, polygon in zip(apartments, apartment_polygons):
            if point_in_polygon(center, polygon):
                apartment_feature = candidate
                break
        room_spec = build_room_spec(
            project_id=floor_input.project_id,
            floor=floor_input.floor,
            room_index=idx,
            room_feature=room_feature,
            local_points=local_points,
            apartment_feature=apartment_feature,
            source_geojson=floor_input.geojson_path,
        )
        room_path = rooms_out / f"{room_spec['room']['id']}.json"
        write_json(room_path, room_spec)
        room_rows.append(
            {
                "room_json": str(room_path),
                "room_id": room_spec["room"]["id"],
                "room_type": room_spec["room"]["room_type"],
                "source_room_type": room_spec["room"]["source_room_type"],
                "area_m2": room_spec["room"]["area_m2"],
                "apartment_id": room_spec["room"]["meta"]["apartment_id"],
            }
        )
        apartment_id = room_spec["room"]["meta"]["apartment_id"]
        if apartment_id:
            room_frame = make_oriented_room_frame(local_points)
            room_local_polygon = normalize_polygon_oriented(local_points, room_frame)
            child_rooms_by_apartment.setdefault(str(apartment_id), []).append(
                {
                    "room_json": str(room_path),
                    "room_id": room_spec["room"]["id"],
                    "room_type": room_spec["room"]["room_type"],
                    "source_room_type": room_spec["room"]["source_room_type"],
                    "area_m2": room_spec["room"]["area_m2"],
                    "_global_polygon": local_points,
                    "_coordinate_offset": local_offset(local_points),
                    "_room_frame": room_frame,
                    "_local_polygon": room_local_polygon,
                    "_feature_properties": room_feature.get("properties") or {},
                }
            )

    apartments_out = floor_out / "apartments"
    apartment_rows: list[dict[str, Any]] = []
    response_blocks_path = floor_input.project_dir / f"{floor_input.project_id}_response_blocks.json"
    response_windows = extract_response_block_windows(
        response_blocks_path=response_blocks_path if response_blocks_path.is_file() else None,
        floor=floor_input.floor,
        origin=origin,
        apartment_polygons=apartment_polygons,
    )
    floor_windows = response_windows if response_windows else dxf_windows
    nearest_apartment_by_window_id = build_nearest_apartment_by_window_id(floor_windows, apartment_polygons)
    for idx, (apartment_feature, apartment_local_points) in enumerate(zip(apartments, apartment_polygons), start=1):
        apartment_id = str(apartment_feature.get("_adapter_id") or f"apt_{idx:04d}")
        child_rooms = child_rooms_by_apartment.get(apartment_id, [])
        min_x = min(x for x, _ in apartment_local_points)
        min_y = min(y for _, y in apartment_local_points)
        synthetic_door_graph = build_synthetic_door_graph(
            apartment_polygon=apartment_local_points,
            child_rooms=child_rooms,
            coordinate_offset=(min_x, min_y),
        )
        real_door_graph = build_real_door_graph(
            apartment_feature=apartment_feature,
            child_rooms=child_rooms,
            origin=origin,
            coordinate_offset=(min_x, min_y),
        )
        door_graph = _combine_real_with_synthetic_completion(real_door_graph, synthetic_door_graph, child_rooms)
        window_graph = assign_windows_to_apartment(
            windows=floor_windows,
            apartment_index=idx - 1,
            nearest_apartment_by_window_id=nearest_apartment_by_window_id,
            apartment_polygon=apartment_local_points,
            child_rooms=child_rooms,
            coordinate_offset=(min_x, min_y),
        )
        apartment_doors, apartment_windows = build_apartment_openings(
            apartment_polygon_local=normalize_polygon(apartment_local_points),
            door_graph=door_graph,
            window_graph=window_graph,
        )
        for room in child_rooms:
            room_doors, room_windows = build_child_room_openings(
                room=room,
                door_graph=door_graph,
                window_graph=window_graph,
                apartment_offset=(min_x, min_y),
            )
            room["doors"] = room_doors
            room["windows"] = room_windows
            room_path = Path(str(room["room_json"]))
            if room_path.is_file():
                room_payload = read_json(room_path)
                room_payload.setdefault("room", {})
                room_payload["room"]["type"] = room_payload["room"].get("type") or room_payload["room"].get("room_type")
                room_payload["room"]["doors"] = room_doors
                room_payload["room"]["windows"] = room_windows
                write_json(room_path, room_payload)
        public_child_rooms = [
            {key: value for key, value in room.items() if not key.startswith("_")}
            for room in child_rooms
        ]
        apartment_spec = build_apartment_spec(
            project_id=floor_input.project_id,
            floor=floor_input.floor,
            apartment_index=idx,
            apartment_feature=apartment_feature,
            local_points=apartment_local_points,
            child_rooms=public_child_rooms,
            door_graph=door_graph,
            window_graph=window_graph,
            doors=apartment_doors,
            windows=apartment_windows,
            synthetic_door_graph=synthetic_door_graph,
            real_door_graph=real_door_graph,
            source_geojson=floor_input.geojson_path,
        )
        apartment_path = apartments_out / f"{apartment_spec['room']['id']}.json"
        write_json(apartment_path, apartment_spec)

        bundle_dir = floor_out / "apartment_bundles" / apartment_id
        bundle_rooms_dir = bundle_dir / "rooms"
        write_json(bundle_dir / "apartment.json", apartment_spec)
        bundled_rooms: list[dict[str, Any]] = []
        for child_room in public_child_rooms:
            source_room_path = Path(str(child_room["room_json"]))
            if not source_room_path.is_file():
                continue
            room_payload = read_json(source_room_path)
            bundle_room_path = bundle_rooms_dir / f"{child_room['room_id']}.json"
            write_json(bundle_room_path, room_payload)
            bundled_rooms.append(
                {
                    "room_id": child_room["room_id"],
                    "room_type": child_room.get("room_type"),
                    "source_room_type": child_room.get("source_room_type"),
                    "source_room_json": str(source_room_path),
                    "room_json": str(bundle_room_path),
                    "doors_count": len((room_payload.get("room") or {}).get("doors") or []),
                    "windows_count": len((room_payload.get("room") or {}).get("windows") or []),
                }
            )
        bundle_manifest = {
            "project_id": floor_input.project_id,
            "floor": floor_input.floor,
            "apartment_id": apartment_id,
            "apartment_json": str(bundle_dir / "apartment.json"),
            "source_apartment_json": str(apartment_path),
            "rooms_count": len(bundled_rooms),
            "rooms": bundled_rooms,
            "door_graph": door_graph,
            "window_graph": window_graph,
        }
        write_json(bundle_dir / "manifest.json", bundle_manifest)

        apartment_rows.append(
            {
                "apartment_json": str(apartment_path),
                "room_json": str(apartment_path),
                "bundle_dir": str(bundle_dir),
                "bundle_manifest": str(bundle_dir / "manifest.json"),
                "apartment_id": apartment_id,
                "room_id": apartment_spec["room"]["id"],
                "area_m2": apartment_spec["room"]["area_m2"],
                "child_rooms_count": len(apartment_spec["room"]["meta"]["child_rooms"]),
                "child_rooms": apartment_spec["room"]["meta"]["child_rooms"],
                "door_graph": door_graph,
                "window_graph": window_graph,
                "synthetic_door_graph": synthetic_door_graph,
                "real_door_graph": real_door_graph,
            }
        )

    manifest = {
        "project_id": floor_input.project_id,
        "floor": floor_input.floor,
        "source": {
            "geojson": str(floor_input.geojson_path),
            "dxf": str(floor_input.dxf_path) if floor_input.dxf_path else None,
            "response_blocks": str(response_blocks_path) if response_blocks_path.is_file() else None,
            "floor_xlsx": str(floor_input.floor_xlsx_path) if floor_input.floor_xlsx_path else None,
            "project_xlsx": str(floor_input.project_xlsx_path) if floor_input.project_xlsx_path else None,
            "floor_jpg": str(floor_input.floor_jpg_path) if floor_input.floor_jpg_path else None,
            "overview_jpg": str(floor_input.overview_jpg_path) if floor_input.overview_jpg_path else None,
        },
        "projection": {
            "kind": "local_equirectangular_meters",
            "origin_lonlat": [origin[0], origin[1]],
        },
        "window_import": {
            "source": "response_blocks_insolation_points" if response_windows else ("dxf_windows_layer" if dxf_windows else "none"),
            "response_blocks_insolation_points_count": len(response_windows),
            "dxf_windows_count": len(dxf_windows),
            "floor_windows_count": len(floor_windows),
        },
        "dxf_window_import": {
            "enabled": bool(floor_input.dxf_path),
            "transform": {
                key: value
                for key, value in (dxf_transform or {}).items()
                if key not in {"matrix", "src_center", "dst_center"}
            } if dxf_transform else None,
            "floor_windows_count": len(dxf_windows),
        },
        "feature_type_counts": type_counts,
        "apartments_count": len(apartments),
        "apartment_specs_count": len(apartment_rows),
        "rooms_count": len(room_rows),
        "rooms": room_rows,
        "apartments": apartment_rows,
        "xlsx_preview": {
            "floor": read_xlsx_preview(floor_input.floor_xlsx_path) if floor_input.floor_xlsx_path else None,
            "project": read_xlsx_preview(floor_input.project_xlsx_path) if floor_input.project_xlsx_path else None,
        },
    }
    write_json(floor_out / "manifest.json", manifest)
    return manifest


def build_batch_commands(
    manifest: dict[str, Any],
    prompt: str,
    run_root: Path,
    *,
    source_key: str = "rooms",
    placer: str = "infinigen_clean",
    modes: str = "infinigen_clean",
    pipeline_extra_args: str = "",
) -> list[str]:
    commands: list[str] = []
    for room in manifest.get(source_key) or []:
        room_json = Path(room["room_json"])
        room_id = room["room_id"]
        run_dir = run_root / manifest["project_id"] / f"floor_{manifest['floor']}" / room_id
        parts = [
            "python3",
            "src/run_pipeline.py",
            "--room",
            str(room_json),
            "--prompt",
            prompt,
            "--run-dir",
            str(run_dir),
            "--placer",
            placer,
            "--modes",
            modes,
            "--keep-tmp",
            "--keep-blend",
            "--infinigen-task",
            "coarse",
            "--infinigen-configs",
            "singleroom.gin",
            "fast_solve.gin",
        ]
        command = " ".join(shlex.quote(part) for part in parts)
        if pipeline_extra_args.strip():
            command = f"{command} {pipeline_extra_args.strip()}"
        commands.append(command)
    return commands


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Kvartirografiya РФ floor exports to CGS room.json inputs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--include-room-types",
        default=",".join(DEFAULT_INCLUDE_ROOM_TYPES),
        help="Comma-separated source roomType values to export; use 'all' to include every room feature.",
    )
    parser.add_argument("--prompt", default="Современный спокойный интерьер по реальной планировке")
    parser.add_argument("--commands-file", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--apartment-run-root", default=None)
    parser.add_argument("--pipeline-placer", default="infinigen_clean")
    parser.add_argument("--pipeline-modes", default="infinigen_clean")
    parser.add_argument(
        "--pipeline-extra-args",
        default="",
        help="Raw extra args appended to every generated run_pipeline command, e.g. remote SSH flags.",
    )
    return parser


def main() -> None:
    args = build_cli().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    include_raw = str(args.include_room_types or "").strip()
    include_room_types = (
        {"KITCHEN", "BEDROOM", "STUDIO", "LIVING_ROOM", "ROOM", "HALL", "CORRIDOR", "BATHROOM", "JOINT_BATHROOM", "TOILET", "BALCONY"}
        if include_raw.lower() == "all"
        else {part.strip().upper() for part in include_raw.split(",") if part.strip()}
    )

    floors = discover_floors(input_dir)
    response_block_files = discover_response_block_files(input_dir)
    if not floors and not response_block_files:
        raise SystemExit(f"No floor GeoJSON or *_response_blocks.json files found in {input_dir}")

    doc_previews = {
        doc.name: read_docx_text(doc)
        for doc in sorted(input_dir.glob("*.docx"))
    }
    manifests = [convert_floor(floor, out_dir, include_room_types) for floor in floors]
    response_block_manifests = [convert_response_blocks(path, out_dir) for path in response_block_files]
    commands: list[str] = []
    apartment_commands: list[str] = []
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else out_dir / "_runs"
    apartment_run_root = (
        Path(args.apartment_run_root).expanduser().resolve()
        if args.apartment_run_root
        else out_dir / "_apartment_runs"
    )
    for manifest in manifests:
        commands.extend(
            build_batch_commands(
                manifest,
                args.prompt,
                run_root,
                placer=str(args.pipeline_placer),
                modes=str(args.pipeline_modes),
                pipeline_extra_args=str(args.pipeline_extra_args or ""),
            )
        )
        apartment_commands.extend(
            build_batch_commands(
                manifest,
                args.prompt,
                apartment_run_root,
                source_key="apartments",
                placer=str(args.pipeline_placer),
                modes=str(args.pipeline_modes),
                pipeline_extra_args=str(args.pipeline_extra_args or ""),
            )
        )

    index = {
        "source_dir": str(input_dir),
        "out_dir": str(out_dir),
        "docs": doc_previews,
        "floors_count": len(manifests),
        "rooms_count": sum(int(m.get("rooms_count") or 0) for m in manifests),
        "apartment_specs_count": sum(int(m.get("apartment_specs_count") or 0) for m in manifests),
        "response_blocks_count": len(response_block_manifests),
        "response_block_projects_count": len(response_block_manifests),
        "response_block_apartments_count": sum(int(m.get("apartments_count") or 0) for m in response_block_manifests),
        "response_block_rooms_count": sum(int(m.get("rooms_count") or 0) for m in response_block_manifests),
        "floor_manifests": [
            str(out_dir / m["project_id"] / f"floor_{m['floor']}" / "manifest.json")
            for m in manifests
        ],
        "response_block_manifests": [
            str(out_dir / "_response_blocks" / m["project_id"] / "manifest.json")
            for m in response_block_manifests
        ],
        "run_root": str(run_root),
        "apartment_run_root": str(apartment_run_root),
        "pipeline_defaults": {
            "placer": str(args.pipeline_placer),
            "modes": str(args.pipeline_modes),
            "keep_tmp": True,
            "keep_blend": True,
            "infinigen_task": "coarse",
            "infinigen_configs": ["singleroom.gin", "fast_solve.gin"],
            "pipeline_extra_args": str(args.pipeline_extra_args or ""),
            "baseline_policy": (
                "Generated commands build a full infinigen_clean baseline first. "
                "Supplier replacement is not used as the fallback scene."
            ),
        },
        "run_commands": commands,
        "run_apartment_commands": apartment_commands,
    }
    write_json(out_dir / "index.json", index)

    commands_file = Path(args.commands_file).expanduser().resolve() if args.commands_file else out_dir / "run_rooms.sh"
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    commands_file.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n", encoding="utf-8")
    apartment_commands_file = out_dir / "run_apartments.sh"
    apartment_commands_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(apartment_commands) + "\n",
        encoding="utf-8",
    )
    print(f"Converted floors: {len(manifests)}")
    print(f"Exported rooms: {index['rooms_count']}")
    print(f"Exported apartment specs: {index['apartment_specs_count']}")
    print(f"Converted response block files: {index['response_blocks_count']}")
    print(f"Response block apartments: {index['response_block_apartments_count']}")
    print(f"Response block rooms: {index['response_block_rooms_count']}")
    print(f"Index: {out_dir / 'index.json'}")
    print(f"Run commands: {commands_file}")
    print(f"Run apartment commands: {apartment_commands_file}")


if __name__ == "__main__":
    main()
