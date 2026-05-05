from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .kitchen_roles import normalize_material_record


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_raw_catalog(path: str | Path) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported catalog file extension: {suffix}")


def normalize_kitchen_material_catalog(raw_records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in raw_records:
        material = normalize_material_record(raw)
        if material["kitchen_role"] != "unknown":
            normalized.append(material)
    return normalized


def load_kitchen_material_catalog(path: str | Path) -> list[dict[str, Any]]:
    return normalize_kitchen_material_catalog(load_raw_catalog(path))


def group_by_role(materials: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for material in materials:
        grouped.setdefault(material.get("kitchen_role") or "unknown", []).append(material)
    return grouped
