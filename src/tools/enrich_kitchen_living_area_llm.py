#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any


CATALOG_PATH = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
IMODERN_ROOT = Path("data/sourse/imodern")
SUPPORTED_MESH = {".fbx", ".obj", ".glb", ".gltf"}
GENERATED_SOURCE = "kitchen_living_llm_enrichment"


ROLE_TARGETS = {
    "dining_table": {"size": (1.20, 0.78, 0.75), "category": "DiningTableFactory", "semantic": "dining_table"},
    "chair": {"size": (0.48, 0.48, 0.86), "category": "ChairFactory", "semantic": "chair"},
    "armchair": {"size": (0.66, 0.66, 0.82), "category": "ArmchairFactory", "semantic": "armchair"},
    "sofa": {"size": (1.55, 0.78, 0.82), "category": "SofaFactory", "semantic": "sofa"},
}
ROLE_ALIASES = {
    "dining chair": "chair",
    "dining chairs": "chair",
    "dining_chair": "chair",
    "dining_chairs": "chair",
    "chair": "chair",
    "chairs": "chair",
    "seat": "chair",
    "seating": "chair",
    "arm chair": "armchair",
    "arm_chair": "armchair",
    "armchair": "armchair",
    "lounge chair": "armchair",
    "sofa": "sofa",
    "couch": "sofa",
    "loveseat": "sofa",
    "dining table": "dining_table",
    "dining_table": "dining_table",
    "table": "dining_table",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def normalize_role(role: Any) -> str:
    value = str(role or "").strip().lower().replace("-", " ").replace("/", " ")
    value = re.sub(r"\s+", " ", value)
    return ROLE_ALIASES.get(value, value.replace(" ", "_"))


def norm(value: Any) -> str:
    return str(value or "").replace("ё", "е").lower()


def item_aabb(item: dict[str, Any]) -> dict[str, float]:
    aabb = item.get("aabb") if isinstance(item.get("aabb"), dict) else {}
    if {"x_min", "x_max", "y_min", "y_max"} <= set(aabb):
        z_min = float(aabb.get("z_min", 0.0) or 0.0)
        z_max = float(aabb.get("z_max", z_min + 1.0) or z_min + 1.0)
        return {
            "x_min": float(aabb["x_min"]),
            "x_max": float(aabb["x_max"]),
            "y_min": float(aabb["y_min"]),
            "y_max": float(aabb["y_max"]),
            "z_min": z_min,
            "z_max": z_max,
        }
    pos = list(item.get("position_m") or [0.0, 0.0, 0.0])
    size = list(item.get("size_m") or [0.5, 0.5, 0.5])
    cx, cy, cz = [float(x or 0.0) for x in (pos + [0, 0, 0])[:3]]
    sx, sy, sz = [float(x or 0.0) for x in (size + [0.5, 0.5, 0.5])[:3]]
    return {
        "x_min": cx - sx / 2.0,
        "x_max": cx + sx / 2.0,
        "y_min": cy - sy / 2.0,
        "y_max": cy + sy / 2.0,
        "z_min": cz - sz / 2.0,
        "z_max": cz + sz / 2.0,
    }


def aabb_intersects(a: dict[str, float], b: dict[str, float], margin: float = 0.0) -> bool:
    return not (
        a["x_max"] + margin <= b["x_min"]
        or a["x_min"] - margin >= b["x_max"]
        or a["y_max"] + margin <= b["y_min"]
        or a["y_min"] - margin >= b["y_max"]
    )


def room_size(room: dict[str, Any]) -> tuple[float, float]:
    width = float(room.get("width_m") or 0.0)
    depth = float(room.get("depth_m") or 0.0)
    if width > 0 and depth > 0:
        return width, depth
    poly = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    xs = [float(p.get("x", 0.0)) for p in poly if isinstance(p, dict)]
    ys = [float(p.get("y", p.get("z", 0.0))) for p in poly if isinstance(p, dict)]
    return (max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else (3.2, 3.0)


def is_kitchen_scene(scene: dict[str, Any]) -> bool:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    text = " ".join(str(x or "") for x in (room.get("id"), room.get("room_type"), room.get("source_room_type")))
    return "kitchen" in norm(text) or "кух" in norm(text)


def is_kitchen_item(item: dict[str, Any]) -> bool:
    text = " ".join(str(item.get(k) or "") for k in ("id", "name", "category", "semantic_group", "type", "assembly_type"))
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
    text += " " + " ".join(str(meta.get(k) or "") for k in ("procedural_assembly", "assembly_type"))
    text += " " + " ".join(str(asset.get(k) or "") for k in ("kind", "assembly_type"))
    return "kitchen" in norm(text)


def is_generated_living_item(item: dict[str, Any]) -> bool:
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return source.get("placement_source") == GENERATED_SOURCE or bool(meta.get("kitchen_living_llm_generated"))


def is_table_item(item: dict[str, Any]) -> bool:
    text = norm(" ".join(str(item.get(k) or "") for k in ("id", "name", "category", "semantic_group")))
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    text += " " + norm(" ".join(str(candidate.get(k) or "") for k in ("title", "category_norm", "semantic_group")))
    return "dining_table" in text or "стол" in text or "dining table" in text


def opening_blockers(room: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    width, depth = room_size(room)
    for door in room.get("doors") or []:
        if not isinstance(door, dict):
            continue
        seg = door.get("segment") if isinstance(door.get("segment"), dict) else {}
        xs = [float(seg.get("x1", 0.0) or 0.0), float(seg.get("x2", 0.0) or 0.0)]
        ys = [float(seg.get("y1", 0.0) or 0.0), float(seg.get("y2", 0.0) or 0.0)]
        horizontal = abs(ys[0] - ys[1]) < abs(xs[0] - xs[1])
        margin = 0.22
        if horizontal:
            y = min(max(sum(ys) * 0.5, 0.0), depth)
            y0, y1 = (max(0.0, y - 0.85), min(depth, y + 0.10)) if y > depth * 0.5 else (max(0.0, y - 0.10), min(depth, y + 0.85))
            blockers.append({"id": door.get("id") or "door", "kind": "door_clearance", "x_min": min(xs) - margin, "x_max": max(xs) + margin, "y_min": y0, "y_max": y1})
        else:
            x = min(max(sum(xs) * 0.5, 0.0), width)
            x0, x1 = (max(0.0, x - 0.85), min(width, x + 0.10)) if x > width * 0.5 else (max(0.0, x - 0.10), min(width, x + 0.85))
            blockers.append({"id": door.get("id") or "door", "kind": "door_clearance", "x_min": x0, "x_max": x1, "y_min": min(ys) - margin, "y_max": max(ys) + margin})
    for window in room.get("windows") or []:
        if not isinstance(window, dict):
            continue
        seg = window.get("segment") if isinstance(window.get("segment"), dict) else {}
        xs = [float(seg.get("x1", 0.0) or 0.0), float(seg.get("x2", 0.0) or 0.0)]
        ys = [float(seg.get("y1", 0.0) or 0.0), float(seg.get("y2", 0.0) or 0.0)]
        if abs(ys[0] - ys[1]) < abs(xs[0] - xs[1]):
            y = min(max(sum(ys) * 0.5, 0.0), depth)
            y0, y1 = (0.0, min(depth, 0.95)) if y < depth * 0.5 else (max(0.0, depth - 0.95), depth)
            blockers.append({"id": window.get("id") or "window", "kind": "window_clearance", "x_min": min(xs) - 0.18, "x_max": max(xs) + 0.18, "y_min": y0, "y_max": y1})
    return blockers


def blockers_for_scene(scene: dict[str, Any]) -> list[dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    width, depth = room_size(room)
    blockers = opening_blockers(room)
    for item in scene.get("placements") or []:
        if not isinstance(item, dict):
            continue
        if is_kitchen_item(item):
            a = item_aabb(item)
            blockers.append(
                {
                    "id": item.get("id") or "kitchen",
                    "kind": "kitchen_fixed",
                    "x_min": max(0.0, a["x_min"] - 0.03),
                    "x_max": min(width, a["x_max"] + 0.03),
                    "y_min": max(0.0, a["y_min"] - 0.03),
                    "y_max": min(depth, a["y_max"] + 0.55),
                }
            )
    return blockers


def mesh_exists(path_value: Any) -> bool:
    path = Path(str(path_value or "")).expanduser()
    return path.is_file() and path.suffix.lower() in SUPPORTED_MESH


def find_mesh(root: Path) -> Path | None:
    if not root.exists():
        return None
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_MESH]
    if not files:
        return None
    order = {".fbx": 0, ".obj": 1, ".glb": 2, ".gltf": 3}
    files.sort(key=lambda p: (order.get(p.suffix.lower(), 99), len(p.parts), str(p).lower()))
    return files[0]


def dims_from_row(row: dict[str, Any], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    dims = row.get("dimensions_cm") if isinstance(row.get("dimensions_cm"), dict) else {}
    vals = []
    for key, fallback_value in (("width_cm", fallback[0]), ("depth_cm", fallback[1]), ("height_cm", fallback[2])):
        raw = row.get(key)
        if raw is None:
            raw = dims.get(key.replace("_cm", ""))
        try:
            value = float(raw) / 100.0
        except Exception:
            value = 0.0
        vals.append(value if value > 0.05 else fallback_value)
    return tuple(vals)  # type: ignore[return-value]


def normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    mesh = out.get("asset_local_path") or out.get("mesh_local_path")
    if mesh and mesh_exists(mesh):
        out["asset_local_path"] = str(Path(str(mesh)).expanduser().resolve())
        out.setdefault("asset_status", "local_supplier_asset")
        out.setdefault("asset_format", Path(str(mesh)).suffix.lstrip(".").lower())
    return out


def local_table_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not IMODERN_ROOT.exists():
        return rows
    for folder in IMODERN_ROOT.iterdir():
        if not folder.is_dir():
            continue
        text = norm(folder.name)
        if ("стол" not in text and "table" not in text) or any(x in text for x in ("лампа", "lamp", "журн", "coffee")):
            continue
        mesh = find_mesh(folder)
        if mesh is None:
            continue
        nums = [float(x) for x in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", folder.name)]
        width_cm = nums[0] if nums else 120.0
        depth_cm = nums[1] if len(nums) > 1 and nums[1] <= 120.0 else 78.0
        title = folder.name.replace("_", " ")
        rows.append(
            {
                "unique_key": f"local_imodern::{folder.name}",
                "source_site": "imodern_local",
                "title": title,
                "category_norm": "dining_table",
                "semantic_group": "dining_table",
                "asset_local_path": str(mesh.resolve()),
                "asset_status": "local_supplier_asset",
                "asset_format": mesh.suffix.lstrip(".").lower(),
                "width_cm": width_cm,
                "depth_cm": depth_cm,
                "height_cm": 76.0,
                "style": "современный",
            }
        )
    return rows


def load_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if CATALOG_PATH.is_file():
        payload = read_json(CATALOG_PATH)
        raw = payload.get("items") if isinstance(payload, dict) else payload
        rows.extend(normalize_candidate(x) for x in raw if isinstance(x, dict))
    rows.extend(local_table_candidates())
    return rows


def role_match(row: dict[str, Any], role: str) -> bool:
    text = norm(" ".join(str(row.get(k) or "") for k in ("title", "name", "category_norm", "category_raw", "semantic_group")))
    category = norm(row.get("category_norm"))
    semantic = norm(row.get("semantic_group"))
    if role == "dining_table":
        return category in {"dining_table", "table"} or semantic == "dining_table" or ("стол" in text and "журн" not in text and "lamp" not in text)
    if role == "chair":
        return category in {"chair", "dining_chair", "armchair"} or semantic in {"chair", "armchair"} or "стул" in text or "кресло" in text
    if role == "armchair":
        return category == "armchair" or semantic == "armchair" or "кресло" in text
    if role == "sofa":
        return category in {"sofa", "sectional_sofa", "modular_sofa"} or semantic == "sofa" or "диван" in text
    return False


def style_score(row: dict[str, Any], room_text: str) -> float:
    hay = norm(" ".join(str(row.get(k) or "") for k in ("title", "name", "style", "color", "materials", "description")))
    tokens = [x for x in re.split(r"[^a-zа-я0-9]+", norm(room_text)) if len(x) > 2]
    return -sum(1 for token in tokens[:30] if token and token in hay) * 0.04


def candidate_shortlist(rows: list[dict[str, Any]], role: str, target_size: tuple[float, float, float], room_text: str, limit: int = 8) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        if not role_match(row, role):
            continue
        if not mesh_exists(row.get("asset_local_path")):
            continue
        cw, cd, ch = dims_from_row(row, target_size)
        tw, td, th = target_size
        size_score = abs(math.log(max(cw, 0.05) / max(tw, 0.05))) + abs(math.log(max(cd, 0.05) / max(td, 0.05))) + 0.35 * abs(math.log(max(ch, 0.05) / max(th, 0.05)))
        price = row.get("price_value")
        price_score = 0.0
        try:
            price_score = min(float(price) / 300000.0, 1.2)
        except Exception:
            price_score = 0.25
        ranked.append((size_score + price_score + style_score(row, room_text), row))
    ranked.sort(key=lambda x: x[0])
    return [row for _, row in ranked[:limit]]


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "unique_key": row.get("unique_key"),
        "title": row.get("title") or row.get("name"),
        "category_norm": row.get("category_norm"),
        "semantic_group": row.get("semantic_group"),
        "source_site": row.get("source_site"),
        "asset_local_path": row.get("asset_local_path"),
        "asset_format": row.get("asset_format"),
        "width_cm": row.get("width_cm"),
        "depth_cm": row.get("depth_cm"),
        "height_cm": row.get("height_cm"),
        "price_value": row.get("price_value"),
        "price_currency": row.get("price_currency"),
        "style": row.get("style"),
        "color": row.get("color"),
    }


def extract_response_text(resp: dict[str, Any]) -> str:
    message = resp.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(resp.get("response"), str):
        return str(resp["response"])
    return json.dumps(resp, ensure_ascii=False)


def parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def chat_ollama_http(base_url: str, model: str, system: str, payload: dict[str, Any], schema: dict[str, Any], timeout: int, temperature: float) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1536},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_ollama_ssh(args: argparse.Namespace, system: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    remote_code = r'''
import json, sys, urllib.request
cfg=json.loads(sys.stdin.read())
body=json.dumps({
  "model": cfg["model"],
  "stream": False,
  "format": cfg["schema"],
  "messages": [
    {"role": "system", "content": cfg["system"]},
    {"role": "user", "content": json.dumps(cfg["payload"], ensure_ascii=False, indent=2)}
  ],
  "options": {"temperature": cfg["temperature"], "num_ctx": 8192, "num_predict": 1536}
}, ensure_ascii=False).encode("utf-8")
req=urllib.request.Request(cfg["base_url"].rstrip("/")+"/api/chat", data=body, headers={"Content-Type":"application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=cfg["timeout"]) as resp:
    sys.stdout.write(resp.read().decode("utf-8"))
'''
    cfg = {
        "base_url": args.ollama_url,
        "model": args.ollama_model,
        "system": system,
        "payload": payload,
        "schema": schema,
        "timeout": int(args.ollama_timeout),
        "temperature": float(args.ollama_temperature),
    }
    cmd = ["ssh", "-p", str(args.ssh_port)]
    if args.ssh_key:
        cmd += ["-i", str(Path(args.ssh_key).expanduser())]
    remote_cmd = "python3 -c " + shlex.quote(remote_code)
    cmd += [f"{args.ssh_user}@{args.ssh_host}", remote_cmd]
    proc = subprocess.run(cmd, input=json.dumps(cfg, ensure_ascii=False), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(f"ssh ollama command failed rc={proc.returncode} stderr={stderr[:1000]!r} stdout={stdout[:1000]!r}")
    parsed = parse_json_object(proc.stdout)
    if not parsed:
        raise RuntimeError(f"ssh ollama returned non-json stdout={(proc.stdout or '')[:1000]!r}")
    return parsed


def llm_json(args: argparse.Namespace, system: str, payload: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.llm_provider == "none":
        return {}, {"status": "skipped", "provider": "none"}
    try:
        if args.llm_provider == "ssh-ollama":
            resp = chat_ollama_ssh(args, system, payload, schema)
        else:
            resp = chat_ollama_http(args.ollama_url, args.ollama_model, system, payload, schema, args.ollama_timeout, args.ollama_temperature)
        parsed = parse_json_object(extract_response_text(resp))
        return parsed, {"status": "ok", "provider": args.llm_provider, "model": args.ollama_model, "raw_response": resp}
    except Exception as exc:
        return {}, {"status": "failed", "provider": args.llm_provider, "model": args.ollama_model, "reason": f"{type(exc).__name__}:{exc}"}


def desired_roles_with_llm(args: argparse.Namespace, scene: dict[str, Any], blockers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    items = scene.get("placements") or []
    schema = {
        "type": "object",
        "properties": {
            "roles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "count": {"type": "integer"},
                        "reason": {"type": "string"},
                    },
                    "required": ["role", "count"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["roles"],
        "additionalProperties": False,
    }
    payload = {
        "room": {"id": room.get("id"), "width_m": room_size(room)[0], "depth_m": room_size(room)[1], "doors": room.get("doors"), "windows": room.get("windows")},
        "fixed_items": [{"id": x.get("id"), "name": x.get("name"), "category": x.get("category"), "aabb": item_aabb(x)} for x in items if isinstance(x, dict)],
        "blockers": blockers,
        "allowed_roles": ["dining_table", "chair", "armchair", "sofa"],
        "rules": [
            "Kitchen set is fixed and must not move.",
            "Use free floor area without blocking door or window clearances.",
            "If a dining table already exists, keep it and add seating instead of duplicating it.",
            "For this compact kitchen prefer 2 dining chairs/armchairs; add sofa only if it clearly fits.",
        ],
    }
    parsed, info = llm_json(args, "You are a pragmatic interior layout planner. Return strict JSON.", payload, schema)
    roles = parsed.get("roles") if isinstance(parsed.get("roles"), list) else []
    clean: list[dict[str, Any]] = []
    for row in roles:
        if not isinstance(row, dict):
            continue
        role = normalize_role(row.get("role"))
        if role not in ROLE_TARGETS:
            continue
        count = max(0, min(4, int(float(row.get("count") or 0))))
        if count:
            clean.append({"role": role, "count": count, "reason": row.get("reason")})
    if not clean:
        clean = [{"role": "chair", "count": 2, "reason": "fallback: existing dining table needs seating"}]
        info = {**info, "fallback_roles_used": True}
    return clean, info


def approve_candidates_with_llm(args: argparse.Namespace, roles: list[dict[str, Any]], candidates: dict[str, list[dict[str, Any]]], existing_table: dict[str, Any] | None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "approved": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"role": {"type": "string"}, "unique_key": {"type": "string"}, "count": {"type": "integer"}, "reason": {"type": "string"}},
                    "required": ["role", "unique_key", "count"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["approved"],
        "additionalProperties": False,
    }
    payload = {
        "desired_roles": roles,
        "existing_table": {"id": existing_table.get("id"), "name": existing_table.get("name"), "aabb": item_aabb(existing_table)} if existing_table else None,
        "candidates_by_role": candidates,
        "rules": [
            "Choose only unique_key values present in candidates_by_role.",
            "Prefer local assets and dimensions close to target.",
            "Do not approve sofa if it will crowd the compact kitchen.",
        ],
    }
    parsed, info = llm_json(args, "You approve supplier assets for a compact kitchen-living area. Return strict JSON.", payload, schema)
    approved: dict[str, list[dict[str, Any]]] = {}
    rows_by_key = {str(row.get("unique_key")): row for role_rows in candidates.values() for row in role_rows}
    for row in parsed.get("approved", []) if isinstance(parsed.get("approved"), list) else []:
        if not isinstance(row, dict):
            continue
        role = normalize_role(row.get("role"))
        key = str(row.get("unique_key") or "")
        if role not in ROLE_TARGETS or key not in rows_by_key:
            continue
        count = max(1, min(4, int(float(row.get("count") or 1))))
        approved.setdefault(role, [])
        for _ in range(count):
            approved[role].append(rows_by_key[key])
    if not approved:
        for role_spec in roles:
            role = str(role_spec.get("role"))
            rows = candidates.get(role) or []
            if not rows:
                continue
            count = max(1, min(4, int(role_spec.get("count") or 1)))
            approved.setdefault(role, [])
            for idx in range(count):
                approved[role].append(rows[min(idx, len(rows) - 1)])
        info = {**info, "fallback_candidate_selection_used": True}
    return approved, info


def table_from_existing(scene: dict[str, Any]) -> dict[str, Any] | None:
    for item in scene.get("placements") or []:
        if isinstance(item, dict) and not is_generated_living_item(item) and is_table_item(item):
            return item
    return None


def proposed_items_for_placement(approved: dict[str, list[dict[str, Any]]], existing_table: dict[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if existing_table:
        out.append({"id": existing_table.get("id"), "role": "existing_dining_table", "fixed": True, "aabb": item_aabb(existing_table)})
    counters: dict[str, int] = {}
    for role, rows in approved.items():
        if role == "dining_table" and existing_table:
            continue
        for row in rows:
            counters[role] = counters.get(role, 0) + 1
            idx = counters[role]
            target = ROLE_TARGETS[role]["size"]
            sx, sy, sz = dims_from_row(row, target)
            sx = min(max(sx, target[0] * 0.75), target[0] * 1.20)
            sy = min(max(sy, target[1] * 0.75), target[1] * 1.20)
            sz = min(max(sz, target[2] * 0.65), target[2] * 1.25)
            out.append({"id": f"kitchen_living_{role}_{idx:02d}", "role": role, "size_m": [round(sx, 4), round(sy, 4), round(sz, 4)], "candidate": compact_candidate(row)})
    return out


def place_with_llm(args: argparse.Namespace, scene: dict[str, Any], proposed: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    width, depth = room_size(room)
    schema = {
        "type": "object",
        "properties": {
            "placements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "x_m": {"type": "number"}, "y_m": {"type": "number"}, "yaw_deg": {"type": "number"}, "reason": {"type": "string"}},
                    "required": ["id", "x_m", "y_m", "yaw_deg"],
                    "additionalProperties": False,
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["placements"],
        "additionalProperties": False,
    }
    payload = {
        "room": {"width_m": width, "depth_m": depth, "doors": room.get("doors"), "windows": room.get("windows")},
        "proposed_items": proposed,
        "blockers": blockers,
        "rules": [
            "Do not move existing_dining_table.",
            "Do not overlap kitchen_fixed, door_clearance, window_clearance.",
            "Keep at least 0.08 m gap between furniture except chairs may be close to dining table.",
            "Prefer chairs around the existing table.",
            "Return center coordinates in room meters.",
        ],
    }
    parsed, info = llm_json(args, "You place supplier furniture in a compact kitchen-living area. Return strict JSON only.", payload, schema)
    out: dict[str, dict[str, Any]] = {}
    for row in parsed.get("placements", []) if isinstance(parsed.get("placements"), list) else []:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id") or "")
        out[item_id] = {"x_m": float(row.get("x_m") or 0.0), "y_m": float(row.get("y_m") or 0.0), "yaw_deg": float(row.get("yaw_deg") or 0.0), "reason": row.get("reason")}
    if not out:
        info = {**info, "fallback_placement_used": True}
    return out, info


def candidate_aabb(cx: float, cy: float, size: tuple[float, float, float]) -> dict[str, float]:
    sx, sy, sz = size
    return {"x_min": cx - sx / 2.0, "x_max": cx + sx / 2.0, "y_min": cy - sy / 2.0, "y_max": cy + sy / 2.0, "z_min": 0.0, "z_max": sz}


def valid_aabb(aabb: dict[str, float], room_size_xy: tuple[float, float], blockers: list[dict[str, Any]], occupied: list[dict[str, float]], allow_table_touch: bool = False) -> bool:
    width, depth = room_size_xy
    if aabb["x_min"] < 0.06 or aabb["y_min"] < 0.06 or aabb["x_max"] > width - 0.06 or aabb["y_max"] > depth - 0.06:
        return False
    for blocker in blockers:
        if aabb_intersects(aabb, blocker, margin=0.0):
            return False
    for other in occupied:
        margin = -0.04 if allow_table_touch else 0.04
        if aabb_intersects(aabb, other, margin=margin):
            return False
    return True


def fallback_position(item: dict[str, Any], room_size_xy: tuple[float, float], blockers: list[dict[str, Any]], occupied: list[dict[str, float]], table: dict[str, Any] | None, index: int) -> tuple[float, float, float]:
    width, depth = room_size_xy
    role = str(item["role"])
    sx, sy, _ = [float(x) for x in item["size_m"]]
    table_aabb = item_aabb(table) if table else None
    candidates: list[tuple[float, float, float]] = []
    if role in {"chair", "armchair"} and table_aabb:
        tx = 0.5 * (table_aabb["x_min"] + table_aabb["x_max"])
        ty = 0.5 * (table_aabb["y_min"] + table_aabb["y_max"])
        table_w = table_aabb["x_max"] - table_aabb["x_min"]
        table_d = table_aabb["y_max"] - table_aabb["y_min"]
        gap = 0.08
        candidates.extend(
            [
                (table_aabb["x_min"] - sx / 2.0 - gap, ty, 90.0),
                (table_aabb["x_max"] + sx / 2.0 + gap, ty, 270.0),
                (tx, table_aabb["y_max"] + sy / 2.0 + gap, 180.0),
                (tx, table_aabb["y_min"] - sy / 2.0 - gap, 0.0),
                (tx - table_w * 0.32, table_aabb["y_max"] + sy / 2.0 + gap, 180.0),
                (tx + table_w * 0.32, table_aabb["y_max"] + sy / 2.0 + gap, 180.0),
            ]
        )
    if role == "sofa":
        candidates.extend([(sx / 2.0 + 0.08, depth * 0.58, 90.0), (width - sx / 2.0 - 0.08, depth * 0.58, 270.0)])
    for x, y, yaw in candidates:
        aabb = candidate_aabb(x, y, (sx, sy, float(item["size_m"][2])))
        if valid_aabb(aabb, room_size_xy, blockers, occupied, allow_table_touch=role in {"chair", "armchair"}):
            return x, y, yaw
    step = 0.18
    y = depth - sy / 2.0 - 0.08
    while y > sy / 2.0 + 0.08:
        x = width - sx / 2.0 - 0.08
        while x > sx / 2.0 + 0.08:
            aabb = candidate_aabb(x, y, (sx, sy, float(item["size_m"][2])))
            if valid_aabb(aabb, room_size_xy, blockers, occupied, allow_table_touch=role in {"chair", "armchair"}):
                return x, y, 180.0 if role in {"chair", "armchair"} else 0.0
            x -= step
        y -= step
    return width * 0.5, depth * 0.5, 0.0


def make_item(item: dict[str, Any], row: dict[str, Any], cx: float, cy: float, yaw: float, table_id: str | None, llm_meta: dict[str, Any]) -> dict[str, Any]:
    role = str(item["role"])
    sx, sy, sz = [float(x) for x in item["size_m"]]
    aabb = candidate_aabb(cx, cy, (sx, sy, sz))
    candidate = compact_candidate(row)
    meta = {
        "kitchen_living_llm_generated": True,
        "required_role": role,
        "supplier_candidate": candidate,
        "supplier_candidate_pool": [candidate],
        "llm_enrichment": llm_meta,
    }
    if role in {"chair", "armchair"} and table_id:
        meta.update({"affordance": "table_chair", "target_table_id": table_id, "support_group": "kitchen_dining"})
    return {
        "id": str(item["id"]),
        "name": str(row.get("title") or row.get("name") or role),
        "category": ROLE_TARGETS[role]["category"],
        "semantic_group": ROLE_TARGETS[role]["semantic"],
        "position_m": [round(cx, 4), round(cy, 4), round(sz / 2.0, 4)],
        "size_m": [round(sx, 4), round(sy, 4), round(sz, 4)],
        "yaw_deg": round(yaw, 4),
        "rotation_deg": round(yaw, 4),
        "yaw_rad": round(math.radians(yaw), 8),
        "aabb": {k: round(v, 4) for k, v in aabb.items()},
        "constraints": {"mount_type": "floor", "touch_floor": {"side": "bottom"}},
        "asset": {"mesh_path": str(row.get("asset_local_path") or ""), "mesh_fit_mode": "uniform"},
        "source": {
            "placement_source": GENERATED_SOURCE,
            "asset_source": "supplier_catalog_local_asset",
            "supplier_replaced": True,
            "supplier_target_id": str(item["id"]),
            "supplier_unique_key": row.get("unique_key"),
            "supplier_source_site": row.get("source_site"),
            "supplier_product_url": row.get("product_url") or row.get("model_page_url"),
            "placeholder_bbox": False,
        },
        "meta": meta,
    }


def enrich_scene(scene: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(scene)
    if not is_kitchen_scene(out):
        return out, {"status": "skipped", "reason": "not_kitchen"}
    items = [x for x in out.get("placements", []) if isinstance(x, dict) and not is_generated_living_item(x)]
    out["placements"] = items
    room = out.get("room") if isinstance(out.get("room"), dict) else {}
    width, depth = room_size(room)
    blockers = blockers_for_scene(out)
    existing_table = table_from_existing(out)

    roles, plan_info = desired_roles_with_llm(args, out, blockers)
    if existing_table:
        roles = [r for r in roles if r.get("role") != "dining_table"] or [{"role": "chair", "count": 2, "reason": "existing table retained"}]

    room_text = " ".join(str(x or "") for x in (room.get("style"), room.get("room_type"), read_prompt_near_scene(args.scene_json)))
    catalog = load_catalog()
    shortlists: dict[str, list[dict[str, Any]]] = {}
    for role_spec in roles:
        role = str(role_spec.get("role"))
        target = ROLE_TARGETS[role]["size"]
        shortlists[role] = [compact_candidate(row) for row in candidate_shortlist(catalog, role, target, room_text, limit=8)]
    approved_compact, approve_info = approve_candidates_with_llm(args, roles, shortlists, existing_table)
    row_by_key = {str(row.get("unique_key")): row for row in catalog if row.get("unique_key")}
    approved_full: dict[str, list[dict[str, Any]]] = {}
    for role, rows in approved_compact.items():
        for compact in rows:
            row = row_by_key.get(str(compact.get("unique_key")))
            if row:
                approved_full.setdefault(role, []).append(row)

    proposed = proposed_items_for_placement(approved_full, existing_table)
    placement_map, placement_info = place_with_llm(args, out, proposed, blockers)

    occupied = [item_aabb(x) for x in items if isinstance(x, dict)]
    new_items: list[dict[str, Any]] = []
    table_id = str(existing_table.get("id")) if existing_table else None
    for idx, item in enumerate(x for x in proposed if not x.get("fixed")):
        role = str(item["role"])
        approved_rows = approved_full.get(role) or []
        row = approved_rows[min(idx, len(approved_rows) - 1)] if approved_rows else None
        if row is None:
            continue
        size = tuple(float(x) for x in item["size_m"])
        planned = placement_map.get(str(item["id"])) or {}
        cx = float(planned.get("x_m") or 0.0)
        cy = float(planned.get("y_m") or 0.0)
        yaw = float(planned.get("yaw_deg") or 0.0)
        aabb = candidate_aabb(cx, cy, size)
        if not valid_aabb(aabb, (width, depth), blockers, occupied, allow_table_touch=role in {"chair", "armchair"}):
            cx, cy, yaw = fallback_position(item, (width, depth), blockers, occupied, existing_table, idx)
            aabb = candidate_aabb(cx, cy, size)
        if not valid_aabb(aabb, (width, depth), blockers, occupied, allow_table_touch=role in {"chair", "armchair"}):
            continue
        placed = make_item(
            item,
            row,
            cx,
            cy,
            yaw,
            table_id,
            {"plan": plan_info, "approval": approve_info, "placement": placement_info, "llm_planned": bool(planned)},
        )
        occupied.append(item_aabb(placed))
        new_items.append(placed)

    out.setdefault("placements", []).extend(new_items)
    meta = out.setdefault("meta", {})
    meta["kitchen_living_llm_enrichment"] = {
        "status": "ok",
        "generated_count": len(new_items),
        "preserved_kitchen_position": True,
        "existing_table_id": table_id,
        "roles": roles,
        "blockers": blockers,
        "plan_llm": plan_info,
        "approval_llm": approve_info,
        "placement_llm": placement_info,
        "added": [{"id": x.get("id"), "role": x.get("meta", {}).get("required_role"), "name": x.get("name")} for x in new_items],
    }
    return out, meta["kitchen_living_llm_enrichment"]


def read_prompt_near_scene(scene_json: Path | None) -> str:
    if scene_json is None:
        return ""
    for name in ("prompt.styled.txt", "prompt.txt"):
        path = scene_json.parent / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def iter_kitchen_scenes(apt_dir: Path, mode: str) -> list[Path]:
    manifest = read_json(apt_dir / "manifest.json")
    out: list[Path] = []
    for room_meta in manifest.get("rooms") or []:
        room_id = str(room_meta.get("room_id") or "")
        room_type = norm(room_meta.get("room_type") or room_meta.get("prompt_room_type") or room_id)
        if "kitchen" not in room_type and "кух" not in room_type:
            continue
        scene = apt_dir / "rooms" / room_id / "pipeline" / mode / "scene_requirements.v1.json"
        if scene.is_file():
            out.append(scene)
    return out


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add LLM-approved supplier seating/table items to kitchen free area without moving the kitchen set.")
    p.add_argument("--apt-dir", default=None)
    p.add_argument("--scene-json", default=None)
    p.add_argument("--mode", default="optimal")
    p.add_argument("--llm-provider", choices=["none", "ollama", "ssh-ollama"], default="ssh-ollama")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ollama-model", default="mistral-small3.2:24b")
    p.add_argument("--ollama-timeout", type=int, default=240)
    p.add_argument("--ollama-temperature", type=float, default=0.1)
    p.add_argument("--ssh-host", default="1.208.108.242")
    p.add_argument("--ssh-port", type=int, default=32172)
    p.add_argument("--ssh-user", default="root")
    p.add_argument("--ssh-key", default="~/.ssh/id_ed25519")
    p.add_argument("--out-report", default=None)
    return p


def main() -> None:
    args = build_cli().parse_args()
    scene_paths: list[Path] = []
    if args.scene_json:
        scene_paths = [Path(args.scene_json).expanduser().resolve()]
    elif args.apt_dir:
        scene_paths = iter_kitchen_scenes(Path(args.apt_dir).expanduser().resolve(), args.mode)
    else:
        raise SystemExit("Pass --scene-json or --apt-dir")
    reports: list[dict[str, Any]] = []
    for scene_path in scene_paths:
        args.scene_json = scene_path
        scene = read_json(scene_path)
        enriched, report = enrich_scene(scene, args)
        write_json(scene_path, enriched)
        reports.append({"scene_json": str(scene_path), **report})
    if args.out_report:
        write_json(args.out_report, {"reports": reports})
    print(json.dumps({"processed": len(reports), "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
