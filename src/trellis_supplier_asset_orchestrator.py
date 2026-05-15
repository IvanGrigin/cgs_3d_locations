#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trellis_supplier_asset_orchestrator.py

Локальный orchestrator для Mac:
1. Принимает карточку supplier-товара в JSON.
2. Извлекает ссылки/пути изображений, описание, размеры, категорию, стиль, цвет.
3. Создаёт локальную job-папку.
4. Скачивает/копирует preview images локально.
5. Отправляет job на GPU-сервер по SSH/SCP.
6. На сервере запускает TRELLIS image-to-3D / multi-image-to-3D.
7. Скачивает итоговый .glb обратно на Mac.
8. Пишет enriched JSON, который дальше можно использовать в supplier pipeline:
   asset_status=trellis_generated_local_asset,
   asset_format=glb,
   asset_local_path=<локальный путь к glb>,
   dimensions/orientation/target AABB metadata preserved.

Предполагаемое состояние сервера:
- TRELLIS.2 находится в /workspace/TRELLIS.2
- env доступен через /venv/trellis2/bin/python
- модель скачана в /workspace/models/TRELLIS.2-4B
- worker script находится в /workspace/TRELLIS.2/run_trellis2_persistent_worker.py

Пример запуска с Mac:
python3 trellis_supplier_asset_orchestrator.py \
  --card-json data/sourse/suppliers/one_card.json \
  --out-dir out/trellis_supplier_assets \
  --server-host 1.208.108.242 \
  --server-port 32172 \
  --server-user root \
  --ssh-key ~/.ssh/id_ed25519 \
  --remote-root /workspace/trellis2_supplier_jobs \
  --mode multi_image \
  --max-images 4 \
  --sparse-steps 4 \
  --slat-steps 4 \
  --texture-size 256

Пример запуска, если карточка передаётся прямо из supplier_catalog_canonical.json по unique_key:
python3 trellis_supplier_asset_orchestrator.py \
  --catalog-json data/sourse/suppliers/supplier_catalog_canonical.json \
  --unique-key '3ddd::url::https://3ddd.ru/3dmodels/show/012-lamp-zava-pendant-lamp-1' \
  --out-dir out/trellis_supplier_assets \
  --server-host 1.208.108.242 \
    --server-port 28553 \
  --server-user root \
  --ssh-key ~/.ssh/id_ed25519
"""

from __future__ import annotations


try:
    from src.trellis_progress import (
        ProgressETA,
        StageTimer,
        TrellisCandidateBlacklist,
        apply_candidate_to_binding,
        candidate_unique_key,
        extract_candidate_pool,
    )
except Exception:
    from trellis_progress import (
        ProgressETA,
        StageTimer,
        TrellisCandidateBlacklist,
        apply_candidate_to_binding,
        candidate_unique_key,
        extract_candidate_pool,
    )
import argparse
import tarfile
import zipfile
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:
    raise SystemExit("Install requests first: python3 -m pip install requests") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_ASSET_EXTS = {".fbx", ".obj", ".glb", ".gltf"}
MODEL_SIDECAR_EXTS = IMAGE_EXTS | {".mtl", ".bin", ".tga", ".tif", ".tiff", ".dds", ".exr", ".hdr", ".ktx2"}
ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".7z", ".rar"}
ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tgz", ".tbz2", ".txz", ".7z", ".rar", ".tar")
final_generation_mode = "image"
DEFAULT_REMOTE_TRELLIS_ROOT = "/workspace/TRELLIS.2"
DEFAULT_REMOTE_MODEL_DIR = "/workspace/models/TRELLIS.2-4B"
DEFAULT_REMOTE_ROOT = "/workspace/trellis2_supplier_jobs"
DEFAULT_REMOTE_PYTHON = "/venv/trellis2/bin/python"
DEFAULT_REMOTE_WORKER_ROOT = "/workspace/trellis2_worker"
TRUSTED_PRODUCT_IMAGE_PATTERNS = (
    "ikea.",
    "ikea.com",
    "ikea com",
    "ikea_de",
    "moyamebel",
    "moya-mebel",
    "moya_mebel",
    "moya mebel",
    "моя мебель",
    "mebel.ru",
    "mebel ru",
    "mebelru",
    "mebel_ru",
)
SCENE_RENDER_IMAGE_PATTERNS = (
    "3ddd",
    "zeelproject",
)


def validate_trellis2_only_args(args: argparse.Namespace) -> None:
    """Reject legacy TRELLIS backends before any remote job can start."""
    problems: list[str] = []

    def value(name: str) -> str:
        return str(getattr(args, name, "") or "").strip()

    root = value("remote_trellis_root")
    root_l = root.lower().rstrip("/")
    if root_l in {"/workspace/trellis", "/workspace/trellis-box"} or any(
        token in root_l for token in ("trellis-box", "ltrellis", "trellis1", "trellis.1")
    ):
        problems.append(f"remote_trellis_root={root!r} is legacy; use /workspace/TRELLIS.2")

    model = value("remote_model_dir")
    model_l = model.lower()
    if any(
        token in model_l
        for token in (
            "trellis-image-large",
            "trellis-text-base",
            "trellis-small",
            "ltrellis",
            "trellis1",
            "trellis.1",
            "/trellis_models/",
        )
    ):
        problems.append(f"remote_model_dir={model!r} is legacy; use /workspace/models/TRELLIS.2-4B")

    text_model = value("remote_text_model_dir")
    text_model_l = text_model.lower()
    if text_model and any(
        token in text_model_l
        for token in (
            "trellis-image-large",
            "trellis-text-base",
            "trellis-small",
            "ltrellis",
            "trellis1",
            "trellis.1",
            "/trellis_models/",
        )
    ):
        problems.append(
            f"remote_text_model_dir={text_model!r} is legacy; use /workspace/models/TRELLIS.2-4B or leave empty"
        )

    remote_python = value("remote_python")
    remote_python_l = remote_python.lower()
    if remote_python_l == "/venv/trellis/bin/python" or any(
        token in remote_python_l for token in ("ltrellis", "trellis1", "trellis.1")
    ):
        problems.append(f"remote_python={remote_python!r} is legacy; use /venv/trellis2/bin/python")

    runner = value("remote_runner_path")
    if runner:
        problems.append("--remote-runner-path is disabled; TRELLIS.2 jobs must use run_trellis2_persistent_worker.py")

    if problems:
        raise SystemExit("Legacy TRELLIS backend is disabled:\n- " + "\n- ".join(problems))


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def _cleanup_failed_job_dir(job_dir: Path) -> None:
    p = Path(job_dir)
    if p.exists():
        try:
            shutil.rmtree(p)
        except Exception:
            pass


def _proxy_category_from_card(card: dict[str, Any]) -> str:
    return item_category_label(card)


def _proxy_size_from_card(card: dict[str, Any]) -> tuple[float, float, float]:
    raw = card.get("target_size_m")
    if isinstance(raw, list) and len(raw) >= 3:
        try:
            return (
                max(float(raw[0]), 0.05),
                max(float(raw[1]), 0.05),
                max(float(raw[2]), 0.05),
            )
        except Exception:
            pass

    dims = card.get("dimensions_cm") if isinstance(card.get("dimensions_cm"), dict) else {}
    try:
        w = float(dims.get("width") or 80)
        d = float(dims.get("depth") or 45)
        h = float(dims.get("height") or 75)
        if max(w, d, h) > 10:
            w, d, h = w / 100.0, d / 100.0, h / 100.0
        return (max(w, 0.05), max(d, 0.05), max(h, 0.05))
    except Exception:
        return 0.8, 0.45, 0.75


def _proxy_mesh_from_card(card: dict[str, Any], out_glb: Path) -> None:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required for proxy fallback") from exc

    category = _proxy_category_from_card(card)
    sx, sy, sz = _proxy_size_from_card(card)
    parts = []

    def add_box(name: str, center: tuple[float, float, float], extents: tuple[float, float, float]) -> None:
        mesh = trimesh.creation.box(extents=[max(float(v), 0.01) for v in extents])
        mesh.apply_translation(center)
        mesh.metadata["name"] = name
        parts.append(mesh)

    def add_cyl(name: str, radius: float, depth: float, center: tuple[float, float, float], sections: int = 24) -> None:
        mesh = trimesh.creation.cylinder(
            radius=max(float(radius), 0.01),
            height=max(float(depth), 0.01),
            sections=sections,
        )
        mesh.apply_translation(center)
        mesh.metadata["name"] = name
        parts.append(mesh)

    if "bed" in category or "кровать" in category:
        add_box("base", (0.0, 0.0, sz * 0.20), (sx, sy, sz * 0.28))
        add_box("mattress", (0.0, -sy * 0.03, sz * 0.46), (sx * 0.96, sy * 0.92, sz * 0.24))
        add_box("headboard", (0.0, sy * 0.47, sz * 0.55), (sx, sy * 0.08, sz * 0.70))
        add_box("pillow_left", (-sx * 0.22, sy * 0.30, sz * 0.64), (sx * 0.30, sy * 0.16, sz * 0.09))
        add_box("pillow_right", (sx * 0.22, sy * 0.30, sz * 0.64), (sx * 0.30, sy * 0.16, sz * 0.09))
    elif any(k in category for k in ["nightstand", "dresser", "wardrobe", "cabinet", "комод", "шкаф", "тумб"]):
        add_box("case", (0.0, 0.0, sz * 0.52), (sx, sy, sz * 0.92))
        n = 2 if ("nightstand" in category or "тумб" in category) else 4
        for i in range(n):
            z = sz * (0.20 + 0.58 * (i + 0.5) / n)
            add_box(f"drawer_{i+1}", (0.0, -sy * 0.51, z), (sx * 0.86, sy * 0.035, sz * 0.10))
            add_box(f"handle_{i+1}", (0.0, -sy * 0.55, z), (sx * 0.22, sy * 0.025, sz * 0.025))
    elif "desk" in category or "table" in category or "стол" in category:
        top_h = max(sz * 0.08, 0.045)
        leg_h = max(sz - top_h, 0.10)
        add_box("top", (0.0, 0.0, leg_h + top_h * 0.5), (sx, sy, top_h))
        for x in (-sx * 0.42, sx * 0.42):
            for y in (-sy * 0.38, sy * 0.38):
                add_box("leg", (x, y, leg_h * 0.5), (sx * 0.06, sy * 0.06, leg_h))
    elif "chair" in category or "стул" in category:
        seat_z = sz * 0.46
        add_box("seat", (0.0, 0.0, seat_z), (sx, sy, sz * 0.10))
        add_box("back", (0.0, sy * 0.42, sz * 0.70), (sx, sy * 0.10, sz * 0.55))
        for x in (-sx * 0.38, sx * 0.38):
            for y in (-sy * 0.34, sy * 0.34):
                add_box("leg", (x, y, seat_z * 0.5), (sx * 0.06, sy * 0.06, seat_z))
    elif any(k in category for k in ["lamp", "light", "свет", "торшер"]):
        add_cyl("base", min(sx, sy) * 0.30, sz * 0.04, (0.0, 0.0, sz * 0.02))
        add_cyl("pole", min(sx, sy) * 0.04, sz * 0.72, (0.0, 0.0, sz * 0.38))
        add_cyl("shade", min(sx, sy) * 0.28, sz * 0.20, (0.0, 0.0, sz * 0.82), sections=32)
    elif "plant" in category or "раст" in category:
        add_cyl("pot", min(sx, sy) * 0.28, sz * 0.28, (0.0, 0.0, sz * 0.14))
        for i in range(7):
            import math as _m
            ang = i * _m.tau / 7.0
            add_box(
                "leaf",
                (_m.cos(ang) * sx * 0.12, _m.sin(ang) * sy * 0.12, sz * 0.55),
                (sx * 0.10, sy * 0.035, sz * 0.45),
            )
    else:
        add_box("body", (0.0, 0.0, sz * 0.5), (sx, sy, sz))

    scene = trimesh.Scene()
    for mesh in parts:
        scene.add_geometry(mesh, node_name=mesh.metadata.get("name") or "part")
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_glb)


def _run_local_proxy_fallback(local_job_dir: Path, card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    local_job_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_card_for_job(card, orientation_yaw_deg=getattr(args, "orientation_yaw_deg", None))
    write_json(local_job_dir / "card.raw.json", card)
    write_json(local_job_dir / "card.normalized.json", normalized)

    local_output_dir = local_job_dir / "output"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    local_glb = local_output_dir / "asset.trellis.glb"
    local_report = local_output_dir / "trellis.report.json"

    if local_glb.exists():
        local_glb.unlink(missing_ok=True)

    started = time.monotonic()
    _proxy_mesh_from_card(card, local_glb)
    elapsed = round(time.monotonic() - started, 4)

    category = _proxy_category_from_card(card)
    sx, sy, sz = _proxy_size_from_card(card)
    remote_report = {
        "ok": True,
        "mode": "proxy_glb_fallback",
        "asset_format": "glb",
        "asset_path": str(local_glb),
        "out_glb": str(local_glb),
        "glb_size_mb": round(local_glb.stat().st_size / 1024 / 1024, 3),
        "proxy_category_text": category,
        "proxy_size_m": [sx, sy, sz],
        "total_sec": elapsed,
        "generated_at": time.time(),
        "generation_source": "proxy_fallback",
    }
    write_json(local_report, remote_report)

    result_payload = build_enriched_result(
        card=card,
        normalized_card=normalized,
        local_job_dir=local_job_dir,
        local_glb_path=local_glb,
        remote_report=remote_report,
        args=args,
    )
    enriched_path = write_json(local_job_dir / "asset.enriched.json", result_payload)
    patched_card = patch_card_with_asset(card, result_payload)
    patched_card_path = write_json(local_job_dir / "card.with_trellis_asset.json", patched_card)

    summary = {
        "ok": True,
        "job_id": str(args.job_id or local_job_dir.name),
        "local_job_dir": str(local_job_dir),
        "asset_glb": str(local_glb),
        "asset_path": str(local_glb),
        "asset_enriched_json": str(enriched_path),
        "card_with_trellis_asset_json": str(patched_card_path),
        "remote_report_json": str(local_report),
        "generation_source": "proxy",
        "final_generation_mode": "procedural_proxy",
        "asset_generation_mode": "procedural_proxy",
        "candidate_cascade": {
            "procedural_proxy": {
                "tried": True,
                "ok": True,
                "asset_local_path": str(local_glb),
                "reason": "all_supplier_candidates_failed",
            }
        },
        "proxy_category": category,
        "proxy_target_size": [sx, sy, sz],
    }
    write_json(local_job_dir / "summary.json", summary)
    return summary


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def slugify(value, max_len=None):
    """
    Stable filesystem-safe slug.

    Backward compatible:
      slugify(value)
      slugify(value, 64)
      slugify(value, max_len=64)
    """
    import re

    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z0-9а-яё._-]+", "_", s, flags=re.IGNORECASE)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "item"

    if max_len is not None:
        try:
            n = int(max_len)
        except Exception:
            n = 0
        if n > 0 and len(s) > n:
            s = s[:n].rstrip("._-") or s[:n]

    return s



def stable_hash(value: Any, n: int = 12) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:n]


def run_cmd(cmd: list[str], *, cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def run_cmd_stream(cmd: list[str], *, cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    ret = proc.wait()
    out = "".join(lines)
    if check and ret != 0:
        raise subprocess.CalledProcessError(ret, cmd, output=out)
    return subprocess.CompletedProcess(cmd, ret, out, None)


# -----------------------------------------------------------------------------
# Product card normalization
# -----------------------------------------------------------------------------


def load_card(args: argparse.Namespace) -> dict[str, Any]:
    if args.card_json:
        card = read_json(args.card_json)
        if not isinstance(card, dict):
            raise RuntimeError("--card-json must contain one JSON object")
        return card

    if args.catalog_json and args.unique_key:
        payload = read_json(args.catalog_json)
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("--catalog-json must contain {'items': [...]} structure")
        for item in items:
            if isinstance(item, dict) and str(item.get("unique_key") or "") == str(args.unique_key):
                return item
        raise RuntimeError(f"unique_key not found in catalog: {args.unique_key}")

    raise RuntimeError("Provide either --card-json or both --catalog-json and --unique-key")


def dimensions_cm(card: dict[str, Any]) -> dict[str, float | None]:
    dims = card.get("dimensions_cm") if isinstance(card.get("dimensions_cm"), dict) else {}

    def f(*keys: str) -> float | None:
        for key in keys:
            value = card.get(key)
            if value is None and dims:
                value = dims.get(key.replace("_cm", "")) or dims.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    return {
        "width": f("width_cm", "width"),
        "depth": f("depth_cm", "depth"),
        "height": f("height_cm", "height"),
    }


def target_size_m(card: dict[str, Any]) -> list[float] | None:
    dims = dimensions_cm(card)
    if dims["width"] and dims["depth"] and dims["height"]:
        return [
            round(float(dims["width"]) / 100.0, 6),
            round(float(dims["depth"]) / 100.0, 6),
            round(float(dims["height"]) / 100.0, 6),
        ]
    return None


def item_category_label(card: dict[str, Any]) -> str:
    raw = safe_text(card.get("category_norm") or card.get("category_raw") or card.get("title") or "object")
    raw = raw.split(">")[-1].strip()
    raw = raw.replace("_", " ").replace("-", " ")
    raw = re.sub(r"\s+", " ", raw).strip().lower()
    if raw in {"столы", "table", "tables"}:
        return "dining table" if "dining" in safe_text(card.get("title")).lower() else "table"
    if raw in {"диваны", "sofas"}:
        return "sofa"
    if raw in {"кресла", "chairs"}:
        return "chair"
    return raw or "object"


def candidate_image_sources(card: dict[str, Any]) -> list[str]:
    sources: list[str] = []

    for key in ("preview_local_path", "image_local_path", "thumbnail_local_path"):
        value = safe_text(card.get(key))
        if value:
            sources.append(value)

    images = card.get("images")
    if isinstance(images, list):
        for value in images:
            if isinstance(value, str) and value.strip():
                sources.append(value.strip())
            elif isinstance(value, dict):
                for key in ("url", "src", "path", "local_path"):
                    v = safe_text(value.get(key))
                    if v:
                        sources.append(v)
                        break

    # Some cards store richer image metadata under image_color_features.source_image.path
    icf = card.get("image_color_features") if isinstance(card.get("image_color_features"), dict) else {}
    src_img = icf.get("source_image") if isinstance(icf.get("source_image"), dict) else {}
    for key in ("path", "value"):
        v = safe_text(src_img.get(key))
        if v:
            sources.append(v)

    out: list[str] = []
    seen: set[str] = set()
    for s in sources:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_text_prompt(card: dict[str, Any]) -> str:
    dims = dimensions_cm(card)
    parts = [
        f"Title: {safe_text(card.get('title'))}",
        f"Category: {safe_text(card.get('category_norm') or card.get('category_raw'))}",
        f"Brand: {safe_text(card.get('brand'))}",
        f"Style: {safe_text(card.get('style'))}",
        f"Color: {safe_text(card.get('color'))}",
        f"Materials: {safe_text(card.get('materials'))}",
        f"Dimensions cm: width={dims['width']}, depth={dims['depth']}, height={dims['height']}",
        f"Description: {safe_text(card.get('description'))}",
        f"VLM summary: {safe_text(card.get('vlm_description_summary'))}",
        f"VLM text: {safe_text(card.get('vlm_description_text'))[:1000]}",
    ]
    return "\n".join(p for p in parts if p and not p.endswith(": "))


def build_trellis_text_prompt(card: dict[str, Any]) -> str:
    category = item_category_label(card)
    dims = dimensions_cm(card)
    parts = [
        f"A single standalone {category}.",
        "Exactly one product, no duplicates, no collage, no showroom, no room background, no people.",
        f"Product title: {safe_text(card.get('title'))}.",
        f"Style: {safe_text(card.get('style'))}.",
        f"Color: {safe_text(card.get('color'))}.",
        f"Materials: {safe_text(card.get('materials'))}.",
        f"Dimensions: width {dims['width']} cm, depth {dims['depth']} cm, height {dims['height']} cm.",
        f"Description: {safe_text(card.get('description'))[:500]}",
        f"Visual summary: {safe_text(card.get('vlm_description_summary') or card.get('vlm_description_text'))[:500]}",
    ]
    return " ".join(p for p in parts if p and not p.endswith(": ."))


def has_text_description_for_trellis(card: dict[str, Any]) -> bool:
    fields = (
        "title",
        "category_norm",
        "category_raw",
        "description",
        "materials",
        "color",
        "style",
        "vlm_description_summary",
        "vlm_description_text",
    )
    text = " ".join(safe_text(card.get(key)) for key in fields)
    dims = dimensions_cm(card)
    return bool(text.strip()) or any(dims.get(key) is not None for key in ("width", "depth", "height"))


def normalize_card_for_job(card: dict[str, Any], *, orientation_yaw_deg: float | None = None) -> dict[str, Any]:
    dims = dimensions_cm(card)
    return {
        "schema": "trellis_supplier_card_job/v1",
        "unique_key": card.get("unique_key"),
        "title": card.get("title"),
        "source_site": card.get("source_site"),
        "product_url": card.get("product_url") or card.get("model_page_url") or card.get("source_url"),
        "category_raw": card.get("category_raw"),
        "category_norm": card.get("category_norm"),
        "brand": card.get("brand"),
        "style": card.get("style"),
        "color": card.get("color"),
        "materials": card.get("materials"),
        "description": card.get("description"),
        "vlm_description_summary": card.get("vlm_description_summary"),
        "vlm_description_text": card.get("vlm_description_text"),
        "dimensions_cm": dims,
        "target_size_m": target_size_m(card),
        "orientation": {
            "yaw_deg": orientation_yaw_deg,
            "source": "user_or_pipeline" if orientation_yaw_deg is not None else "not_provided",
        },
        "prompt_text": build_text_prompt(card),
        "trellis_text_prompt": build_trellis_text_prompt(card),
        "raw_card_compact": compact_card(card),
    }


def compact_card(card: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "unique_key",
        "source_site",
        "title",
        "brand",
        "collection",
        "category_raw",
        "category_norm",
        "product_url",
        "model_page_url",
        "model_download_url",
        "model_download_landing_url",
        "model_vendor_url",
        "asset_status",
        "asset_format",
        "asset_local_path",
        "preview_local_path",
        "price_value",
        "price_currency",
        "style",
        "color",
        "materials",
        "description",
        "dimensions_cm",
        "images",
        "vlm_description_summary",
        "vlm_description_text",
        "image_color_features",
    ]
    return {k: card.get(k) for k in keys if k in card}



# -----------------------------------------------------------------------------
# Direct supplier asset resolution before TRELLIS fallback
# -----------------------------------------------------------------------------


def _url_or_path_suffix(value: str) -> str:
    text = safe_text(value)
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        path = urllib.parse.urlparse(text).path.lower()
    else:
        path = text.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if path.endswith(suffix):
            return suffix
    return Path(path).suffix.lower()


def _looks_like_model_or_archive_ref(value: str) -> bool:
    suffix = _url_or_path_suffix(value)
    return suffix in MODEL_ASSET_EXTS or suffix in ARCHIVE_EXTS or suffix in ARCHIVE_SUFFIXES


def _collect_direct_asset_sources(card: dict[str, Any]) -> list[dict[str, str]]:
    fields = (
        "asset_local_path",
        "local_asset_path",
        "mesh_path",
        "mesh_local_path",
        "obj_path",
        "fbx_path",
        "glb_path",
        "gltf_path",
        "file_path",
        "downloaded_path",
        "model_download_url",
        "model_file_url",
        "archive_url",
        "source_archive_url",
        "model_archive_url",
        "asset_source_url",
    )

    raw: list[dict[str, str]] = []

    def add(value: Any, field: str) -> None:
        if not value:
            return
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                add(item, f"{field}[{idx}]")
            return
        if isinstance(value, dict):
            for k in ("url", "href", "src", "path", "local_path", "download_url", "file_url"):
                if value.get(k):
                    add(value.get(k), f"{field}.{k}")
            return
        text = safe_text(value)
        if text:
            raw.append({"field": field, "value": text})

    for key in fields:
        add(card.get(key), key)

    for nested_key in ("asset", "source", "extra"):
        nested = card.get(nested_key)
        if isinstance(nested, dict):
            for key in fields:
                add(nested.get(key), f"{nested_key}.{key}")
            for k, v in nested.items():
                lk = str(k).lower()
                if any(tok in lk for tok in ("asset", "model", "archive", "download", "fbx", "obj", "glb", "gltf")):
                    add(v, f"{nested_key}.{k}")

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        value = item["value"]
        if value in seen:
            continue
        seen.add(value)
        if value.startswith(("http://", "https://")) and not _looks_like_model_or_archive_ref(value):
            continue
        out.append(item)
    return out


def _find_supported_model_file(root: Path) -> Path | None:
    if root.is_file() and root.suffix.lower() in MODEL_ASSET_EXTS:
        return root.resolve()
    if not root.is_dir():
        return None

    candidates: list[Path] = []
    for ext in ("*.fbx", "*.FBX", "*.glb", "*.GLB", "*.gltf", "*.GLTF", "*.obj", "*.OBJ"):
        candidates.extend(root.rglob(ext))

    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, int, str]:
        ext = path.suffix.lower()
        priority = {".fbx": 0, ".glb": 1, ".gltf": 2, ".obj": 3}.get(ext, 9)
        try:
            size_rank = -path.stat().st_size
        except Exception:
            size_rank = 0
        return priority, size_rank, str(path).lower()

    return sorted(candidates, key=rank)[0].resolve()


def _copy_direct_model_sidecars(selected: Path, final_dir: Path) -> list[str]:
    copied: list[str] = []
    try:
        siblings = list(selected.parent.iterdir())
    except Exception:
        return copied
    mtl_sources: list[Path] = []

    for sibling in siblings:
        if not sibling.is_file() or sibling.resolve() == selected.resolve():
            continue
        if sibling.suffix.lower() not in MODEL_SIDECAR_EXTS:
            continue
        dest = final_dir / sibling.name
        try:
            if sibling.resolve() != dest.resolve():
                shutil.copy2(sibling, dest)
            copied.append(str(dest.resolve()))
            if sibling.suffix.lower() == ".mtl":
                mtl_sources.append(sibling)
        except Exception:
            continue

    texture_keys = {"map_kd", "map_ka", "map_ks", "map_bump", "bump", "norm", "disp", "decal", "refl"}
    for mtl_path in mtl_sources:
        try:
            lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(None, 1)
            if len(parts) != 2 or parts[0].lower() not in texture_keys:
                continue
            rel = parts[1].strip().strip('"').strip("'")
            if not rel:
                continue
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                continue

            source_path = mtl_path.parent / rel_path
            if not source_path.is_file():
                matches = [p for p in mtl_path.parent.rglob(rel_path.name) if p.is_file()]
                source_path = matches[0] if matches else source_path
            if not source_path.is_file():
                continue

            dest = final_dir / rel_path
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest)
                copied.append(str(dest.resolve()))
            except Exception:
                continue
    return copied


def _download_direct_asset(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _url_or_path_suffix(url)
    filename = Path(urllib.parse.urlparse(url).path).name
    if not filename:
        filename = f"downloaded_asset{suffix or '.bin'}"
    if suffix and not filename.lower().endswith(suffix):
        filename += suffix
    out_path = out_dir / filename

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://3ddd.ru/",
    }
    r = requests.get(url, headers=headers, timeout=180)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def _extract_direct_archive(archive_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _url_or_path_suffix(str(archive_path))
    report: dict[str, Any] = {
        "archive_path": str(archive_path),
        "extract_dir": str(out_dir),
        "method": None,
        "ok": False,
    }

    if suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
        report["method"] = "zipfile"
        report["ok"] = True
        return report

    if suffix in {".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(out_dir)
        report["method"] = "tarfile"
        report["ok"] = True
        return report

    if suffix in {".7z", ".rar"}:
        commands = []
        if shutil.which("7z"):
            commands.append(["7z", "x", "-y", f"-o{out_dir}", str(archive_path)])
        if shutil.which("unar"):
            commands.append(["unar", "-f", "-o", str(out_dir), str(archive_path)])
        if not commands:
            raise RuntimeError(f"Archive requires 7z or unar, but neither is available: {archive_path}")
        last_error = None
        for cmd in commands:
            try:
                run_cmd_stream(cmd, check=True)
                report["method"] = cmd[0]
                report["ok"] = True
                return report
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Archive extraction failed: {archive_path}: {last_error}")

    raise RuntimeError(f"Unsupported archive type: {archive_path}")


def try_resolve_direct_model_asset(card: dict[str, Any], local_job_dir: Path) -> dict[str, Any] | None:
    sources = _collect_direct_asset_sources(card)
    if not sources:
        return None

    work_dir = local_job_dir / "direct_model_asset"
    downloads_dir = work_dir / "downloads"
    extract_root = work_dir / "extracted"
    final_dir = work_dir / "selected"
    final_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []

    for src in sources:
        raw = src["value"]
        field = src["field"]
        attempt: dict[str, Any] = {"field": field, "source": raw, "status": "running"}
        attempts.append(attempt)

        try:
            if raw.startswith(("http://", "https://")):
                if not _looks_like_model_or_archive_ref(raw):
                    attempt["status"] = "skipped_not_direct_model_or_archive_url"
                    continue
                src_path = _download_direct_asset(raw, downloads_dir)
                attempt["downloaded_path"] = str(src_path)
            else:
                src_path = Path(raw).expanduser()
                if not src_path.exists():
                    attempt["status"] = "skipped_local_path_not_found"
                    continue
                src_path = src_path.resolve()

            if src_path.is_dir():
                selected = _find_supported_model_file(src_path)
                if selected is None:
                    attempt["status"] = "failed_no_supported_model_in_dir"
                    continue
            elif src_path.suffix.lower() in MODEL_ASSET_EXTS:
                selected = src_path
            elif _url_or_path_suffix(str(src_path)) in ARCHIVE_EXTS or _url_or_path_suffix(str(src_path)) in ARCHIVE_SUFFIXES:
                one_extract_dir = extract_root / f"{len(attempts):03d}_{slugify(src_path.stem, max_len=48)}"
                extract_report = _extract_direct_archive(src_path, one_extract_dir)
                attempt["extract_report"] = extract_report
                selected = _find_supported_model_file(one_extract_dir)
                if selected is None:
                    attempt["status"] = "failed_no_supported_model_in_archive"
                    continue
            else:
                attempt["status"] = "skipped_unsupported_suffix"
                continue

            final_path = final_dir / selected.name
            if selected.resolve() != final_path.resolve():
                shutil.copy2(selected, final_path)
            sidecar_paths = _copy_direct_model_sidecars(selected, final_dir)

            asset_format = final_path.suffix.lower().lstrip(".")
            payload = {
                "schema": "direct_supplier_model_asset_result/v1",
                "ok": True,
                "asset_status": "ready_existing_or_downloaded_model_asset",
                "asset_format": asset_format,
                "asset_local_path": str(final_path.resolve()),
                "asset_source_url": raw if raw.startswith(("http://", "https://")) else None,
                "asset_generation_method": "direct_supplier_model_or_archive",
                "source_field": field,
                "source_value": raw,
                "selected_source_model_path": str(selected),
                "asset_sidecar_paths": sidecar_paths,
                "attempts": attempts,
            }
            attempt["status"] = "success"
            attempt["selected_model_path"] = str(selected)
            attempt["final_asset_path"] = str(final_path)
            attempt["copied_sidecar_paths"] = sidecar_paths
            return payload

        except Exception as exc:
            attempt["status"] = "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"

    write_json(work_dir / "direct_model_asset_attempts.json", {"attempts": attempts})
    return None


def patch_card_with_direct_asset(card: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out["asset_status"] = payload["asset_status"]
    out["asset_format"] = payload["asset_format"]
    out["asset_local_path"] = payload["asset_local_path"]
    out["asset_source_url"] = payload.get("asset_source_url")
    extra = out.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        out["extra"] = extra
    extra["direct_supplier_model_asset"] = {
        "asset_local_path": payload["asset_local_path"],
        "asset_format": payload["asset_format"],
        "asset_status": payload["asset_status"],
        "asset_generation_method": payload["asset_generation_method"],
        "source_field": payload.get("source_field"),
        "source_value": payload.get("source_value"),
        "selected_source_model_path": payload.get("selected_source_model_path"),
    }
    return out



# -----------------------------------------------------------------------------
# Local image preparation
# -----------------------------------------------------------------------------


def image_ext_from_response(url: str, content_type: str | None) -> str:
    path = urllib.parse.urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return ext
    ct = (content_type or "").lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    return ".jpg"


def _single_object_crop_bbox(
    image_path: Path,
    *,
    component: str,
    background_threshold: int = 245,
    min_component_area_ratio: float = 0.001,
) -> tuple[int, int, int, int] | None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --single-object-crop: python3 -m pip install pillow") from exc

    with Image.open(image_path) as src:
        rgba = src.convert("RGBA")
        full_w, full_h = rgba.size
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        rgb = bg.convert("RGB")
        small = rgb.copy()
        small.thumbnail((768, 768))

    scale_x = full_w / float(small.size[0])
    scale_y = full_h / float(small.size[1])
    width, height = small.size
    pixels = small.load()
    mask = bytearray(width * height)
    for y in range(height):
        row = y * width
        for x in range(width):
            r, g, b = pixels[x, y]
            # Product renders are normally on white/transparent backgrounds.
            # Keep shadows and dark/colored furniture, reject near-white canvas.
            if min(r, g, b) < background_threshold:
                mask[row + x] = 1

    visited = bytearray(width * height)
    min_area = max(16, int(width * height * min_component_area_ratio))
    components: list[dict[str, float | int]] = []

    for start in range(width * height):
        if not mask[start] or visited[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = 1
        area = 0
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0
        while q:
            idx = q.popleft()
            x = idx % width
            y = idx // width
            area += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                nidx = ny * width + nx
                if mask[nidx] and not visited[nidx]:
                    visited[nidx] = 1
                    q.append(nidx)
        if area >= min_area:
            components.append(
                {
                    "area": area,
                    "min_x": min_x,
                    "min_y": min_y,
                    "max_x": max_x,
                    "max_y": max_y,
                    "center_y": (min_y + max_y) / 2.0,
                }
            )

    if not components:
        return None

    if component == "top":
        chosen = min(components, key=lambda c: (float(c["center_y"]), -int(c["area"])))
    elif component == "middle":
        by_y = sorted(components, key=lambda c: float(c["center_y"]))
        chosen = by_y[len(by_y) // 2]
    elif component == "bottom":
        chosen = max(components, key=lambda c: (float(c["center_y"]), int(c["area"])))
    else:
        chosen = max(components, key=lambda c: int(c["area"]))

    left = int(max(0, int(chosen["min_x"]) * scale_x))
    top = int(max(0, int(chosen["min_y"]) * scale_y))
    right = int(min(full_w, (int(chosen["max_x"]) + 1) * scale_x))
    bottom = int(min(full_h, (int(chosen["max_y"]) + 1) * scale_y))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_single_object_image(
    image_path: Path,
    out_path: Path,
    *,
    component: str = "largest",
    padding_ratio: float = 0.16,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --single-object-crop: python3 -m pip install pillow") from exc

    bbox = _single_object_crop_bbox(image_path, component=component)
    if bbox is None:
        raise RuntimeError(f"Could not detect foreground object for crop: {image_path}")

    with Image.open(image_path) as src:
        rgba = src.convert("RGBA")
        width, height = rgba.size
        left, top, right, bottom = bbox
        pad = int(max(right - left, bottom - top) * max(0.0, padding_ratio))
        crop_box = (
            max(0, left - pad),
            max(0, top - pad),
            min(width, right + pad),
            min(height, bottom + pad),
        )
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        crop = bg.crop(crop_box).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=95)

    return {
        "source_image": str(image_path),
        "cropped_image": str(out_path),
        "component": component,
        "bbox": list(bbox),
        "crop_box": list(crop_box),
        "padding_ratio": padding_ratio,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = safe_text(text)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _encode_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _single_object_vlm_prompt(category_label: str) -> str:
    return f"""
You are filtering supplier product images before 3D reconstruction.

Question: Is this image suitable as a reference for generating exactly one single {category_label}?
Главная проверка: "на фото один объект категории: {category_label}".
Ответь true только если на изображении ровно один основной предмет этой категории.

Return only JSON with this schema:
{{
  "single_object": true,
  "object_count": 1,
  "has_multiple_variants": false,
  "has_collage_or_comparison": false,
  "confidence": 0.0,
  "reason": "short reason"
}}

Set "single_object" to false if the image contains multiple {category_label} instances,
multiple color/material variants of the product, a collage, comparison layout, showroom,
room scene, styled interior, people, props that dominate the view, or any repeated product
that could make a 3D generator create more than one object.
Prefer clean catalog packshots or product-page images on a plain/simple background.
""".strip()


def ask_ollama_single_object_vlm(
    image_path: Path,
    *,
    category_label: str,
    ollama_url: str,
    model: str,
    timeout_sec: int,
) -> dict[str, Any]:
    prompt = _single_object_vlm_prompt(category_label)
    endpoint = ollama_url.rstrip("/") + "/api/chat"
    response = requests.post(
        endpoint,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [_encode_image_base64(image_path)],
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        },
        timeout=timeout_sec,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    content = str(message.get("content") or payload.get("response") or "")
    parsed = _extract_json_object(content)
    single = bool(parsed.get("single_object") is True)
    return {
        "provider": "ollama",
        "model": model,
        "image": str(image_path),
        "category_label": category_label,
        "accepted": single,
        "raw_response": content,
        "parsed": parsed,
    }


def unload_ollama_model(*, ollama_url: str, model: str, timeout_sec: int = 30) -> dict[str, Any]:
    endpoint = ollama_url.rstrip("/") + "/api/generate"
    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=timeout_sec,
        )
        response.raise_for_status()
        return {"ok": True, "provider": "ollama", "model": model, "response": response.json()}
    except Exception as exc:
        return {"ok": False, "provider": "ollama", "model": model, "error": str(exc)}


def ask_openai_compatible_single_object_vlm(
    image_path: Path,
    *,
    provider: str,
    category_label: str,
    model: str,
    timeout_sec: int,
) -> dict[str, Any]:
    from topview_vlm_orientation_repair import call_openai_compatible_vlm

    prompt = _single_object_vlm_prompt(category_label)
    payload = call_openai_compatible_vlm(
        provider=provider,
        model=model or None,
        prompt=prompt,
        image_path=image_path,
        temperature=0.0,
        max_tokens=700,
        timeout_sec=timeout_sec,
    )
    content = ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(message, dict):
            content = str(message.get("content") or "")
    parsed = _extract_json_object(content)
    single = bool(parsed.get("single_object") is True)
    return {
        "provider": provider,
        "model": model,
        "image": str(image_path),
        "category_label": category_label,
        "accepted": single,
        "raw_response": content,
        "parsed": parsed,
    }


def filter_images_with_single_object_vlm(
    images: list[Path],
    *,
    card: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[Path], list[dict[str, Any]]]:
    if not args.vlm_single_object_filter:
        return images, []

    if os.environ.get("CGS_TRELLIS33_SKIP_VLM_FILTER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return images, [
            {
                "stage": "trellis2_vlm_filter_skipped_by_env",
                "reason": "CGS_TRELLIS33_SKIP_VLM_FILTER is enabled",
                "accepted_count": len(images),
            }
        ]
    category = item_category_label(card)
    reviews: list[dict[str, Any]] = []
    accepted: list[Path] = []
    for image in images:
        if args.vlm_provider == "ollama":
            review = ask_ollama_single_object_vlm(
                image,
                category_label=category,
                ollama_url=args.vlm_ollama_url,
                model=args.vlm_model,
                timeout_sec=int(args.vlm_timeout),
            )
        elif args.vlm_provider in {"openai", "openrouter"}:
            review = ask_openai_compatible_single_object_vlm(
                image,
                provider=args.vlm_provider,
                category_label=category,
                model=args.vlm_model,
                timeout_sec=int(args.vlm_timeout),
            )
        else:
            raise RuntimeError(f"Unsupported --vlm-provider={args.vlm_provider!r}")
        reviews.append(review)
        if review.get("accepted"):
            accepted.append(image)
        else:
            reason = (review.get("parsed") or {}).get("reason") or "not a single object"
            print(f"[VLM] rejected {image.name}: {reason}", file=sys.stderr)
    if args.vlm_provider == "ollama" and args.vlm_unload_after_filter:
        unload_report = unload_ollama_model(
            ollama_url=args.vlm_ollama_url,
            model=args.vlm_model,
            timeout_sec=min(int(args.vlm_timeout), 60),
        )
        reviews.append({"stage": "vlm_unload_after_filter", **unload_report})
        if unload_report.get("ok"):
            print(f"[VLM] unloaded Ollama model after filtering: {args.vlm_model}", file=sys.stderr)
        else:
            print(f"[WARN] failed to unload Ollama model after filtering: {unload_report.get('error')}", file=sys.stderr)
    return accepted, reviews


def prepare_images(
    card: dict[str, Any],
    local_job_dir: Path,
    *,
    max_images: int,
    image_source_index: int = 0,
    single_object_crop: bool = False,
    single_object_crop_component: str = "largest",
    single_object_crop_padding: float = 0.16,
    args: argparse.Namespace | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    sources = candidate_image_sources(card)
    if not sources:
        raise RuntimeError("No images found in product card: expected images[] or preview_local_path")
    if image_source_index < 0:
        raise RuntimeError("--image-source-index must be 0 or a positive 1-based index")
    selected_sources = sources
    if image_source_index:
        if image_source_index > len(sources):
            raise RuntimeError(f"--image-source-index={image_source_index} is out of range; card has {len(sources)} image sources")
        selected_sources = [sources[image_source_index - 1]]

    out_dir = local_job_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": safe_text(card.get("product_url") or card.get("source_url") or "https://3ddd.ru/"),
    }

    local_images: list[Path] = []
    prepared: list[dict[str, Any]] = []
    download_limit = len(selected_sources) if (args is not None and bool(getattr(args, "vlm_single_object_filter", False))) else max(max_images, 1)
    for idx, src in enumerate(selected_sources[:download_limit], 1):
        parsed = urllib.parse.urlparse(src)
        try:
            if parsed.scheme in {"http", "https"}:
                r = requests.get(src, headers=headers, timeout=90)
                r.raise_for_status()
                ext = image_ext_from_response(src, r.headers.get("content-type"))
                out = out_dir / f"image_{idx:02d}{ext}"
                out.write_bytes(r.content)
                local_images.append(out)
            else:
                p = Path(src).expanduser()
                if not p.is_file():
                    print(f"[WARN] Local image not found, skipping: {p}", file=sys.stderr)
                    continue
                ext = p.suffix.lower() if p.suffix.lower() in IMAGE_EXTS else ".jpg"
                out = out_dir / f"image_{idx:02d}{ext}"
                shutil.copy2(p, out)
                local_images.append(out)
            final_path = local_images[-1]
            entry: dict[str, Any] = {
                "source": src,
                "downloaded_or_copied": str(final_path),
            }
            if single_object_crop:
                cropped_path = out_dir / f"image_{idx:02d}.single_object.png"
                entry["single_object_crop"] = crop_single_object_image(
                    final_path,
                    cropped_path,
                    component=single_object_crop_component,
                    padding_ratio=single_object_crop_padding,
                )
                local_images[-1] = cropped_path
                final_path = cropped_path
            entry["prepared_image"] = str(final_path)
            prepared.append(entry)
        except Exception as exc:
            print(f"[WARN] Failed to fetch image {idx}: {src}: {exc}", file=sys.stderr)

    if not local_images:
        raise RuntimeError("All product images failed to download/copy")

    vlm_reviews: list[dict[str, Any]] = []
    trellis_input_dir = out_dir
    if args is not None:
        local_images, vlm_reviews = filter_images_with_single_object_vlm(local_images, card=card, args=args)
        if args.vlm_single_object_filter:
            trellis_input_dir = local_job_dir / "trellis_input_images"
            if trellis_input_dir.exists():
                shutil.rmtree(trellis_input_dir)
            trellis_input_dir.mkdir(parents=True, exist_ok=True)
            remapped_images: list[Path] = []
            for idx, image_path in enumerate(local_images[: max(max_images, 1)], 1):
                ext = image_path.suffix.lower() if image_path.suffix.lower() in IMAGE_EXTS else ".jpg"
                dst = trellis_input_dir / f"image_{idx:02d}{ext}"
                shutil.copy2(image_path, dst)
                remapped_images.append(dst)
            local_images = remapped_images

    manifest = {
        "images": [str(p) for p in local_images],
        "count": len(local_images),
        "trellis_input_dir": str(trellis_input_dir),
        "sources": sources,
        "selected_sources": selected_sources[: max(max_images, 1)],
        "image_source_index": image_source_index,
        "single_object_crop": single_object_crop,
        "single_object_crop_component": single_object_crop_component,
        "single_object_crop_padding": single_object_crop_padding,
        "vlm_single_object_filter": bool(args.vlm_single_object_filter) if args is not None else False,
        "vlm_reviews": vlm_reviews,
        "prepared": prepared,
    }
    return local_images, manifest


# -----------------------------------------------------------------------------
# SSH/SCP transport
# -----------------------------------------------------------------------------


def ssh_base(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh", "-p", str(args.server_port)]
    if args.ssh_key:
        cmd.extend(["-i", str(Path(args.ssh_key).expanduser())])
    cmd.extend([
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{args.server_user}@{args.server_host}",
    ])
    return cmd


def scp_base(args: argparse.Namespace) -> list[str]:
    cmd = ["scp", "-P", str(args.server_port), "-r"]
    if args.ssh_key:
        cmd.extend(["-i", str(Path(args.ssh_key).expanduser())])
    cmd.extend([
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
    ])
    return cmd


def ssh_run(args: argparse.Namespace, remote_script: str) -> str:
    proc = run_cmd_stream(ssh_base(args) + [remote_script], check=True)
    return proc.stdout or ""


def scp_to_remote(args: argparse.Namespace, local_path: Path, remote_path: str) -> None:
    remote_spec = f"{args.server_user}@{args.server_host}:{remote_path}"
    run_cmd_stream(scp_base(args) + [str(local_path), remote_spec], check=True)


def scp_from_remote(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_spec = f"{args.server_user}@{args.server_host}:{remote_path}"
    run_cmd_stream(scp_base(args) + [remote_spec, str(local_path)], check=True)


def remote_worker_root(args: argparse.Namespace) -> str:
    return str(getattr(args, "remote_worker_root", "") or DEFAULT_REMOTE_WORKER_ROOT).rstrip("/")


def ensure_remote_worker(args: argparse.Namespace) -> None:
    start_script = f"{str(args.remote_trellis_root).rstrip('/')}/start_trellis2_worker.sh"
    command = "\n".join(
        [
            remote_env_prefix(args),
            f"test -f {shell_quote(start_script)}",
            f"bash {shell_quote(start_script)}",
            f"cat {shell_quote(remote_worker_root(args) + '/worker.state.json')} 2>/dev/null || true",
        ]
    )
    ssh_run(args, command)


def build_trellis2_worker_job_payload(
    args: argparse.Namespace,
    job_id: str,
    remote_job_dir: str,
    remote_glb: str,
    remote_report: str,
) -> dict[str, Any]:
    return {
        "schema": "trellis2_persistent_job/v1",
        "job_id": job_id,
        "remote_job_dir": remote_job_dir,
        "image_dir": remote_job_dir.rstrip("/") + "/images",
        "output_dir": remote_job_dir.rstrip("/") + "/output",
        "out_glb": remote_glb,
        "out_report": remote_report,
        "model": str(args.remote_model_dir),
        "mode": str(args.mode),
        "max_images": int(args.max_images),
        "seed": int(args.seed),
        "pipeline_type": int(getattr(args, "pipeline_type", 512) or 512),
        "sparse_steps": int(args.sparse_steps),
        "slat_steps": int(args.slat_steps),
        "ss_guidance_strength": float(getattr(args, "ss_guidance_strength", 7.5) or 7.5),
        "slat_guidance_strength": float(getattr(args, "slat_guidance_strength", 3.0) or 3.0),
        "decimation_target": int(getattr(args, "decimation_target", 50000) or 50000),
        "texture_size": int(args.texture_size),
        "pre_export_simplify_target": int(getattr(args, "pre_export_simplify_target", 0) or 0),
        "no_remesh": bool(getattr(args, "no_remesh", False)),
        "remesh_band": int(getattr(args, "remesh_band", 1) or 1),
        "remesh_project": float(getattr(args, "remesh_project", 0.0) or 0.0),
        "no_webp": bool(getattr(args, "no_webp", True)),
        "legacy_params": {
            "multi_mode": str(getattr(args, "multi_mode", "")),
            "simplify": float(getattr(args, "simplify", 0.0) or 0.0),
            "image_size": int(getattr(args, "image_size", 0) or 0),
            "fill_holes_resolution": int(getattr(args, "fill_holes_resolution", 0) or 0),
            "fill_holes_num_views": int(getattr(args, "fill_holes_num_views", 0) or 0),
        },
    }


def enqueue_remote_worker_job(args: argparse.Namespace, payload: dict[str, Any], local_job_dir: Path) -> str:
    worker_root = remote_worker_root(args)
    job_id = str(payload["job_id"])
    local_payload = local_job_dir / "trellis2_worker_job.json"
    write_json(local_payload, payload)
    remote_tmp = f"{worker_root}/queue/{job_id}.json.tmp"
    remote_final = f"{worker_root}/queue/{job_id}.json"
    ssh_run(args, f"mkdir -p {shell_quote(worker_root + '/queue')}")
    scp_to_remote(args, local_payload, remote_tmp)
    ssh_run(args, f"mv {shell_quote(remote_tmp)} {shell_quote(remote_final)}")
    return remote_final


def wait_remote_worker_job(args: argparse.Namespace, remote_report: str, remote_glb: str) -> str:
    timeout_sec = float(getattr(args, "remote_worker_timeout_sec", 1800.0) or 1800.0)
    poll_sec = max(0.5, float(getattr(args, "remote_worker_poll_sec", 2.0) or 2.0))
    deadline = time.monotonic() + timeout_sec
    last_status = ""
    while True:
        probe = "\n".join(
            [
                f"if [ -s {shell_quote(remote_report)} ]; then",
                f"  if [ -s {shell_quote(remote_glb)} ]; then echo done; else echo report_only; fi",
                "else",
                "  echo wait",
                "fi",
            ]
        )
        try:
            out = ssh_run(args, probe).strip()
        except Exception as exc:
            out = f"probe_error:{type(exc).__name__}:{exc}"
        last_status = out.splitlines()[-1] if out else "wait"
        if last_status in {"done", "report_only"}:
            return last_status
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for TRELLIS.2 worker job after {timeout_sec:.1f}s; "
                f"last_status={last_status!r}, report={remote_report}"
            )
        time.sleep(poll_sec)


def run_remote_trellis2_persistent(
    args: argparse.Namespace,
    job_id: str,
    remote_job_dir: str,
    remote_glb: str,
    remote_report: str,
    local_job_dir: Path,
) -> str:
    ensure_remote_worker(args)
    payload = build_trellis2_worker_job_payload(args, job_id, remote_job_dir, remote_glb, remote_report)
    remote_queue_job_json = enqueue_remote_worker_job(args, payload, local_job_dir)
    status = wait_remote_worker_job(args, remote_report, remote_glb)
    stdout = (
        "[TRELLIS2][persistent] "
        f"worker_root={remote_worker_root(args)} queue_job={remote_queue_job_json} wait_status={status}\n"
    )
    (local_job_dir / "trellis2_worker_queue_path.txt").write_text(remote_queue_job_json, encoding="utf-8")
    if status != "done":
        raise RuntimeError(
            f"TRELLIS.2 persistent worker finished without GLB: status={status}; "
            f"remote_report={remote_report}; remote_glb={remote_glb}"
        )
    return stdout


def run_remote_trellis2_single_run(
    args: argparse.Namespace,
    job_id: str,
    remote_job_dir: str,
    remote_glb: str,
    remote_report: str,
    local_job_dir: Path,
) -> str:
    payload = build_trellis2_worker_job_payload(args, job_id, remote_job_dir, remote_glb, remote_report)
    remote_queue_job_json = enqueue_remote_worker_job(args, payload, local_job_dir)
    worker_root = remote_worker_root(args)
    worker_script = f"{str(args.remote_trellis_root).rstrip('/')}/run_trellis2_persistent_worker.py"
    log_file = f"{worker_root}/logs/trellis2_single_run_{job_id}.log"
    gpu_index = int(str(args.remote_cuda_visible_devices).split(",")[0])
    command = "\n".join(
        [
            remote_env_prefix(args),
            f"mkdir -p {shell_quote(worker_root + '/logs')}",
            f"test -f {shell_quote(worker_script)}",
            f"{shell_quote(args.remote_python)} -u -X faulthandler {shell_quote(worker_script)} "
            f"--worker-root {shell_quote(worker_root)} "
            f"--model {shell_quote(args.remote_model_dir)} "
            f"--gpu-index {gpu_index} --poll-sec 1.0 --idle-exit-sec 1 "
            f"--log-file {shell_quote(log_file)}",
        ]
    )
    remote_stdout = ssh_run(args, command)
    (local_job_dir / "trellis2_worker_queue_path.txt").write_text(remote_queue_job_json, encoding="utf-8")
    status = wait_remote_worker_job(args, remote_report, remote_glb)
    stdout = (
        "[TRELLIS2][single-run] "
        f"worker_root={worker_root} queue_job={remote_queue_job_json} wait_status={status} "
        f"log={log_file}\n{remote_stdout or ''}"
    )
    if status != "done":
        raise RuntimeError(
            f"TRELLIS.2 single-run worker finished without GLB: status={status}; "
            f"remote_report={remote_report}; remote_glb={remote_glb}"
        )
    return stdout


# -----------------------------------------------------------------------------
# Remote runner generation
# -----------------------------------------------------------------------------


def remote_runner_code() -> str:
    # This code is intentionally self-contained and written to remote job dir.
    return r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils


def gpu_mem_mib(gpu_index: int = 0) -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


class VramMonitor:
    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.2) -> None:
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.time(), gpu_mem_mib(self.gpu_index)))
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def peak_mib(self) -> int:
        vals = [v for _, v in self.samples if v >= 0]
        return max(vals) if vals else -1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-model", default="")
    ap.add_argument("--generation-source", choices=["image", "text"], default="image")
    ap.add_argument("--mode", choices=["single_image", "multi_image"], default="multi_image")
    ap.add_argument("--multi-mode", choices=["stochastic", "multidiffusion"], default="stochastic")
    ap.add_argument("--max-images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--sparse-steps", type=int, default=4)
    ap.add_argument("--slat-steps", type=int, default=4)
    ap.add_argument("--texture-size", type=int, default=256)
    ap.add_argument("--simplify", type=float, default=0.98)
    args = ap.parse_args()

    job_dir = Path(args.job_dir).resolve()
    images_dir = job_dir / "images"
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    card_path = job_dir / "card.normalized.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    )[: max(args.max_images, 1)]
    if args.generation_source == "image" and not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    if args.generation_source == "image" and args.mode == "single_image":
        image_paths = image_paths[:1]

    out_glb = out_dir / "asset.trellis.glb"
    out_report = out_dir / "trellis.report.json"

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    report: dict = {
        "schema": "trellis_remote_generation_report/v1",
        "job_dir": str(job_dir),
        "image_paths": [str(p) for p in image_paths],
        "image_count": len(image_paths),
        "generation_source": args.generation_source,
        "mode": args.mode,
        "multi_mode": args.multi_mode,
        "model": args.model,
        "seed": args.seed,
        "sparse_steps": args.sparse_steps,
        "slat_steps": args.slat_steps,
        "texture_size": args.texture_size,
        "simplify": args.simplify,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "card": card,
    }

    monitor = VramMonitor(gpu_index=args.gpu_index, interval_sec=0.2)
    t_total0 = time.perf_counter()
    monitor.start()

    try:
        t0 = time.perf_counter()
        if args.generation_source == "text":
            pipe = TrellisTextTo3DPipeline.from_pretrained(args.text_model or args.model)
        else:
            pipe = TrellisImageTo3DPipeline.from_pretrained(args.model)
        pipe.cuda()
        torch.cuda.synchronize()
        report["load_model_sec"] = time.perf_counter() - t0

        images = [Image.open(p).convert("RGB") for p in image_paths]

        t0 = time.perf_counter()
        if args.generation_source == "text":
            prompt = str(card.get("trellis_text_prompt") or card.get("prompt_text") or "").strip()
            if not prompt:
                raise RuntimeError("Text fallback requested, but card has no trellis_text_prompt/prompt_text")
            report["text_prompt"] = prompt
            outputs = pipe.run(
                prompt,
                seed=args.seed,
                sparse_structure_sampler_params={"steps": args.sparse_steps, "cfg_strength": 7.5},
                slat_sampler_params={"steps": args.slat_steps, "cfg_strength": 3.0},
            )
        elif args.mode == "single_image":
            outputs = pipe.run(
                images[0],
                seed=args.seed,
                sparse_structure_sampler_params={"steps": args.sparse_steps, "cfg_strength": 7.5},
                slat_sampler_params={"steps": args.slat_steps, "cfg_strength": 3.0},
            )
        else:
            outputs = pipe.run_multi_image(
                images,
                seed=args.seed,
                mode=args.multi_mode,
                sparse_structure_sampler_params={"steps": args.sparse_steps, "cfg_strength": 7.5},
                slat_sampler_params={"steps": args.slat_steps, "cfg_strength": 3.0},
            )
        torch.cuda.synchronize()
        report["generation_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=args.simplify,
            texture_size=args.texture_size,
        )
        glb.export(str(out_glb))
        torch.cuda.synchronize()
        report["glb_export_sec"] = time.perf_counter() - t0

    finally:
        monitor.stop()

    report["total_sec"] = time.perf_counter() - t_total0
    report["nvidia_smi_peak_vram_mib"] = monitor.peak_mib
    report["torch_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    report["torch_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024 / 1024, 2)
    report["out_glb"] = str(out_glb)
    report["glb_size_mb"] = round(out_glb.stat().st_size / 1024 / 1024, 3) if out_glb.exists() else None
    report["ok"] = out_glb.exists()

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def remote_env_prefix(args: argparse.Namespace) -> str:
    # Commands are executed on remote server. Keep explicit env to avoid contamination
    # from other conda envs such as m3dlayout.
    cuda_device = int(str(args.remote_cuda_visible_devices).split(",")[0])
    remote_python = str(getattr(args, "remote_python", "") or DEFAULT_REMOTE_PYTHON)
    venv_root = str(Path(remote_python).parent.parent) if remote_python.startswith("/") else "/venv/trellis2"
    lines = [
        "set -euo pipefail",
        f"cd {shell_quote(args.remote_trellis_root)}",
        f"export PATH={shell_quote(str(Path(remote_python).parent))}:$PATH",
        f"export CUDA_HOME={shell_quote(venv_root)}",
        f"export CUDA_PATH={shell_quote(venv_root)}",
        f"export PYTHONPATH={shell_quote(args.remote_trellis_root)}:${{PYTHONPATH:-}}",
        "export LD_LIBRARY_PATH=" + shell_quote(f"{venv_root}/lib") + ":${LD_LIBRARY_PATH:-}",
        f"export CUDA_VISIBLE_DEVICES={cuda_device}",
        "export TORCH_CUDA_ARCH_LIST=8.6",
        "export MAX_JOBS=4",
        "export OPENCV_IO_ENABLE_OPENEXR=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "export ATTN_BACKEND=sdpa",
        "export SPARSE_ATTN_BACKEND=xformers",
        "export HF_HOME=/workspace/hf_cache",
        "export HF_HUB_CACHE=/workspace/hf_cache/hub",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
        "export PYTHONUNBUFFERED=1",
        "export TOKENIZERS_PARALLELISM=false",
    ]
    return "\n".join(lines)


def remote_proxy_glb_code() -> str:
    return r'''
import json
import math
import sys
import time
from pathlib import Path

import trimesh

job_dir = Path(sys.argv[1])
out_glb = Path(sys.argv[2])
out_report = Path(sys.argv[3])
card_path = job_dir / "card.normalized.json"
card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}

def text(*keys):
    vals = []
    for key in keys:
        v = card.get(key)
        if v is not None:
            vals.append(str(v).lower())
    return " ".join(vals)

def size_m():
    raw = card.get("target_size_m")
    if isinstance(raw, list) and len(raw) >= 3:
        try:
            vals = [max(float(raw[i]), 0.05) for i in range(3)]
            return vals[0], vals[1], vals[2]
        except Exception:
            pass
    dims = card.get("dimensions_cm") if isinstance(card.get("dimensions_cm"), dict) else {}
    try:
        return (
            max(float(dims.get("width") or 80) / 100.0, 0.05),
            max(float(dims.get("depth") or 45) / 100.0, 0.05),
            max(float(dims.get("height") or 75) / 100.0, 0.05),
        )
    except Exception:
        return 0.8, 0.45, 0.75

category = text("category_norm", "category_raw", "title", "description")
sx, sy, sz = size_m()
parts = []

def box(name, center, extents):
    mesh = trimesh.creation.box(extents=[max(float(v), 0.01) for v in extents])
    mesh.apply_translation(center)
    mesh.metadata["name"] = name
    parts.append(mesh)

def cyl(name, radius, depth, center, sections=24):
    mesh = trimesh.creation.cylinder(radius=max(float(radius), 0.01), height=max(float(depth), 0.01), sections=sections)
    mesh.apply_translation(center)
    mesh.metadata["name"] = name
    parts.append(mesh)

if "bed" in category or "кровать" in category:
    box("base", (0, 0, sz * 0.20), (sx, sy, sz * 0.28))
    box("mattress", (0, -sy * 0.03, sz * 0.46), (sx * 0.96, sy * 0.92, sz * 0.24))
    box("headboard", (0, sy * 0.47, sz * 0.55), (sx, sy * 0.08, sz * 0.70))
    box("pillow_left", (-sx * 0.22, sy * 0.30, sz * 0.64), (sx * 0.30, sy * 0.16, sz * 0.09))
    box("pillow_right", (sx * 0.22, sy * 0.30, sz * 0.64), (sx * 0.30, sy * 0.16, sz * 0.09))
elif any(k in category for k in ["nightstand", "dresser", "wardrobe", "cabinet", "комод", "шкаф", "тумб"]):
    box("case", (0, 0, sz * 0.52), (sx, sy, sz * 0.92))
    n = 2 if "nightstand" in category or "тумб" in category else 4
    for i in range(n):
        z = sz * (0.20 + 0.58 * (i + 0.5) / n)
        box(f"drawer_{i+1}", (0, -sy * 0.51, z), (sx * 0.86, sy * 0.035, sz * 0.10))
        box(f"handle_{i+1}", (0, -sy * 0.55, z), (sx * 0.22, sy * 0.025, sz * 0.025))
elif "desk" in category or "table" in category or "стол" in category:
    top_h = max(sz * 0.08, 0.045)
    leg_h = max(sz - top_h, 0.10)
    box("top", (0, 0, leg_h + top_h * 0.5), (sx, sy, top_h))
    for x in (-sx * 0.42, sx * 0.42):
        for y in (-sy * 0.38, sy * 0.38):
            box("leg", (x, y, leg_h * 0.5), (sx * 0.06, sy * 0.06, leg_h))
elif "chair" in category or "стул" in category:
    seat_z = sz * 0.46
    box("seat", (0, 0, seat_z), (sx, sy, sz * 0.10))
    box("back", (0, sy * 0.42, sz * 0.70), (sx, sy * 0.10, sz * 0.55))
    for x in (-sx * 0.38, sx * 0.38):
        for y in (-sy * 0.34, sy * 0.34):
            box("leg", (x, y, seat_z * 0.5), (sx * 0.06, sy * 0.06, seat_z))
elif "lamp" in category or "light" in category or "свет" in category or "торшер" in category:
    cyl("base", min(sx, sy) * 0.30, sz * 0.04, (0, 0, sz * 0.02))
    cyl("pole", min(sx, sy) * 0.04, sz * 0.72, (0, 0, sz * 0.38))
    cyl("shade", min(sx, sy) * 0.28, sz * 0.20, (0, 0, sz * 0.82), sections=32)
elif "plant" in category or "раст" in category:
    cyl("pot", min(sx, sy) * 0.28, sz * 0.28, (0, 0, sz * 0.14))
    for i in range(7):
        ang = i * math.tau / 7.0
        box("leaf", (math.cos(ang) * sx * 0.12, math.sin(ang) * sy * 0.12, sz * 0.55), (sx * 0.10, sy * 0.035, sz * 0.45))
else:
    box("body", (0, 0, sz * 0.5), (sx, sy, sz))

scene = trimesh.Scene()
for mesh in parts:
    scene.add_geometry(mesh, node_name=mesh.metadata.get("name") or "part")
out_glb.parent.mkdir(parents=True, exist_ok=True)
out_report.parent.mkdir(parents=True, exist_ok=True)
scene.export(out_glb)
report = {
    "ok": True,
    "mode": "proxy_glb_fallback",
    "asset_format": "glb",
    "asset_path": str(out_glb),
    "out_glb": str(out_glb),
    "glb_size_mb": round(out_glb.stat().st_size / 1024 / 1024, 3),
    "proxy_category_text": category,
    "proxy_size_m": [sx, sy, sz],
    "total_sec": 0.0,
    "generated_at": time.time(),
}
out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
'''


def shell_quote(value: Any) -> str:
    s = str(value)
    return "'" + s.replace("'", "'\\''") + "'"


# -----------------------------------------------------------------------------
# Result payload for downstream pipeline
# -----------------------------------------------------------------------------


def build_enriched_result(
    *,
    card: dict[str, Any],
    normalized_card: dict[str, Any],
    local_job_dir: Path,
    local_glb_path: Path,
    remote_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dims = dimensions_cm(card)
    size_m = target_size_m(card)
    unique_key = card.get("unique_key") or stable_hash(card)
    report_mode = str(
        remote_report.get("final_generation_mode")
        or remote_report.get("asset_generation_mode")
        or remote_report.get("mode")
        or remote_report.get("generation_source")
        or ""
    ).strip()
    if report_mode == "proxy_glb_fallback":
        generation_method = "proxy_glb_fallback"
        asset_source = "supplier_catalog_procedural_proxy"
    elif report_mode == "text":
        generation_method = "trellis2_text_to_3d"
        asset_source = "trellis_generated"
    else:
        generation_method = "trellis2_image_to_3d"
        asset_source = "trellis_generated"

    asset_payload = {
        "schema": "trellis_supplier_asset_result/v1",
        "unique_key": unique_key,
        "title": card.get("title"),
        "category_norm": card.get("category_norm"),
        "category_raw": card.get("category_raw"),
        "source_site": card.get("source_site"),
        "product_url": card.get("product_url") or card.get("model_page_url") or card.get("source_url"),
        "asset_status": "trellis_generated_local_asset",
        "asset_format": "glb",
        "asset_local_path": str(local_glb_path.expanduser().resolve()),
        "asset_source": asset_source,
        "asset_generation_method": generation_method,
        "asset_generation_mode": remote_report.get("final_generation_mode") or remote_report.get("asset_generation_mode") or args.mode,
        "trellis_multi_mode": args.multi_mode,
        "trellis_model": args.remote_model_dir,
        "trellis_params": {
            "seed": args.seed,
            "sparse_steps": args.sparse_steps,
            "slat_steps": args.slat_steps,
            "texture_size": args.texture_size,
            "simplify": args.simplify,
            "max_images": args.max_images,
            "image_size": int(getattr(args, "image_size", 336) or 336),
            "fill_holes_resolution": int(getattr(args, "fill_holes_resolution", 256) or 256),
            "fill_holes_num_views": int(getattr(args, "fill_holes_num_views", 120) or 120),
            "pipeline_type": int(getattr(args, "pipeline_type", 512) or 512),
            "decimation_target": int(getattr(args, "decimation_target", 50000) or 50000),
        },
        "dimensions_cm": dims,
        "target_size_m": size_m,
        "orientation": normalized_card.get("orientation"),
        "fit_policy": {
            "mode": "fit_generated_glb_to_scene_item_aabb",
            "note": "TRELLIS does not guarantee metric dimensions; downstream Blender builder must scale this GLB to item.size_m/aabb and apply item.rotation_deg/yaw_deg.",
        },
        "local_job_dir": str(local_job_dir),
        "remote_report": remote_report,
        "normalized_card": normalized_card,
        "raw_card_compact": compact_card(card),
    }
    return asset_payload


def patch_card_with_asset(card: dict[str, Any], result_payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(card)
    out["asset_status"] = result_payload["asset_status"]
    out["asset_format"] = result_payload["asset_format"]
    out["asset_local_path"] = result_payload["asset_local_path"]
    out["asset_source_url"] = None
    out.setdefault("extra", {})
    if isinstance(out["extra"], dict):
        out["extra"]["trellis_generated_asset"] = {
            "asset_local_path": result_payload["asset_local_path"],
            "asset_format": result_payload["asset_format"],
            "asset_status": result_payload["asset_status"],
            "asset_generation_mode": result_payload["asset_generation_mode"],
            "asset_generation_method": result_payload.get("asset_generation_method"),
            "asset_source": result_payload.get("asset_source"),
            "trellis_multi_mode": result_payload["trellis_multi_mode"],
            "target_size_m": result_payload["target_size_m"],
            "orientation": result_payload["orientation"],
            "remote_report_summary": {
                "total_sec": result_payload.get("remote_report", {}).get("total_sec"),
                "generation_sec": result_payload.get("remote_report", {}).get("generation_sec"),
                "nvidia_smi_peak_vram_mib": result_payload.get("remote_report", {}).get("nvidia_smi_peak_vram_mib"),
                "glb_size_mb": result_payload.get("remote_report", {}).get("glb_size_mb"),
            },
        }
    return out


def _remote_report_value(report: dict[str, Any], key: str) -> Any:
    if key in report:
        return report.get(key)
    top_level_aliases = {
        "total_sec": "total_job_sec",
        "glb_export_sec": "export_sec",
        "export_sec": "glb_export_sec",
    }
    alias = top_level_aliases.get(key)
    if alias in report:
        return report.get(alias)
    stages = report.get("stages")
    if isinstance(stages, dict):
        aliases = {
            "generation_sec": "generation_sec",
            "glb_export_sec": "export_sec",
            "export_sec": "export_sec",
            "load_model_sec": "load_model_sec",
        }
        alias = aliases.get(key)
        if alias in stages:
            return stages.get(alias)
    return None


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def build_job_id(card: dict[str, Any], args: argparse.Namespace) -> str:
    base = card.get("unique_key") or card.get("title") or card.get("product_url") or "supplier_item"
    digest = stable_hash({
        "card": compact_card(card),
        "mode": args.mode,
        "multi_mode": args.multi_mode,
        "max_images": args.max_images,
        "image_source_index": args.image_source_index,
        "single_object_crop": args.single_object_crop,
        "single_object_crop_component": args.single_object_crop_component,
        "vlm_single_object_filter": args.vlm_single_object_filter,
        "vlm_model": args.vlm_model,
        "text_fallback": args.text_fallback_if_no_single_image,
        "sparse_steps": args.sparse_steps,
        "slat_steps": args.slat_steps,
        "texture_size": args.texture_size,
        "seed": args.seed,
    })
    return f"{now_stamp()}__{slugify(base, max_len=64)}__{digest}"


def trellis2_generation_cache_key(card: dict[str, Any], args: argparse.Namespace) -> str:
    identity = (
        card.get("unique_key")
        or card.get("product_url")
        or card.get("model_page_url")
        or card.get("source_url")
        or card.get("title")
        or stable_hash(compact_card(card), n=16)
    )
    digest = stable_hash(
        {
            "card": compact_card(card),
            "mode": args.mode,
            "multi_mode": args.multi_mode,
            "max_images": args.max_images,
            "image_source_index": args.image_source_index,
            "single_object_crop": args.single_object_crop,
            "single_object_crop_component": args.single_object_crop_component,
            "vlm_single_object_filter": args.vlm_single_object_filter,
            "vlm_model": args.vlm_model,
            "sparse_steps": args.sparse_steps,
            "slat_steps": args.slat_steps,
            "texture_size": args.texture_size,
            "remote_model_dir": args.remote_model_dir,
        },
        n=16,
    )
    return f"{slugify(str(identity), max_len=72)}__{digest}"


def _summary_asset_path(summary: dict[str, Any]) -> Path | None:
    for key in ("asset_glb", "asset_path"):
        raw = str(summary.get(key) or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.is_file() and p.suffix.lower() in MODEL_ASSET_EXTS:
                return p.resolve()
    return None


def try_reuse_local_generated_asset(local_job_dir: Path) -> dict[str, Any] | None:
    summary_path = local_job_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = read_json(summary_path)
    except Exception:
        return None
    if not isinstance(summary, dict) or summary.get("ok") is not True:
        return None
    asset_path = _summary_asset_path(summary)
    if asset_path is None:
        return None
    card_path_raw = str(summary.get("card_with_trellis_asset_json") or "").strip()
    if card_path_raw and not Path(card_path_raw).expanduser().is_file():
        return None
    summary = dict(summary)
    summary["reused_existing_local_asset"] = True
    summary["asset_glb"] = str(asset_path)
    summary["asset_path"] = str(asset_path)
    return summary


def _finalize_reused_remote_trellis_asset(
    *,
    card: dict[str, Any],
    normalized: dict[str, Any],
    local_job_dir: Path,
    local_glb: Path,
    local_report: Path,
    args: argparse.Namespace,
    job_id: str,
    remote_source_glb: str,
    remote_source_report: str,
    reuse_source: str,
    candidate_cascade_report: dict[str, Any],
) -> dict[str, Any]:
    remote_report = read_json(local_report)
    if not isinstance(remote_report, dict):
        remote_report = {"ok": True}
    remote_report.setdefault("final_generation_mode", "image")
    remote_report.setdefault("asset_generation_mode", remote_report["final_generation_mode"])
    remote_report["reused_remote_trellis2_asset"] = True
    remote_report["remote_reuse_source"] = reuse_source
    remote_report["remote_source_glb"] = remote_source_glb
    remote_report["remote_source_report"] = remote_source_report
    write_json(local_report, remote_report)

    result_payload = build_enriched_result(
        card=card,
        normalized_card=normalized,
        local_job_dir=local_job_dir,
        local_glb_path=local_glb,
        remote_report=remote_report,
        args=args,
    )
    enriched_path = write_json(local_job_dir / "asset.enriched.json", result_payload)
    patched_card = patch_card_with_asset(card, result_payload)
    patched_card_path = write_json(local_job_dir / "card.with_trellis_asset.json", patched_card)

    candidate_cascade_report.setdefault("remote_trellis2_cache", {})
    candidate_cascade_report["remote_trellis2_cache"].update(
        {
            "tried": True,
            "ok": True,
            "reuse_source": reuse_source,
            "remote_source_glb": remote_source_glb,
            "asset_local_path": str(local_glb),
        }
    )

    summary = {
        "ok": True,
        "job_id": job_id,
        "local_job_dir": str(local_job_dir),
        "asset_glb": str(local_glb),
        "asset_path": str(local_glb),
        "final_generation_mode": remote_report.get("final_generation_mode") or "image",
        "asset_generation_mode": remote_report.get("asset_generation_mode") or "image",
        "remote_worker_mode": "reused_remote_cache",
        "remote_worker_root": remote_worker_root(args),
        "remote_reuse_source": reuse_source,
        "remote_source_glb": remote_source_glb,
        "candidate_cascade": candidate_cascade_report,
        "asset_enriched_json": str(enriched_path),
        "card_with_trellis_asset_json": str(patched_card_path),
        "remote_report_json": str(local_report),
        "reused_remote_trellis2_asset": True,
    }
    write_json(local_job_dir / "summary.json", summary)
    return summary


def try_reuse_remote_trellis2_asset(
    *,
    args: argparse.Namespace,
    card: dict[str, Any],
    normalized: dict[str, Any],
    local_job_dir: Path,
    job_id: str,
    remote_job_dir: str,
    candidate_cascade_report: dict[str, Any],
) -> dict[str, Any] | None:
    cache_key = trellis2_generation_cache_key(card, args)
    cache_dir = f"{remote_worker_root(args)}/generated_cache/{cache_key}"
    candidates = [
        ("remote_job_dir", f"{remote_job_dir.rstrip('/')}/output/asset.trellis.glb", f"{remote_job_dir.rstrip('/')}/output/trellis.report.json"),
        ("remote_generation_cache", f"{cache_dir}/asset.trellis.glb", f"{cache_dir}/trellis.report.json"),
    ]
    candidate_cascade_report.setdefault("remote_trellis2_cache", {"tried": True, "ok": False, "cache_key": cache_key})
    for reuse_source, remote_glb, remote_report in candidates:
        try:
            status = ssh_run(
                args,
                "\n".join(
                    [
                        f"if [ -s {shell_quote(remote_glb)} ] && [ -s {shell_quote(remote_report)} ]; then",
                        "  echo done",
                        "else",
                        "  echo miss",
                        "fi",
                    ]
                ),
            ).strip().splitlines()[-1]
        except Exception as exc:
            candidate_cascade_report["remote_trellis2_cache"]["error"] = f"{type(exc).__name__}: {exc}"
            continue
        if status != "done":
            continue

        local_output_dir = local_job_dir / "output"
        local_output_dir.mkdir(parents=True, exist_ok=True)
        local_glb = local_output_dir / "asset.trellis.glb"
        local_report = local_output_dir / "trellis.report.json"
        scp_from_remote(args, remote_glb, local_glb)
        scp_from_remote(args, remote_report, local_report)
        print(
            f"[TRELLIS][remote-cache-hit] job={job_id} source={reuse_source} "
            f"remote_glb={remote_glb}",
            flush=True,
        )
        return _finalize_reused_remote_trellis_asset(
            card=card,
            normalized=normalized,
            local_job_dir=local_job_dir,
            local_glb=local_glb,
            local_report=local_report,
            args=args,
            job_id=job_id,
            remote_source_glb=remote_glb,
            remote_source_report=remote_report,
            reuse_source=reuse_source,
            candidate_cascade_report=candidate_cascade_report,
        )
    return None


def store_remote_trellis2_cache(args: argparse.Namespace, card: dict[str, Any], remote_glb: str, remote_report: str) -> str | None:
    cache_key = trellis2_generation_cache_key(card, args)
    cache_dir = f"{remote_worker_root(args)}/generated_cache/{cache_key}"
    try:
        ssh_run(
            args,
            "\n".join(
                [
                    f"mkdir -p {shell_quote(cache_dir)}",
                    f"cp {shell_quote(remote_glb)} {shell_quote(cache_dir + '/asset.trellis.glb')}",
                    f"cp {shell_quote(remote_report)} {shell_quote(cache_dir + '/trellis.report.json')}",
                ]
            ),
        )
        return cache_dir
    except Exception as exc:
        print(f"[TRELLIS][remote-cache-store-warning] key={cache_key} error={type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _run_orchestration_one(args: argparse.Namespace) -> dict[str, Any]:
    validate_trellis2_only_args(args)

    # patched_final_generation_mode_default
    final_generation_mode = str(locals().get("generation_source") or getattr(args, "mode", None) or "image")
    if final_generation_mode not in {"image", "text", "proxy", "direct", "single_image", "multi_image"}:
        final_generation_mode = "image"

    out_root = Path(args.out_dir).expanduser().resolve()
    prepared_job_dir_raw = str(getattr(args, "prepared_job_dir", "") or "").strip()
    if prepared_job_dir_raw:
        local_job_dir = Path(prepared_job_dir_raw).expanduser().resolve()
        job_id = args.job_id or local_job_dir.name
        card = read_json(local_job_dir / "card.raw.json")
        normalized = read_json(local_job_dir / "card.normalized.json")
        image_manifest = read_json(local_job_dir / "image_manifest.json")
        local_images = [Path(str(p)).expanduser() for p in image_manifest.get("images", []) if Path(str(p)).expanduser().is_file()]
        prepare_summary_path = local_job_dir / "summary.json"
        prepare_summary = read_json(prepare_summary_path) if prepare_summary_path.is_file() else {}
        generation_source = str(prepare_summary.get("generation_source") or ("image" if local_images else "text"))
        candidate_cascade_report = prepare_summary.get("candidate_cascade") if isinstance(prepare_summary, dict) else {}
        if not isinstance(candidate_cascade_report, dict):
            candidate_cascade_report = {}
        if not bool(getattr(args, "prepare_only", False)):
            reused_local = try_reuse_local_generated_asset(local_job_dir)
            if reused_local is not None:
                print(f"[TRELLIS][local-cache-hit] job={job_id} asset={reused_local.get('asset_glb')}", flush=True)
                return reused_local
    else:
        card = load_card(args)
        job_id = args.job_id or build_job_id(card, args)

        local_job_dir = out_root / job_id
        local_job_dir.mkdir(parents=True, exist_ok=True)
        if not bool(getattr(args, "prepare_only", False)):
            reused_local = try_reuse_local_generated_asset(local_job_dir)
            if reused_local is not None:
                print(f"[TRELLIS][local-cache-hit] job={job_id} asset={reused_local.get('asset_glb')}", flush=True)
                return reused_local

        normalized = normalize_card_for_job(card, orientation_yaw_deg=args.orientation_yaw_deg)
        write_json(local_job_dir / "card.raw.json", card)
        write_json(local_job_dir / "card.normalized.json", normalized)

        candidate_cascade_report: dict[str, Any] = {
            "direct_model": {"tried": bool(_collect_direct_asset_sources(card)), "ok": False},
            "image_trellis": {"tried": False, "ok": False},
            "text_trellis": {"tried": False, "ok": False},
        }
        direct_asset_payload = try_resolve_direct_model_asset(card, local_job_dir)
        if direct_asset_payload is not None:
            candidate_cascade_report["direct_model"] = {
                "tried": True,
                "ok": True,
                "asset_format": direct_asset_payload.get("asset_format"),
                "asset_local_path": direct_asset_payload.get("asset_local_path"),
                "source_field": direct_asset_payload.get("source_field"),
                "source_value": direct_asset_payload.get("source_value"),
            }
            asset_payload_path = write_json(local_job_dir / "asset.direct_model.json", direct_asset_payload)
            patched_card = patch_card_with_direct_asset(card, direct_asset_payload)
            patched_card_path = write_json(local_job_dir / "card.with_trellis_asset.json", patched_card)
            summary = {
                "ok": True,
                "job_id": job_id,
                "local_job_dir": str(local_job_dir),
                "asset_glb": direct_asset_payload.get("asset_local_path"),
                "asset_path": direct_asset_payload.get("asset_local_path"),
                "asset_enriched_json": str(asset_payload_path),
                "card_with_trellis_asset_json": str(patched_card_path),
                "remote_report_json": None,
                "final_generation_mode": "direct_model",
                "asset_generation_mode": "direct_model",
                "candidate_cascade": candidate_cascade_report,
                "used_direct_supplier_asset": True,
                "direct_supplier_asset": {
                    "asset_status": direct_asset_payload.get("asset_status"),
                    "asset_format": direct_asset_payload.get("asset_format"),
                    "asset_local_path": direct_asset_payload.get("asset_local_path"),
                    "source_field": direct_asset_payload.get("source_field"),
                    "source_value": direct_asset_payload.get("source_value"),
                },
            }
            write_json(local_job_dir / "summary.json", summary)
            return summary
        direct_attempts_path = local_job_dir / "direct_model_asset" / "direct_model_asset_attempts.json"
        if direct_attempts_path.is_file():
            candidate_cascade_report["direct_model"]["attempts_json"] = str(direct_attempts_path)

        remote_cache_job_dir = f"{args.remote_root.rstrip('/')}/{job_id}"
        reused_remote = try_reuse_remote_trellis2_asset(
            args=args,
            card=card,
            normalized=normalized,
            local_job_dir=local_job_dir,
            job_id=job_id,
            remote_job_dir=remote_cache_job_dir,
            candidate_cascade_report=candidate_cascade_report,
        )
        if reused_remote is not None:
            return reused_remote

        image_sources = candidate_image_sources(card)
        generation_source = "image"
        if image_sources:
            try:
                local_images, image_manifest = prepare_images(
                    card,
                    local_job_dir,
                    max_images=args.max_images,
                    image_source_index=args.image_source_index,
                    single_object_crop=args.single_object_crop,
                    single_object_crop_component=args.single_object_crop_component,
                    single_object_crop_padding=args.single_object_crop_padding,
                    args=args,
                )
                candidate_cascade_report["image_trellis"] = {
                    "tried": True,
                    "ok": False,
                    "accepted_image_count": len(local_images),
                    "source_count": len(image_sources),
                }
            except RuntimeError as exc:
                msg = str(exc)
                if "All product images failed to download/copy" not in msg and "No images found in product card" not in msg:
                    raise
                print(f"[TRELLIS][image-skip] {msg}; product images are required in TRELLIS.2-only mode", file=sys.stderr)
                local_images = []
                images_dir = local_job_dir / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                image_manifest = {
                    "images": [],
                    "count": 0,
                    "trellis_input_dir": str(images_dir),
                    "sources": image_sources,
                    "selected_sources": [],
                    "image_source_index": args.image_source_index,
                    "skipped": True,
                    "skip_reason": msg,
                }
                candidate_cascade_report["image_trellis"] = {
                    "tried": True,
                    "ok": False,
                    "reason": msg,
                }
        else:
            print("[TRELLIS][image-skip] no image sources in product card; product images are required in TRELLIS.2-only mode", file=sys.stderr)
            local_images = []
            images_dir = local_job_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            image_manifest = {
                "images": [],
                "count": 0,
                "trellis_input_dir": str(images_dir),
                "sources": [],
                "selected_sources": [],
                "image_source_index": args.image_source_index,
                "skipped": True,
                "skip_reason": "no_image_sources",
            }
            candidate_cascade_report["image_trellis"] = {
                "tried": False,
                "ok": False,
                "reason": "skipped_no_images",
            }

        if not local_images:
            candidate_cascade_report["text_trellis"] = {
                "tried": False,
                "ok": False,
                "reason": "disabled_legacy_text_backend",
            }
            raise RuntimeError("TRELLIS.2 generation requires product images; legacy text/TRELLIS1 fallback is disabled")
        if generation_source == "image" and args.mode == "single_image" and args.vlm_single_object_filter:
            print("[WARN] --vlm-single-object-filter is intended for multi_image; consider --mode multi_image", file=sys.stderr)
        write_json(local_job_dir / "image_manifest.json", image_manifest)

    if args.prepare_only:
        summary = {
            "ok": True,
            "prepare_only": True,
            "job_id": job_id,
            "local_job_dir": str(local_job_dir),
            "generation_source": generation_source,
            "accepted_image_count": len(local_images),
            "accepted_images": [str(p) for p in local_images],
            "image_manifest_json": str(local_job_dir / "image_manifest.json"),
            "card_normalized_json": str(local_job_dir / "card.normalized.json"),
            "candidate_cascade": candidate_cascade_report,
        }
        write_json(local_job_dir / "summary.json", summary)
        return summary

    # patched_final_generation_mode_before_remote
    final_generation_mode = str(locals().get("generation_source") or getattr(args, "mode", None) or "image")
    if final_generation_mode not in {"image", "text", "proxy", "direct", "single_image", "multi_image"}:
        final_generation_mode = "image"

    remote_job_dir = f"{args.remote_root.rstrip('/')}/{job_id}"
    ssh_run(
        args,
        "\n".join(
            [
                f"rm -rf {shell_quote(remote_job_dir)}",
                f"mkdir -p {shell_quote(remote_job_dir)} {shell_quote(remote_job_dir + '/output')}",
            ]
        ),
    )
    scp_to_remote(args, local_job_dir / "card.normalized.json", f"{remote_job_dir}/card.normalized.json")
    scp_to_remote(args, local_job_dir / "image_manifest.json", f"{remote_job_dir}/image_manifest.json")
    trellis_input_dir = Path(str(image_manifest.get("trellis_input_dir") or (local_job_dir / "images")))
    scp_to_remote(args, trellis_input_dir, f"{remote_job_dir}/images")

    remote_glb = remote_job_dir + "/output/asset.trellis.glb"
    remote_report = remote_job_dir + "/output/trellis.report.json"
    text_prompt = str(
        normalized.get("trellis_text_prompt")
        or normalized.get("prompt_text")
        or build_trellis_text_prompt(normalized)
        or build_text_prompt(normalized)
    )

    if bool(getattr(args, "remote_persistent_worker", True)):
        remote_stdout = run_remote_trellis2_persistent(
            args=args,
            job_id=job_id,
            remote_job_dir=remote_job_dir,
            remote_glb=remote_glb,
            remote_report=remote_report,
            local_job_dir=local_job_dir,
        )
    else:
        try:
            remote_stdout = run_remote_trellis2_single_run(
                args=args,
                job_id=job_id,
                remote_job_dir=remote_job_dir,
                remote_glb=remote_glb,
                remote_report=remote_report,
                local_job_dir=local_job_dir,
            )
        except subprocess.CalledProcessError as exc:
            remote_stdout = exc.output or ""
            (local_job_dir / "remote_stdout.log").write_text(remote_stdout, encoding="utf-8")
            probe_script = "\n".join(
                [
                    f"test -s {shell_quote(remote_glb)}",
                    f"test -s {shell_quote(remote_report)}",
                    "echo '[TRELLIS][recover] remote artifacts present after non-zero remote exit'",
                ]
            )
            try:
                probe_stdout = ssh_run(args, probe_script)
            except Exception:
                raise exc
            remote_stdout = remote_stdout + "\n" + (probe_stdout or "")
            print(
                "[TRELLIS][recover] remote command returned non-zero, "
                "but asset.trellis.glb and trellis.report.json exist; continuing",
                flush=True,
            )

    # patched_extract_final_generation_mode_from_remote_stdout
    final_generation_mode = "image"
    if (
        "[TRELLIS][remote-end] ok=True mode=text" in remote_stdout
        or "TRELLIS_FINAL_GENERATION_MODE=text" in remote_stdout
        or '"mode": "text"' in remote_stdout
        or "'mode': 'text'" in remote_stdout
    ):
        final_generation_mode = "text"
    elif "[TRELLIS][remote-end] ok=True mode=image" in remote_stdout or "TRELLIS_FINAL_GENERATION_MODE=image" in remote_stdout:
        final_generation_mode = "image"
    elif "[TRELLIS][remote-end] ok=False mode=image" in remote_stdout and "trying text fallback" in remote_stdout:
        final_generation_mode = "text_attempted"
    print(f"[TRELLIS][local-mode] final_generation_mode={final_generation_mode}", flush=True)
    (local_job_dir / "remote_stdout.log").write_text(remote_stdout, encoding="utf-8")

    local_output_dir = local_job_dir / "output"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    local_glb = local_output_dir / "asset.trellis.glb"
    local_report = local_output_dir / "trellis.report.json"

    scp_from_remote(args, remote_glb, local_glb)
    scp_from_remote(args, remote_report, local_report)

    remote_report_path = remote_report
    remote_report = read_json(local_report)

    # patched_store_final_generation_mode_in_report
    if isinstance(remote_report, dict):
        reported_mode = str(
            remote_report.get("final_generation_mode")
            or remote_report.get("asset_generation_mode")
            or remote_report.get("mode")
            or ""
        ).strip()
        if reported_mode in {"image", "text"}:
            final_generation_mode = reported_mode
        elif reported_mode == "proxy_glb_fallback":
            final_generation_mode = "procedural_proxy"
        remote_report["final_generation_mode"] = str(locals().get("final_generation_mode") or "image")
        remote_report.setdefault("asset_generation_mode", remote_report["final_generation_mode"])
    remote_cache_dir = None
    if str(remote_report.get("final_generation_mode") or "") != "procedural_proxy":
        remote_cache_dir = store_remote_trellis2_cache(args, card, remote_glb, remote_report_path)
        if remote_cache_dir:
            remote_report["remote_trellis2_cache_dir"] = remote_cache_dir
            write_json(local_report, remote_report)
    if not local_glb.is_file() or local_glb.stat().st_size <= 0:
        raise RuntimeError(f"remote generation finished but local GLB is missing or empty: {local_glb}")
    if remote_report.get("ok") is False:
        raise RuntimeError(f"remote generation report is not ok: {remote_report.get('error') or remote_report}")
    result_payload = build_enriched_result(
        card=card,
        normalized_card=normalized,
        local_job_dir=local_job_dir,
        local_glb_path=local_glb,
        remote_report=remote_report,
        args=args,
    )
    enriched_path = write_json(local_job_dir / "asset.enriched.json", result_payload)

    patched_card = patch_card_with_asset(card, result_payload)
    patched_card_path = write_json(local_job_dir / "card.with_trellis_asset.json", patched_card)

    if isinstance(candidate_cascade_report, dict):
        if final_generation_mode == "image":
            candidate_cascade_report.setdefault("image_trellis", {})
            candidate_cascade_report["image_trellis"].update(
                {"tried": True, "ok": True, "asset_local_path": str(local_glb)}
            )
        elif final_generation_mode == "text":
            if "[TRELLIS][remote-end] ok=False mode=image" in remote_stdout:
                candidate_cascade_report.setdefault("image_trellis", {})
                candidate_cascade_report["image_trellis"].update({"tried": True, "ok": False, "reason": "remote_image_failed"})
            candidate_cascade_report.setdefault("text_trellis", {})
            candidate_cascade_report["text_trellis"].update(
                {"tried": True, "ok": True, "prompt": text_prompt, "asset_local_path": str(local_glb)}
            )

    summary = {
        "ok": True,
        "job_id": job_id,
        "local_job_dir": str(local_job_dir),
        "asset_glb": str(local_glb),
        "final_generation_mode": final_generation_mode,
        "asset_generation_mode": final_generation_mode,
        "remote_worker_mode": "persistent" if bool(getattr(args, "remote_persistent_worker", True)) else "single_run",
        "remote_worker_root": remote_worker_root(args),
        "remote_queue_job_json": (
            (local_job_dir / "trellis2_worker_queue_path.txt").read_text(encoding="utf-8").strip()
            if (local_job_dir / "trellis2_worker_queue_path.txt").is_file()
            else None
        ),
        "remote_trellis2_cache_dir": remote_cache_dir,
        "candidate_cascade": candidate_cascade_report,
        "asset_enriched_json": str(enriched_path),
        "card_with_trellis_asset_json": str(patched_card_path),
        "remote_report_json": str(local_report),
        "remote_report_summary": {
            "total_sec": _remote_report_value(remote_report, "total_sec"),
            "generation_sec": _remote_report_value(remote_report, "generation_sec"),
            "glb_export_sec": _remote_report_value(remote_report, "glb_export_sec"),
            "nvidia_smi_peak_vram_mib": _remote_report_value(remote_report, "nvidia_smi_peak_vram_mib"),
            "torch_peak_allocated_mib": _remote_report_value(remote_report, "torch_peak_allocated_mib"),
            "torch_peak_reserved_mib": _remote_report_value(remote_report, "torch_peak_reserved_mib"),
            "glb_size_mb": _remote_report_value(remote_report, "glb_size_mb"),
        },
    }
    write_json(local_job_dir / "summary.json", summary)
    return summary



# --- patched: runtime candidate fallback wrapper v2 ---
def _trellis_fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes:02d}m{rest:02d}s"


def _trellis_candidate_unique_key_v2(card: dict[str, Any]) -> str:
    if not isinstance(card, dict):
        return "unknown"
    for key in (
        "unique_key",
        "model_download_url",
        "model_page_url",
        "product_url",
        "source_url",
        "title",
    ):
        value = card.get(key)
        if value:
            return str(value)
    return stable_hash(card)


def _trellis_extract_candidate_pool_v2(card: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(card, dict):
        return []

    pool_fields = (
        "supplier_candidate_pool",
        "candidate_pool",
        "top_candidates",
        "candidates",
        "supplier_candidates",
        "alternatives",
    )

    sources: list[dict[str, Any]] = [card]
    for nested_key in ("extra", "meta", "source", "supplier_candidate", "binding"):
        nested = card.get(nested_key)
        if isinstance(nested, dict):
            sources.append(nested)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        cand = raw.get("candidate") if isinstance(raw.get("candidate"), dict) else raw
        if not isinstance(cand, dict):
            return
        uk = _trellis_candidate_unique_key_v2(cand)
        if not uk or uk in seen:
            return
        seen.add(uk)
        out.append(cand)

    # Текущий выбранный кандидат должен идти первым.
    add_candidate(card)

    for src in sources:
        for field in pool_fields:
            val = src.get(field)
            if isinstance(val, list):
                for item in val:
                    add_candidate(item)
            elif isinstance(val, dict):
                # Иногда pool хранится как {"items": [...]} или {"candidates": [...]}.
                for inner_key in ("items", "candidates", "rows", "results"):
                    inner = val.get(inner_key)
                    if isinstance(inner, list):
                        for item in inner:
                            add_candidate(item)

    return out


def _trellis_make_candidate_card_v2(base: dict[str, Any], cand: dict[str, Any], *, candidate_index: int, candidate_total: int) -> dict[str, Any]:
    # Кандидат задаёт supplier/product fields, base сохраняет target/layout metadata.
    out = dict(base)

    pool_snapshot = {
        k: base[k]
        for k in (
            "supplier_candidate_pool",
            "candidate_pool",
            "top_candidates",
            "candidates",
            "supplier_candidates",
        )
        if k in base
    }

    target_snapshot = {
        k: base[k]
        for k in (
            "target_id",
            "supplier_target_id",
            "layout_target_id",
            "target_size_m",
            "orientation_yaw_deg",
            "orientation",
            "placement",
            "position_m",
            "size_m",
            "rotation_deg",
            "yaw_deg",
            "yaw_rad",
            "aabb",
        )
        if k in base
    }

    out.update(cand)
    out.update(target_snapshot)
    out.update(pool_snapshot)

    out["unique_key"] = _trellis_candidate_unique_key_v2(cand)
    out.setdefault("extra", {})
    if isinstance(out["extra"], dict):
        out["extra"]["trellis_candidate_fallback"] = {
            "candidate_index": candidate_index,
            "candidate_total": candidate_total,
            "candidate_unique_key": out["unique_key"],
            "candidate_title": cand.get("title"),
            "candidate_source_site": cand.get("source_site"),
            "candidate_product_url": cand.get("product_url") or cand.get("model_page_url") or cand.get("source_url"),
        }
    return out


def _trellis_blacklist_load_v2(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": "trellis_candidate_blacklist/v2", "failures": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"schema": "trellis_candidate_blacklist/v2", "failures": {}}
        data.setdefault("schema", "trellis_candidate_blacklist/v2")
        data.setdefault("failures", {})
        if not isinstance(data["failures"], dict):
            data["failures"] = {}
        return data
    except Exception:
        return {"schema": "trellis_candidate_blacklist/v2", "failures": {}}


def _trellis_blacklist_key_v2(target_id: str, unique_key: str) -> str:
    return f"{target_id}||{unique_key}"


def _trellis_failure_count_v2(path: Path, target_id: str, unique_key: str) -> int:
    data = _trellis_blacklist_load_v2(path)
    rec = data.get("failures", {}).get(_trellis_blacklist_key_v2(target_id, unique_key), {})
    if isinstance(rec, dict):
        return int(rec.get("count", 0) or 0)
    return 0


def _trellis_mark_failure_v2(path: Path, target_id: str, unique_key: str, error: Exception, *, max_failures: int) -> int:
    data = _trellis_blacklist_load_v2(path)
    failures = data.setdefault("failures", {})
    key = _trellis_blacklist_key_v2(target_id, unique_key)
    rec = failures.get(key)
    if not isinstance(rec, dict):
        rec = {"count": 0, "errors": []}
    rec["count"] = int(rec.get("count", 0) or 0) + 1
    rec["last_error"] = f"{type(error).__name__}: {str(error)[-1000:]}"
    rec.setdefault("errors", [])
    if isinstance(rec["errors"], list):
        rec["errors"].append({
            "ts": time.time(),
            "error": rec["last_error"],
        })
        rec["errors"] = rec["errors"][-10:]
    failures[key] = rec
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "blocked" if rec["count"] >= int(max_failures) else "retry_allowed"
    print(
        f"[TRELLIS][candidate-fail] target={target_id} failures={rec['count']}/{max_failures} "
        f"status={status} unique_key={unique_key} error={rec['last_error']}",
        flush=True,
    )
    return int(rec["count"])


def _trellis_progress_v2(
    *,
    t0: float,
    done_attempts: int,
    total_attempts: int,
    target_id: str,
    candidate_index: int,
    candidate_total: int,
    attempt_index: int,
    max_failures: int,
    status: str,
    unique_key: str,
) -> None:
    elapsed = time.monotonic() - t0
    avg = elapsed / max(1, done_attempts)
    eta = avg * max(0, total_attempts - done_attempts)
    print(
        f"[TRELLIS][progress] done={done_attempts}/{total_attempts} "
        f"elapsed={_trellis_fmt_duration(elapsed)} eta≈{_trellis_fmt_duration(eta)} "
        f"target={target_id} candidate={candidate_index}/{candidate_total} "
        f"attempt={attempt_index}/{max_failures} status={status} unique_key={unique_key}",
        flush=True,
    )


def run_orchestration(args: argparse.Namespace) -> dict[str, Any]:
    validate_trellis2_only_args(args)

    # prepared-job/prepare-only modes are fixed single-card operations; candidate
    # fallback must only report success for full GLB generation.
    if str(getattr(args, "prepared_job_dir", "") or "").strip() or bool(getattr(args, "prepare_only", False)):
        return _run_orchestration_one(args)

    import copy

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    base_card = load_card(args)
    target_id = str(
        base_card.get("target_id")
        or base_card.get("supplier_target_id")
        or base_card.get("layout_target_id")
        or getattr(args, "job_id", "")
        or base_card.get("title")
        or base_card.get("unique_key")
        or "target"
    )

    max_failures = int(
        getattr(args, "max_failures_per_candidate", None)
        or getattr(args, "trellis_max_failures_per_candidate", None)
        or 2
    )
    max_failures = max(1, max_failures)
    max_candidate_pool = int(
        getattr(args, "trellis_max_candidate_pool", None)
        or os.environ.get("TRELLIS_MAX_CANDIDATE_POOL", "0")
        or 0
    )

    blacklist_path = out_root / "trellis_candidate_blacklist.json"
    candidates = _trellis_fallback_candidate_sequence(
        binding=base_card,
        target_id=target_id,
        blacklist_path=blacklist_path,
        max_failures_per_candidate=max_failures,
        max_candidate_pool=max_candidate_pool,
    )
    if not candidates:
        candidates = _trellis_extract_candidate_pool_v2(base_card)
    if not candidates:
        candidates = [base_card]

    candidate_cards_dir = out_root / "_trellis_candidate_cards"
    candidate_cards_dir.mkdir(parents=True, exist_ok=True)

    base_job_id = str(getattr(args, "job_id", "") or build_job_id(base_card, args))
    # retry14 already performs the per-image low-VRAM retry sequence for one candidate.
    # The outer loop is candidate-level: direct -> image -> text is tried once, then we
    # move to the next supplier candidate.
    candidate_attempts_per_candidate = 1
    total_attempts = len(candidates) * candidate_attempts_per_candidate
    done_attempts = 0
    t0 = time.monotonic()
    last_error: Exception | None = None

    print(
        f"[TRELLIS][target-start] target={target_id} candidates={len(candidates)} "
        f"cascade_attempts_per_candidate={candidate_attempts_per_candidate} "
        f"image_runner_attempts_per_candidate={max_failures} out={out_root}",
        flush=True,
    )

    for ci, cand in enumerate(candidates, start=1):
        unique_key = _trellis_candidate_unique_key_v2(cand)
        candidate_card = _trellis_make_candidate_card_v2(
            base_card,
            cand,
            candidate_index=ci,
            candidate_total=len(candidates),
        )

        for ai in range(1, candidate_attempts_per_candidate + 1):
            attempt_started = time.monotonic()
            done_attempts += 1
            _trellis_progress_v2(
                t0=t0,
                done_attempts=done_attempts - 1,
                total_attempts=total_attempts,
                target_id=target_id,
                candidate_index=ci,
                candidate_total=len(candidates),
                attempt_index=ai,
                max_failures=candidate_attempts_per_candidate,
                status="attempt_start",
                unique_key=unique_key,
            )

            card_path = candidate_cards_dir / f"{slugify(target_id, 48)}__cand_{ci:03d}__try_{ai:02d}.json"
            write_json(card_path, candidate_card)

            args2 = copy.copy(args)
            args2.card_json = str(card_path)
            args2.catalog_json = ""
            args2.unique_key = ""
            args2.job_id = f"{base_job_id}__cand{ci:03d}__try{ai:02d}"

            try:
                base_seed = int(getattr(args, "seed", 1) or 1)
                args2.seed = base_seed + (ci - 1) * 100 + (ai - 1)
            except Exception:
                pass

            try:
                summary = _run_orchestration_one(args2)
                if summary.get("prepare_only"):
                    raise RuntimeError("prepare-only result returned during remote TRELLIS attempt")
                local_glb = Path(str(summary.get("asset_glb") or summary.get("asset_path") or "")).expanduser()
                if summary.get("ok") is not True or not local_glb.is_file():
                    raise RuntimeError(
                        "TRELLIS attempt returned without a local GLB: "
                        f"ok={summary.get('ok')!r} asset={local_glb}"
                    )
                dt = time.monotonic() - attempt_started
                elapsed = time.monotonic() - t0
                print(
                    f"[TRELLIS][candidate-success] target={target_id} candidate={ci}/{len(candidates)} "
                    f"attempt={ai}/{candidate_attempts_per_candidate} dt={_trellis_fmt_duration(dt)} "
                    f"elapsed={_trellis_fmt_duration(elapsed)} unique_key={unique_key}",
                    flush=True,
                )
                summary["candidate_fallback"] = {
                    "enabled": True,
                    "target_id": target_id,
                    "selected_candidate_index": ci,
                    "candidate_total": len(candidates),
                    "attempt_index": ai,
                    "max_failures_per_candidate": max_failures,
                    "selected_unique_key": unique_key,
                    "blacklist_json": str(blacklist_path),
                    "elapsed_sec": round(elapsed, 4),
                }
                return summary
            except Exception as e:
                last_error = e
                _cleanup_failed_job_dir(out_root / f"{base_job_id}__cand{ci:03d}__try{ai:02d}")
                dt = time.monotonic() - attempt_started
                print(
                    f"[TRELLIS][candidate-error] target={target_id} candidate={ci}/{len(candidates)} "
                    f"attempt={ai}/{candidate_attempts_per_candidate} dt={_trellis_fmt_duration(dt)} "
                    f"unique_key={unique_key} error={type(e).__name__}: {str(e)[-500:]}",
                    flush=True,
                )
                _trellis_mark_failure_v2(
                    blacklist_path,
                    target_id,
                    unique_key,
                    e,
                    max_failures=candidate_attempts_per_candidate,
                )
                _trellis_progress_v2(
                    t0=t0,
                    done_attempts=done_attempts,
                    total_attempts=total_attempts,
                    target_id=target_id,
                    candidate_index=ci,
                    candidate_total=len(candidates),
                    attempt_index=ai,
                    max_failures=candidate_attempts_per_candidate,
                    status="attempt_failed",
                    unique_key=unique_key,
                )

        print(
            f"[TRELLIS][candidate-next] target={target_id} candidate={ci}/{len(candidates)} "
            f"reason=cascade_failed unique_key={unique_key}",
            flush=True,
        )

    allow_proxy_fallback = bool(
        getattr(args, "allow_proxy_fallback", False)
        or getattr(args, "trellis_allow_proxy_fallback", False)
    )
    print(
        f"[TRELLIS][target-failed] target={target_id} tried_candidates={len(candidates)} "
        f"elapsed={_trellis_fmt_duration(time.monotonic() - t0)} last_error={type(last_error).__name__ if last_error else 'None'} "
        f"fallback={'proxy' if allow_proxy_fallback else 'disabled'}",
        flush=True,
    )

    if not allow_proxy_fallback:
        elapsed = time.monotonic() - t0
        raise RuntimeError(
            f"TRELLIS failed for target={target_id}: all {len(candidates)} candidates exhausted "
            f"after {done_attempts} attempts, elapsed={_trellis_fmt_duration(elapsed)}; proxy fallback is disabled. "
            f"Last error: {type(last_error).__name__ if last_error else 'None'}: {str(last_error)[-1000:] if last_error else ''}"
        )

    try:
        proxy_card = base_card
        if candidates:
            # Keep last candidate metadata in card context for deterministic fallback naming.
            proxy_card = _trellis_make_candidate_card_v2(
                base_card,
                candidates[-1],
                candidate_index=len(candidates),
                candidate_total=len(candidates),
            )
        print(
            f"[TRELLIS][proxy-fallback-start] target={target_id} job={base_job_id} "
            f"candidate_total={len(candidates)}",
            flush=True,
        )
        proxy_summary = _run_local_proxy_fallback(
            local_job_dir=out_root / f"{base_job_id}__proxy",
            card=proxy_card,
            args=args,
        )
        proxy_elapsed = time.monotonic() - t0
        print(
            f"[TRELLIS][target-success] target={target_id} mode=proxy_glb_fallback "
            f"dt={_trellis_fmt_duration(proxy_elapsed)} asset={proxy_summary.get('asset_glb')}",
            flush=True,
        )
        return proxy_summary
    except Exception as proxy_error:
        print(
            f"[TRELLIS][proxy-fallback-failed] target={target_id} error={type(proxy_error).__name__}: {str(proxy_error)[:500]}",
            flush=True,
        )

    elapsed = time.monotonic() - t0
    raise RuntimeError(
        f"TRELLIS failed for target={target_id}: all {len(candidates)} candidates exhausted "
        f"after {done_attempts} attempts, elapsed={_trellis_fmt_duration(elapsed)}. "
        f"Last error: {type(last_error).__name__ if last_error else 'None'}: {str(last_error)[-1000:] if last_error else ''}"
    )
# --- end patched: runtime candidate fallback wrapper v2 ---


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate a TRELLIS GLB asset from a supplier product card using a remote GPU server."
    )

    src = ap.add_argument_group("Product card source")
    src.add_argument("--card-json", default="", help="Path to one product-card JSON object")
    src.add_argument("--catalog-json", default="", help="Path to supplier_catalog_canonical.json")
    src.add_argument("--unique-key", default="", help="unique_key to select from --catalog-json")

    out = ap.add_argument_group("Local output")
    out.add_argument("--out-dir", required=True, help="Local output root on Mac")
    out.add_argument("--job-id", default="", help="Optional deterministic job id")
    out.add_argument("--prepared-job-dir", default="", help="Use an already prepared local job dir and skip image/VLM preparation")
    out.add_argument("--prepare-only", action="store_true", help="Only prepare/filter local images and write manifests; do not run remote TRELLIS")

    ssh = ap.add_argument_group("Remote server")
    ssh.add_argument("--server-host", required=True)
    ssh.add_argument("--server-port", type=int, default=22)
    ssh.add_argument("--server-user", default="root")
    ssh.add_argument("--ssh-key", default="", help="SSH private key path")
    ssh.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    ssh.add_argument("--remote-trellis-root", default=DEFAULT_REMOTE_TRELLIS_ROOT)
    ssh.add_argument("--remote-model-dir", default=DEFAULT_REMOTE_MODEL_DIR)
    ssh.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    ssh.add_argument("--remote-worker-root", default=DEFAULT_REMOTE_WORKER_ROOT)
    ssh.add_argument("--remote-worker-timeout-sec", type=float, default=1800.0)
    ssh.add_argument("--remote-worker-poll-sec", type=float, default=2.0)
    ssh.add_argument("--remote-persistent-worker", action=argparse.BooleanOptionalAction, default=True)
    ssh.add_argument(
        "--remote-text-model-dir",
        default="",
        help="Deprecated. Legacy text-to-3D fallback is disabled for TRELLIS.2-only runs.",
    )
    ssh.add_argument("--remote-cuda-visible-devices", type=int, default=0)

    tr = ap.add_argument_group("TRELLIS generation")
    tr.add_argument("--mode", choices=["single_image", "multi_image"], default="multi_image")
    tr.add_argument("--multi-mode", choices=["stochastic", "multidiffusion"], default="stochastic")
    tr.add_argument("--max-images", type=int, default=4)
    tr.add_argument("--seed", type=int, default=1)
    tr.add_argument("--sparse-steps", type=int, default=4)
    tr.add_argument("--slat-steps", type=int, default=4)
    tr.add_argument("--texture-size", type=int, default=256)
    tr.add_argument("--simplify", type=float, default=0.98)
    tr.add_argument("--pipeline-type", type=int, default=512)
    tr.add_argument("--ss-guidance-strength", type=float, default=7.5)
    tr.add_argument("--slat-guidance-strength", type=float, default=3.0)
    tr.add_argument("--decimation-target", type=int, default=50000)
    tr.add_argument("--pre-export-simplify-target", type=int, default=0)
    tr.add_argument("--no-remesh", action=argparse.BooleanOptionalAction, default=False)
    tr.add_argument("--remesh-band", type=int, default=1)
    tr.add_argument("--remesh-project", type=float, default=0.0)
    tr.add_argument("--no-webp", action=argparse.BooleanOptionalAction, default=True)
    tr.add_argument("--image-size", type=int, default=336, help="Prepared reference image size; must be divisible by DINOv2 patch size 14.")
    tr.add_argument("--fill-holes-resolution", type=int, default=256)
    tr.add_argument("--fill-holes-num-views", type=int, default=120)
    tr.add_argument("--remote-runner-path", default="", help="Deprecated; disabled in TRELLIS.2-only mode.")
    tr.add_argument("--max-failures-per-candidate", type=int, default=2, help="Try one supplier candidate at most N full TRELLIS runs before switching to the next catalog candidate.")
    tr.add_argument("--progress-log", action=argparse.BooleanOptionalAction, default=True, help="Print TRELLIS candidate progress, elapsed time and ETA.")
    tr.add_argument("--allow-proxy-fallback", action="store_true", help="Debug only: emit a simple local proxy GLB if all real/TRELLIS candidates fail.")

    prep = ap.add_argument_group("Local image preprocessing")
    prep.add_argument(
        "--image-source-index",
        type=int,
        default=0,
        help="1-based source image index from the product card; 0 means use the first --max-images sources.",
    )
    prep.add_argument(
        "--single-object-crop",
        action="store_true",
        help="Crop each prepared image to one foreground object before sending it to TRELLIS.",
    )
    prep.add_argument(
        "--single-object-crop-component",
        choices=["largest", "top", "middle", "bottom"],
        default="largest",
        help="Foreground component to keep when --single-object-crop is enabled.",
    )
    prep.add_argument(
        "--single-object-crop-padding",
        type=float,
        default=0.16,
        help="Relative padding added around the detected foreground object crop.",
    )

    vlm = ap.add_argument_group("VLM image suitability filter")
    vlm.add_argument(
        "--vlm-single-object-filter",
        action="store_true",
        help="Ask a VLM whether each prepared image contains exactly one target category object before TRELLIS.",
    )
    vlm.add_argument("--vlm-provider", choices=["ollama", "openai", "openrouter"], default="ollama")
    vlm.add_argument("--vlm-ollama-url", default="http://127.0.0.1:11435")
    vlm.add_argument("--vlm-model", default="llama3.2-vision:11b")
    vlm.add_argument("--vlm-timeout", type=int, default=120)
    vlm.add_argument(
        "--vlm-unload-after-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unload the Ollama VLM model after image filtering so TRELLIS can use GPU VRAM.",
    )
    vlm.add_argument(
        "--text-fallback-if-no-single-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated; legacy text fallback is disabled in TRELLIS.2-only mode.",
    )

    geom = ap.add_argument_group("Downstream geometry metadata")
    geom.add_argument("--orientation-yaw-deg", type=float, default=None)

    return ap

# --- patched: candidate fallback + progress helpers ---
_TRELLIS_CATALOG_CACHE = None


def _trellis_norm_text(v):
    import re
    return re.sub(r"[^a-zа-яё0-9]+", " ", str(v or "").lower()).strip()


def _trellis_candidate_key_safe(cand):
    try:
        return candidate_unique_key(cand)
    except Exception:
        if not isinstance(cand, dict):
            return ""
        return str(
            cand.get("unique_key")
            or cand.get("product_url")
            or cand.get("model_page_url")
            or cand.get("title")
            or ""
        )


def _trellis_target_group_from_target_id(target_id):
    t = _trellis_norm_text(target_id)

    rules = [
        ("nightstand", ["nightstand", "bedside", "тумб"]),
        ("bed", ["bedroom bed", " bed ", "кровать"]),
        ("wardrobe", ["wardrobe", "closet", "шкаф", "гардероб"]),
        ("dresser", ["dresser", "sideboard", "chest", "комод"]),
        ("desk", ["desk", "writing", "рабоч", "стол"]),
        ("chair", ["chair", "стул"]),
        ("floor_lamp", ["floor lamp", "floor_lamp", "торшер"]),
        ("ceiling_light", ["ceiling", "pendant", "люстра", "подвес"]),
        ("wall_light", ["wall light", "sconce", "бра"]),
        ("plant", ["plant", "растен"]),
        ("bench", ["bench", "банкет", "скам"]),
    ]

    padded = f" {t} "
    for group, needles in rules:
        for needle in needles:
            if needle in padded or needle in t:
                return group
    return ""


def _trellis_group_aliases(group):
    aliases = {
        "nightstand": ["nightstand", "bedside", "bedside_table", "side_table", "тумба", "прикроват"],
        "bed": ["bed", "queen_bed", "double_bed", "кровать"],
        "wardrobe": ["wardrobe", "closet", "шкаф", "гардероб"],
        "dresser": ["dresser", "sideboard", "chest", "комод"],
        "desk": ["desk", "writing_table", "work_table", "рабочий стол", "консоль", "console"],
        "chair": ["chair", "dining_chair", "стул"],
        "floor_lamp": ["floor_lamp", "lamp_floor", "floor lamp", "торшер"],
        "ceiling_light": ["ceiling_light", "lamp_ceiling", "pendant_lamp", "ceiling lamp", "pendant", "люстра", "подвес"],
        "wall_light": ["wall_light", "sconce", "бра"],
        "plant": ["plant", "indoor_plant", "растение"],
        "bench": ["bench", "банкетка", "скамья"],
    }
    return aliases.get(group, [group])


def _trellis_candidate_group_text(cand):
    if not isinstance(cand, dict):
        return ""
    parts = [
        cand.get("semantic_group"),
        cand.get("category_norm"),
        cand.get("category_raw"),
        cand.get("title"),
        cand.get("name"),
        cand.get("description"),
    ]
    raw = " ".join(str(x or "") for x in parts).lower()
    return f"{raw} {_trellis_norm_text(raw)}"


def _trellis_candidate_matches_group(cand, group):
    text = _trellis_candidate_group_text(cand)
    if not text:
        return False
    if group == "bed":
        category_norm = _trellis_norm_text(cand.get("category_norm"))
        if category_norm and category_norm not in {"bed", "beds", "queen bed", "double bed"}:
            return False
        strict_text = _trellis_norm_text(
            " ".join(
                str(cand.get(k) or "")
                for k in ("title", "name", "category_norm", "category_raw", "semantic_group", "description")
            )
        )
        padded = f" {strict_text} "
        if "bedside" in padded or "прикроват" in padded:
            return False
        if any(
            bad in strict_text
            for bad in (
                "багет",
                "frame set",
                "стол",
                "table",
                "desk",
                "комод",
                "dresser",
                "tisch",
                "stuhl",
                "st hle",
                "sofa",
                "sessel",
                "gartenm bel",
                "bank",
                "regal",
                "schrank",
                "kommode",
            )
        ):
            return False
        return (
            " кровать" in padded
            or " кровати" in padded
            or " bed " in padded
            or " beds " in padded
            or " queen bed " in padded
            or " double bed " in padded
        )
    for alias in _trellis_group_aliases(group):
        a = _trellis_norm_text(alias)
        if not a:
            continue
        if a in text:
            return True
    return False


def _trellis_has_image_source(cand):
    if not isinstance(cand, dict):
        return False

    for k in ("preview_local_path", "preview_path", "image_path", "image_local_path"):
        v = cand.get(k)
        if isinstance(v, str) and v.strip():
            return True

    for k in ("images", "image_urls", "photos", "preview_images"):
        v = cand.get(k)
        if isinstance(v, list) and len(v) > 0:
            return True

    extra = cand.get("extra")
    if isinstance(extra, dict):
        for k in ("images", "image_urls", "photos", "preview_images"):
            v = extra.get(k)
            if isinstance(v, list) and len(v) > 0:
                return True
        for k in ("preview_local_path", "preview_path", "image_path", "image_local_path"):
            v = extra.get(k)
            if isinstance(v, str) and v.strip():
                return True

    return False


def _trellis_candidate_search_text(cand):
    if not isinstance(cand, dict):
        return ""
    parts = [
        cand.get("unique_key"),
        cand.get("source_site"),
        cand.get("product_url"),
        cand.get("model_page_url"),
        cand.get("source_url"),
        cand.get("brand"),
        cand.get("title"),
    ]
    for key in ("images", "image_urls", "photos", "preview_images"):
        value = cand.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value[:8])
    extra = cand.get("extra")
    if isinstance(extra, dict):
        for key in ("images", "image_urls", "photos", "preview_images"):
            value = extra.get(key)
            if isinstance(value, list):
                parts.extend(str(x) for x in value[:8])
        parts.extend(str(extra.get(k) or "") for k in ("source_site", "product_url", "brand"))
    return _trellis_norm_text(" ".join(str(x or "") for x in parts))


def _trellis_trusted_product_image_score(cand):
    text = _trellis_candidate_search_text(cand)
    if not text:
        return 0.0
    if any(pattern in text for pattern in TRUSTED_PRODUCT_IMAGE_PATTERNS):
        return 240.0
    if any(pattern in text for pattern in SCENE_RENDER_IMAGE_PATTERNS):
        return -60.0
    source = safe_text(cand.get("source_site")).lower() if isinstance(cand, dict) else ""
    # Retail/product catalogs usually have cleaner single-object packshots than
    # 3D model marketplaces rendered as full interior scenes.
    if source and not any(pattern in source for pattern in SCENE_RENDER_IMAGE_PATTERNS):
        return 60.0
    return 0.0


def _trellis_float_or_none(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _trellis_size_m_from_candidate(cand):
    if not isinstance(cand, dict):
        return None
    w = _trellis_float_or_none(cand.get("width_cm") or cand.get("width"))
    d = _trellis_float_or_none(cand.get("depth_cm") or cand.get("depth"))
    h = _trellis_float_or_none(cand.get("height_cm") or cand.get("height"))
    if w and d and h:
        # Если значения похожи на сантиметры.
        if max(w, d, h) > 10:
            return [w / 100.0, d / 100.0, h / 100.0]
        return [w, d, h]
    return None


def _trellis_target_size_from_binding(binding):
    if not isinstance(binding, dict):
        return None

    direct_keys = ("target_size_m", "size_m")
    for k in direct_keys:
        v = binding.get(k)
        if isinstance(v, list) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]

    for k in ("original_generated_item", "target", "target_item", "placement", "item"):
        v = binding.get(k)
        if isinstance(v, dict):
            s = v.get("size_m")
            if isinstance(s, list) and len(s) >= 3:
                return [float(s[0]), float(s[1]), float(s[2])]

    return None


def _trellis_size_score(cand, target_size_m):
    import math
    cs = _trellis_size_m_from_candidate(cand)
    if not cs or not target_size_m:
        return 0.0
    try:
        dist = 0.0
        for a, b in zip(cs[:3], target_size_m[:3]):
            a = max(float(a), 1e-4)
            b = max(float(b), 1e-4)
            dist += abs(math.log(a / b))
        return max(0.0, 30.0 - 10.0 * dist)
    except Exception:
        return 0.0


def _trellis_catalog_paths():
    import os
    paths = []
    env = os.environ.get("TRELLIS_FALLBACK_CATALOG_JSON", "").strip()
    if env:
        paths.append(Path(env).expanduser())

    paths.extend([
        Path("data/sourse/suppliers/supplier_catalog_canonical.json"),
        Path("data/source/suppliers/supplier_catalog_canonical.json"),
    ])

    out = []
    seen = set()
    for path in paths:
        try:
            rp = path.resolve()
        except Exception:
            rp = path
        if str(rp) not in seen and path.exists():
            seen.add(str(rp))
            out.append(path)
    return out


def _trellis_collect_catalog_cards(data):
    cards = []

    if isinstance(data, dict):
        for k in ("items", "rows", "products", "catalog", "data"):
            v = data.get(k)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        cards.append(x)
                if cards:
                    return cards

        # fallback: рекурсивный сбор карточек
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                if (
                    x.get("unique_key")
                    and (x.get("title") or x.get("name") or x.get("product_url") or x.get("model_page_url"))
                    and (x.get("semantic_group") or x.get("category_norm") or x.get("category_raw"))
                ):
                    cards.append(x)
                else:
                    stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)

    elif isinstance(data, list):
        cards = [x for x in data if isinstance(x, dict)]

    return cards


def _trellis_load_catalog_cards():
    global _TRELLIS_CATALOG_CACHE
    if _TRELLIS_CATALOG_CACHE is not None:
        return _TRELLIS_CATALOG_CACHE

    import json
    loaded = []
    used_path = None

    for path in _trellis_catalog_paths():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded = _trellis_collect_catalog_cards(data)
            used_path = str(path)
            if loaded:
                break
        except Exception as e:
            print(f"[TRELLIS][catalog-warning] path={path} error={type(e).__name__}: {e}", flush=True)

    _TRELLIS_CATALOG_CACHE = {
        "path": used_path,
        "cards": loaded,
    }
    print(f"[TRELLIS][catalog-load] path={used_path} cards={len(loaded)}", flush=True)
    return _TRELLIS_CATALOG_CACHE


def _trellis_append_catalog_alternatives(binding, target_id, pool, seen, max_catalog_candidates=30, require_images=False):
    group = _trellis_target_group_from_target_id(target_id)
    if not group and pool:
        # Пытаемся вывести группу из текущего top-1 кандидата.
        text = _trellis_candidate_group_text(pool[0])
        for g in ("nightstand", "bed", "wardrobe", "dresser", "desk", "chair", "floor_lamp", "ceiling_light", "wall_light", "plant", "bench"):
            if any(_trellis_norm_text(a) in text for a in _trellis_group_aliases(g)):
                group = g
                break

    if not group:
        print(f"[TRELLIS][catalog-fallback] target={target_id} skipped reason=no_group", flush=True)
        return pool

    target_size_m = _trellis_target_size_from_binding(binding)
    catalog = _trellis_load_catalog_cards()
    cards = catalog.get("cards") or []

    scored = []
    for cand in cards:
        if not isinstance(cand, dict):
            continue
        uk = _trellis_candidate_key_safe(cand)
        if not uk or uk in seen:
            continue
        if not _trellis_candidate_matches_group(cand, group):
            continue
        has_image = _trellis_has_image_source(cand)
        if require_images and not has_image:
            continue

        score = 0.0
        if has_image:
            score += 100.0
            score += _trellis_trusted_product_image_score(cand)
        else:
            # Без фото такой кандидат почти бесполезен для TRELLIS image-to-3D.
            score -= 100.0

        score += _trellis_size_score(cand, target_size_m)

        # Небольшой бонус за нормальную карточку.
        if cand.get("title") or cand.get("name"):
            score += 5.0
        if cand.get("product_url") or cand.get("model_page_url"):
            score += 2.0

        scored.append((score, cand))

    scored.sort(key=lambda x: x[0], reverse=True)

    appended = 0
    for score, cand in scored:
        if appended >= int(max_catalog_candidates):
            break
        uk = _trellis_candidate_key_safe(cand)
        if not uk or uk in seen:
            continue
        seen.add(uk)
        pool.append(cand)
        appended += 1

    print(
        f"[TRELLIS][catalog-fallback] target={target_id} group={group} "
        f"base_candidates={len(pool) - appended} appended={appended} "
        f"require_images={bool(require_images)} catalog_path={catalog.get('path')}",
        flush=True,
    )
    return pool


def _trellis_fallback_candidate_sequence(
    binding: dict[str, Any],
    target_id: str,
    blacklist_path: Path,
    max_failures_per_candidate: int = 2,
    max_candidate_pool: int = 0,
) -> list[dict[str, Any]]:
    """
    Возвращает список кандидатов для target_id:
    1) кандидаты из binding;
    2) если их мало, альтернативы из supplier_catalog_canonical.json;
    3) исключает кандидатов, которые уже заблокированы после N падений.
    """
    import os

    bl = TrellisCandidateBlacklist(blacklist_path)

    raw_pool = extract_candidate_pool(binding if isinstance(binding, dict) else {})
    pool = []
    seen = set()
    original_order: dict[str, int] = {}

    for cand in raw_pool:
        if not isinstance(cand, dict):
            continue
        uk = _trellis_candidate_key_safe(cand)
        if not uk or uk in seen:
            continue
        seen.add(uk)
        original_order[uk] = len(original_order)
        pool.append(cand)

    def _candidate_has_image_priority(c):  # pragma: no cover - tiny helper
        try:
            return 1 if _trellis_has_image_source(c) else 0
        except Exception:
            return 0

    def _candidate_trusted_image_priority(c):  # pragma: no cover - tiny helper
        try:
            if not _trellis_has_image_source(c):
                return -1000.0
            return float(_trellis_trusted_product_image_score(c))
        except Exception:
            return 0.0

    def _candidate_direct_asset_priority(c):  # pragma: no cover - tiny helper
        try:
            return 1 if _collect_direct_asset_sources(c) else 0
        except Exception:
            return 0

    def _candidate_order(c):  # pragma: no cover - tiny helper
        uk = _trellis_candidate_key_safe(c)
        return original_order.get(uk, 10_000_000)

    def _sort_pool(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Image-rich candidates must dominate the TRELLIS pool because TRELLIS.2 is
        # image-only here. Direct/local asset candidates stay ahead of plain no-image cards.
        return sorted(
            items,
            key=lambda c: (
                _candidate_direct_asset_priority(c),
                _candidate_trusted_image_priority(c),
                _candidate_has_image_priority(c),
                -_candidate_order(c),
            ),
            reverse=True,
        )

    max_catalog = int(os.environ.get("TRELLIS_FALLBACK_MAX_CATALOG_CANDIDATES", "30") or "30")
    effective_pool_size = int(max_candidate_pool) if int(max_candidate_pool or 0) > 0 else max(len(pool), 1)
    min_image_candidates = max(1, (effective_pool_size + 1) // 2)
    image_count = sum(_candidate_has_image_priority(c) for c in pool)
    trusted_image_count = sum(
        1
        for c in pool
        if _candidate_has_image_priority(c) and _candidate_trusted_image_priority(c) > 0
    )
    if len(pool) <= 1 or image_count < min_image_candidates or trusted_image_count < min_image_candidates:
        pool = _trellis_append_catalog_alternatives(
            binding=binding if isinstance(binding, dict) else {},
            target_id=target_id,
            pool=pool,
            seen=seen,
            max_catalog_candidates=max_catalog,
            require_images=True,
        )
        for cand in pool:
            uk = _trellis_candidate_key_safe(cand)
            if uk and uk not in original_order:
                original_order[uk] = len(original_order)

    pool = _sort_pool(pool)

    raw_pool = list(pool)

    selected = []
    for cand in pool:
        if not isinstance(cand, dict):
            continue
        uk = _trellis_candidate_key_safe(cand)
        if not uk:
            continue
        failures = bl.failures(str(target_id), uk)
        if failures >= int(max_failures_per_candidate):
            print(
                f"[TRELLIS][candidate-skip] target={target_id} unique_key={uk} "
                f"reason=blacklisted failures={failures}",
                flush=True,
            )
            continue
        selected.append(cand)

    if max_candidate_pool > 0:
        selected = selected[:max_candidate_pool]

    selected_image_count = sum(_candidate_has_image_priority(cand) for cand in selected)
    selected_trusted_image_count = sum(
        1
        for cand in selected
        if _candidate_has_image_priority(cand) and _candidate_trusted_image_priority(cand) > 0
    )
    print(
        f"[TRELLIS][candidate-pool] target={target_id} "
        f"raw={len(raw_pool)} after_catalog={len(pool)} usable={len(selected)} "
        f"with_images={selected_image_count} trusted_images={selected_trusted_image_count}",
        flush=True,
    )
    return selected


def _trellis_error_is_nonretryable(error: object) -> bool:
    """
    Ошибки, которые не имеет смысла повторять для того же supplier candidate.
    OOM сюда не входит: OOM может зависеть от size/image/seed.
    """
    text = str(error or "").lower()
    patterns = [
        "no images found in product card",
        "expected images[] or preview_local_path",
        "preview_local_path",
        "no preview",
        "no supported model",
        "failed_no_supported_model",
        "unsupported method",
        "unsupported source",
        "unsupported asset",
        "only .max",
        "only max",
        ".max",
        "3ds max",
        "empty image",
        "image file not found",
        "vlm rejected all images",
        "requires product images",
        "legacy text/trellis1 fallback is disabled",
    ]
    return any(p in text for p in patterns)

def _trellis_mark_candidate_failure(blacklist_path, target_id, unique_key, error, max_failures_per_candidate=2):
    bl = TrellisCandidateBlacklist(blacklist_path)

    try:
        max_failures = max(1, int(max_failures_per_candidate))
    except Exception:
        max_failures = 2

    hard_block = _trellis_error_is_nonretryable(error)

    # Для non-retryable ошибок сразу добиваем счётчик до max_failures,
    # чтобы внешний цикл перешёл к следующему кандидату без второй попытки.
    repeat = max_failures if hard_block else 1
    n = 0
    for _ in range(repeat):
        n = bl.add_failure(
            str(target_id),
            str(unique_key),
            error=str(error),
            max_failures=max_failures,
        )

    if hard_block:
        n = max_failures

    status = "blocked" if n >= max_failures else "retry_allowed"
    reason = "nonretryable" if hard_block else "retryable"

    print(
        f"[TRELLIS][candidate-fail] target={target_id} failures={n}/{max_failures} "
        f"status={status} reason={reason} unique_key={unique_key} error={str(error)[-300:]}",
        flush=True,
    )
    return n



def _trellis_progress_line(progress, *, target_id=None, status=None, candidate_index=None, candidate_total=None, unique_key=None, success=False, failed=False, skipped=False):
    try:
        progress.update(
            success_delta=1 if success else 0,
            failed_delta=1 if failed else 0,
            skipped_delta=1 if skipped else 0,
            target_id=target_id,
            status=status,
            candidate_index=candidate_index,
            candidate_total=candidate_total,
            unique_key=unique_key,
        )
    except Exception as e:
        print(f"[TRELLIS][progress-warning] {type(e).__name__}: {e}", flush=True)
# --- end patched helpers ---

def main() -> None:
    args = build_cli().parse_args()
    summary = run_orchestration(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
