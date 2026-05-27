#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create lightweight GLB candidates from supplier catalog product images.

The tool is deliberately file-based:
  card/images -> TRELLIS.2 persistent worker -> GLB -> renders -> quality JSON.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - handled at runtime
    Image = None  # type: ignore

try:
    import requests
except Exception:  # pragma: no cover - optional until network calls
    requests = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_ASSET_EXTS = {
    ".glb",
    ".gltf",
    ".fbx",
    ".obj",
    ".zip",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".7z",
    ".rar",
}
ASSET_FIELDS = [
    "asset_local_path",
    "local_asset_path",
    "model_download_url",
    "model_file_url",
    "glb_path",
    "fbx_path",
    "obj_path",
    "mesh_path",
    "downloaded_path",
]

DEFAULT_REMOTE_HOST = "84.2.13.196"
DEFAULT_REMOTE_PORT = 28553
DEFAULT_REMOTE_ROOT = "/workspace/trellis2_supplier_jobs"
DEFAULT_REMOTE_TRELLIS_ROOT = "/workspace/TRELLIS.2"
DEFAULT_REMOTE_MODEL_DIR = "/workspace/models/TRELLIS.2-4B"
DEFAULT_REMOTE_PYTHON = "/venv/trellis2/bin/python"
DEFAULT_REMOTE_WORKER_ROOT = "/workspace/trellis2_worker"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(safe_text(v) for v in value if safe_text(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def slugify(value: Any, max_len: int = 64) -> str:
    s = safe_text(value).lower()
    s = re.sub(r"[^a-z0-9а-яё._-]+", "_", s, flags=re.IGNORECASE)
    s = re.sub(r"_+", "_", s).strip("._-")
    s = s or "item"
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip("._-") or s[:max_len]
    return s


def stable_hash(value: Any, n: int = 12) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:n]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def extract_cards(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "products", "cards", "catalog", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def card_unique_key(card: dict[str, Any]) -> str:
    for key in ("unique_key", "id", "product_id", "url", "product_url", "title"):
        v = safe_text(card.get(key))
        if v:
            if key == "title":
                return f"title::{stable_hash(v)}::{slugify(v, 40)}"
            return v
    return f"card::{stable_hash(card)}"


def _suffix_for_asset(value: str) -> str:
    lower = value.lower().split("?", 1)[0].split("#", 1)[0]
    for suffix in (".tar.gz",):
        if lower.endswith(suffix):
            return suffix
    return Path(urllib.parse.urlparse(lower).path).suffix.lower()


def has_existing_asset(card: dict[str, Any]) -> tuple[bool, str, str]:
    for field in ASSET_FIELDS:
        value = safe_text(card.get(field))
        if not value:
            continue
        suffix = _suffix_for_asset(value)
        if suffix not in MODEL_ASSET_EXTS:
            continue
        if field.endswith("_path") or field in {"asset_local_path", "downloaded_path", "mesh_path"}:
            p = Path(value).expanduser()
            if p.is_file():
                return True, "has_local_model_asset", field
            if suffix in MODEL_ASSET_EXTS and value.startswith(("http://", "https://")):
                return True, "has_direct_model_asset", field
        else:
            return True, "has_direct_model_asset", field
    if bool(card.get("has_downloadable_asset")):
        return True, "has_direct_model_asset", "has_downloadable_asset"
    return False, "", ""


def collect_cards_without_assets(catalog: Any) -> list[dict[str, Any]]:
    out = []
    for card in extract_cards(catalog):
        has_asset, _, _ = has_existing_asset(card)
        if not has_asset:
            out.append(card)
    return out


def _iter_image_values(value: Any) -> list[str]:
    vals: list[str] = []
    if not value:
        return vals
    if isinstance(value, str):
        vals.append(value)
    elif isinstance(value, dict):
        for key in ("url", "src", "path", "local_path", "image", "preview", "thumbnail"):
            vals.extend(_iter_image_values(value.get(key)))
    elif isinstance(value, list):
        for item in value:
            vals.extend(_iter_image_values(item))
    return vals


def candidate_image_sources(card: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for key in ("preview_local_path", "image_local_path", "thumbnail_local_path", "preview_url", "image_url", "thumbnail_url"):
        sources.extend(_iter_image_values(card.get(key)))
    sources.extend(_iter_image_values(card.get("images")))
    extra = card.get("extra") if isinstance(card.get("extra"), dict) else {}
    sources.extend(_iter_image_values(extra.get("images")))
    sources.extend(_iter_image_values(extra.get("preview_images")))
    color = card.get("image_color_features") if isinstance(card.get("image_color_features"), dict) else {}
    src = color.get("source_image") if isinstance(color.get("source_image"), dict) else {}
    sources.extend(_iter_image_values(src.get("path")))

    seen = set()
    uniq = []
    for src in sources:
        s = safe_text(src)
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _image_ext_from_source(source: str, content_type: str = "") -> str:
    path = urllib.parse.urlparse(source).path
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return ext
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
    return guessed if guessed in IMAGE_EXTS else ".jpg"


def _copy_or_download_image(source: str, out_path_no_ext: Path, referer: str = "") -> Path:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https"}:
        if requests is None:
            raise RuntimeError("requests is required for image downloads")
        headers = {"User-Agent": "Mozilla/5.0"}
        if referer:
            headers["Referer"] = referer
        response = requests.get(source, headers=headers, timeout=90)
        response.raise_for_status()
        ext = _image_ext_from_source(source, response.headers.get("content-type", ""))
        dst = out_path_no_ext.with_suffix(ext)
        dst.write_bytes(response.content)
        return dst
    src = Path(source).expanduser()
    if not src.is_file():
        raise FileNotFoundError(source)
    ext = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".jpg"
    dst = out_path_no_ext.with_suffix(ext)
    shutil.copy2(src, dst)
    return dst


def prepare_card_images(card: dict[str, Any], job_dir: Path, max_images: int) -> dict[str, Any]:
    all_dir = job_dir / "images" / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    sources = candidate_image_sources(card)
    prepared = []
    referer = safe_text(card.get("product_url") or card.get("source_url"))
    for idx, src in enumerate(sources[: max(1, max_images)], 1):
        try:
            dst = _copy_or_download_image(src, all_dir / f"image_{idx:03d}", referer=referer)
            prepared.append({"source": src, "path": str(dst.resolve()), "ok": True})
        except Exception as exc:
            prepared.append({"source": src, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    manifest = {"sources": sources, "prepared": prepared, "images": [x["path"] for x in prepared if x.get("ok")]}
    write_json(job_dir / "images" / "image_prepare_manifest.json", manifest)
    return manifest


def _image_resolution(path: Path) -> tuple[int, int]:
    if Image is None:
        return 0, 0
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


def white_border_metrics(path: Path, border_ratio: float = 0.08) -> dict[str, Any]:
    if Image is None:
        return {"ok": False, "error": "PIL unavailable", "white_border_ratio": 0.0}
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            bw = max(1, int(round(min(w, h) * border_ratio)))
            pixels = rgb.load()
            total = 0
            white = 0
            for y in range(h):
                for x in range(w):
                    if not (x < bw or x >= w - bw or y < bw or y >= h - bw):
                        continue
                    r, g, b = pixels[x, y]
                    total += 1
                    if r >= 238 and g >= 238 and b >= 238 and max(r, g, b) - min(r, g, b) <= 14:
                        white += 1
            ratio = white / max(1, total)
            return {"ok": True, "width": w, "height": h, "border_px": bw, "white_border_ratio": round(ratio, 4)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "white_border_ratio": 0.0}


def select_image_without_vlm(images: list[Path], card: dict[str, Any], job_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(getattr(args, "image_selection_mode", "vlm") or "vlm")
    scores: list[dict[str, Any]] = []
    threshold = float(getattr(args, "white_border_threshold", 0.72) or 0.72)
    for idx, image in enumerate(images, 1):
        score = heuristic_image_score(image, card)
        metrics = white_border_metrics(image)
        score["image_path"] = str(image)
        score["selection_mode"] = mode
        score["white_border_metrics"] = metrics
        score["white_border_ratio"] = metrics.get("white_border_ratio", 0.0)
        score["overall_score"] = 8.0 if float(metrics.get("white_border_ratio") or 0.0) >= threshold else 5.0
        score["reason"] = "white_border_selected" if score["overall_score"] >= 8.0 else "fallback_candidate"
        score["image_rank"] = idx
        scores.append(score)
    write_json(job_dir / "image_scores.json", scores)

    selected_score = None
    if mode == "white-border-first":
        selected_score = next((s for s in scores if float(s.get("white_border_ratio") or 0.0) >= threshold), None)
    if selected_score is None and scores:
        selected_score = scores[0]
    if selected_score is None:
        payload = {"selected": False, "reason": "no_images", "all_scores_path": str(job_dir / "image_scores.json")}
        write_json(job_dir / "selected_image.json", payload)
        return scores, payload

    src = Path(selected_score["image_path"]).expanduser()
    selected_dir = job_dir / "images" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    dst = selected_dir / f"source_selected{src.suffix.lower() if src.suffix else '.png'}"
    shutil.copy2(src, dst)
    payload = {
        "selected": True,
        "selected_image_path": str(dst.resolve()),
        "selected_original_source": str(src.resolve()),
        "score": selected_score,
        "all_scores_path": str((job_dir / "image_scores.json").resolve()),
        "selection_mode": mode,
    }
    write_json(job_dir / "selected_image.json", payload)
    return scores, payload


def _extract_json_object(text: str) -> dict[str, Any]:
    text = safe_text(text)
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        payload = json.loads(m.group(0))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_ollama_vlm(prompt: str, image_paths: list[Path], model: str, base_url: str, timeout: int) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for Ollama calls")
    response = requests.post(
        base_url.rstrip("/") + "/api/chat",
        json={
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt, "images": [image_to_base64(p) for p in image_paths]}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    return {"content": safe_text(message.get("content") or payload.get("response")), "raw": payload}


def _openai_endpoint_and_key(provider: str, base_url: str) -> tuple[str, str]:
    if provider == "openai":
        return (base_url.rstrip("/") + "/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"), os.environ.get("OPENAI_API_KEY", "")
    return (base_url.rstrip("/") + "/chat/completions" if base_url else "https://openrouter.ai/api/v1/chat/completions"), os.environ.get("OPENROUTER_API_KEY", "")


def call_openai_compatible_vlm(
    prompt: str,
    image_paths: list[Path],
    provider: str,
    model: str,
    base_url: str,
    timeout: int,
) -> dict[str, Any]:
    endpoint, key = _openai_endpoint_and_key(provider, base_url)
    if not key:
        raise RuntimeError(f"Missing API key for {provider}")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_to_base64(p)}"}})
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content_text = safe_text(payload.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return {"content": content_text, "raw": payload}


def call_ollama_llm(prompt: str, model: str, base_url: str, timeout: int) -> dict[str, Any]:
    return call_ollama_vlm(prompt, [], model, base_url, timeout)


def call_openai_compatible_llm(prompt: str, provider: str, model: str, base_url: str, timeout: int) -> dict[str, Any]:
    endpoint, key = _openai_endpoint_and_key(provider, base_url)
    if not key:
        raise RuntimeError(f"Missing API key for {provider}")
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content_text = safe_text(payload.get("choices", [{}])[0].get("message", {}).get("content", ""))
    return {"content": content_text, "raw": payload}


def image_scoring_prompt(card: dict[str, Any], image_path: Path) -> str:
    return f"""
Ты оцениваешь supplier-фото для генерации 3D-модели одного предмета.
Лучше всего подходят 3/4 ракурсы, где видны глубина, боковая грань, ножки, спинка, объём.
Плохо подходят коллажи, несколько вариантов, интерьерные сцены, плоский фронтальный вид, детали материала.
Товар: {safe_text(card.get('title'))}
Категория: {safe_text(card.get('category_norm') or card.get('category') or card.get('semantic_group'))}
Файл: {image_path.name}
Верни только JSON:
{{
  "image_path": "{str(image_path)}",
  "is_single_object": true,
  "has_collage": false,
  "has_multiple_variants": false,
  "object_visible": true,
  "view_angle": "three_quarter",
  "structure_visibility_score": 0,
  "single_object_score": 0,
  "angle_score": 0,
  "occlusion_score": 0,
  "resolution_score": 0,
  "background_score": 0,
  "overall_score": 0,
  "reason": "",
  "visible_parts": [],
  "risks_for_3d_generation": []
}}
""".strip()


def heuristic_image_score(image_path: Path, card: dict[str, Any]) -> dict[str, Any]:
    w, h = _image_resolution(image_path)
    pixels = w * h
    name = image_path.name.lower()
    bad_terms = ["detail", "texture", "schema", "drawing", "plan", "схем", "черт"]
    penalty = 2.5 if any(t in name for t in bad_terms) else 0.0
    res_score = 10 if pixels >= 900_000 else 8 if pixels >= 350_000 else 5 if pixels >= 120_000 else 2
    angle_score = 7.0
    if any(t in name for t in ["front", "фасад"]):
        angle_score = 5.0
    if any(t in name for t in ["persp", "angle", "3-4", "34"]):
        angle_score = 8.5
    overall = max(0.0, min(10.0, (res_score * 0.45 + angle_score * 0.35 + 8.0 * 0.20) - penalty))
    return {
        "image_path": str(image_path),
        "is_single_object": True,
        "has_collage": False,
        "has_multiple_variants": False,
        "object_visible": True,
        "view_angle": "unknown",
        "structure_visibility_score": round(angle_score, 2),
        "single_object_score": 8,
        "angle_score": round(angle_score, 2),
        "occlusion_score": 8,
        "resolution_score": res_score,
        "background_score": 7,
        "overall_score": round(overall, 2),
        "reason": "heuristic_no_vlm",
        "visible_parts": [],
        "risks_for_3d_generation": ["VLM disabled; score is heuristic"],
        "evaluation_source": "heuristic_no_vlm",
        "resolution": [w, h],
        "card_title": safe_text(card.get("title")),
    }


def normalize_vlm_image_score(parsed: dict[str, Any], image_path: Path, card: dict[str, Any]) -> dict[str, Any]:
    """Make VLM scores usable even when a model copies numeric zeros from schema.

    Some local VLMs reliably answer booleans/angle but leave all requested scores
    at 0. Treat that as an incomplete numeric answer, not as a rejection.
    """
    parsed = dict(parsed)
    vlm_image_path = safe_text(parsed.get("image_path"))
    parsed["image_path"] = str(image_path)
    if vlm_image_path and vlm_image_path != str(image_path):
        parsed["vlm_returned_image_path"] = vlm_image_path
    numeric_keys = [
        "structure_visibility_score",
        "single_object_score",
        "angle_score",
        "occlusion_score",
        "resolution_score",
        "background_score",
        "overall_score",
    ]
    vals = []
    for key in numeric_keys:
        try:
            vals.append(float(parsed.get(key) or 0))
        except Exception:
            vals.append(0.0)
    if max(vals or [0.0]) > 0.0:
        return parsed

    if parsed.get("object_visible") is False or parsed.get("is_single_object") is False:
        parsed["overall_score"] = 0
        parsed["score_postprocess"] = "vlm_boolean_rejection"
        return parsed

    base = heuristic_image_score(image_path, card)
    angle = safe_text(parsed.get("view_angle")).lower()
    if angle == "three_quarter":
        base["angle_score"] = 9
        base["structure_visibility_score"] = 8.5
    elif angle in {"front", "side"}:
        base["angle_score"] = 5.5
        base["structure_visibility_score"] = 6
    elif angle in {"detail", "top"}:
        base["angle_score"] = 3
        base["structure_visibility_score"] = 3.5
    if parsed.get("has_collage") or parsed.get("has_multiple_variants"):
        base["overall_score"] = min(float(base["overall_score"]), 2.5)
    else:
        base["single_object_score"] = 9 if parsed.get("is_single_object", True) else 2
        base["overall_score"] = round(
            min(
                10.0,
                float(base["overall_score"]) * 0.55
                + float(base["structure_visibility_score"]) * 0.25
                + float(base["single_object_score"]) * 0.20,
            ),
            2,
        )
    for key, value in base.items():
        parsed.setdefault(key, value)
    for key in numeric_keys:
        parsed[key] = base.get(key, parsed.get(key, 0))
    parsed["score_postprocess"] = "filled_zero_numeric_scores_from_vlm_flags_and_resolution"
    return parsed


def score_image_with_vlm(image_path: Path, card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.vlm_provider == "none":
        return heuristic_image_score(image_path, card)
    prompt = image_scoring_prompt(card, image_path)
    try:
        if args.vlm_provider == "ollama":
            result = call_ollama_vlm(prompt, [image_path], args.vlm_model, args.vlm_base_url, int(args.vlm_timeout_sec))
        else:
            result = call_openai_compatible_vlm(prompt, [image_path], args.vlm_provider, args.vlm_model, args.vlm_base_url, int(args.vlm_timeout_sec))
        parsed = _extract_json_object(result.get("content", ""))
        if not parsed:
            raise RuntimeError("VLM returned no JSON object")
        parsed = normalize_vlm_image_score(parsed, image_path, card)
        parsed["provider"] = args.vlm_provider
        parsed["model"] = args.vlm_model
        return parsed
    except Exception as exc:
        score = heuristic_image_score(image_path, card)
        score["vlm_error"] = f"{type(exc).__name__}: {exc}"
        return score


def score_images(images: list[Path], card: dict[str, Any], job_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    scores = [score_image_with_vlm(p, card, args) for p in images]
    write_json(job_dir / "image_scores.json", scores)
    return scores


def is_suitable_image_score(score: dict[str, Any], threshold: float = 6.0) -> bool:
    try:
        overall = float(score.get("overall_score") or 0)
    except Exception:
        overall = 0.0
    return (
        bool(score.get("object_visible", True))
        and bool(score.get("is_single_object", True))
        and not bool(score.get("has_collage", False))
        and not bool(score.get("has_multiple_variants", False))
        and overall >= threshold
    )


def select_best_image(scores: list[dict[str, Any]], job_dir: Path) -> dict[str, Any]:
    viable = [
        s
        for s in scores
        if bool(s.get("object_visible", True))
        and not bool(s.get("has_collage", False))
        and not bool(s.get("has_multiple_variants", False))
        and float(s.get("overall_score") or 0) >= 3.0
    ]
    if not viable:
        payload = {"selected": False, "reason": "no_suitable_image", "all_scores_path": str(job_dir / "image_scores.json")}
        write_json(job_dir / "selected_image.json", payload)
        return payload
    best = max(viable, key=lambda s: float(s.get("overall_score") or 0))
    src = Path(best["image_path"]).expanduser()
    selected_dir = job_dir / "images" / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    dst = selected_dir / f"source_selected{src.suffix.lower() if src.suffix else '.png'}"
    shutil.copy2(src, dst)
    payload = {
        "selected": True,
        "selected_image_path": str(dst.resolve()),
        "selected_original_source": str(src.resolve()),
        "score": best,
        "all_scores_path": str((job_dir / "image_scores.json").resolve()),
    }
    write_json(job_dir / "selected_image.json", payload)
    return payload


def score_images_until_selected(images: list[Path], card: dict[str, Any], job_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if str(getattr(args, "image_selection_mode", "vlm") or "vlm") != "vlm":
        return select_image_without_vlm(images, card, job_dir, args)
    scores_path = job_dir / "image_scores.json"
    selected_path = job_dir / "selected_image.json"
    scores: list[dict[str, Any]] = []
    if args.resume and scores_path.is_file():
        loaded = read_json(scores_path)
        if isinstance(loaded, list):
            scores = [x for x in loaded if isinstance(x, dict)]
            selected = select_best_image(scores, job_dir)
            if selected.get("selected"):
                return scores, selected

    scored_paths = {safe_text(s.get("image_path")) for s in scores}
    threshold = float(getattr(args, "vlm_accept_threshold", 6.0) or 6.0)
    for image in images:
        if str(image) not in scored_paths:
            score = score_image_with_vlm(image, card, args)
            scores.append(score)
            write_json(scores_path, scores)
        else:
            score = next((s for s in scores if safe_text(s.get("image_path")) == str(image)), {})
        if is_suitable_image_score(score, threshold=threshold):
            selected = select_best_image(scores, job_dir)
            selected["early_stop"] = True
            selected["early_stop_after_images"] = len(scores)
            write_json(selected_path, selected)
            return scores, selected

    selected = select_best_image(scores, job_dir)
    selected["early_stop"] = False
    selected["early_stop_after_images"] = len(scores)
    write_json(selected_path, selected)
    return scores, selected


def _import_orchestrator():
    try:
        import src.trellis_supplier_asset_orchestrator as orch
    except Exception:
        import trellis_supplier_asset_orchestrator as orch  # type: ignore
    return orch


def _remote_job_id(job_id: str) -> str:
    return slugify(job_id, 120)


def _remote_artifact_name(card: dict[str, Any], job_id: str, suffix: str) -> str:
    uid = slugify(card_unique_key(card), 56)
    title = slugify(card.get("title") or "asset", 64)
    return f"{job_id}__uid_{uid}__title_{title}{suffix}"


def _make_trellis_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        server_host=args.server_host,
        server_port=args.server_port,
        server_user=args.server_user,
        ssh_key=args.ssh_key,
        remote_root=args.remote_root,
        remote_trellis_root=args.remote_trellis_root,
        remote_model_dir=args.remote_model_dir,
        remote_python=args.remote_python,
        remote_worker_root=args.remote_worker_root,
        remote_worker_timeout_sec=args.remote_worker_timeout_sec,
        remote_worker_poll_sec=args.remote_worker_poll_sec,
        remote_persistent_worker=args.remote_persistent_worker,
        remote_cuda_visible_devices=0,
        mode="single_image",
        multi_mode="stochastic",
        max_images=1,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        sparse_steps=args.sparse_steps,
        slat_steps=args.slat_steps,
        ss_guidance_strength=args.ss_guidance_strength,
        slat_guidance_strength=args.slat_guidance_strength,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        pre_export_simplify_target=0,
        no_remesh=args.no_remesh,
        remesh_band=args.remesh_band,
        remesh_project=args.remesh_project,
        no_webp=args.no_webp,
        simplify=0.0,
        image_size=0,
        fill_holes_resolution=0,
        fill_holes_num_views=0,
    )


def run_trellis2_generation(card: dict[str, Any], selected_image: Path, job_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    orch = _import_orchestrator()
    output_dir = job_dir / "output"
    images_dir = job_dir / "trellis_input_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    image_dst = images_dir / f"image_01{selected_image.suffix.lower() if selected_image.suffix else '.png'}"
    shutil.copy2(selected_image, image_dst)

    card_norm = dict(card)
    card_norm.setdefault("unique_key", card_unique_key(card))
    card_norm["glb_creator_selected_image"] = str(selected_image.resolve())
    write_json(job_dir / "card.normalized.json", card_norm)
    manifest = {"images": [str(image_dst.resolve())], "count": 1, "trellis_input_dir": str(images_dir.resolve())}
    write_json(job_dir / "image_manifest.json", manifest)

    remote_id = _remote_job_id(job_dir.name)
    remote_job_dir = f"{args.remote_root.rstrip('/')}/{remote_id}"
    remote_glb = f"{remote_job_dir}/output/asset.trellis.glb"
    remote_report = f"{remote_job_dir}/output/trellis.report.json"
    remote_named_glb = f"{remote_job_dir}/output/{_remote_artifact_name(card, remote_id, '.glb')}"
    targs = _make_trellis_args(args)

    if args.resume:
        status = orch.ssh_run(
            targs,
            "\n".join(
                [
                    f"if [ -s {orch.shell_quote(remote_glb)} ] && [ -s {orch.shell_quote(remote_report)} ]; then",
                    "  echo done",
                    "else",
                    "  echo missing",
                    "fi",
                ]
            ),
        ).strip().splitlines()[-1:]
        if status and status[0] == "done":
            logs_dir = job_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "remote_stdout.log").write_text("[resume] remote asset already exists; generation skipped\n", encoding="utf-8")
            if bool(getattr(args, "remote_artifacts_only", False)):
                orch.ssh_run(
                    targs,
                    "\n".join(
                        [
                            f"mkdir -p {orch.shell_quote(remote_job_dir + '/output')}",
                            f"test -s {orch.shell_quote(remote_glb)}",
                            f"test -s {orch.shell_quote(remote_report)}",
                            f"if [ ! -s {orch.shell_quote(remote_named_glb)} ]; then cp {orch.shell_quote(remote_glb)} {orch.shell_quote(remote_named_glb)}; fi",
                        ]
                    ),
                )
                return {
                    "ok": True,
                    "asset_glb": remote_named_glb,
                    "canonical_asset_glb": remote_glb,
                    "remote_report_json": remote_report,
                    "remote_job_dir": remote_job_dir,
                    "remote_worker_mode": "persistent",
                    "remote_stdout_log": str((logs_dir / "remote_stdout.log").resolve()),
                    "remote_artifacts_only": True,
                    "remote_render_manifest": {},
                    "resumed_remote_asset": True,
                }
            orch.scp_from_remote(targs, remote_glb, output_dir / "asset.trellis.glb")
            orch.scp_from_remote(targs, remote_report, output_dir / "trellis.report.json")
            return {
                "ok": True,
                "asset_glb": str((output_dir / "asset.trellis.glb").resolve()),
                "remote_report_json": str((output_dir / "trellis.report.json").resolve()),
                "remote_job_dir": remote_job_dir,
                "remote_worker_mode": "persistent",
                "remote_stdout_log": str((logs_dir / "remote_stdout.log").resolve()),
                "resumed_remote_asset": True,
            }

    orch.ssh_run(targs, f"rm -rf {orch.shell_quote(remote_job_dir)}\nmkdir -p {orch.shell_quote(remote_job_dir)} {orch.shell_quote(remote_job_dir + '/output')}")
    orch.scp_to_remote(targs, job_dir / "card.normalized.json", f"{remote_job_dir}/card.normalized.json")
    orch.scp_to_remote(targs, job_dir / "image_manifest.json", f"{remote_job_dir}/image_manifest.json")
    orch.scp_to_remote(targs, images_dir, f"{remote_job_dir}/images")

    if bool(getattr(args, "enqueue_only", False)):
        if not bool(args.remote_persistent_worker):
            raise RuntimeError("--enqueue-only requires --remote-persistent-worker")
        orch.ensure_remote_worker(targs)
        payload = orch.build_trellis2_worker_job_payload(targs, remote_id, remote_job_dir, remote_glb, remote_report)
        queue_path = orch.enqueue_remote_worker_job(targs, payload, job_dir)
        (job_dir / "trellis2_worker_queue_path.txt").write_text(queue_path, encoding="utf-8")
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "remote_stdout.log").write_text(
            f"[enqueue-only] queued={queue_path}\nremote_job_dir={remote_job_dir}\n",
            encoding="utf-8",
        )
        return {
            "ok": True,
            "queued_only": True,
            "asset_glb": remote_named_glb,
            "canonical_asset_glb": remote_glb,
            "remote_report_json": remote_report,
            "remote_job_dir": remote_job_dir,
            "remote_worker_mode": "persistent",
            "remote_queue_job_json": queue_path,
            "remote_stdout_log": str((logs_dir / "remote_stdout.log").resolve()),
            "remote_artifacts_only": bool(getattr(args, "remote_artifacts_only", False)),
            "remote_render_manifest": {},
        }

    if bool(args.remote_persistent_worker):
        remote_stdout = orch.run_remote_trellis2_persistent(targs, remote_id, remote_job_dir, remote_glb, remote_report, job_dir)
    else:
        payload = orch.build_trellis2_worker_job_payload(targs, remote_id, remote_job_dir, remote_glb, remote_report)
        queue_path = orch.enqueue_remote_worker_job(targs, payload, job_dir)
        worker_script = f"{args.remote_trellis_root.rstrip('/')}/run_trellis2_persistent_worker.py"
        log_file = f"{args.remote_worker_root.rstrip('/')}/logs/glb_creator_single_run_{remote_id}.log"
        cmd = "\n".join(
            [
                orch.remote_env_prefix(targs),
                f"mkdir -p {orch.shell_quote(args.remote_worker_root.rstrip('/') + '/logs')}",
                f"test -f {orch.shell_quote(worker_script)}",
                f"{orch.shell_quote(args.remote_python)} -u -X faulthandler {orch.shell_quote(worker_script)} "
                f"--worker-root {orch.shell_quote(args.remote_worker_root)} "
                f"--model {orch.shell_quote(args.remote_model_dir)} "
                f"--gpu-index 0 --poll-sec 1.0 --idle-exit-sec 1 "
                f"--log-file {orch.shell_quote(log_file)}",
            ]
        )
        remote_stdout = orch.ssh_run(targs, cmd)
        (job_dir / "trellis2_worker_queue_path.txt").write_text(queue_path, encoding="utf-8")
        status = orch.wait_remote_worker_job(targs, remote_report, remote_glb)
        if status != "done":
            raise RuntimeError(f"TRELLIS.2 single-run worker finished without GLB: {status}")
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "remote_stdout.log").write_text(remote_stdout, encoding="utf-8")

    if bool(getattr(args, "remote_artifacts_only", False)):
        orch.ssh_run(
            targs,
            "\n".join(
                [
                    f"test -s {orch.shell_quote(remote_glb)}",
                    f"test -s {orch.shell_quote(remote_report)}",
                    f"cp {orch.shell_quote(remote_glb)} {orch.shell_quote(remote_named_glb)}",
                ]
            ),
        )
        remote_render_manifest = (
            {}
            if bool(getattr(args, "skip_remote_renders", False))
            else run_remote_glb_renders(card, remote_glb, remote_job_dir, remote_id, job_dir, args, orch, targs)
        )
        return {
            "ok": True,
            "asset_glb": remote_named_glb,
            "canonical_asset_glb": remote_glb,
            "remote_report_json": remote_report,
            "remote_job_dir": remote_job_dir,
            "remote_worker_mode": "persistent",
            "remote_stdout_log": str((logs_dir / "remote_stdout.log").resolve()),
            "remote_artifacts_only": True,
            "remote_render_manifest": remote_render_manifest,
        }

    orch.scp_from_remote(targs, remote_glb, output_dir / "asset.trellis.glb")
    orch.scp_from_remote(targs, remote_report, output_dir / "trellis.report.json")
    report = read_json(output_dir / "trellis.report.json")
    return {
        "ok": bool(report.get("ok")) and (output_dir / "asset.trellis.glb").is_file(),
        "asset_glb": str((output_dir / "asset.trellis.glb").resolve()),
        "remote_report_json": str((output_dir / "trellis.report.json").resolve()),
        "remote_job_dir": remote_job_dir,
        "remote_worker_mode": "persistent",
        "remote_stdout_log": str((logs_dir / "remote_stdout.log").resolve()),
    }


def run_remote_glb_renders(
    card: dict[str, Any],
    remote_glb: str,
    remote_job_dir: str,
    remote_id: str,
    job_dir: Path,
    args: argparse.Namespace,
    orch: Any,
    targs: argparse.Namespace,
) -> dict[str, Any]:
    local_script = Path(__file__).with_name("glb_creator_render_glb_blender.py")
    remote_script = f"{remote_job_dir}/glb_creator_render_glb_blender.py"
    remote_renders_dir = f"{remote_job_dir}/renders"
    remote_manifest = f"{remote_renders_dir}/render_manifest.json"
    orch.scp_to_remote(targs, local_script, remote_script)
    render_cmd = "\n".join(
        [
            f"mkdir -p {orch.shell_quote(remote_renders_dir)}",
            f"{orch.shell_quote(args.remote_blender_path)} --background --python {orch.shell_quote(remote_script)} -- "
            f"--glb {orch.shell_quote(remote_glb)} "
            f"--out-dir {orch.shell_quote(remote_renders_dir)} "
            f"--resolution {int(args.render_resolution)}",
            f"test -s {orch.shell_quote(remote_manifest)}",
        ]
    )
    stdout = orch.ssh_run(targs, render_cmd)
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "remote_render_stdout.log").write_text(stdout, encoding="utf-8")

    prefix = _remote_artifact_name(card, remote_id, "")
    rename_script = "\n".join(
        [
            f"cd {orch.shell_quote(remote_renders_dir)}",
            f"cp render_front.png {orch.shell_quote(prefix + '__view_front.png')} 2>/dev/null || true",
            f"cp render_left.png {orch.shell_quote(prefix + '__view_left.png')} 2>/dev/null || true",
            f"cp render_right.png {orch.shell_quote(prefix + '__view_right.png')} 2>/dev/null || true",
            f"cp render_three_quarter.png {orch.shell_quote(prefix + '__view_three_quarter.png')} 2>/dev/null || true",
            "python3 - <<'PY'\n"
            "import json\n"
            "from pathlib import Path\n"
            "p=Path('render_manifest.json')\n"
            "d=json.loads(p.read_text())\n"
            "for r in d.get('renders', []):\n"
            "    view=r.get('view')\n"
            f"    named='{prefix}__view_' + str(view) + '.png'\n"
            "    if Path(named).is_file():\n"
            "        r['named_path']=str(Path.cwd()/named)\n"
            "p.write_text(json.dumps(d, ensure_ascii=False, indent=2))\n"
            "PY",
        ]
    )
    orch.ssh_run(targs, rename_script)
    return {
        "glb": remote_glb,
        "manifest": remote_manifest,
        "renders_dir": remote_renders_dir,
        "renders": [
            {"view": "front", "path": f"{remote_renders_dir}/{prefix}__view_front.png", "azimuth_deg": 0},
            {"view": "left", "path": f"{remote_renders_dir}/{prefix}__view_left.png", "azimuth_deg": 90},
            {"view": "right", "path": f"{remote_renders_dir}/{prefix}__view_right.png", "azimuth_deg": -90},
            {"view": "three_quarter", "path": f"{remote_renders_dir}/{prefix}__view_three_quarter.png", "azimuth_deg": 45},
        ],
    }


def render_glb_views(glb_path: Path, job_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    renders_dir = job_dir / "renders"
    manifest_path = renders_dir / "render_manifest.json"
    if args.resume and manifest_path.is_file() and all(Path(r.get("path", "")).is_file() for r in read_json(manifest_path).get("renders", [])):
        return read_json(manifest_path)
    if args.render_backend == "blender" and Path(args.blender_path).is_file():
        script = Path(__file__).with_name("glb_creator_render_glb_blender.py")
        cmd = [
            args.blender_path,
            "--background",
            "--python",
            str(script),
            "--",
            "--glb",
            str(glb_path),
            "--out-dir",
            str(renders_dir),
            "--resolution",
            str(args.render_resolution),
        ]
        (job_dir / "logs").mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (job_dir / "logs" / "render_blender.log").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Blender render failed with code {proc.returncode}; see {job_dir / 'logs' / 'render_blender.log'}")
        return read_json(manifest_path)
    return render_glb_views_trimesh(glb_path, renders_dir, args.render_resolution)


def render_glb_views_trimesh(glb_path: Path, renders_dir: Path, resolution: int) -> dict[str, Any]:
    import trimesh

    renders_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(str(glb_path), force="scene")
    views = [("front", 0), ("left", 90), ("right", -90), ("three_quarter", 45)]
    manifest = {"glb": str(glb_path.resolve()), "renders": []}
    for view, azimuth in views:
        out = renders_dir / f"render_{view}.png"
        try:
            png = scene.save_image(resolution=(resolution, resolution), visible=True)
            out.write_bytes(png)
        except Exception as exc:
            out.write_bytes(b"")
            manifest.setdefault("warnings", []).append(f"trimesh render failed for {view}: {exc}")
        manifest["renders"].append({"view": view, "path": str(out.resolve()), "azimuth_deg": azimuth})
    write_json(renders_dir / "render_manifest.json", manifest)
    return manifest


def description_prompt(card: dict[str, Any], image_type: str, view: str) -> str:
    return f"""
Опиши предмет на изображении для последующего сравнения 3D-модели.
Не фантазируй. Описывай только видимые признаки: категория, форма, силуэт, детали, материалы, цвета.
Карточка товара: title={safe_text(card.get('title'))}; category={safe_text(card.get('category_norm') or card.get('category') or card.get('semantic_group'))}
Верни только JSON:
{{
  "image_path": "",
  "image_type": "{image_type}",
  "view": "{view}",
  "object_category": "",
  "object_count": 1,
  "shape_description": "",
  "visible_geometry": [],
  "materials": [],
  "colors": [],
  "style": "",
  "distinctive_details": [],
  "possible_mismatches_or_uncertainties": [],
  "short_caption": ""
}}
""".strip()


def describe_image_with_vlm(image_path: Path, card: dict[str, Any], args: argparse.Namespace, image_type: str, view: str) -> dict[str, Any]:
    if args.vlm_provider == "none":
        return {
            "image_path": str(image_path),
            "image_type": image_type,
            "view": view,
            "object_category": safe_text(card.get("category_norm") or card.get("category") or card.get("semantic_group")),
            "object_count": 1,
            "shape_description": safe_text(card.get("description") or card.get("title")),
            "visible_geometry": [],
            "materials": _as_list(card.get("materials") or card.get("material")),
            "colors": _as_list(card.get("color") or card.get("colors")),
            "style": safe_text(card.get("style")),
            "distinctive_details": [],
            "possible_mismatches_or_uncertainties": ["VLM disabled; description from card metadata"],
            "short_caption": safe_text(card.get("title")),
            "evaluation_source": "heuristic_no_vlm",
        }
    prompt = description_prompt(card, image_type, view)
    try:
        if args.vlm_provider == "ollama":
            result = call_ollama_vlm(prompt, [image_path], args.vlm_model, args.vlm_base_url, int(args.vlm_timeout_sec))
        else:
            result = call_openai_compatible_vlm(prompt, [image_path], args.vlm_provider, args.vlm_model, args.vlm_base_url, int(args.vlm_timeout_sec))
        parsed = _extract_json_object(result.get("content", ""))
        if not parsed:
            raise RuntimeError("VLM returned no JSON object")
        parsed["image_path"] = str(image_path)
        parsed.setdefault("image_type", image_type)
        parsed.setdefault("view", view)
        return parsed
    except Exception as exc:
        fallback = describe_image_with_vlm(image_path, card, argparse.Namespace(vlm_provider="none"), image_type, view)
        fallback["vlm_error"] = f"{type(exc).__name__}: {exc}"
        return fallback


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [safe_text(x) for x in value if safe_text(x)]
    txt = safe_text(value)
    return [txt] if txt else []


def describe_source_and_renders(card: dict[str, Any], selected_image: Path, render_manifest: dict[str, Any], job_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = job_dir / "vlm_descriptions"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = describe_image_with_vlm(selected_image, card, args, "source", "source")
    write_json(out_dir / "source_description.json", source)
    descriptions = {"source": source, "renders": []}
    for item in render_manifest.get("renders", []):
        path = Path(item.get("path", ""))
        view = safe_text(item.get("view"))
        desc = describe_image_with_vlm(path, card, args, "render", view)
        descriptions["renders"].append(desc)
        write_json(out_dir / f"render_{view}_description.json", desc)
    write_json(out_dir / "all_descriptions.json", descriptions)
    return descriptions


def similarity_prompt(card: dict[str, Any], selected_score: dict[str, Any], descriptions: dict[str, Any]) -> str:
    compact = {
        "card": {k: card.get(k) for k in ("unique_key", "title", "category_norm", "category", "semantic_group", "materials", "color", "colors", "dimensions_cm", "description")},
        "selected_image_score": selected_score,
        "descriptions": descriptions,
    }
    return f"""
Сравни описание исходного supplier-фото и описания рендеров 3D-модели.
Нужно понять, это тот же предмет или другой. Оцени форму, силуэт, части, пропорции, материалы/цвета.
Данные:
{json.dumps(compact, ensure_ascii=False)}
Верни только JSON:
{{
  "same_object_likelihood": 0.0,
  "similarity_score_1_to_10": 0,
  "geometry_score_1_to_10": 0,
  "silhouette_score_1_to_10": 0,
  "parts_score_1_to_10": 0,
  "material_color_score_1_to_10": 0,
  "category_match": true,
  "is_same_category": true,
  "looks_like_different_object": false,
  "major_mismatches": [],
  "minor_mismatches": [],
  "matched_features": [],
  "verdict": "review",
  "reason": ""
}}
""".strip()


def compare_descriptions_with_llm(card: dict[str, Any], selected_score: dict[str, Any], descriptions: dict[str, Any], job_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metrics_dir = job_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.llm_provider == "none":
        score = float(selected_score.get("overall_score") or 5)
        category = safe_text(card.get("category_norm") or card.get("category") or card.get("semantic_group"))
        sim = max(4, min(8, int(round(score * 0.75 + (1.0 if category else 0.0)))))
        report = {
            "same_object_likelihood": round(sim / 10, 2),
            "similarity_score_1_to_10": sim,
            "geometry_score_1_to_10": sim,
            "silhouette_score_1_to_10": sim,
            "parts_score_1_to_10": sim,
            "material_color_score_1_to_10": max(4, sim - 1),
            "category_match": bool(category),
            "is_same_category": bool(category),
            "looks_like_different_object": False,
            "major_mismatches": [],
            "minor_mismatches": ["LLM disabled; similarity is heuristic"],
            "matched_features": [category] if category else [],
            "verdict": "accept" if sim >= 7 and category else "review",
            "reason": "heuristic_no_vlm",
            "evaluation_source": "heuristic_no_vlm",
        }
    else:
        prompt = similarity_prompt(card, selected_score, descriptions)
        try:
            if args.llm_provider == "ollama":
                result = call_ollama_llm(prompt, args.llm_model, args.llm_base_url, int(args.vlm_timeout_sec))
            else:
                result = call_openai_compatible_llm(prompt, args.llm_provider, args.llm_model, args.llm_base_url, int(args.vlm_timeout_sec))
            report = _extract_json_object(result.get("content", ""))
            if not report:
                raise RuntimeError("LLM returned no JSON object")
        except Exception as exc:
            fallback_args = argparse.Namespace(llm_provider="none")
            report = compare_descriptions_with_llm(card, selected_score, descriptions, job_dir, fallback_args)
            report["llm_error"] = f"{type(exc).__name__}: {exc}"
            return report
    sim = int(float(report.get("similarity_score_1_to_10") or 0))
    verdict = safe_text(report.get("verdict")).lower()
    expected_verdict = "accept" if sim >= 7 and report.get("category_match", True) else "review" if sim >= 5 else "reject"
    if verdict not in {"accept", "review", "reject"}:
        verdict = expected_verdict
    elif verdict != expected_verdict and (sim < 5 or sim >= 7):
        report["llm_original_verdict"] = verdict
        verdict = expected_verdict
    report["verdict"] = verdict
    write_json(metrics_dir / "similarity_report.json", report)
    write_json(metrics_dir / "quality_summary.json", {"similarity_score_1_to_10": sim, "verdict": report["verdict"], "category_match": report.get("category_match")})
    return report


def build_summary(card: dict[str, Any], job_id: str, job_dir: Path, selected: dict[str, Any], trellis: dict[str, Any], render_manifest: dict[str, Any], descriptions: dict[str, Any], similarity: dict[str, Any]) -> dict[str, Any]:
    score = selected.get("score") if isinstance(selected.get("score"), dict) else {}
    return {
        "schema": "supplier_glb_creator_summary/v1",
        "ok": bool(trellis.get("ok")),
        "job_id": job_id,
        "unique_key": card_unique_key(card),
        "title": safe_text(card.get("title")),
        "source_site": safe_text(card.get("source_site") or card.get("site")),
        "product_url": safe_text(card.get("product_url") or card.get("url")),
        "local_job_dir": str(job_dir.resolve()),
        "selected_image": selected.get("selected_image_path"),
        "selected_image_score": score.get("overall_score"),
        "asset_glb": trellis.get("asset_glb"),
        "canonical_asset_glb": trellis.get("canonical_asset_glb"),
        "asset_format": "glb",
        "asset_status": "trellis2_generated_candidate",
        "renders_dir": str((job_dir / "renders").resolve()),
        "image_scores_json": str((job_dir / "image_scores.json").resolve()),
        "vlm_descriptions_json": str((job_dir / "vlm_descriptions" / "all_descriptions.json").resolve()),
        "similarity_report_json": str((job_dir / "metrics" / "similarity_report.json").resolve()),
        "similarity_score_1_to_10": similarity.get("similarity_score_1_to_10"),
        "verdict": similarity.get("verdict"),
        "remote_report_json": trellis.get("remote_report_json"),
        "remote_job_dir": trellis.get("remote_job_dir"),
        "remote_artifacts_only": bool(trellis.get("remote_artifacts_only")),
        "logs": {
            "main_log": str((job_dir / "logs" / "glb_creator.log").resolve()),
            "remote_stdout": trellis.get("remote_stdout_log"),
        },
        "render_manifest": render_manifest,
    }


def patch_card_with_generated_glb(card: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    patched = dict(card)
    patched["asset_status"] = "trellis2_generated_candidate"
    patched["asset_format"] = "glb"
    patched["asset_local_path"] = summary.get("asset_glb")
    patched["asset_source"] = "trellis2_generated_from_supplier_image"
    patched["asset_generation_method"] = "glb_creator_v1"
    patched["asset_quality_score"] = summary.get("similarity_score_1_to_10")
    patched["asset_quality_verdict"] = summary.get("verdict")
    extra = dict(patched.get("extra") or {})
    extra["glb_creator"] = {
        "summary_json": str(Path(summary["local_job_dir"]) / "glb_creator.summary.json"),
        "selected_image": summary.get("selected_image"),
        "similarity_score_1_to_10": summary.get("similarity_score_1_to_10"),
        "verdict": summary.get("verdict"),
    }
    patched["extra"] = extra
    return patched


def _job_id_for_card(card: dict[str, Any], prefix: str = "supplier_glb") -> str:
    key = card_unique_key(card)
    title = safe_text(card.get("title")) or key
    uid_slug = slugify(key, 42)
    title_slug = slugify(title, 42)
    return f"{prefix}_{stable_hash(key)}_{uid_slug}_{title_slug}"


def ensure_named_glb_copy(card: dict[str, Any], job_id: str, glb_path: Path) -> Path:
    """Keep the worker contract path and also create a human-readable GLB name."""
    unique = slugify(card_unique_key(card), 56)
    title = slugify(card.get("title") or "asset", 48)
    named = glb_path.parent / f"{job_id}__uid_{unique}__title_{title}.glb"
    if named.resolve() != glb_path.resolve() and glb_path.is_file():
        if not named.is_file() or named.stat().st_size != glb_path.stat().st_size:
            shutil.copy2(glb_path, named)
    return named


def _skip_payload(card: dict[str, Any], job_dir: Path, reason: str, detail: str = "") -> dict[str, Any]:
    payload = {
        "schema": "supplier_glb_creator_summary/v1",
        "ok": False,
        "skipped": True,
        "skipped_reason": reason,
        "skip_detail": detail,
        "job_id": job_dir.name,
        "unique_key": card_unique_key(card),
        "title": safe_text(card.get("title")),
        "local_job_dir": str(job_dir.resolve()),
    }
    write_json(job_dir / "glb_creator.summary.json", payload)
    return payload


def _log(job_dir: Path, message: str) -> None:
    logs = job_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    line = f"{now_iso()} {message}\n"
    with (logs / "glb_creator.log").open("a", encoding="utf-8") as f:
        f.write(line)
    print(message, flush=True)


def process_existing_glb(args: argparse.Namespace) -> dict[str, Any]:
    glb = Path(args.existing_glb).expanduser().resolve()
    source = Path(args.source_image).expanduser().resolve()
    card = read_json(args.card_json) if args.card_json else {"title": glb.stem, "unique_key": f"existing::{stable_hash(str(glb))}"}
    job_id = args.job_id or f"existing_{stable_hash(str(glb))}_{slugify(glb.stem, 40)}"
    job_dir = Path(args.out_dir).expanduser().resolve() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_glb = job_dir / "output" / "asset.trellis.glb"
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite or not out_glb.is_file():
        shutil.copy2(glb, out_glb)
    selected = {"selected": True, "selected_image_path": str(source), "score": heuristic_image_score(source, card)}
    write_json(job_dir / "selected_image.json", selected)
    render_manifest = render_glb_views(out_glb, job_dir, args)
    descriptions = describe_source_and_renders(card, source, render_manifest, job_dir, args)
    similarity = compare_descriptions_with_llm(card, selected["score"], descriptions, job_dir, args)
    trellis = {"ok": out_glb.is_file(), "asset_glb": str(out_glb.resolve()), "remote_report_json": ""}
    summary = build_summary(card, job_id, job_dir, selected, trellis, render_manifest, descriptions, similarity)
    write_json(job_dir / "glb_creator.summary.json", summary)
    write_json(job_dir / "card.with_generated_glb.json", patch_card_with_generated_glb(card, summary))
    return summary


def process_one_card(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    job_id = args.job_id or _job_id_for_card(card)
    job_dir = Path(args.out_dir).expanduser().resolve() / job_id
    summary_path = job_dir / "glb_creator.summary.json"
    if summary_path.is_file() and not args.overwrite and not args.resume:
        return read_json(summary_path)
    job_dir.mkdir(parents=True, exist_ok=True)
    _log(job_dir, f"[start] job_id={job_id} unique_key={card_unique_key(card)}")
    try:
        has_asset, reason, field = has_existing_asset(card)
        if has_asset:
            return _skip_payload(card, job_dir, "has_direct_model_asset", field or reason)

        if not (args.resume and (job_dir / "image_scores.json").is_file() and (job_dir / "selected_image.json").is_file()):
            manifest = prepare_card_images(card, job_dir, args.max_images_per_card)
            images = [Path(p) for p in manifest.get("images", []) if Path(p).is_file()]
            if not images:
                return _skip_payload(card, job_dir, "no_images")
            _, selected = score_images_until_selected(images, card, job_dir, args)
        else:
            selected = read_json(job_dir / "selected_image.json")

        if not selected.get("selected"):
            return _skip_payload(card, job_dir, "no_suitable_image", safe_text(selected.get("reason")))
        selected_image = Path(selected["selected_image_path"])

        if args.prepare_only or args.dry_run:
            payload = {
                "schema": "supplier_glb_creator_summary/v1",
                "ok": True,
                "prepare_only": True,
                "job_id": job_id,
                "unique_key": card_unique_key(card),
                "local_job_dir": str(job_dir.resolve()),
                "selected_image": str(selected_image.resolve()),
                "selected_image_score": (selected.get("score") or {}).get("overall_score"),
            }
            write_json(summary_path, payload)
            return payload

        glb_path = job_dir / "output" / "asset.trellis.glb"
        report_path = job_dir / "output" / "trellis.report.json"
        if args.resume and glb_path.is_file() and report_path.is_file():
            trellis = {"ok": True, "asset_glb": str(glb_path.resolve()), "remote_report_json": str(report_path.resolve())}
        else:
            trellis = run_trellis2_generation(card, selected_image, job_dir, args)
        if not trellis.get("ok"):
            raise RuntimeError("TRELLIS.2 generation did not produce a valid GLB")
        if bool(getattr(args, "remote_artifacts_only", False)):
            render_manifest = trellis.get("remote_render_manifest") or {}
            descriptions = {"source": {}, "renders": []}
            similarity = {
                "similarity_score_1_to_10": None,
                "verdict": "remote_artifacts_only",
                "category_match": None,
                "reason": "local VLM/LLM evaluation skipped; GLB and renders are stored on remote",
            }
        else:
            named_glb = ensure_named_glb_copy(card, job_id, Path(trellis["asset_glb"]))
            trellis["canonical_asset_glb"] = trellis["asset_glb"]
            trellis["asset_glb"] = str(named_glb.resolve())

            render_manifest = render_glb_views(Path(trellis["canonical_asset_glb"]), job_dir, args)
            descriptions_path = job_dir / "vlm_descriptions" / "all_descriptions.json"
            descriptions = read_json(descriptions_path) if args.resume and descriptions_path.is_file() else describe_source_and_renders(card, selected_image, render_manifest, job_dir, args)
            similarity_path = job_dir / "metrics" / "similarity_report.json"
            similarity = read_json(similarity_path) if args.resume and similarity_path.is_file() else compare_descriptions_with_llm(card, selected.get("score") or {}, descriptions, job_dir, args)

        summary = build_summary(card, job_id, job_dir, selected, trellis, render_manifest, descriptions, similarity)
        write_json(summary_path, summary)
        write_json(job_dir / "card.with_generated_glb.json", patch_card_with_generated_glb(card, summary))
        _log(job_dir, f"[done] ok={summary.get('ok')} glb={summary.get('asset_glb')} score={summary.get('similarity_score_1_to_10')} verdict={summary.get('verdict')}")
        return summary
    except Exception as exc:
        err = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "job_id": job_id, "unique_key": card_unique_key(card)}
        write_json(job_dir / "error.json", err)
        _log(job_dir, f"[error] {err['error']}")
        if not args.continue_on_error:
            raise
        return err


def _filter_cards(cards: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for card in cards:
        if args.category_filter:
            text = " ".join(safe_text(card.get(k)) for k in ("category_norm", "category", "semantic_group", "title")).lower()
            if args.category_filter.lower() not in text:
                continue
        if args.only_source_site:
            site = safe_text(card.get("source_site") or card.get("site") or card.get("supplier")).lower()
            if args.only_source_site.lower() not in site:
                continue
        out.append(card)
    return out


def _batch_item(card: dict[str, Any], item: dict[str, Any], out_root: Path) -> dict[str, Any]:
    local_job_dir = safe_text(item.get("local_job_dir"))
    return {
        "unique_key": card_unique_key(card),
        "job_id": item.get("job_id"),
        "ok": bool(item.get("ok")) and not bool(item.get("skipped")),
        "skipped": bool(item.get("skipped")),
        "skipped_reason": item.get("skipped_reason"),
        "summary_json": str((Path(local_job_dir) / "glb_creator.summary.json").resolve()) if local_job_dir else "",
        "asset_glb": item.get("asset_glb"),
        "canonical_asset_glb": item.get("canonical_asset_glb"),
        "remote_job_dir": item.get("remote_job_dir"),
        "remote_artifacts_only": item.get("remote_artifacts_only"),
        "similarity_score_1_to_10": item.get("similarity_score_1_to_10"),
        "verdict": item.get("verdict"),
        "error": item.get("error"),
    }


def _finalize_batch_report(report: dict[str, Any], out_root: Path) -> dict[str, Any]:
    report["finished_at"] = now_iso()
    report["ok_count"] = sum(1 for x in report["items"] if x.get("ok"))
    report["failed_count"] = sum(1 for x in report["items"] if x.get("error"))
    report["skipped_count"] = sum(1 for x in report["items"] if x.get("skipped"))
    write_json(out_root / "glb_creator.batch_report.json", report)
    return report


def _select_catalog_cards(args: argparse.Namespace) -> list[dict[str, Any]]:
    catalog = read_json(args.catalog_json)
    cards = collect_cards_without_assets(catalog)
    cards = _filter_cards(cards, args)
    if args.unique_key:
        cards = [c for c in cards if card_unique_key(c) == args.unique_key or safe_text(c.get("unique_key")) == args.unique_key]
    if args.shuffle:
        random.Random(args.seed).shuffle(cards)
    if args.skip_existing_remote_assets:
        done_job_ids = _remote_existing_asset_job_ids(args)
        before = len(cards)
        cards = [c for c in cards if _remote_job_id(_job_id_for_card(c)) not in done_job_ids]
        print(f"[filter:remote] existing={len(done_job_ids)} kept={len(cards)} skipped={before - len(cards)}", flush=True)
    if args.offset:
        cards = cards[args.offset :]
    if args.limit:
        cards = cards[: args.limit]
    return cards


def _remote_existing_asset_job_ids(args: argparse.Namespace) -> set[str]:
    orch = _import_orchestrator()
    targs = _make_trellis_args(args)
    root = args.remote_root.rstrip("/")
    script = "\n".join(
        [
            f"ROOT={orch.shell_quote(root)}",
            'if [ ! -d "$ROOT" ]; then exit 0; fi',
            'find "$ROOT" -mindepth 2 -path "*/output/asset.trellis.glb" -type f | while IFS= read -r glb; do',
            '  job_dir="$(dirname "$(dirname "$glb")")"',
            '  report="$job_dir/output/trellis.report.json"',
            '  if [ -s "$glb" ] && [ -s "$report" ]; then basename "$job_dir"; fi',
            "done",
        ]
    )
    stdout = orch.ssh_run(targs, script)
    return {line.strip() for line in stdout.splitlines() if line.strip()}


def _job_context(card: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    job_id = args.job_id or _job_id_for_card(card)
    job_dir = Path(args.out_dir).expanduser().resolve() / job_id
    return {"card": card, "job_id": job_id, "job_dir": job_dir}


def _stage_prepare_images(ctx: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    card = ctx["card"]
    job_id = ctx["job_id"]
    job_dir: Path = ctx["job_dir"]
    summary_path = job_dir / "glb_creator.summary.json"
    if summary_path.is_file() and not args.overwrite and not args.resume:
        ctx["status"] = "done"
        ctx["summary"] = read_json(summary_path)
        return ctx
    job_dir.mkdir(parents=True, exist_ok=True)
    _log(job_dir, f"[stage-prepare] job_id={job_id} unique_key={card_unique_key(card)}")
    has_asset, reason, field = has_existing_asset(card)
    if has_asset:
        ctx["status"] = "skipped"
        ctx["summary"] = _skip_payload(card, job_dir, "has_direct_model_asset", field or reason)
        return ctx
    try:
        manifest_path = job_dir / "images" / "image_prepare_manifest.json"
        if args.resume and manifest_path.is_file():
            manifest = read_json(manifest_path)
        else:
            manifest = prepare_card_images(card, job_dir, args.max_images_per_card)
        images = [Path(p) for p in manifest.get("images", []) if Path(p).is_file()]
        if not images:
            ctx["status"] = "skipped"
            ctx["summary"] = _skip_payload(card, job_dir, "no_images")
            return ctx
        ctx["images"] = images
        ctx["status"] = "prepared"
    except Exception as exc:
        ctx["status"] = "failed"
        ctx["summary"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "job_id": job_id, "unique_key": card_unique_key(card), "local_job_dir": str(job_dir.resolve())}
        write_json(job_dir / "error.json", ctx["summary"])
    return ctx


def _stage_score_images(ctx: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if ctx.get("status") != "prepared":
        return ctx
    card = ctx["card"]
    job_id = ctx["job_id"]
    job_dir: Path = ctx["job_dir"]
    _log(job_dir, f"[stage-vlm] job_id={job_id} images={len(ctx.get('images') or [])}")
    try:
        selected_path = job_dir / "selected_image.json"
        if args.resume and selected_path.is_file() and (job_dir / "image_scores.json").is_file():
            selected = read_json(selected_path)
        else:
            _, selected = score_images_until_selected(list(ctx.get("images") or []), card, job_dir, args)
        if not selected.get("selected"):
            ctx["status"] = "skipped"
            ctx["summary"] = _skip_payload(card, job_dir, "no_suitable_image", safe_text(selected.get("reason")))
            return ctx
        ctx["selected"] = selected
        ctx["selected_image"] = Path(selected["selected_image_path"])
        if args.prepare_only or args.dry_run:
            payload = {
                "schema": "supplier_glb_creator_summary/v1",
                "ok": True,
                "prepare_only": True,
                "job_id": job_id,
                "unique_key": card_unique_key(card),
                "local_job_dir": str(job_dir.resolve()),
                "selected_image": str(ctx["selected_image"].resolve()),
                "selected_image_score": (selected.get("score") or {}).get("overall_score"),
            }
            write_json(job_dir / "glb_creator.summary.json", payload)
            ctx["summary"] = payload
            ctx["status"] = "done"
            return ctx
        ctx["status"] = "scored"
    except Exception as exc:
        ctx["status"] = "failed"
        ctx["summary"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "job_id": job_id, "unique_key": card_unique_key(card), "local_job_dir": str(job_dir.resolve())}
        write_json(job_dir / "error.json", ctx["summary"])
    return ctx


def _stage_generate_and_evaluate(ctx: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if ctx.get("status") != "scored":
        return ctx
    card = ctx["card"]
    job_id = ctx["job_id"]
    job_dir: Path = ctx["job_dir"]
    selected = ctx["selected"]
    selected_image = Path(ctx["selected_image"])
    _log(job_dir, f"[stage-glb] job_id={job_id} selected={selected_image}")
    try:
        glb_path = job_dir / "output" / "asset.trellis.glb"
        report_path = job_dir / "output" / "trellis.report.json"
        if args.resume and glb_path.is_file() and report_path.is_file():
            trellis = {"ok": True, "asset_glb": str(glb_path.resolve()), "remote_report_json": str(report_path.resolve())}
        else:
            trellis = run_trellis2_generation(card, selected_image, job_dir, args)
        if not trellis.get("ok"):
            raise RuntimeError("TRELLIS.2 generation did not produce a valid GLB")
        if bool(getattr(args, "remote_artifacts_only", False)):
            render_manifest = trellis.get("remote_render_manifest") or {}
            descriptions = {"source": {}, "renders": []}
            similarity = {
                "similarity_score_1_to_10": None,
                "verdict": "remote_artifacts_only",
                "category_match": None,
                "reason": "local VLM/LLM evaluation skipped; GLB and renders are stored on remote",
            }
        else:
            named_glb = ensure_named_glb_copy(card, job_id, Path(trellis["asset_glb"]))
            trellis["canonical_asset_glb"] = trellis["asset_glb"]
            trellis["asset_glb"] = str(named_glb.resolve())
            render_manifest = render_glb_views(Path(trellis["canonical_asset_glb"]), job_dir, args)
            descriptions_path = job_dir / "vlm_descriptions" / "all_descriptions.json"
            descriptions = read_json(descriptions_path) if args.resume and descriptions_path.is_file() else describe_source_and_renders(card, selected_image, render_manifest, job_dir, args)
            similarity_path = job_dir / "metrics" / "similarity_report.json"
            similarity = read_json(similarity_path) if args.resume and similarity_path.is_file() else compare_descriptions_with_llm(card, selected.get("score") or {}, descriptions, job_dir, args)
        summary = build_summary(card, job_id, job_dir, selected, trellis, render_manifest, descriptions, similarity)
        write_json(job_dir / "glb_creator.summary.json", summary)
        write_json(job_dir / "card.with_generated_glb.json", patch_card_with_generated_glb(card, summary))
        _log(job_dir, f"[done] ok={summary.get('ok')} glb={summary.get('asset_glb')} score={summary.get('similarity_score_1_to_10')} verdict={summary.get('verdict')}")
        ctx["summary"] = summary
        ctx["status"] = "done"
    except Exception as exc:
        ctx["status"] = "failed"
        ctx["summary"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "job_id": job_id, "unique_key": card_unique_key(card), "local_job_dir": str(job_dir.resolve())}
        write_json(job_dir / "error.json", ctx["summary"])
        _log(job_dir, f"[error] {ctx['summary']['error']}")
        if not args.continue_on_error:
            raise
    return ctx


def process_catalog_batch_staged(args: argparse.Namespace) -> dict[str, Any]:
    cards = _select_catalog_cards(args)
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "supplier_glb_creator_batch_report/v1",
        "catalog_json": str(Path(args.catalog_json).expanduser().resolve()),
        "started_at": now_iso(),
        "mode": "chunked_staged",
        "chunk_size": int(args.chunk_size),
        "total_selected": len(cards),
        "items": [],
    }
    chunk_size = max(1, int(args.chunk_size or 10))
    for start in range(0, len(cards), chunk_size):
        chunk_cards = cards[start : start + chunk_size]
        chunk_no = start // chunk_size + 1
        print(f"[batch][chunk-start] {chunk_no} items={len(chunk_cards)} stage=prepare", flush=True)
        contexts = [_stage_prepare_images(_job_context(card, args), args) for card in chunk_cards]
        print(f"[batch][chunk-stage] {chunk_no} stage=vlm items={sum(1 for c in contexts if c.get('status') == 'prepared')}", flush=True)
        contexts = [_stage_score_images(ctx, args) for ctx in contexts]
        print(f"[batch][chunk-stage] {chunk_no} stage=glb items={sum(1 for c in contexts if c.get('status') == 'scored')}", flush=True)
        contexts = [_stage_generate_and_evaluate(ctx, args) for ctx in contexts]
        for ctx in contexts:
            card = ctx["card"]
            item = ctx.get("summary") or {}
            report["items"].append(_batch_item(card, item, out_root))
        write_json(out_root / "glb_creator.batch_report.json", report)
        print(f"[batch][chunk-end] {chunk_no}", flush=True)
    return _finalize_batch_report(report, out_root)


def process_catalog_batch(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "staged_batch", True)):
        return process_catalog_batch_staged(args)
    cards = _select_catalog_cards(args)
    report = {
        "schema": "supplier_glb_creator_batch_report/v1",
        "catalog_json": str(Path(args.catalog_json).expanduser().resolve()),
        "started_at": now_iso(),
        "mode": "sequential_full_card",
        "total_selected": len(cards),
        "items": [],
    }
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for card in cards:
        item = process_one_card(card, args)
        report["items"].append(_batch_item(card, item, out_root))
        write_json(out_root / "glb_creator.batch_report.json", report)
    return _finalize_batch_report(report, out_root)


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate lightweight GLB candidates from supplier catalog cards.")
    src = ap.add_argument_group("Input")
    src.add_argument("--catalog-json", default="")
    src.add_argument("--unique-key", default="")
    src.add_argument("--card-json", default="")
    src.add_argument("--existing-glb", default="")
    src.add_argument("--source-image", default="")

    out = ap.add_argument_group("Output/control")
    out.add_argument("--out-dir", required=True)
    out.add_argument("--job-id", default="")
    out.add_argument("--limit", type=int, default=0)
    out.add_argument("--offset", type=int, default=0)
    out.add_argument("--shuffle", action="store_true")
    out.add_argument("--category-filter", default="")
    out.add_argument("--only-source-site", default="")
    out.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    out.add_argument("--overwrite", action="store_true")
    out.add_argument("--resume", action="store_true")
    out.add_argument("--prepare-only", action="store_true")
    out.add_argument("--dry-run", action="store_true")
    out.add_argument("--skip-vlm", action="store_true")
    out.add_argument("--image-selection-mode", choices=["vlm", "white-border-first", "first"], default="vlm")
    out.add_argument("--white-border-threshold", type=float, default=0.72)
    out.add_argument("--remote-artifacts-only", action="store_true", help="Keep generated GLB/renders on the remote server; only local JSON summaries are written.")
    out.add_argument("--skip-remote-renders", action="store_true", help="With --remote-artifacts-only, skip remote Blender render stage and generate only GLB artifacts.")
    out.add_argument("--skip-existing-remote-assets", action="store_true", help="Before processing a catalog, list remote completed assets once and skip matching cards locally.")
    out.add_argument("--enqueue-only", action="store_true", help="Upload missing jobs to the persistent remote worker queue and do not wait for GLB completion.")
    out.add_argument("--chunk-size", type=int, default=10, help="Batch chunk size for staged mode: prepare all, VLM all, then GLB all.")
    out.add_argument("--staged-batch", action=argparse.BooleanOptionalAction, default=True)

    prep = ap.add_argument_group("Image/VLM/LLM")
    prep.add_argument("--max-images-per-card", type=int, default=8)
    prep.add_argument("--vlm-provider", choices=["ollama", "openai", "openrouter", "none"], default="none")
    prep.add_argument("--vlm-model", default="llama3.2-vision:11b")
    prep.add_argument("--vlm-base-url", default="http://127.0.0.1:11435")
    prep.add_argument("--vlm-timeout-sec", type=int, default=120)
    prep.add_argument("--vlm-accept-threshold", type=float, default=6.0, help="Stop scoring a card once an image reaches this suitability score.")
    prep.add_argument("--llm-provider", choices=["ollama", "openai", "openrouter", "none"], default="none")
    prep.add_argument("--llm-model", default="llama3.1:8b")
    prep.add_argument("--llm-base-url", default="http://127.0.0.1:11435")

    render = ap.add_argument_group("Rendering")
    render.add_argument("--render-backend", choices=["blender", "trimesh"], default="blender")
    render.add_argument("--blender-path", default=DEFAULT_BLENDER)
    render.add_argument("--render-resolution", type=int, default=1024)

    ssh = ap.add_argument_group("Remote TRELLIS.2")
    ssh.add_argument("--server-host", default=DEFAULT_REMOTE_HOST)
    ssh.add_argument("--server-port", type=int, default=DEFAULT_REMOTE_PORT)
    ssh.add_argument("--server-user", default="root")
    ssh.add_argument("--ssh-key", default="~/.ssh/id_ed25519")
    ssh.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    ssh.add_argument("--remote-trellis-root", default=DEFAULT_REMOTE_TRELLIS_ROOT)
    ssh.add_argument("--remote-model-dir", default=DEFAULT_REMOTE_MODEL_DIR)
    ssh.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    ssh.add_argument("--remote-worker-root", default=DEFAULT_REMOTE_WORKER_ROOT)
    ssh.add_argument("--remote-blender-path", default="blender")
    ssh.add_argument("--remote-persistent-worker", action=argparse.BooleanOptionalAction, default=True)
    ssh.add_argument("--remote-worker-timeout-sec", type=float, default=1800.0)
    ssh.add_argument("--remote-worker-poll-sec", type=float, default=2.0)

    tr = ap.add_argument_group("TRELLIS.2 params")
    tr.add_argument("--pipeline-type", type=int, default=512)
    tr.add_argument("--sparse-steps", type=int, default=4)
    tr.add_argument("--slat-steps", type=int, default=4)
    tr.add_argument("--ss-guidance-strength", type=float, default=7.5)
    tr.add_argument("--slat-guidance-strength", type=float, default=3.0)
    tr.add_argument("--decimation-target", type=int, default=50000)
    tr.add_argument("--texture-size", type=int, default=256)
    tr.add_argument("--no-remesh", action=argparse.BooleanOptionalAction, default=False)
    tr.add_argument("--remesh-band", type=int, default=1)
    tr.add_argument("--remesh-project", type=float, default=0.0)
    tr.add_argument("--no-webp", action=argparse.BooleanOptionalAction, default=True)
    tr.add_argument("--seed", type=int, default=1)
    return ap


def main() -> None:
    args = build_cli().parse_args()
    if args.skip_vlm:
        args.vlm_provider = "none"
        args.llm_provider = "none"
    if args.existing_glb:
        if not args.source_image:
            raise SystemExit("--existing-glb requires --source-image")
        result = process_existing_glb(args)
    elif args.card_json:
        result = process_one_card(read_json(args.card_json), args)
    elif args.catalog_json:
        result = process_catalog_batch(args)
    else:
        raise SystemExit("Provide --card-json, --catalog-json, or --existing-glb")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
