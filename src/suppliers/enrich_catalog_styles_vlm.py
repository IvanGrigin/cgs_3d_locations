#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Enrich supplier catalog items with visual style labels from product images.

The script is intentionally resumable:
- product images are cached on disk by catalog id;
- VLM results are appended to JSONL one item at a time;
- existing result ids are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/sourse/suppliers/supplier_catalog_one_table.json"
DEFAULT_OUT_JSONL = "data/sourse/suppliers/supplier_catalog_one_table.vlm_styles.jsonl"
DEFAULT_IMAGE_CACHE = "data/sourse/suppliers/vlm_style_images"
DEFAULT_TOP_SAMPLES_DIR = "data/sourse/suppliers/vlm_style_top30"

OBJECT_CATEGORIES = [
    "furniture",
    "cabinet",
    "hardware",
    "kitchenware",
    "appliance",
    "washer",
    "decor",
]

ROOM_TYPES = [
    "kitchen",
    "bathroom",
    "bedroom",
    "living_room",
    "hallway",
    "laundry",
]

INTERIOR_STYLES = [
    "contemporary",
    "modern",
    "minimal",
    "scandinavian",
    "japandi",
    "loft",
    "classic",
    "neoclassic",
    "art_deco",
    "rustic",
    "farmhouse",
    "provence",
    "high_tech",
    "luxury",
    "eco",
    "retro",
    "unknown",
]

FORM_STYLES = [
    "straight",
    "rounded",
    "curved",
    "geometric",
    "massive",
    "slim",
    "low_profile",
    "built_in",
]

FINISHES = [
    "matte",
    "glossy",
    "wood",
    "veneer",
    "lacquer",
    "laminate",
    "metal",
    "glass",
    "mirror",
    "stone",
    "ceramic",
    "rattan",
]

COLOR_FAMILIES = [
    "white",
    "black",
    "gray",
    "beige",
    "brown",
    "natural_wood",
    "brass",
    "gold",
    "silver",
    "pastel",
    "bright",
]

HARDWARE_STYLES = [
    "handleless",
    "bar_pull",
    "knob",
    "cup_pull",
    "edge_pull",
    "gola_profile",
    "classic_handle",
    "industrial_handle",
]

INSTALLATION_TYPES = [
    "freestanding",
    "built_in",
    "integrated",
    "wall_mounted",
    "floor_mounted",
    "under_counter",
    "countertop",
]

APPLIANCE_STYLES = [
    "stainless_steel",
    "black_glass",
    "white_standard",
    "retro",
    "smart",
    "professional",
    "compact",
    "panel_ready",
]

OPTIONAL_UNKNOWN = [
    "unknown",
]


def load_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items") or []
    if not isinstance(data, list):
        raise RuntimeError(f"Catalog must be a list or object with items[]: {path}")
    return [x for x in data if isinstance(x, dict)]


def safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def item_id(item: dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "unique_key", "external_id", "title"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row_{fallback_index}"


def safe_fs_name(value: str, max_len: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    if not text:
        text = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return text[:max_len]


def image_urls(item: dict[str, Any], max_images: int) -> list[str]:
    raw = safe_json_loads(item.get("images_json"), [])
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if isinstance(value, dict):
            url = str(value.get("url") or value.get("src") or value.get("image") or "").strip()
        else:
            url = str(value or "").strip()
        if not url or url in seen:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_images:
            break
    return out


def infer_extension(url: str, content_type: str | None) -> str:
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if ext in {".jpg", ".jpeg", ".png", ".webp"}:
            return ".jpg" if ext == ".jpeg" else ext
    guessed = Path(url.split("?", 1)[0]).suffix.lower()
    if guessed in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if guessed == ".jpeg" else guessed
    return ".jpg"


def download_url(url: str, out_prefix: Path, timeout_sec: int, retries: int) -> Path:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; cgs-vlm-style-enricher/1.0)",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                content_type = resp.headers.get("Content-Type")
                body = resp.read()
            if len(body) < 128:
                raise RuntimeError(f"downloaded body too small: {len(body)} bytes")
            ext = infer_extension(url, content_type)
            out_path = out_prefix.with_suffix(ext)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(body)
            return out_path
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError("unreachable download retry loop")


def cached_downloads(
    *,
    item: dict[str, Any],
    index: int,
    image_cache_dir: Path,
    max_images: int,
    timeout_sec: int,
    retries: int,
) -> tuple[list[Path], list[dict[str, str]]]:
    iid = safe_fs_name(item_id(item, index))
    title = safe_fs_name(str(item.get("title") or "item"), max_len=48)
    item_dir = image_cache_dir / f"{iid}_{title}"
    paths: list[Path] = []
    errors: list[dict[str, str]] = []
    for image_index, url in enumerate(image_urls(item, max_images=max_images), start=1):
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        prefix = item_dir / f"{image_index:02d}_{url_hash}"
        existing = sorted(prefix.parent.glob(prefix.name + ".*"))
        if existing:
            paths.append(existing[0])
            continue
        try:
            paths.append(download_url(url, prefix, timeout_sec=timeout_sec, retries=retries))
        except Exception as exc:
            errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    return paths, errors


def image_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_prompt(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "")
    category = str(item.get("category_norm") or item.get("category_raw") or "")
    color = str(item.get("color") or "")
    material = str(item.get("materials") or "")
    description = str(item.get("description") or "")[:900]
    return (
        "You classify interior furniture/product style from product photos. "
        "Use visible shape, materials, ornament, silhouette, finish, installation hints, hardware, appliance appearance, "
        "and product text only as weak context. "
        "Choose only labels from the allowed values. If a field is not visible or not applicable, use unknown or an empty list.\n"
        "CRITICAL OUTPUT RULES:\n"
        "- Return exactly one JSON object and nothing else.\n"
        "- Do not write markdown, explanations, bullet lists, or '*Answer*' lines.\n"
        "- Do not copy the full list of allowed values into any field.\n"
        "- For array fields, return only selected labels, not all possible labels.\n\n"
        f"object_category: {' / '.join(OBJECT_CATEGORIES)} / unknown\n"
        f"room_type: {' / '.join(ROOM_TYPES)} / unknown\n"
        f"interior_style: {' / '.join(INTERIOR_STYLES)}\n"
        f"form_style: {' / '.join(FORM_STYLES)} / unknown\n"
        f"finish: {' / '.join(FINISHES)} / unknown\n"
        f"color_family: {' / '.join(COLOR_FAMILIES)} / unknown\n"
        f"hardware_style: {' / '.join(HARDWARE_STYLES)} / unknown\n"
        f"installation_type: {' / '.join(INSTALLATION_TYPES)} / unknown\n"
        f"appliance_style: {' / '.join(APPLIANCE_STYLES)} / unknown\n\n"
        "Catalog context:\n"
        f"title: {title}\n"
        f"category: {category}\n"
        f"color: {color}\n"
        f"materials: {material}\n"
        f"description: {description}\n\n"
        "Return this exact JSON shape with selected values:\n"
        "{"
        "\"object_category\":\"decor\","
        "\"room_type\":[\"living_room\"],"
        "\"interior_style\":[\"modern\"],"
        "\"form_style\":[\"rounded\"],"
        "\"finish\":[\"metal\"],"
        "\"color_family\":[\"black\"],"
        "\"hardware_style\":\"unknown\","
        "\"installation_type\":\"freestanding\","
        "\"appliance_style\":\"unknown\","
        "\"confidence\":0.75,"
        "\"rationale\":\"one short sentence\""
        "}"
    )


def build_validation_prompt(item: dict[str, Any], vlm_style: dict[str, Any]) -> str:
    title = str(item.get("title") or "")
    category = str(item.get("category_norm") or item.get("category_raw") or "")
    color = str(item.get("color") or "")
    material = str(item.get("materials") or "")
    return (
        "You are auditing a previous visual classification of a furniture/product image. "
        "Look at the image again and compare it with the proposed JSON labels. "
        "Judge whether the labels match the visible object, details, finish, and colors. "
        "Scores are integers from 1 to 10, where 10 means excellent match.\n"
        "CRITICAL OUTPUT RULES:\n"
        "- Return exactly one JSON object and nothing else.\n"
        "- Do not write markdown, explanations outside JSON, bullet lists, or placeholder text.\n"
        "- issues must contain real issues or be an empty list.\n\n"
        "Catalog context:\n"
        f"title: {title}\n"
        f"category: {category}\n"
        f"color: {color}\n"
        f"materials: {material}\n\n"
        "Proposed VLM labels:\n"
        f"{json.dumps(vlm_style, ensure_ascii=False)}\n\n"
        "Return this exact JSON shape:\n"
        "{"
        "\"matches_image\":true,"
        "\"overall_score\":8,"
        "\"detail_score\":8,"
        "\"color_score\":8,"
        "\"category_score\":8,"
        "\"style_score\":8,"
        "\"issues\":[],"
        "\"corrected_summary\":\"short corrected description if needed\","
        "\"rationale\":\"one short audit sentence\""
        "}"
    )


def build_description_prompt(item: dict[str, Any]) -> str:
    title = str(item.get("title") or "")
    category = str(item.get("category_norm") or item.get("category_raw") or "")
    color = str(item.get("color") or "")
    material = str(item.get("materials") or "")
    return (
        "Describe only what is visually visible in the product image. "
        "Do not classify interior style. Do not use style names like modern, classic, loft, "
        "scandinavian, luxury, etc. unless the word is printed in the image itself. "
        "Do not infer hidden properties, brand, exact material, or room if not visible. "
        "Focus on concrete visual evidence: object type, shape, proportions, visible parts, "
        "wood-like/metal-like/glass/fabric/stone-like surfaces, finish, colors, texture, "
        "handles, legs/base, cords, mounts, decorative details, and uncertainty.\n"
        "Return plain text only. Write 5-7 compact sentences. No markdown table. No JSON. "
        "Keep the answer under 170 words.\n\n"
        "Catalog context:\n"
        f"title: {title}\n"
        f"category: {category}\n"
        f"color hint: {color}\n"
        f"materials hint: {material}\n"
    )


def build_json_retry_prompt(
    *,
    item: dict[str, Any],
    previous_text: str,
    mode: str,
    vlm_style: dict[str, Any] | None = None,
) -> str:
    title = str(item.get("title") or "")
    category = str(item.get("category_norm") or item.get("category_raw") or "")
    if mode == "validation":
        return (
            "Your previous answer was not valid JSON. Look at the image and return only the audit JSON object. "
            "Do not add markdown or prose outside JSON.\n"
            f"title: {title}\ncategory: {category}\n"
            f"proposed labels: {json.dumps(vlm_style or {}, ensure_ascii=False)}\n"
            f"previous invalid answer: {previous_text[:1200]}\n"
            "{"
            "\"matches_image\":true,"
            "\"overall_score\":8,"
            "\"detail_score\":8,"
            "\"color_score\":8,"
            "\"category_score\":8,"
            "\"style_score\":8,"
            "\"issues\":[],"
            "\"corrected_summary\":\"\","
            "\"rationale\":\"short audit sentence\""
            "}"
        )
    if mode == "description":
        return (
            "Your previous answer was not valid JSON. Look at the image and return only the visual description JSON object. "
            "Describe visible appearance only. Do not classify style. Do not add markdown or prose outside JSON.\n"
            f"title: {title}\ncategory: {category}\n"
            f"previous invalid answer: {previous_text[:1200]}\n"
            "{"
            "\"object_summary\":\"one sentence visible-object summary\","
            "\"visible_objects\":[\"object name\"],"
            "\"detailed_description\":\"5-8 sentences about visible appearance only\","
            "\"materials_visible\":[\"unknown\"],"
            "\"finish_visible\":[\"unknown\"],"
            "\"colors_visible\":[\"unknown\"],"
            "\"forms_visible\":[\"unknown\"],"
            "\"hardware_visible\":\"unknown\","
            "\"installation_or_support_visible\":\"unknown\","
            "\"room_context_visible\":\"unknown\","
            "\"uncertainty\":\"short uncertainty note\""
            "}"
        )
    return (
        "Your previous answer was not valid JSON. Look at the image and return only the classification JSON object. "
        "Do not add markdown or prose outside JSON. Select labels, do not copy allowed-value lists.\n"
        f"title: {title}\ncategory: {category}\n"
        f"previous invalid answer: {previous_text[:1200]}\n"
        "{"
        "\"object_category\":\"decor\","
        "\"room_type\":[\"living_room\"],"
        "\"interior_style\":[\"modern\"],"
        "\"form_style\":[\"rounded\"],"
        "\"finish\":[\"metal\"],"
        "\"color_family\":[\"black\"],"
        "\"hardware_style\":\"unknown\","
        "\"installation_type\":\"freestanding\","
        "\"appliance_style\":\"unknown\","
        "\"confidence\":0.75,"
        "\"rationale\":\"one short sentence\""
        "}"
    )


def post_ollama_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_paths: list[Path],
    timeout_sec: int,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    response_format: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_to_b64(p) for p in image_paths],
            }
        ],
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if response_format != "none":
        payload["format"] = response_format
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_ollama_content(resp: dict[str, Any]) -> str:
    msg = resp.get("message") or {}
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"].strip()
    if isinstance(resp.get("response"), str):
        return resp["response"].strip()
    return json.dumps(resp, ensure_ascii=False)


def parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw, strict=False)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            raise
        data = json.loads(m.group(0), strict=False)
    if not isinstance(data, dict):
        raise RuntimeError("VLM response JSON is not an object")
    return data


def validate_classification_payload(data: dict[str, Any]) -> None:
    """Reject common non-answers where the model copied the whole taxonomy."""

    multi_fields = {
        "room_type": ROOM_TYPES,
        "interior_style": INTERIOR_STYLES,
        "form_style": FORM_STYLES,
        "finish": FINISHES,
        "color_family": COLOR_FAMILIES,
    }
    for key, allowed in multi_fields.items():
        value = data.get(key)
        if isinstance(value, str) and "/" in value:
            raise RuntimeError(f"VLM copied allowed options instead of choosing {key}")
        if isinstance(value, list):
            cleaned = [str(x).strip().lower() for x in value if str(x).strip()]
            allowed_without_unknown = [x for x in allowed if x != "unknown"]
            if len(cleaned) >= max(5, len(allowed_without_unknown) // 2):
                raise RuntimeError(f"VLM copied too many allowed values for {key}")


def validate_validation_payload(data: dict[str, Any]) -> None:
    required = ["matches_image", "overall_score", "detail_score", "color_score", "category_score", "style_score"]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"VLM validation JSON misses fields: {', '.join(missing)}")


def validate_description_payload(data: dict[str, Any]) -> None:
    required = ["object_summary", "detailed_description"]
    missing = [key for key in required if not str(data.get(key) or "").strip()]
    if missing:
        raise RuntimeError(f"VLM description JSON misses fields: {', '.join(missing)}")


def call_vlm_json(
    *,
    args: argparse.Namespace,
    item: dict[str, Any],
    image_paths: list[Path],
    prompt: str,
    mode: str,
    vlm_style: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    raw_text = ""
    raw_attempts: list[str] = []
    last_error: Exception | None = None
    for attempt in range(max(1, int(args.parse_retries) + 1)):
        use_prompt = prompt
        if attempt > 0:
            use_prompt = build_json_retry_prompt(
                item=item,
                previous_text=raw_text,
                mode=mode,
                vlm_style=vlm_style,
            )
        response_format = str(args.ollama_format)
        if attempt > 0 and response_format == "none":
            response_format = "json"
        resp = post_ollama_chat(
            base_url=str(args.ollama_url),
            model=str(args.ollama_model),
            prompt=use_prompt,
            image_paths=image_paths,
            timeout_sec=int(args.ollama_timeout),
            temperature=float(args.ollama_temperature),
            num_ctx=int(args.ollama_num_ctx),
            num_predict=int(args.ollama_num_predict),
            response_format=response_format,
        )
        raw_text = extract_ollama_content(resp)
        raw_attempts.append(f"ATTEMPT {attempt + 1} format={response_format}\n{raw_text}")
        try:
            data = parse_json_object(raw_text)
            if mode == "classification":
                validate_classification_payload(data)
            elif mode == "validation":
                validate_validation_payload(data)
            elif mode == "description":
                validate_description_payload(data)
            return data, "\n\n---\n\n".join(raw_attempts)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("call_vlm_json failed without captured error")


def normalize_vlm_result(data: dict[str, Any]) -> dict[str, Any]:
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    def clean_one(key: str, allowed: list[str], default: str = "unknown") -> str:
        value = str(data.get(key) or default).strip().lower()
        return value if value in allowed else default

    def clean_list(key: str, allowed: list[str], max_items: int) -> list[str]:
        value = data.get(key)
        if isinstance(value, str):
            value = [x.strip() for x in re.split(r"[,;/]", value) if x.strip()]
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for x in value:
            sx = str(x).strip().lower()
            if sx in allowed and sx not in out:
                out.append(sx)
        return out[:max_items]

    return {
        "object_category": clean_one("object_category", OBJECT_CATEGORIES),
        "room_type": clean_list("room_type", ROOM_TYPES, 3),
        "interior_style": clean_list("interior_style", INTERIOR_STYLES, 4),
        "form_style": clean_list("form_style", FORM_STYLES, 4),
        "finish": clean_list("finish", FINISHES, 5),
        "color_family": clean_list("color_family", COLOR_FAMILIES, 4),
        "hardware_style": clean_one("hardware_style", HARDWARE_STYLES),
        "installation_type": clean_one("installation_type", INSTALLATION_TYPES),
        "appliance_style": clean_one("appliance_style", APPLIANCE_STYLES),
        "confidence": round(confidence, 4),
        "rationale": str(data.get("rationale") or "").strip()[:400],
    }


def normalize_validation_result(data: dict[str, Any]) -> dict[str, Any]:
    def score(key: str) -> int:
        try:
            value = int(round(float(data.get(key, 1))))
        except Exception:
            value = 1
        return max(1, min(value, 10))

    issues = data.get("issues")
    if isinstance(issues, str):
        issues = [x.strip() for x in re.split(r"[,;/]", issues) if x.strip()]
    if not isinstance(issues, list):
        issues = []
    clean_issues: list[str] = []
    for issue in issues:
        text = str(issue).strip()
        if text and text not in clean_issues:
            clean_issues.append(text[:160])

    return {
        "matches_image": bool(data.get("matches_image", False)),
        "overall_score": score("overall_score"),
        "detail_score": score("detail_score"),
        "color_score": score("color_score"),
        "category_score": score("category_score"),
        "style_score": score("style_score"),
        "issues": clean_issues[:5],
        "corrected_summary": str(data.get("corrected_summary") or "").strip()[:500],
        "rationale": str(data.get("rationale") or "").strip()[:400],
    }


def normalize_description_result(data: dict[str, Any]) -> dict[str, Any]:
    def clean_text(key: str, limit: int) -> str:
        return str(data.get(key) or "").strip()[:limit]

    def clean_list(key: str, limit: int = 12) -> list[str]:
        value = data.get(key)
        if isinstance(value, str):
            value = [x.strip() for x in re.split(r"[,;/]", value) if x.strip()]
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in out:
                out.append(text[:80])
            if len(out) >= limit:
                break
        return out

    return {
        "object_summary": clean_text("object_summary", 300),
        "visible_objects": clean_list("visible_objects"),
        "detailed_description": clean_text("detailed_description", 1400),
        "materials_visible": clean_list("materials_visible"),
        "finish_visible": clean_list("finish_visible"),
        "colors_visible": clean_list("colors_visible"),
        "forms_visible": clean_list("forms_visible"),
        "hardware_visible": clean_text("hardware_visible", 300),
        "installation_or_support_visible": clean_text("installation_or_support_visible", 300),
        "room_context_visible": clean_text("room_context_visible", 300),
        "uncertainty": clean_text("uncertainty", 500),
    }


def description_from_text(text: str) -> dict[str, Any]:
    clean = re.sub(r"\s+", " ", text).strip()
    clean = re.sub(r"^```(?:text)?", "", clean).strip()
    clean = re.sub(r"```$", "", clean).strip()
    first_sentence = clean
    m = re.search(r"(.+?[.!?])(?:\s|$)", clean)
    if m:
        first_sentence = m.group(1)
    return {
        "object_summary": first_sentence[:300],
        "visible_objects": [],
        "detailed_description": clean[:2000],
        "materials_visible": [],
        "finish_visible": [],
        "colors_visible": [],
        "forms_visible": [],
        "hardware_visible": "",
        "installation_or_support_visible": "",
        "room_context_visible": "",
        "uncertainty": "",
    }


def call_vlm_description(
    *,
    args: argparse.Namespace,
    item: dict[str, Any],
    image_paths: list[Path],
) -> tuple[dict[str, Any], str]:
    resp = post_ollama_chat(
        base_url=str(args.ollama_url),
        model=str(args.ollama_model),
        prompt=build_description_prompt(item),
        image_paths=image_paths,
        timeout_sec=int(args.ollama_timeout),
        temperature=float(args.ollama_temperature),
        num_ctx=int(args.ollama_num_ctx),
        num_predict=int(args.ollama_num_predict),
        response_format="none",
    )
    raw_text = extract_ollama_content(resp)
    if not raw_text.strip():
        raise RuntimeError("VLM returned empty description")
    try:
        return normalize_description_result(parse_json_object(raw_text)), raw_text
    except Exception:
        return description_from_text(raw_text), raw_text


def validation_rank_score(row: dict[str, Any]) -> float:
    validation = row.get("vlm_validation") or {}
    if row.get("vlm_description") and not validation:
        description = row.get("vlm_description") or {}
        score = 0.0
        if description.get("detailed_description"):
            score += 5.0
        score += min(2.0, len(description.get("materials_visible") or []) * 0.5)
        score += min(2.0, len(description.get("colors_visible") or []) * 0.5)
        score += min(1.0, len(description.get("forms_visible") or []) * 0.25)
        return round(score, 6)
    vlm_style = row.get("vlm_style") or {}
    values = [
        float(validation.get("overall_score") or 0),
        float(validation.get("detail_score") or 0),
        float(validation.get("color_score") or 0),
        float(validation.get("category_score") or 0),
        float(validation.get("style_score") or 0),
    ]
    avg = sum(values) / len(values) if values else 0.0
    confidence = float(vlm_style.get("confidence") or 0.0)
    match_bonus = 0.25 if validation.get("matches_image") else 0.0
    return round(avg + confidence + match_bonus, 6)


def top_description(row: dict[str, Any]) -> str:
    vlm_description = row.get("vlm_description") or {}
    if vlm_description:
        parts = [
            str(vlm_description.get("object_summary") or "").strip(),
            "materials=" + ",".join(vlm_description.get("materials_visible") or []),
            "finish=" + ",".join(vlm_description.get("finish_visible") or []),
            "colors=" + ",".join(vlm_description.get("colors_visible") or []),
            "forms=" + ",".join(vlm_description.get("forms_visible") or []),
            str(vlm_description.get("detailed_description") or "").strip(),
        ]
        return " | ".join(x for x in parts if x)
    vlm_style = row.get("vlm_style") or {}
    validation = row.get("vlm_validation") or {}
    parts = [
        f"object_category={vlm_style.get('object_category')}",
        f"room_type={','.join(vlm_style.get('room_type') or [])}",
        f"interior_style={','.join(vlm_style.get('interior_style') or [])}",
        f"form_style={','.join(vlm_style.get('form_style') or [])}",
        f"finish={','.join(vlm_style.get('finish') or [])}",
        f"color_family={','.join(vlm_style.get('color_family') or [])}",
        f"scores=overall:{validation.get('overall_score')},detail:{validation.get('detail_score')},color:{validation.get('color_score')}",
    ]
    rationale = str(vlm_style.get("rationale") or validation.get("rationale") or "").strip()
    if rationale:
        parts.append(rationale)
    return " | ".join(x for x in parts if x)


def update_top_samples(
    *,
    row: dict[str, Any],
    local_images: list[Path],
    top_samples_dir: Path,
    top_n: int,
) -> None:
    if top_n <= 0 or row.get("status") != "ok" or not local_images:
        return

    top_samples_dir.mkdir(parents=True, exist_ok=True)
    images_dir = top_samples_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = top_samples_dir / "top30.json"

    manifest: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                manifest = [x for x in loaded if isinstance(x, dict)]
        except Exception:
            manifest = []

    iid = safe_fs_name(str(row.get("id") or "item"))
    title = safe_fs_name(str(row.get("title") or "item"), max_len=48)
    src_image = local_images[0]
    ext = src_image.suffix.lower() if src_image.suffix else ".jpg"
    dst_image = images_dir / f"{iid}_{title}{ext}"
    shutil.copy2(src_image, dst_image)

    candidate = {
        "id": row.get("id"),
        "row_index": row.get("row_index"),
        "title": row.get("title"),
        "source_site": row.get("source_site"),
        "unique_key": row.get("unique_key"),
        "category_norm": row.get("category_norm"),
        "rank_score": validation_rank_score(row),
        "image_path": str(dst_image),
        "image_urls": row.get("image_urls") or [],
        "description": top_description(row),
        "vlm_style": row.get("vlm_style"),
        "vlm_validation": row.get("vlm_validation"),
        "vlm_description": row.get("vlm_description"),
    }

    by_id: dict[str, dict[str, Any]] = {
        str(x.get("id")): x for x in manifest if x.get("id") not in (None, "")
    }
    by_id[str(row.get("id"))] = candidate
    ranked = sorted(
        by_id.values(),
        key=lambda x: (float(x.get("rank_score") or 0.0), str(x.get("id") or "")),
        reverse=True,
    )
    keep = ranked[:top_n]
    keep_paths = {str(x.get("image_path") or "") for x in keep}

    for old in ranked[top_n:]:
        old_path = str(old.get("image_path") or "")
        if old_path and old_path not in keep_paths:
            try:
                Path(old_path).unlink(missing_ok=True)
            except Exception:
                pass

    manifest_path.write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl_path = top_samples_dir / "top30.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in keep:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            iid = row.get("id")
            if iid not in (None, ""):
                done.add(str(iid))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def maybe_write_enriched_catalog(
    *,
    input_path: Path,
    results_jsonl: Path,
    output_path: Path,
) -> None:
    catalog = load_catalog(input_path)
    by_id: dict[str, dict[str, Any]] = {}
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("id") not in (None, ""):
                by_id[str(row["id"])] = row

    for idx, item in enumerate(catalog):
        iid = item_id(item, idx)
        result = by_id.get(iid)
        if not result:
            continue
        vlm = result.get("vlm_style") or {}
        item["object_category_vlm"] = vlm.get("object_category")
        item["room_type_vlm_json"] = json.dumps(vlm.get("room_type") or [], ensure_ascii=False)
        item["interior_style_vlm_json"] = json.dumps(vlm.get("interior_style") or [], ensure_ascii=False)
        item["form_style_vlm_json"] = json.dumps(vlm.get("form_style") or [], ensure_ascii=False)
        item["finish_vlm_json"] = json.dumps(vlm.get("finish") or [], ensure_ascii=False)
        item["color_family_vlm_json"] = json.dumps(vlm.get("color_family") or [], ensure_ascii=False)
        item["hardware_style_vlm"] = vlm.get("hardware_style")
        item["installation_type_vlm"] = vlm.get("installation_type")
        item["appliance_style_vlm"] = vlm.get("appliance_style")
        item["confidence_vlm"] = vlm.get("confidence")
        item["rationale_vlm"] = vlm.get("rationale")
        validation = result.get("vlm_validation") or {}
        item["vlm_matches_image"] = validation.get("matches_image")
        item["vlm_overall_score"] = validation.get("overall_score")
        item["vlm_detail_score"] = validation.get("detail_score")
        item["vlm_color_score"] = validation.get("color_score")
        item["vlm_category_score"] = validation.get("category_score")
        item["vlm_style_score"] = validation.get("style_score")
        item["vlm_validation_json"] = json.dumps(validation, ensure_ascii=False)
        item["style_vlm_result_json"] = json.dumps(vlm, ensure_ascii=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download supplier product photos and classify style with a VLM via Ollama.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    p.add_argument("--image-cache-dir", default=DEFAULT_IMAGE_CACHE)
    p.add_argument("--top-samples-dir", default=DEFAULT_TOP_SAMPLES_DIR)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--enriched-output", default=None, help="Optional full catalog JSON with *_vlm fields merged in.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-images", type=int, default=1)
    p.add_argument("--download-timeout", type=int, default=40)
    p.add_argument("--download-retries", type=int, default=3)
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ollama-model", default="qwen2.5vl:7b")
    p.add_argument("--ollama-timeout", type=int, default=240)
    p.add_argument("--ollama-temperature", type=float, default=0.0)
    p.add_argument("--ollama-num-ctx", type=int, default=4096)
    p.add_argument("--ollama-num-predict", type=int, default=384)
    p.add_argument("--ollama-format", choices=["json", "none"], default="json")
    p.add_argument("--parse-retries", type=int, default=1)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--describe-only", action="store_true", help="Only ask VLM for a detailed visual description; skip style labels and validation.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-images", action="store_true", help="Keep downloaded images after successful VLM labeling.")
    p.add_argument("--skip-validation", action="store_true", help="Skip the second VLM audit dialog.")
    return p


def main() -> None:
    args = build_cli().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    image_cache_dir = Path(args.image_cache_dir).expanduser().resolve()
    top_samples_dir = Path(args.top_samples_dir).expanduser().resolve()

    catalog = load_catalog(input_path)
    start = max(0, int(args.offset))
    end = len(catalog) if args.limit is None else min(len(catalog), start + max(0, int(args.limit)))
    done = set() if args.force else load_done_ids(out_jsonl)
    print(
        f"INFO: catalog={input_path} records={len(catalog)} range={start}:{end} "
        f"done={len(done)} out={out_jsonl}",
        flush=True,
    )

    for idx in range(start, end):
        item = catalog[idx]
        iid = item_id(item, idx)
        if iid in done:
            continue
        title = str(item.get("title") or "")[:120]
        if args.dry_run:
            urls = image_urls(item, max_images=int(args.max_images))
            print(f"DRY {idx} id={iid} title={title!r} image_urls={len(urls)}", flush=True)
            continue

        local_images: list[Path] = []
        download_errors: list[dict[str, str]] = []
        status = "ok"
        error = ""
        vlm_style: dict[str, Any] | None = None
        vlm_validation: dict[str, Any] | None = None
        vlm_description: dict[str, Any] | None = None
        raw_text = ""
        raw_validation_text = ""
        try:
            local_images, download_errors = cached_downloads(
                item=item,
                index=idx,
                image_cache_dir=image_cache_dir,
                max_images=int(args.max_images),
                timeout_sec=int(args.download_timeout),
                retries=int(args.download_retries),
            )
            if not local_images:
                status = "no_image"
            elif args.download_only:
                status = "downloaded"
            elif args.describe_only:
                vlm_description, raw_text = call_vlm_description(
                    args=args,
                    item=item,
                    image_paths=local_images,
                )
            else:
                parsed, raw_text = call_vlm_json(
                    args=args,
                    item=item,
                    image_paths=local_images,
                    prompt=build_prompt(item),
                    mode="classification",
                )
                vlm_style = normalize_vlm_result(parsed)
                if args.skip_validation:
                    vlm_validation = {
                        "matches_image": None,
                        "overall_score": None,
                        "detail_score": None,
                        "color_score": None,
                        "category_score": None,
                        "style_score": None,
                        "issues": [],
                        "corrected_summary": "",
                        "rationale": "validation skipped",
                    }
                else:
                    validation_parsed, raw_validation_text = call_vlm_json(
                        args=args,
                        item=item,
                        image_paths=local_images,
                        prompt=build_validation_prompt(item, vlm_style),
                        mode="validation",
                        vlm_style=vlm_style,
                    )
                    vlm_validation = normalize_validation_result(validation_parsed)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        row = {
            "id": iid,
            "row_index": idx,
            "status": status,
            "error": error,
            "title": item.get("title"),
            "source_site": item.get("source_site"),
            "unique_key": item.get("unique_key"),
            "category_norm": item.get("category_norm"),
            "existing_style": item.get("style"),
            "existing_style_primary_web": item.get("style_primary_web"),
            "existing_style_family_web": item.get("style_family_web"),
            "image_urls": image_urls(item, max_images=int(args.max_images)),
            "local_images": [str(p) for p in local_images],
            "download_errors": download_errors,
            "vlm_description": vlm_description,
            "vlm_style": vlm_style,
            "vlm_validation": vlm_validation,
            "raw_vlm_text": raw_text,
            "raw_validation_text": raw_validation_text,
            "model": str(args.ollama_model),
            "processed_at_unix": time.time(),
        }
        append_jsonl(out_jsonl, row)
        update_top_samples(
            row=row,
            local_images=local_images,
            top_samples_dir=top_samples_dir,
            top_n=int(args.top_n),
        )
        if status == "ok" and not args.keep_images:
            for local_path in local_images:
                try:
                    local_path.unlink(missing_ok=True)
                except Exception:
                    pass
        print(f"{status.upper()} {idx + 1}/{len(catalog)} id={iid} title={title!r}", flush=True)
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    if args.enriched_output:
        maybe_write_enriched_catalog(
            input_path=input_path,
            results_jsonl=out_jsonl,
            output_path=Path(args.enriched_output).expanduser().resolve(),
        )
        print(f"OK: enriched catalog -> {args.enriched_output}", flush=True)


if __name__ == "__main__":
    main()
