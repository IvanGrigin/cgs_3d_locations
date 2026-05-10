#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_PROVIDERS = {"openai", "openrouter", "ollama", "none"}
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"


@dataclass(frozen=True)
class SceneObjectRef:
    object_id: str
    category: str
    name: str
    path: tuple[Any, ...]
    data: dict[str, Any]
    yaw_deg: float | None
    position_xy: tuple[float | None, float | None]
    size_xy: tuple[float | None, float | None]


@dataclass(frozen=True)
class OrientationDecision:
    object_id: str
    object_name: str
    action: str
    clockwise_delta_deg: float | None
    target_yaw_deg: float | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class VlmReviewResult:
    summary: str
    decisions: list[OrientationDecision]
    raw_response: dict[str, Any]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _get_first(d: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _norm_angle_deg(angle: float) -> float:
    angle = float(angle) % 360.0
    if angle < 0:
        angle += 360.0
    return round(angle, 6)


def _angle_delta_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _extract_yaw_deg(obj: dict[str, Any]) -> float | None:
    direct = _as_float(_get_first(obj, ["yaw_deg", "rotation_deg", "rotation_z_deg", "angle_deg", "theta_deg"]))
    if direct is not None:
        return _norm_angle_deg(direct)

    yaw_rad = _as_float(_get_first(obj, ["yaw_rad", "yaw", "rotation_z", "angle", "theta"]))
    if yaw_rad is not None:
        return _norm_angle_deg(math.degrees(yaw_rad))

    rotation = obj.get("rotation")
    if isinstance(rotation, dict):
        direct = _as_float(_get_first(rotation, ["yaw_deg", "z_deg", "rotation_z_deg"]))
        if direct is not None:
            return _norm_angle_deg(direct)
        rad = _as_float(_get_first(rotation, ["yaw", "z", "rotation_z"]))
        if rad is not None:
            return _norm_angle_deg(math.degrees(rad))

    rotation_euler = obj.get("rotation_euler")
    if isinstance(rotation_euler, dict):
        direct = _as_float(_get_first(rotation_euler, ["z_deg", "yaw_deg"]))
        if direct is not None:
            return _norm_angle_deg(direct)
        rad = _as_float(_get_first(rotation_euler, ["z", "yaw"]))
        if rad is not None:
            return _norm_angle_deg(math.degrees(rad))
    if isinstance(rotation_euler, list) and len(rotation_euler) >= 3:
        rad = _as_float(rotation_euler[2])
        if rad is not None:
            return _norm_angle_deg(math.degrees(rad))

    transform = obj.get("transform")
    if isinstance(transform, dict):
        return _extract_yaw_deg(transform)

    return None


def _set_yaw_deg(obj: dict[str, Any], yaw_deg: float) -> str:
    yaw_deg = _norm_angle_deg(yaw_deg)
    yaw_rad = math.radians(yaw_deg)

    updated: list[str] = []
    for key in ("yaw_deg", "rotation_deg", "rotation_z_deg", "angle_deg", "theta_deg"):
        if key in obj:
            obj[key] = yaw_deg
            updated.append(key)
    for key in ("yaw_rad", "yaw", "rotation_z", "angle", "theta"):
        if key in obj:
            obj[key] = yaw_rad
            updated.append(key)
    if updated:
        return "/".join(updated)

    rotation = obj.get("rotation")
    if isinstance(rotation, dict):
        for key in ("yaw_deg", "z_deg", "rotation_z_deg"):
            if key in rotation:
                rotation[key] = yaw_deg
                updated.append(f"rotation.{key}")
        for key in ("yaw", "z", "rotation_z"):
            if key in rotation:
                rotation[key] = yaw_rad
                updated.append(f"rotation.{key}")
        if updated:
            return "/".join(updated)

    rotation_euler = obj.get("rotation_euler")
    if isinstance(rotation_euler, dict):
        for key in ("z_deg", "yaw_deg"):
            if key in rotation_euler:
                rotation_euler[key] = yaw_deg
                updated.append(f"rotation_euler.{key}")
        for key in ("z", "yaw"):
            if key in rotation_euler:
                rotation_euler[key] = yaw_rad
                updated.append(f"rotation_euler.{key}")
        if updated:
            return "/".join(updated)

    if isinstance(rotation_euler, list) and len(rotation_euler) >= 3:
        rotation_euler[2] = yaw_rad
        return "rotation_euler[2]"

    transform = obj.get("transform")
    if isinstance(transform, dict):
        field = _set_yaw_deg(transform, yaw_deg)
        return f"transform.{field}"

    obj["yaw_deg"] = yaw_deg
    obj["rotation_deg"] = yaw_deg
    obj["yaw_rad"] = yaw_rad
    return "yaw_deg/rotation_deg/yaw_rad"


def _extract_xy_from_mapping(d: dict[str, Any], keys: Iterable[str]) -> tuple[float | None, float | None]:
    for key in keys:
        value = d.get(key)
        if isinstance(value, dict):
            x = _as_float(value.get("x"))
            y = _as_float(value.get("y"))
            if x is not None or y is not None:
                return x, y
        if isinstance(value, list) and len(value) >= 2:
            x = _as_float(value[0])
            y = _as_float(value[1])
            if x is not None or y is not None:
                return x, y
    return None, None


def _extract_position_xy(obj: dict[str, Any]) -> tuple[float | None, float | None]:
    x = _as_float(_get_first(obj, ["x", "cx", "center_x"]))
    y = _as_float(_get_first(obj, ["y", "cy", "center_y"]))
    if x is not None or y is not None:
        return x, y

    xy = _extract_xy_from_mapping(obj, ["position_m", "position", "center_m", "center", "location"])
    if xy != (None, None):
        return xy

    transform = obj.get("transform")
    if isinstance(transform, dict):
        return _extract_position_xy(transform)
    return None, None


def _extract_size_xy(obj: dict[str, Any]) -> tuple[float | None, float | None]:
    w = _as_float(_get_first(obj, ["width_m", "width", "w", "sx", "size_x", "depth_x"]))
    d = _as_float(_get_first(obj, ["depth_m", "depth", "d", "sy", "size_y", "depth_y"]))
    if w is not None or d is not None:
        return w, d

    xy = _extract_xy_from_mapping(obj, ["size_m", "size", "dimensions_m", "dimensions", "bbox_size_m"])
    if xy != (None, None):
        return xy

    aabb = obj.get("aabb_m") or obj.get("aabb")
    if isinstance(aabb, dict):
        min_v = aabb.get("min") or aabb.get("min_m")
        max_v = aabb.get("max") or aabb.get("max_m")
        if isinstance(min_v, dict) and isinstance(max_v, dict):
            min_x = _as_float(min_v.get("x"))
            min_y = _as_float(min_v.get("y"))
            max_x = _as_float(max_v.get("x"))
            max_y = _as_float(max_v.get("y"))
            if None not in (min_x, min_y, max_x, max_y):
                return abs(max_x - min_x), abs(max_y - min_y)

    transform = obj.get("transform")
    if isinstance(transform, dict):
        return _extract_size_xy(transform)
    return None, None


def _extract_object_id(obj: dict[str, Any], fallback: str) -> str:
    value = _get_first(obj, ["id", "object_id", "instance_id", "uid", "name"])
    return fallback if value is None else str(value)


def _extract_category(obj: dict[str, Any]) -> str:
    value = _get_first(obj, ["category", "category_norm", "semantic", "semantic_group", "type", "class", "label"])
    return "unknown" if value is None else str(value)


def _extract_name(obj: dict[str, Any]) -> str:
    value = _get_first(obj, ["name", "title", "asset_name", "model_name", "label"])
    return "" if value is None else str(value)


def _iter_candidate_lists(root: Any, path: tuple[Any, ...] = ()) -> Iterable[tuple[tuple[Any, ...], list[Any]]]:
    if isinstance(root, dict):
        for key, value in root.items():
            next_path = path + (key,)
            if isinstance(value, list):
                yield next_path, value
            yield from _iter_candidate_lists(value, next_path)
    elif isinstance(root, list):
        for i, value in enumerate(root):
            yield from _iter_candidate_lists(value, path + (i,))


def collect_scene_objects(scene: dict[str, Any], max_objects: int = 10000) -> list[SceneObjectRef]:
    preferred_names = {
        "placements",
        "objects",
        "items",
        "placed_items",
        "furniture",
        "scene_objects",
        "layout_objects",
        "assets",
    }

    best_list_path: tuple[Any, ...] | None = None
    best_list: list[Any] | None = None
    best_score = -1

    for path, value in _iter_candidate_lists(scene):
        dict_items = [x for x in value if isinstance(x, dict)]
        if not dict_items:
            continue
        score = len(dict_items)
        if path and str(path[-1]) in preferred_names:
            score += 10000
        score += sum(1 for x in dict_items if _extract_yaw_deg(x) is not None)
        score += sum(1 for x in dict_items if _extract_position_xy(x) != (None, None))
        if score > best_score:
            best_score = score
            best_list_path = path
            best_list = value

    if best_list_path is None or best_list is None:
        return []

    refs: list[SceneObjectRef] = []
    for i, obj in enumerate(best_list):
        if not isinstance(obj, dict):
            continue
        refs.append(
            SceneObjectRef(
                object_id=_extract_object_id(obj, fallback=f"object_{i:04d}"),
                category=_extract_category(obj),
                name=_extract_name(obj),
                path=best_list_path + (i,),
                data=obj,
                yaw_deg=_extract_yaw_deg(obj),
                position_xy=_extract_position_xy(obj),
                size_xy=_extract_size_xy(obj),
            )
        )
        if len(refs) >= max_objects:
            break
    return refs


def _semantic_blob(ref: SceneObjectRef) -> str:
    chunks = [ref.object_id, ref.category, ref.name]
    source = ref.data.get("source")
    if isinstance(source, dict):
        chunks.extend(str(v or "") for v in source.values())
    meta = ref.data.get("meta")
    if isinstance(meta, dict):
        chunks.extend(str(meta.get(k) or "") for k in ("affordance", "semantic_group", "target_table_id"))
        candidate = meta.get("supplier_candidate")
        if isinstance(candidate, dict):
            chunks.extend(str(candidate.get(k) or "") for k in ("category_norm", "semantic_group", "title"))
    return " ".join(chunks).lower()


def is_chair_like(ref: SceneObjectRef, *, include_armchairs: bool = False) -> bool:
    blob = _semantic_blob(ref)
    compact = re.sub(r"[^a-zа-я0-9]+", " ", blob)
    words = set(compact.split())
    if "armchairfactory" in blob or "armchair" in words or "кресло" in words or "кресл" in blob:
        return include_armchairs
    if "chairfactory" in blob or "chair" in words or "стул" in words or "стуль" in blob:
        return True
    if "dining chair" in blob or "office chair" in blob or "table_chair" in blob:
        return True
    return False


def _is_context_ref(ref: SceneObjectRef) -> bool:
    blob = _semantic_blob(ref)
    tokens = (
        "table",
        "desk",
        "nightstand",
        "bed",
        "sofa",
        "cabinet",
        "wardrobe",
        "dresser",
        "tv",
        "телевиз",
        "стол",
        "кровать",
        "шкаф",
        "тумб",
    )
    return any(token in blob for token in tokens)


def filter_target_objects(
    objects: list[SceneObjectRef],
    *,
    scope: str,
    include_armchairs: bool = False,
) -> list[SceneObjectRef]:
    scope = str(scope or "chairs").strip().lower()
    if scope == "all":
        return list(objects)
    if scope == "chairs":
        return [obj for obj in objects if is_chair_like(obj, include_armchairs=include_armchairs)]
    raise ValueError(f"Unsupported object scope: {scope}")


def _get_by_path(root: Any, path: tuple[Any, ...]) -> Any:
    cur = root
    for p in path:
        cur = cur[p]
    return cur


def image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _compact_ref(ref: SceneObjectRef) -> dict[str, Any]:
    return {
        "object_id": ref.object_id,
        "category": ref.category,
        "name": ref.name,
        "yaw_deg": ref.yaw_deg,
        "position_xy_m": ref.position_xy,
        "size_xy_m": ref.size_xy,
    }


def build_vlm_prompt(
    scene: dict[str, Any],
    target_objects: list[SceneObjectRef],
    context_objects: list[SceneObjectRef],
    *,
    scope: str,
) -> str:
    room = scene.get("room", {}) if isinstance(scene.get("room"), dict) else {}
    room_type = room.get("type") or room.get("room_type") or room.get("name") or scene.get("room_type") or "unknown"

    target_payload = [_compact_ref(obj) for obj in target_objects]
    context_payload = [_compact_ref(obj) for obj in context_objects[:80]]
    target_ids = [obj.object_id for obj in target_objects]

    return (
        "You are checking a top-view render of an automatically generated interior layout.\n"
        "Task: find only orientation problems of the allowed target objects. Do not propose moving, deleting, adding, or resizing objects.\n"
        "This first repair pass is chair-only. Return decisions only for the target chair objects listed below.\n"
        "In the top-view image, target chairs may be highlighted, labeled by object_id, and drawn with an arrow showing current facing direction.\n"
        "Chair rules:\n"
        "- A desk chair should face the desk/table, not the wall or open walkway.\n"
        "- A dining or side chair should face the matching table edge.\n"
        "- If a chair is already plausible, keep it.\n"
        "- If a chair is wrong, first describe what is wrong, then provide how many degrees it should rotate clockwise.\n"
        "- If you cannot infer the global yaw confidently, use rotate_clockwise with rotate_clockwise_deg=90, 180, or 270.\n"
        "Return strict JSON only. No markdown. No comments outside JSON.\n"
        "JSON schema:\n"
        "{\n"
        "  \"summary\": \"short diagnosis\",\n"
        "  \"problems\": [\"short human-readable issues\"],\n"
        "  \"objects\": [\n"
        "    {\n"
        "      \"object_id\": \"id from target list\",\n"
        "      \"object_name\": \"name from target list\",\n"
        "      \"problem\": \"what is wrong with this object orientation, or empty for keep\",\n"
        "      \"action\": \"keep|rotate_clockwise|set_yaw\",\n"
        "      \"rotate_clockwise_deg\": 0,\n"
        "      \"target_yaw_deg\": 0,\n"
        "      \"confidence\": 0.0,\n"
        "      \"reason\": \"why this orientation should be changed or kept\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Include only these target IDs: "
        f"{json.dumps(target_ids, ensure_ascii=False)}\n"
        "- rotate_clockwise_deg must be one of 0, 90, 180, 270. Use 0 for keep.\n"
        "- target_yaw_deg may be null unless action=set_yaw.\n"
        "- Use target_yaw_deg in [0, 360) only when action=set_yaw and you are confident about global axes.\n"
        "- Prefer keep when unsure. confidence must be between 0 and 1.\n"
        f"Room type: {room_type}\n"
        f"Scope: {scope}\n"
        "Target objects allowed to modify:\n"
        f"{json.dumps(target_payload, ensure_ascii=False, indent=2)}\n"
        "Context objects for orientation reasoning:\n"
        f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n"
    )


def _env_value(names: Iterable[str]) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _openrouter_keys_from_env() -> list[str]:
    keys: list[str] = []
    direct = _env_value(["OPENROUTER_API_KEY", "ivangrigin_OPENROUTER_API_KEY"])
    if direct:
        keys.append(direct)
    indexed: list[tuple[int, str]] = []
    for key, value in os.environ.items():
        match = re.fullmatch(r"ivangrigin_OPENROUTER_API_KEY_(\d+)", key)
        if match and value:
            indexed.append((int(match.group(1)), value))
    keys.extend(value for _, value in sorted(indexed))
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _openrouter_key_from_env() -> str:
    keys = _openrouter_keys_from_env()
    return keys[0] if keys else ""


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    for dotenv_path in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if not dotenv_path.is_file():
            continue
        try:
            lines = dotenv_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _provider_config(provider: str, model: str | None) -> tuple[str, str, str]:
    _load_dotenv_once()
    provider = provider.lower().strip()
    if provider == "openai":
        api_key = _env_value(["OPENAI_API_KEY"])
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return "https://api.openai.com/v1/chat/completions", api_key, model or DEFAULT_OPENAI_MODEL
    if provider == "openrouter":
        api_key = _openrouter_key_from_env()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY or ivangrigin_OPENROUTER_API_KEY_* is not set")
        return "https://openrouter.ai/api/v1/chat/completions", api_key, model or DEFAULT_OPENROUTER_MODEL
    raise ValueError(f"Unsupported VLM provider: {provider}")


def call_openai_compatible_vlm(
    *,
    provider: str,
    model: str | None,
    prompt: str,
    image_path: Path,
    temperature: float = 0.0,
    max_tokens: int = 1800,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    provider = provider.lower().strip()
    if provider == "openrouter":
        _load_dotenv_once()
        keys = _openrouter_keys_from_env()
        if not keys:
            raise RuntimeError("OPENROUTER_API_KEY or ivangrigin_OPENROUTER_API_KEY_* is not set")
        configs = [("https://openrouter.ai/api/v1/chat/completions", key, model or DEFAULT_OPENROUTER_MODEL) for key in keys]
    else:
        configs = [_provider_config(provider, model)]

    last_error: str | None = None
    for key_index, (endpoint, api_key, resolved_model) in enumerate(configs, start=1):
        payload = {
            "model": resolved_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/IvanGrigin/cgs_3d_locations"
            headers["X-Title"] = "cgs_3d_locations topview repair"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {e.code}: {detail}"
            if provider == "openrouter" and e.code in {401, 402, 429} and key_index < len(configs):
                continue
            raise RuntimeError(f"VLM request failed: {last_error}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"VLM request failed: {e}") from e
    raise RuntimeError(f"VLM request failed: {last_error or 'all keys failed'}")


def call_ollama_vlm(
    *,
    model: str | None,
    prompt: str,
    image_path: Path,
    temperature: float = 0.0,
    timeout_sec: int = 240,
) -> dict[str, Any]:
    _load_dotenv_once()
    base_url = (
        os.environ.get("TOPVIEW_VLM_OLLAMA_URL")
        or os.environ.get("OLLAMA_URL")
        or "http://127.0.0.1:11435"
    ).rstrip("/")
    payload = {
        "model": model or DEFAULT_OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_to_base64(image_path)],
            }
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama VLM request failed: HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama VLM request failed: {e}") from e
    data = json.loads(raw)
    content = ""
    message = data.get("message")
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    return {
        "choices": [{"message": {"content": content}}],
        "ollama_response": data,
    }


def _extract_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError("VLM response does not contain a JSON object")


def parse_vlm_response(api_response: dict[str, Any]) -> VlmReviewResult:
    choices = api_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("VLM response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        content_text = "\n".join(text_parts)
    else:
        content_text = str(content or "")

    parsed = json.loads(_extract_json_text(content_text))
    summary = str(parsed.get("summary", ""))
    items = parsed.get("objects", [])
    if not isinstance(items, list):
        items = []

    decisions: list[OrientationDecision] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id", "")).strip()
        if not object_id:
            continue
        target_yaw_deg = _as_float(item.get("target_yaw_deg"))
        clockwise_delta_deg = _as_float(item.get("clockwise_delta_deg") if "clockwise_delta_deg" in item else item.get("rotate_clockwise_deg"))
        decisions.append(
            OrientationDecision(
                object_id=object_id,
                object_name=str(item.get("object_name") or item.get("name") or ""),
                action=str(item.get("action", "keep")).strip().lower(),
                clockwise_delta_deg=_norm_angle_deg(clockwise_delta_deg) if clockwise_delta_deg is not None else None,
                target_yaw_deg=_norm_angle_deg(target_yaw_deg) if target_yaw_deg is not None else None,
                confidence=max(0.0, min(1.0, _as_float(item.get("confidence")) or 0.0)),
                reason=str(item.get("reason") or item.get("problem") or ""),
            )
        )
    return VlmReviewResult(summary=summary, decisions=decisions, raw_response=parsed)


def load_review_from_file(path: Path) -> VlmReviewResult:
    raw = load_json(path)
    if isinstance(raw, dict) and "choices" in raw:
        return parse_vlm_response(raw)
    summary = str(raw.get("summary", "")) if isinstance(raw, dict) else ""
    items = raw.get("objects", []) if isinstance(raw, dict) else []
    decisions: list[OrientationDecision] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "")).strip()
            if not object_id:
                continue
            target = _as_float(item.get("target_yaw_deg"))
            clockwise_delta = _as_float(item.get("clockwise_delta_deg") if "clockwise_delta_deg" in item else item.get("rotate_clockwise_deg"))
            decisions.append(
                OrientationDecision(
                    object_id=object_id,
                    object_name=str(item.get("object_name") or item.get("name") or ""),
                    action=str(item.get("action", "keep")).strip().lower(),
                    clockwise_delta_deg=_norm_angle_deg(clockwise_delta) if clockwise_delta is not None else None,
                    target_yaw_deg=_norm_angle_deg(target) if target is not None else None,
                    confidence=max(0.0, min(1.0, _as_float(item.get("confidence")) or 0.0)),
                    reason=str(item.get("reason") or item.get("problem") or ""),
                )
            )
    return VlmReviewResult(summary=summary, decisions=decisions, raw_response=raw)


def _snap_yaw(yaw_deg: float, step_deg: float) -> float:
    if step_deg <= 0:
        return _norm_angle_deg(yaw_deg)
    return _norm_angle_deg(round(yaw_deg / step_deg) * step_deg)


def _decision_target_yaw(decision: OrientationDecision, current_yaw: float | None) -> float | None:
    action = decision.action
    if decision.clockwise_delta_deg is not None and action in {"rotate_clockwise", "clockwise", "rotate"}:
        if current_yaw is None:
            return decision.target_yaw_deg
        # In room/Blender top-down coordinates, positive yaw is counter-clockwise.
        # A user-facing clockwise delta therefore subtracts from current yaw.
        return _norm_angle_deg(current_yaw - decision.clockwise_delta_deg)
    if action == "set_yaw":
        return decision.target_yaw_deg
    if current_yaw is None:
        return decision.target_yaw_deg
    if action == "rotate_90":
        return _norm_angle_deg(current_yaw + 90.0)
    if action == "rotate_minus_90":
        return _norm_angle_deg(current_yaw - 90.0)
    if action == "rotate_180":
        return _norm_angle_deg(current_yaw + 180.0)
    if decision.target_yaw_deg is not None and action not in {"keep"}:
        return decision.target_yaw_deg
    return None


def apply_orientation_decisions(
    scene: dict[str, Any],
    decisions: list[OrientationDecision],
    *,
    target_scope: str = "chairs",
    include_armchairs: bool = False,
    min_confidence: float = 0.70,
    max_delta_deg: float = 180.0,
    snap_step_deg: float = 90.0,
    keep_large_delta_only_if_confident: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(scene)
    refs = collect_scene_objects(result, max_objects=10000)
    allowed_refs = filter_target_objects(refs, scope=target_scope, include_armchairs=include_armchairs)
    by_id = {ref.object_id: ref for ref in allowed_refs}

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for decision in decisions:
        if decision.object_id not in by_id:
            skipped.append({"object_id": decision.object_id, "reason": "object_id_not_allowed_or_not_found", "decision": asdict(decision)})
            continue
        if decision.action == "keep":
            skipped.append({"object_id": decision.object_id, "reason": "action_keep", "decision": asdict(decision)})
            continue
        if decision.confidence < min_confidence:
            skipped.append(
                {
                    "object_id": decision.object_id,
                    "reason": "low_confidence",
                    "min_confidence": min_confidence,
                    "decision": asdict(decision),
                }
            )
            continue

        ref = by_id[decision.object_id]
        current_yaw = ref.yaw_deg
        target_yaw = _decision_target_yaw(decision, current_yaw)
        if target_yaw is None:
            skipped.append({"object_id": decision.object_id, "reason": "missing_target_yaw_deg", "decision": asdict(decision)})
            continue
        target_yaw = _snap_yaw(target_yaw, snap_step_deg)

        if current_yaw is not None:
            delta = _angle_delta_deg(target_yaw, current_yaw)
            if delta > max_delta_deg:
                skipped.append(
                    {
                        "object_id": decision.object_id,
                        "reason": "delta_too_large",
                        "current_yaw_deg": current_yaw,
                        "target_yaw_deg": target_yaw,
                        "delta_deg": delta,
                        "max_delta_deg": max_delta_deg,
                        "decision": asdict(decision),
                    }
                )
                continue
            if keep_large_delta_only_if_confident and delta >= 135.0 and decision.confidence < 0.85:
                skipped.append(
                    {
                        "object_id": decision.object_id,
                        "reason": "large_delta_requires_higher_confidence",
                        "current_yaw_deg": current_yaw,
                        "target_yaw_deg": target_yaw,
                        "delta_deg": delta,
                        "decision": asdict(decision),
                    }
                )
                continue

        obj = _get_by_path(result, ref.path)
        if not isinstance(obj, dict):
            skipped.append({"object_id": decision.object_id, "reason": "path_does_not_resolve_to_object", "decision": asdict(decision)})
            continue

        field = _set_yaw_deg(obj, target_yaw)
        applied.append(
            {
                "object_id": decision.object_id,
                "category": ref.category,
                "name": ref.name,
                "field": field,
                "current_yaw_deg": current_yaw,
                "target_yaw_deg": target_yaw,
                "confidence": decision.confidence,
                "action": decision.action,
                "reason": decision.reason,
            }
        )

    report = {
        "stage": "topview_vlm_orientation_repair",
        "created_at_unix": int(time.time()),
        "policy": {
            "target_scope": target_scope,
            "include_armchairs": include_armchairs,
            "min_confidence": min_confidence,
            "max_delta_deg": max_delta_deg,
            "snap_step_deg": snap_step_deg,
            "keep_large_delta_only_if_confident": keep_large_delta_only_if_confident,
            "orientation_only": True,
        },
        "counts": {
            "target_objects": len(allowed_refs),
            "decisions": len(decisions),
            "applied": len(applied),
            "skipped": len(skipped),
        },
        "target_object_ids": [ref.object_id for ref in allowed_refs],
        "applied": applied,
        "skipped": skipped,
    }
    return result, report


def run_topview_vlm_orientation_repair(
    *,
    scene_path: Path,
    image_path: Path,
    out_scene_path: Path,
    out_review_path: Path,
    out_report_path: Path,
    provider: str,
    model: str | None,
    max_objects: int,
    target_scope: str,
    include_armchairs: bool,
    min_confidence: float,
    max_delta_deg: float,
    snap_step_deg: float,
    out_prompt_path: Path | None = None,
    review_json_path: Path | None = None,
    apply: bool = True,
) -> dict[str, Any]:
    scene = load_json(scene_path)
    if not isinstance(scene, dict):
        raise ValueError("Scene root must be a JSON object")

    all_objects = collect_scene_objects(scene, max_objects=max_objects)
    target_objects = filter_target_objects(all_objects, scope=target_scope, include_armchairs=include_armchairs)
    context_objects = [obj for obj in all_objects if obj.object_id not in {t.object_id for t in target_objects} and _is_context_ref(obj)]
    prompt = build_vlm_prompt(scene, target_objects, context_objects, scope=target_scope) if target_objects else ""
    if out_prompt_path is not None:
        out_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        out_prompt_path.write_text(prompt, encoding="utf-8")

    if not target_objects:
        review = VlmReviewResult(
            summary=f"no target objects for scope={target_scope}",
            decisions=[],
            raw_response={"summary": f"no target objects for scope={target_scope}", "objects": []},
        )
        write_json(out_review_path, review.raw_response)
    elif review_json_path is not None:
        review = load_review_from_file(review_json_path)
        write_json(out_review_path, review.raw_response)
    elif provider == "none":
        review = VlmReviewResult(
            summary="provider=none; VLM call skipped",
            decisions=[],
            raw_response={"summary": "provider=none; VLM call skipped", "objects": []},
        )
        write_json(out_review_path, review.raw_response)
    elif provider == "ollama":
        api_response = call_ollama_vlm(
            model=model,
            prompt=prompt,
            image_path=image_path,
        )
        review = parse_vlm_response(api_response)
        write_json(out_review_path, review.raw_response)
    else:
        api_response = call_openai_compatible_vlm(
            provider=provider,
            model=model,
            prompt=prompt,
            image_path=image_path,
        )
        review = parse_vlm_response(api_response)
        write_json(out_review_path, review.raw_response)

    if apply:
        repaired_scene, report = apply_orientation_decisions(
            scene,
            review.decisions,
            target_scope=target_scope,
            include_armchairs=include_armchairs,
            min_confidence=min_confidence,
            max_delta_deg=max_delta_deg,
            snap_step_deg=snap_step_deg,
        )
    else:
        repaired_scene = scene
        report = {
            "stage": "topview_vlm_orientation_repair",
            "created_at_unix": int(time.time()),
            "apply": False,
            "policy": {
                "target_scope": target_scope,
                "include_armchairs": include_armchairs,
                "orientation_only": True,
            },
            "counts": {
                "target_objects": len(target_objects),
                "decisions": len(review.decisions),
                "applied": 0,
                "skipped": len(review.decisions),
            },
            "target_object_ids": [ref.object_id for ref in target_objects],
            "skipped": [asdict(d) for d in review.decisions],
        }

    report["summary"] = review.summary
    report["input_scene"] = str(scene_path)
    report["input_topview_image"] = str(image_path)
    report["output_scene"] = str(out_scene_path)
    report["output_review"] = str(out_review_path)
    report["provider"] = provider
    report["model"] = model

    write_json(out_scene_path, repaired_scene)
    write_json(out_report_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run top-view VLM orientation review and apply orientation-only scene repair.")
    parser.add_argument("--scene", required=True, type=Path, help="Input scene.v1 JSON path")
    parser.add_argument("--topview-image", required=True, type=Path, help="Top-view render PNG/JPG path")
    parser.add_argument("--out-scene", required=True, type=Path, help="Output repaired scene.v1 JSON path")
    parser.add_argument("--out-review", required=True, type=Path, help="Output VLM review JSON path")
    parser.add_argument("--out-report", required=True, type=Path, help="Output repair report JSON path")
    parser.add_argument("--out-prompt", type=Path, default=None, help="Output prompt text sent to VLM")
    parser.add_argument("--provider", choices=sorted(SUPPORTED_PROVIDERS), default="none")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-objects", type=int, default=10000)
    parser.add_argument("--target-scope", choices=["chairs", "all"], default="chairs")
    parser.add_argument("--include-armchairs", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--max-delta-deg", type=float, default=180.0)
    parser.add_argument("--snap-step-deg", type=float, default=90.0)
    parser.add_argument("--review-json", type=Path, default=None, help="Use already saved review JSON instead of calling VLM")
    parser.add_argument("--no-apply", action="store_true", help="Write review/report but keep scene unchanged")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_topview_vlm_orientation_repair(
        scene_path=args.scene,
        image_path=args.topview_image,
        out_scene_path=args.out_scene,
        out_review_path=args.out_review,
        out_report_path=args.out_report,
        out_prompt_path=args.out_prompt or args.out_report.with_suffix(".prompt.txt"),
        provider=args.provider,
        model=args.model,
        max_objects=args.max_objects,
        target_scope=args.target_scope,
        include_armchairs=bool(args.include_armchairs),
        min_confidence=args.min_confidence,
        max_delta_deg=args.max_delta_deg,
        snap_step_deg=args.snap_step_deg,
        review_json_path=args.review_json,
        apply=not args.no_apply,
    )
    print(json.dumps(report.get("counts", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
