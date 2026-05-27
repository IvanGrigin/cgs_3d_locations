from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .kitchen_catalog_loader import load_kitchen_material_catalog
from .kitchen_pipeline import generate_kitchen_variants


DEFAULT_MATERIAL_CATALOG = "data/floor_materials/basisrf/basisrf_surface_materials.jsonl"
DEFAULT_APPLIANCE_CATALOG = "data/sourse/suppliers/supplier_catalog_canonical.json"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_RENDER_SCRIPT = "src/tools/render_kitchen_assembly_blender.py"
DEFAULT_FLOORING_MATERIALS = Path("data/floor_materials")
DEFAULT_FLOORING_STYLE_RULES = Path("config/flooring_style_rules.json")
DEFAULT_WALL_MATERIALS = Path("data/floor_materials")
KITCHEN_MIN_WIDTH_M = 1.5
KITCHEN_MAX_AUTO_WIDTH_M = 3.6
KITCHEN_DOOR_SWING_CLEARANCE_M = 1.05
KITCHEN_CABINET_DEPTH_M = 0.60
SUPPORTED_DINING_ASSET_SUFFIXES = {".fbx", ".obj", ".glb", ".gltf"}
LOCAL_DINING_TABLE_ROOT = Path("data/sourse/imodern")


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_material_by_sku(material_catalog: str | Path, sku: str | None) -> dict[str, Any] | None:
    if not sku:
        return None
    materials = load_kitchen_material_catalog(material_catalog)
    for material in materials:
        if material.get("sku") == sku:
            return material
    raise ValueError(f"facade_sku_not_found:{sku}")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_room(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("room"), dict):
        return data["room"]
    return data if isinstance(data, dict) else {}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("ё", "е").lower()).strip()


def _load_supplier_catalog(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    catalog_path = Path(path).expanduser()
    if not catalog_path.exists():
        return []
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = data.get("items") if isinstance(data, dict) else data
    result = [row for row in (rows or []) if isinstance(row, dict)]
    result.extend(_local_dining_supplier_candidates())
    return result


def _find_preferred_local_mesh(root: Path) -> Path | None:
    if not root.exists():
        return None
    for suffix in (".glb", ".fbx", ".obj", ".gltf"):
        matches = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == suffix)
        if matches:
            return matches[0]
    return None


def _local_dining_supplier_candidates() -> list[dict[str, Any]]:
    root = LOCAL_DINING_TABLE_ROOT
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        text = _norm(folder.name)
        is_table = ("стол" in text or "table" in text) and not any(
            term in text for term in ("журн", "coffee", "письмен", "desk", "лампа", "lamp")
        )
        is_chair = ("стул" in text or "chair" in text) and not any(
            term in text for term in ("барный", "полубарный", "bar", "counter", "office", "офис")
        )
        if not is_table and not is_chair:
            continue
        mesh = _find_preferred_local_mesh(folder)
        if mesh is None:
            continue
        key = str(mesh.resolve())
        if key in seen:
            continue
        seen.add(key)
        nums = [float(x) for x in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", folder.name)]
        if is_table:
            width_cm = nums[0] if nums else 140.0
            depth_cm = nums[1] if len(nums) > 1 and nums[1] <= 120.0 else min(90.0, max(70.0, width_cm * 0.62))
            height_cm = 76.0
            category_norm = "dining_table"
            category_raw = "Столы"
            unique_prefix = "local_imodern_table"
        else:
            width_cm = nums[0] if nums and 35.0 <= nums[0] <= 75.0 else 48.0
            depth_cm = nums[1] if len(nums) > 1 and 35.0 <= nums[1] <= 75.0 else 54.0
            height_cm = nums[2] if len(nums) > 2 and 65.0 <= nums[2] <= 110.0 else 84.0
            category_norm = "dining_chair"
            category_raw = "Стулья"
            unique_prefix = "local_imodern_dining_chair"
        rows.append(
            {
                "unique_key": f"{unique_prefix}::{folder.name}",
                "source_site": "imodern_local",
                "title": folder.name.replace("_", " "),
                "category_raw": category_raw,
                "category_norm": category_norm,
                "asset_local_path": str(mesh.resolve()),
                "asset_status": "local_supplier_asset",
                "asset_format": mesh.suffix.lstrip(".").lower(),
                "width_cm": width_cm,
                "depth_cm": depth_cm,
                "height_cm": height_cm,
                "price_currency": "RUB",
            }
        )
    return rows


def _existing_asset_path(row: dict[str, Any]) -> str | None:
    raw = str(row.get("asset_local_path") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in SUPPORTED_DINING_ASSET_SUFFIXES:
            return str(candidate.resolve())
    return None


def _row_dims_m(row: dict[str, Any], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    dims = row.get("dimensions_cm") if isinstance(row.get("dimensions_cm"), dict) else {}

    def get_cm(key: str) -> float:
        value = row.get(f"{key}_cm")
        if value is None:
            value = dims.get(key)
        try:
            parsed = float(value)
            return parsed if parsed > 0 else 0.0
        except Exception:
            return 0.0

    values = tuple(get_cm(key) / 100.0 for key in ("width", "depth", "height"))
    return tuple(value if value > 0 else fallback[idx] for idx, value in enumerate(values))


def _compact_supplier_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "unique_key": row.get("unique_key"),
        "title": row.get("title") or row.get("name"),
        "category_norm": row.get("category_norm"),
        "source_site": row.get("source_site"),
        "price": row.get("price") if row.get("price") is not None else row.get("price_value"),
        "price_currency": row.get("price_currency"),
        "product_url": row.get("product_url") or row.get("source_url") or row.get("model_page_url"),
        "model_download_url": row.get("model_download_url") or row.get("model_download_landing_url"),
        "asset_local_path": _existing_asset_path(row),
        "asset_status": row.get("asset_status"),
        "asset_format": row.get("asset_format"),
        "asset_download_error": row.get("asset_download_error"),
        "dining_match_score": row.get("dining_match_score"),
        "dining_llm_selection": row.get("dining_llm_selection"),
    }


def _candidate_dimension(row: dict[str, Any], key: str) -> float | None:
    dims = row.get("dimensions_cm") if isinstance(row.get("dimensions_cm"), dict) else {}
    value = row.get(f"{key}_cm")
    if value is None:
        value = dims.get(key)
    try:
        parsed = float(value)
        return parsed if parsed > 0 else None
    except Exception:
        return None


def _ensure_downloaded_dining_asset(
    candidate: dict[str, Any] | None,
    *,
    asset_out_dir: Path | None,
    blender: str | None,
) -> str | None:
    if not candidate:
        return None
    existing = _existing_asset_path(candidate)
    if existing:
        candidate["asset_local_path"] = existing
        return existing
    if not candidate.get("model_download_url") or asset_out_dir is None:
        return None

    try:
        from src.suppliers.acquire_site_assets import acquire_asset_for_record
        from src.suppliers.db_core import init_db
        from src.suppliers.models import ProductRecord
    except Exception as exc:
        candidate["asset_download_error"] = f"import_failed:{type(exc).__name__}:{exc}"
        return None

    db_path = asset_out_dir / "dining_assets.db"
    out_dir = asset_out_dir / "assets"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        init_db(db_path)
        record = ProductRecord(
            unique_key=str(candidate.get("unique_key") or candidate.get("model_download_url") or candidate.get("title") or "dining_item"),
            source_site=str(candidate.get("source_site") or "supplier_catalog"),
            source_url=str(candidate.get("product_url") or candidate.get("model_download_url") or ""),
            parsed_at=datetime.now(timezone.utc).isoformat(),
            category_raw=str(candidate.get("category_raw") or candidate.get("category_norm") or ""),
            category_norm=str(candidate.get("category_norm") or ""),
            title=str(candidate.get("title") or candidate.get("name") or "Dining item"),
            product_url=candidate.get("product_url"),
            model_link_type="direct_file",
            model_download_url=candidate.get("model_download_url"),
            model_download_landing_url=candidate.get("model_download_landing_url"),
            model_download_filename=candidate.get("model_download_filename"),
            model_format=candidate.get("model_format"),
            price_value=_as_price(candidate.get("price") if candidate.get("price") is not None else candidate.get("price_value")) or None,
            price_currency=candidate.get("price_currency"),
            width_cm=_candidate_dimension(candidate, "width"),
            depth_cm=_candidate_dimension(candidate, "depth"),
            height_cm=_candidate_dimension(candidate, "height"),
            images_json=json.dumps(candidate.get("image_urls") or candidate.get("images") or [], ensure_ascii=False),
            extra_json=json.dumps({"source": "kitchen_dining_supplier_auto_download"}, ensure_ascii=False),
        )
        asset = acquire_asset_for_record(record, db_path=db_path, out_dir=out_dir, blender_bin=blender)
    except Exception as exc:
        candidate["asset_download_error"] = f"{type(exc).__name__}:{exc}"
        return None

    path = str(getattr(asset, "asset_local_path", "") or "").strip()
    if not path:
        candidate["asset_download_error"] = "download_finished_without_local_asset"
        return None
    local = Path(path).expanduser()
    if not local.exists() or local.suffix.lower() not in SUPPORTED_DINING_ASSET_SUFFIXES:
        candidate["asset_download_error"] = f"downloaded_asset_not_importable:{path}"
        return None
    candidate["asset_local_path"] = str(local.resolve())
    candidate["asset_status"] = getattr(asset, "asset_status", None)
    candidate["asset_format"] = getattr(asset, "asset_format", None)
    return candidate["asset_local_path"]


def _dining_candidate_score(row: dict[str, Any], role: str, target_size: tuple[float, float, float]) -> float | None:
    category = _norm(row.get("category_norm"))
    title = _norm(row.get("title") or row.get("name"))
    text = _norm(" ".join(str(row.get(key) or "") for key in ("title", "name", "category_raw", "category_norm", "description")))

    if role == "table":
        if category not in {"dining_table", "table"} and not any(term in text for term in ("dining table", "обеденный стол")):
            return None
        if any(
            term in text
            for term in (
                "coffee",
                "журн",
                "console",
                "консоль",
                "side table",
                "bedside",
                "прикроват",
                "барный",
                "bar table",
                "стуль",
                "chair",
                "обеденная группа",
                "dining set",
            )
        ):
            return None
        category_bonus = -0.7 if category == "dining_table" else 0.0
    else:
        if category not in {"chair", "dining_chair", "stool", "bar_stool", "armchair"} and not any(term in text for term in ("dining chair", "стул")):
            return None
        if any(term in text for term in ("office", "офис", "gaming", "pool table")):
            return None
        category_bonus = -0.8 if category in {"chair", "dining_chair"} else (-0.25 if category == "stool" else 0.35)

    cw, cd, ch = _row_dims_m(row, target_size)
    tw, td, th = target_size
    direct = (
        abs(math.log(max(cw, 0.02) / max(tw, 0.02)))
        + abs(math.log(max(cd, 0.02) / max(td, 0.02)))
        + 0.5 * abs(math.log(max(ch, 0.02) / max(th, 0.02)))
    )
    swapped = (
        abs(math.log(max(cd, 0.02) / max(tw, 0.02)))
        + abs(math.log(max(cw, 0.02) / max(td, 0.02)))
        + 0.5 * abs(math.log(max(ch, 0.02) / max(th, 0.02)))
    )
    download_url = str(row.get("model_download_url") or "")
    download_ext = Path(download_url.split("?", 1)[0]).suffix.lower()
    if _existing_asset_path(row):
        availability_bonus = -2.0
    elif download_ext in {".zip", ".rar", ".7z", ".fbx", ".obj", ".glb", ".gltf"}:
        availability_bonus = -0.45
    elif "attachments.getfile" in download_url:
        availability_bonus = -0.25
    elif download_url:
        availability_bonus = 0.45
        if "disk.360.yandex" in download_url:
            availability_bonus += 0.8
    elif row.get("model_download_landing_url"):
        availability_bonus = 0.65
    else:
        availability_bonus = 0.8
    price_bonus = -0.05 if row.get("price") or row.get("price_value") else 0.0
    return min(direct, swapped) + category_bonus + availability_bonus + price_bonus


def _compact_dining_top_candidate(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "unique_key": row.get("unique_key"),
        "title": row.get("title") or row.get("name"),
        "category_norm": row.get("category_norm"),
        "source_site": row.get("source_site"),
        "color": row.get("color"),
        "style": row.get("style"),
        "materials": row.get("materials"),
        "dimensions_cm": row.get("dimensions_cm"),
        "width_cm": row.get("width_cm"),
        "depth_cm": row.get("depth_cm"),
        "height_cm": row.get("height_cm"),
        "price": row.get("price") if row.get("price") is not None else row.get("price_value"),
        "has_local_asset": bool(_existing_asset_path(row)),
        "asset_format": row.get("asset_format"),
        "score": round(float(score), 4),
    }


def _extract_llm_text(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(response.get("response"), str):
        return str(response["response"]).strip()
    return json.dumps(response, ensure_ascii=False)


def _parse_llm_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
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


def _choose_dining_candidate_with_llm(
    *,
    role: str,
    top_candidates: list[tuple[float, dict[str, Any]]],
    room: dict[str, Any],
    user_prompt: str,
    target_size: tuple[float, float, float],
    llm_settings: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not top_candidates:
        return None, {"status": "empty_top_k"}
    settings = dict(llm_settings or {})
    if str(settings.get("provider") or "none").strip().lower() != "ollama":
        return top_candidates[0][1], {"status": "skipped", "reason": "provider_none", "top_k": len(top_candidates)}

    chat_json = None
    import_error: Exception | None = None
    for module_name in ("src.LLMModule.ollama_client", "LLMModule.ollama_client"):
        try:
            module = __import__(module_name, fromlist=["chat_json"])
            chat_json = getattr(module, "chat_json", None)
            if callable(chat_json):
                break
        except Exception as exc:
            import_error = exc
            chat_json = None
    if not callable(chat_json):
        return top_candidates[0][1], {
            "status": "failed",
            "reason": f"ollama_import_failed:{type(import_error).__name__ if import_error else 'RuntimeError'}:{import_error or 'chat_json_not_found'}",
            "fallback": "heuristic_top1",
            "top_k": len(top_candidates),
        }

    candidate_payload = [_compact_dining_top_candidate(row, score) for score, row in top_candidates]
    schema = {
        "type": "object",
        "properties": {
            "unique_key": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["unique_key"],
        "additionalProperties": False,
    }
    payload = {
        "role": role,
        "user_prompt": user_prompt,
        "room": {
            "id": room.get("id"),
            "room_type": room.get("room_type") or room.get("type"),
            "width_m": room.get("width_m"),
            "depth_m": room.get("depth_m"),
        },
        "target_size_m": {"width": target_size[0], "depth": target_size[1], "height": target_size[2]},
        "top_candidates": candidate_payload,
        "rules": [
            "Choose only a unique_key from top_candidates.",
            "Prefer real dining tables for role=table and real dining chairs for role=chair.",
            "Prefer coherent kitchen dining style and compatible colors/materials.",
            "Reject coffee tables, console tables, bedside tables, bar stools/tables unless no normal dining option exists.",
            "Prefer candidates with local importable assets.",
        ],
    }
    try:
        response = chat_json(
            base_url=str(settings.get("ollama_url") or "http://127.0.0.1:11434"),
            model=str(settings.get("ollama_model") or "gpt-oss:20b"),
            system_prompt="You select one supplier dining furniture asset from top-k candidates. Return strict JSON only.",
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            json_schema=schema,
            timeout_sec=int(settings.get("ollama_timeout") or 180),
            temperature=float(settings.get("ollama_temperature") or 0.1),
            think=str(settings.get("ollama_think") or "low"),
            extra_options={"num_ctx": int(settings.get("ollama_num_ctx") or 8192), "num_predict": 512},
        )
        parsed = _parse_llm_json_object(_extract_llm_text(response))
        selected_key = str(parsed.get("unique_key") or "").strip()
        for _score, row in top_candidates:
            if str(row.get("unique_key") or "") == selected_key:
                return row, {
                    "status": "ok",
                    "reason": str(parsed.get("reason") or ""),
                    "unique_key": selected_key,
                    "top_k": len(top_candidates),
                }
        return top_candidates[0][1], {
            "status": "failed",
            "reason": f"llm_selected_unknown_unique_key:{selected_key}",
            "fallback": "heuristic_top1",
            "top_k": len(top_candidates),
        }
    except Exception as exc:
        return top_candidates[0][1], {
            "status": "failed",
            "reason": f"ollama_dining_rerank_failed:{type(exc).__name__}:{exc}",
            "fallback": "heuristic_top1",
            "top_k": len(top_candidates),
        }


def _rank_dining_suppliers(
    rows: list[dict[str, Any]],
    role: str,
    target_size: tuple[float, float, float],
    used_keys: set[str],
) -> list[tuple[float, dict[str, Any]]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        key = str(row.get("unique_key") or row.get("product_url") or row.get("title") or id(row))
        if key in used_keys:
            continue
        score = _dining_candidate_score(row, role, target_size)
        if score is None:
            continue
        ranked.append((score, row))
    return sorted(ranked, key=lambda x: x[0])


def _select_dining_supplier(
    rows: list[dict[str, Any]],
    role: str,
    target_size: tuple[float, float, float],
    used_keys: set[str],
    *,
    room: dict[str, Any],
    user_prompt: str,
    llm_settings: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    top_k = _rank_dining_suppliers(rows, role, target_size, used_keys)[:8]
    selected, llm_report = _choose_dining_candidate_with_llm(
        role=role,
        top_candidates=top_k,
        room=room,
        user_prompt=user_prompt,
        target_size=target_size,
        llm_settings=llm_settings,
    )
    if selected is None:
        return None, llm_report
    row = dict(selected)
    key = str(row.get("unique_key") or row.get("product_url") or row.get("title") or id(selected))
    used_keys.add(key)
    if top_k:
        row["dining_match_score"] = round(float(next((score for score, candidate in top_k if candidate is selected), top_k[0][0])), 6)
    row["dining_llm_selection"] = {
        **llm_report,
        "selected_unique_key": key,
        "top_candidates": [_compact_dining_top_candidate(candidate, score) for score, candidate in top_k],
    }
    return row, row["dining_llm_selection"]


def _room_polygon_xy(room: dict[str, Any]) -> list[tuple[float, float]]:
    points = room.get("floor_polygon") if isinstance(room.get("floor_polygon"), list) else []
    out: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        x = _float(point.get("x"), float("nan"))
        y = _float(point.get("y", point.get("z")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    if len(out) >= 3:
        return out
    width = _float(room.get("width_m") or room.get("width"), 3.2)
    depth = _float(room.get("depth_m") or room.get("depth"), 3.0)
    return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]


def _polygon_signed_area(poly: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )


def _wall_candidates(room: dict[str, Any]) -> list[dict[str, Any]]:
    poly = _room_polygon_xy(room)
    walls = room.get("walls") if isinstance(room.get("walls"), list) else []
    if not walls:
        walls = [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % len(poly)} for i in range(len(poly))]
    ccw = _polygon_signed_area(poly) > 0
    out: list[dict[str, Any]] = []
    for idx, wall in enumerate(walls):
        if not isinstance(wall, dict):
            continue
        a_idx = int(_float(wall.get("from_vertex"), idx))
        b_idx = int(_float(wall.get("to_vertex"), (idx + 1) % len(poly)))
        if a_idx < 0 or b_idx < 0 or a_idx >= len(poly) or b_idx >= len(poly):
            continue
        ax, ay = poly[a_idx]
        bx, by = poly[b_idx]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = (-uy, ux) if ccw else (uy, -ux)
        out.append(
            {
                "id": str(wall.get("id") or f"w{idx}"),
                "a": (ax, ay),
                "b": (bx, by),
                "u": (ux, uy),
                "n": (nx, ny),
                "length": length,
                "yaw_deg": math.degrees(math.atan2(uy, ux)),
            }
        )
    return out


def _opening_interval_on_wall(opening: dict[str, Any], wall: dict[str, Any], margin: float = 0.18) -> tuple[float, float] | None:
    if str(opening.get("wall_id") or "") != wall["id"]:
        return None
    center = opening.get("s")
    if center is None:
        return None
    width = _float(opening.get("width"), 0.8)
    s = _float(center)
    return max(0.0, s - width * 0.5 - margin), min(float(wall["length"]), s + width * 0.5 + margin)


def _wall_point_at_s(wall: dict[str, Any], s: float) -> tuple[float, float]:
    ax, ay = wall["a"]
    ux, uy = wall["u"]
    return ax + ux * s, ay + uy * s


def _project_point_to_wall_s(point: tuple[float, float], wall: dict[str, Any]) -> tuple[float, float]:
    ax, ay = wall["a"]
    ux, uy = wall["u"]
    px, py = point
    s = (px - ax) * ux + (py - ay) * uy
    closest = _wall_point_at_s(wall, max(0.0, min(float(wall["length"]), s)))
    distance = math.hypot(px - closest[0], py - closest[1])
    return s, distance


def _door_clearance_interval_on_wall(
    door: dict[str, Any],
    *,
    door_wall: dict[str, Any],
    candidate_wall: dict[str, Any],
) -> tuple[float, float] | None:
    if str(door.get("wall_id") or "") == candidate_wall["id"]:
        return None

    door_width = max(0.65, _float(door.get("width"), 0.85))
    door_s = _float(door.get("s"), float("nan"))
    if not math.isfinite(door_s):
        return None

    door_points = [
        _wall_point_at_s(door_wall, door_s),
        _wall_point_at_s(door_wall, door_s - door_width * 0.5),
        _wall_point_at_s(door_wall, door_s + door_width * 0.5),
    ]

    projected: list[float] = []
    min_distance = float("inf")
    for point in door_points:
        s, distance = _project_point_to_wall_s(point, candidate_wall)
        min_distance = min(min_distance, distance)
        projected.append(s)

    # A closed door near an adjacent wall can still swing into a base cabinet.
    # If the door leaf can reach the candidate wall plus the 600 mm cabinet
    # depth, reserve the projected sweep corridor on that wall.
    reach = door_width + KITCHEN_CABINET_DEPTH_M + 0.12
    if min_distance > reach:
        return None

    center_s = max(0.0, min(float(candidate_wall["length"]), sum(projected) / len(projected)))
    reserve = max(KITCHEN_DOOR_SWING_CLEARANCE_M, door_width + 0.25)
    return max(0.0, center_s - reserve), min(float(candidate_wall["length"]), center_s + reserve)


def _subtract_intervals(free: list[tuple[float, float]], blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    for start, end in sorted(blocked):
        next_free: list[tuple[float, float]] = []
        for a, b in free:
            if end <= a or start >= b:
                next_free.append((a, b))
            else:
                if start > a:
                    next_free.append((a, start))
                if end < b:
                    next_free.append((end, b))
        free = next_free
    return free


def _free_wall_intervals(room: dict[str, Any], wall: dict[str, Any]) -> list[tuple[float, float]]:
    wall_len = float(wall["length"])
    free = [(0.0, wall_len)]
    blocked: list[tuple[float, float]] = []
    walls_by_id = {candidate["id"]: candidate for candidate in _wall_candidates(room)}
    for key in ("doors", "windows", "openings"):
        for opening in room.get(key) or []:
            if isinstance(opening, dict):
                interval = _opening_interval_on_wall(opening, wall)
                if interval and interval[1] > interval[0]:
                    blocked.append(interval)
    for door in room.get("doors") or []:
        if not isinstance(door, dict):
            continue
        door_wall = walls_by_id.get(str(door.get("wall_id") or ""))
        if door_wall is None:
            continue
        interval = _door_clearance_interval_on_wall(door, door_wall=door_wall, candidate_wall=wall)
        if interval and interval[1] > interval[0]:
            blocked.append(interval)
    free = _subtract_intervals(free, blocked)
    return [(a, b) for a, b in free if b - a >= KITCHEN_MIN_WIDTH_M]


def _select_room_kitchen_placement(room: dict[str, Any], requested_width_m: float | None) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any], tuple[float, float]] | None = None
    for wall in _wall_candidates(room):
        for interval in _free_wall_intervals(room, wall):
            score = interval[1] - interval[0]
            if best is None or score > best[0]:
                best = (score, wall, interval)
    if best is None:
        return None
    _, wall, interval = best
    free_len = interval[1] - interval[0]
    auto_width = min(free_len, KITCHEN_MAX_AUTO_WIDTH_M)
    width = max(KITCHEN_MIN_WIDTH_M, min(requested_width_m or auto_width, free_len, KITCHEN_MAX_AUTO_WIDTH_M))
    start = interval[0] + max(0.0, (free_len - width) * 0.5)
    ax, ay = wall["a"]
    ux, uy = wall["u"]
    origin = (ax + ux * start, ay + uy * start)
    return {
        "wall_id": wall["id"],
        "available_width_mm": int(round(width * 1000.0)),
        "position": [origin[0], origin[1], 0.0],
        "rotation": [0.0, 0.0, float(wall["yaw_deg"])],
        "wall": wall,
        "start_m": start,
        "end_m": start + width,
    }


def _build_dining_items(
    room: dict[str, Any],
    placement: dict[str, Any] | None,
    supplier_catalog: str | Path | None = DEFAULT_APPLIANCE_CATALOG,
    asset_out_dir: Path | None = None,
    blender: str | None = None,
    user_prompt: str = "",
    llm_settings: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not room or not placement:
        return [], []
    catalog_rows = _load_supplier_catalog(supplier_catalog)
    warnings: list[str] = []
    used_supplier_keys: set[str] = set()
    poly = _room_polygon_xy(room)
    min_x, max_x = min(x for x, _ in poly), max(x for x, _ in poly)
    min_y, max_y = min(y for _, y in poly), max(y for _, y in poly)
    wall = placement["wall"]
    ux, uy = wall["u"]
    nx, ny = wall["n"]
    width_m = placement["available_width_mm"] / 1000.0
    wall_mid = float(placement["start_m"]) + width_m * 0.5
    ax, ay = wall["a"]
    base_x = ax + ux * wall_mid
    base_y = ay + uy * wall_mid
    room_w, room_d = max_x - min_x, max_y - min_y
    compact = min(room_w, room_d) < 2.7
    table_w, table_d = ((0.72, 0.58) if compact else (1.15, 0.78))
    distance = 0.6 + (0.72 if compact else 1.05)
    cx = min(max(base_x + nx * distance, min_x + table_w * 0.5 + 0.12), max_x - table_w * 0.5 - 0.12)
    cy = min(max(base_y + ny * distance, min_y + table_d * 0.5 + 0.12), max_y - table_d * 0.5 - 0.12)
    yaw = float(wall["yaw_deg"])
    table_target = (table_w, table_d, 0.75)
    table_candidate, table_selection = _select_dining_supplier(
        catalog_rows,
        "table",
        table_target,
        used_supplier_keys,
        room=room,
        user_prompt=user_prompt,
        llm_settings=llm_settings,
    )
    if table_candidate is None:
        warnings.append("no_catalog_dining_table_candidate")
    elif not _ensure_downloaded_dining_asset(table_candidate, asset_out_dir=asset_out_dir, blender=blender):
        warnings.append("dining_table_selected_without_local_asset")
    items = [
        {
            "id": "kitchen_dining_table_001",
            "type": "dining_table",
            "x_m": cx,
            "y_m": cy,
            "z_m": 0.0,
            "width_m": table_w,
            "depth_m": table_d,
            "height_m": 0.75,
            "yaw_deg": yaw,
            "render_policy": "supplier_asset_only",
            "supplier_candidate": _compact_supplier_candidate(table_candidate) if table_candidate else None,
        }
    ]
    chair_gap = 0.48 if compact else 0.58
    chair_target = (0.46, 0.52, 0.82)
    chair_candidate, chair_selection = _select_dining_supplier(
        catalog_rows,
        "chair",
        chair_target,
        used_supplier_keys,
        room=room,
        user_prompt=user_prompt,
        llm_settings=llm_settings,
    )
    if chair_candidate is None:
        warnings.append("no_catalog_dining_chair_candidate")
    elif not _ensure_downloaded_dining_asset(chair_candidate, asset_out_dir=asset_out_dir, blender=blender):
        warnings.append("dining_chair_selected_without_local_asset")
    for role, report in (("table", table_selection), ("chair", chair_selection)):
        if isinstance(report, dict) and report.get("status") == "failed":
            warnings.append(f"dining_{role}_llm_fallback:{report.get('reason')}")

    def face_table_yaw(x: float, y: float) -> float:
        # Supplier dining chairs use local +Y as the seating/front direction.
        return (math.degrees(math.atan2(cx - x, cy - y)) + 360.0) % 360.0

    for idx, side in enumerate((-1.0, 1.0), start=1):
        chair_x = min(max(cx + nx * side * chair_gap, min_x + 0.25), max_x - 0.25)
        chair_y = min(max(cy + ny * side * chair_gap, min_y + 0.25), max_y - 0.25)
        items.append(
            {
                "id": f"kitchen_dining_chair_{idx:03d}",
                "type": "dining_chair",
                "x_m": chair_x,
                "y_m": chair_y,
                "z_m": 0.0,
                "width_m": chair_target[0],
                "depth_m": chair_target[1],
                "height_m": chair_target[2],
                "yaw_deg": face_table_yaw(chair_x, chair_y),
                "render_policy": "supplier_asset_only",
                "supplier_candidate": _compact_supplier_candidate(chair_candidate) if chair_candidate else None,
            }
        )
    return items, warnings


def _apply_facade_override(assembly: dict[str, Any], material: dict[str, Any]) -> None:
    binding = (assembly.get("material_bindings") or {}).get("facade")
    if binding:
        binding["chosen_material"] = material
        binding["final_score"] = 1.0
        binding["score_breakdown"] = {"manual_color_override": 1.0}

    for item in assembly.get("bill_of_materials", {}).get("items", []):
        if item.get("role") != "facade_sheet":
            continue
        item["sku"] = material.get("sku")
        item["name"] = material.get("name")
        item["kitchen_role"] = material.get("kitchen_role")
        item["unit_price"] = round(float(material.get("price") or item.get("unit_price") or 0), 2)
        item["total_price"] = round(float(item["unit_price"]) * float(item.get("quantity") or 1), 2)
        item["note"] = (item.get("note") or "") + "; manual_color_override"

    bom = assembly.get("bill_of_materials") or {}
    items = bom.get("items") or []
    total_material = round(sum(float(item.get("total_price") or 0) for item in items), 2)
    old_total = float(bom.get("total_material_price") or 0)
    delta = total_material - old_total
    bom["total_material_price"] = total_material
    bom["total_estimated_price"] = round(float(bom.get("total_estimated_price") or 0) + delta, 2)

    estimate = assembly.setdefault("price_estimate", {})
    estimate["currency"] = bom.get("currency", "RUB")
    estimate["total_material_price"] = bom["total_material_price"]
    estimate["total_estimated_price"] = bom["total_estimated_price"]
    assembly.setdefault("warnings", []).append(f"manual_facade_color_override:{material.get('sku')}")


def _as_price(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").replace("\xa0", " ").replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(" ", ""))
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except Exception:
        return 0.0


def _apply_dining_bom(assembly: dict[str, Any], dining_items: list[dict[str, Any]]) -> None:
    bom = assembly.setdefault("bill_of_materials", {})
    items = bom.setdefault("items", [])
    added = 0
    for item in dining_items:
        candidate = item.get("supplier_candidate") if isinstance(item.get("supplier_candidate"), dict) else {}
        price = _as_price(candidate.get("price"))
        if price <= 0:
            continue
        role = str(item.get("type") or "dining_item")
        items.append(
            {
                "role": role,
                "sku": candidate.get("unique_key"),
                "name": candidate.get("title") or role,
                "source_site": candidate.get("source_site"),
                "unit": "pcs",
                "quantity": 1,
                "unit_price": round(price, 2),
                "total_price": round(price, 2),
                "currency": candidate.get("price_currency") or bom.get("currency", "RUB"),
                "note": "supplier_catalog_dining_item",
            }
        )
        added += 1
    if not added:
        return
    total_material = round(sum(float(row.get("total_price") or 0.0) for row in items), 2)
    old_material = float(bom.get("total_material_price") or 0.0)
    delta = total_material - old_material
    bom["total_material_price"] = total_material
    bom["total_estimated_price"] = round(float(bom.get("total_estimated_price") or 0.0) + delta, 2)
    estimate = assembly.setdefault("price_estimate", {})
    estimate["currency"] = bom.get("currency", "RUB")
    estimate["total_material_price"] = bom["total_material_price"]
    estimate["total_estimated_price"] = bom["total_estimated_price"]


def _select_kitchen_room_surfaces(
    *,
    room: dict[str, Any],
    assembly: dict[str, Any],
    out_dir: Path,
    slug: str,
    prompt: str,
    llm_settings: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not room:
        return room, []
    scene = {"schema": "scene.v1", "room": deepcopy(room), "placements": []}
    warnings: list[str] = []
    design = assembly.get("design_spec") if isinstance(assembly.get("design_spec"), dict) else {}
    style = str(design.get("style") or "modern")
    palette = design.get("palette") if isinstance(design.get("palette"), dict) else {}
    material_prompt = "\n".join(
        part
        for part in (
            prompt,
            "Room type: kitchen. Choose durable water-resistant floor and washable wall material compatible with the kitchen set.",
            "Kitchen palette: " + json.dumps(palette, ensure_ascii=False) if palette else "",
        )
        if part
    )
    try:
        from src.pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection
        from src.pipeline.wall_stage import apply_wall_material_to_scene_with_catalog, run_wall_selection
    except Exception as exc:
        return deepcopy(room), [f"kitchen_surface_import_failed:{type(exc).__name__}:{exc}"]

    if DEFAULT_FLOORING_MATERIALS.exists() and DEFAULT_FLOORING_STYLE_RULES.is_file():
        try:
            floor_path = out_dir / f"{slug}.flooring.selection.v1.json"
            floor_selection = run_flooring_selection(
                prompt=material_prompt,
                style=style,
                room_type="kitchen",
                room_description="kitchen room with procedural cabinet set and supplier dining furniture",
                room_id=str(room.get("id") or slug),
                materials_path=DEFAULT_FLOORING_MATERIALS,
                style_rules_path=DEFAULT_FLOORING_STYLE_RULES,
                out_path=floor_path,
                top_k=10,
                llm_settings=llm_settings,
            )
            scene = apply_flooring_to_scene(scene, floor_selection)
        except Exception as exc:
            warnings.append(f"kitchen_flooring_selection_failed:{type(exc).__name__}:{exc}")

    if DEFAULT_WALL_MATERIALS.exists():
        try:
            wall_path = out_dir / f"{slug}.wall_material.selection.v1.json"
            wall_selection = run_wall_selection(
                prompt=material_prompt,
                style=style,
                room_type="kitchen",
                room_description="kitchen room with washable walls",
                room_id=str(room.get("id") or slug),
                materials_path=DEFAULT_WALL_MATERIALS,
                out_path=wall_path,
                top_k=10,
                llm_settings=llm_settings,
            )
            scene = apply_wall_material_to_scene_with_catalog(scene, wall_selection, materials_path=DEFAULT_WALL_MATERIALS)
        except Exception as exc:
            warnings.append(f"kitchen_wall_selection_failed:{type(exc).__name__}:{exc}")

    out_room = scene.get("room") if isinstance(scene.get("room"), dict) else deepcopy(room)
    return out_room, warnings


def _build_required(args: argparse.Namespace, width_m: float) -> dict[str, Any]:
    width_mm = int(round(width_m * 1000.0))
    allow_cooktop = not args.no_cooktop and width_mm >= args.min_cooktop_width_mm
    return {
        "sink": True,
        "faucet": True,
        "cooktop": allow_cooktop,
        "oven": allow_cooktop and not args.no_oven,
        "hood": allow_cooktop and not args.no_hood,
        "fridge": bool(args.fridge),
        "dishwasher": bool(args.dishwasher),
        "washing_machine": bool(args.washing_machine),
        "microwave": True,
        "decor_accessories": bool(args.decor),
    }


def _render_with_blender(args: argparse.Namespace, json_path: Path, blend_path: Path, png_path: Path) -> None:
    cmd = [
        args.blender,
        "-b",
        "--python",
        args.render_script,
        "--",
        "--input",
        str(json_path),
        "--out-blend",
        str(blend_path),
        "--render-png",
        str(png_path),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and render a straight procedural kitchen.")
    parser.add_argument("--width-m", type=float, default=None, help="Kitchen width in meters, for example 1.5 or 4.5. Optional when --room-json is used.")
    parser.add_argument("--room-json", default=None, help="Optional room or scene JSON with floor_polygon, walls, doors and windows.")
    parser.add_argument("--prompt", required=True, help="Kitchen description.")
    parser.add_argument("--slug", default=None, help="Output filename stem. Default is derived from width.")
    parser.add_argument("--out-dir", default="out/kitchen_demo", help="Directory for JSON, Blend and PNG.")
    parser.add_argument("--mode", default="optimal", choices=("cheapest", "optimal", "best_match"))
    parser.add_argument("--facade-colors", default="", help="Comma-separated desired facade colors.")
    parser.add_argument("--countertop-colors", default="white marble,light stone,gray")
    parser.add_argument("--backsplash-colors", default="light gray,stone,white")
    parser.add_argument("--accent-colors", default="black metal")
    parser.add_argument("--facade-sku", default=None, help="Optional exact BasisRF facade material SKU override.")
    parser.add_argument("--budget", type=float, default=120000.0)
    parser.add_argument("--material-catalog", default=DEFAULT_MATERIAL_CATALOG)
    parser.add_argument("--appliance-catalog", default=DEFAULT_APPLIANCE_CATALOG)
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--render-script", default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--fridge", action="store_true")
    parser.add_argument("--dishwasher", action="store_true")
    parser.add_argument("--washing-machine", action="store_true")
    parser.add_argument("--decor", action="store_true")
    parser.add_argument("--no-cooktop", action="store_true")
    parser.add_argument("--no-oven", action="store_true")
    parser.add_argument("--no-hood", action="store_true")
    parser.add_argument("--min-cooktop-width-mm", type=int, default=1800)
    parser.add_argument("--no-render", action="store_true", help="Only write JSON, do not launch Blender.")
    parser.add_argument("--kitchen-llm-provider", choices=("none", "ollama"), default="none")
    parser.add_argument("--kitchen-ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--kitchen-ollama-model", default="gpt-oss:20b")
    parser.add_argument("--kitchen-ollama-timeout", type=int, default=180)
    parser.add_argument("--kitchen-ollama-temperature", type=float, default=0.1)
    parser.add_argument("--kitchen-ollama-num-ctx", type=int, default=8192)
    parser.add_argument("--kitchen-ollama-think", default="low")
    parser.add_argument("--no-download-dining-assets", action="store_true", help="Do not download missing supplier table/chair assets for the dining zone.")
    args = parser.parse_args(argv)

    room = _load_room(args.room_json)
    placement = _select_room_kitchen_placement(room, args.width_m) if room else None
    if args.width_m is None and placement is None:
        parser.error("--width-m is required unless --room-json contains a usable free wall")
    width_m = (placement["available_width_mm"] / 1000.0) if placement else float(args.width_m)
    width_mm = int(round(width_m * 1000.0))
    slug = args.slug or f"kitchen_{width_mm}mm"
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = _build_required(args, width_m)
    llm_settings = {
        "provider": args.kitchen_llm_provider,
        "ollama_url": args.kitchen_ollama_url,
        "ollama_model": args.kitchen_ollama_model,
        "ollama_timeout": args.kitchen_ollama_timeout,
        "ollama_temperature": args.kitchen_ollama_temperature,
        "ollama_num_ctx": args.kitchen_ollama_num_ctx,
        "ollama_think": args.kitchen_ollama_think,
    }
    variants = generate_kitchen_variants(
        material_catalog=args.material_catalog,
        appliance_catalog=args.appliance_catalog,
        user_prompt=args.prompt,
        room=room or {"width_m": width_m, "depth_m": 2.6, "height_m": 2.7},
        kitchen_zone={
            "layout_type": "straight",
            "wall_id": placement.get("wall_id") if placement else None,
            "available_width_mm": width_mm,
        },
        required_appliances=required,
        recommended_colors={
            "facades": _split_csv(args.facade_colors),
            "countertop": _split_csv(args.countertop_colors),
            "backsplash": _split_csv(args.backsplash_colors),
            "accent": _split_csv(args.accent_colors),
        },
        budget={"total": args.budget, "currency": "RUB"},
        modes=(args.mode,),
        target_id=slug,
        llm_settings=llm_settings,
        position=placement.get("position") if placement else None,
        rotation=placement.get("rotation") if placement else None,
    )
    assembly = variants[args.mode]

    facade_override = _load_material_by_sku(args.material_catalog, args.facade_sku)
    if facade_override:
        _apply_facade_override(assembly, facade_override)

    if room:
        dining_items, dining_warnings = _build_dining_items(
            room,
            placement,
            args.appliance_catalog,
            asset_out_dir=None if args.no_download_dining_assets else out_dir / "_dining_assets",
            blender=args.blender,
            user_prompt=args.prompt,
            llm_settings=llm_settings,
        )
        room_with_surfaces, surface_warnings = _select_kitchen_room_surfaces(
            room=room,
            assembly=assembly,
            out_dir=out_dir,
            slug=slug,
            prompt=args.prompt,
            llm_settings=llm_settings,
        )
        assembly["room_context"] = {
            "room": room_with_surfaces,
            "kitchen_wall_id": placement.get("wall_id") if placement else None,
            "kitchen_free_interval_m": [round(float(placement["start_m"]), 4), round(float(placement["end_m"]), 4)] if placement else None,
            "dining_items": dining_items,
        }
        assembly.setdefault("warnings", []).append("room_context:full_shell_with_openings")
        assembly.setdefault("warnings", []).extend(dining_warnings)
        assembly.setdefault("warnings", []).extend(surface_warnings)
        _apply_dining_bom(assembly, dining_items)
        if isinstance(assembly.get("price_estimate"), dict):
            assembly["price"] = dict(assembly["price_estimate"])

    json_path = out_dir / f"{slug}.json"
    blend_path = out_dir / f"{slug}.blend"
    png_path = out_dir / f"{slug}_preview.png"
    json_path.write_text(json.dumps(assembly, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_render:
        _render_with_blender(args, json_path, blend_path, png_path)

    print(f"json={json_path.resolve()}")
    if not args.no_render:
        print(f"blend={blend_path.resolve()}")
        print(f"png={png_path.resolve()}")
    print(f"price={assembly.get('price_estimate')}")
    print(f"warnings={assembly.get('warnings') or []}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
