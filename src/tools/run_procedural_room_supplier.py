from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any



# === CGS TRELLIS33 RETRY14 PATCH START ===
# Runtime patch:
# 1) for procedural-room JSONs, make all procedural items supplier-replaceable
#    except pillow/blanket, giving ~33 replacement targets from 39 bedroom items;
# 2) expose subprocess output inside CalledProcessError.__str__, so wrapped
#    CUDA OOM messages are detectable by existing code;
# 3) keep the patch disable-able through env vars.
import copy as _cgs_copy
import os as _cgs_os
import subprocess as _cgs_subprocess

_CGS_TRELLIS33_SKIP_CATEGORIES = {
    "pillow",
    "blanket",
}

_CGS_TRELLIS33_DISABLE = _cgs_os.environ.get("CGS_TRELLIS33_PATCH_DISABLE", "").strip().lower() in {
    "1", "true", "yes", "on"
}

def _cgs_trellis33_item_is_procedural(item):
    if not isinstance(item, dict):
        return False
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    return bool(
        meta.get("procedural")
        or source.get("placement_source") == "procedural_room_stage"
        or source.get("generator") == "bedroom_generator"
    )

def _cgs_trellis33_category(item):
    if not isinstance(item, dict):
        return ""
    return str(item.get("category") or item.get("type") or item.get("name") or "").strip().lower()

def _cgs_trellis33_force_supplier_targets_payload(payload):
    if _CGS_TRELLIS33_DISABLE:
        return payload
    if not isinstance(payload, dict):
        return payload

    items = payload.get("placements")
    if not isinstance(items, list):
        items = payload.get("items")
    if not isinstance(items, list):
        return payload

    has_procedural_items = any(_cgs_trellis33_item_is_procedural(item) for item in items if isinstance(item, dict))
    if not has_procedural_items:
        return payload

    changed = False
    for item in items:
        if not isinstance(item, dict):
            continue
        if not _cgs_trellis33_item_is_procedural(item):
            continue

        cat = _cgs_trellis33_category(item)
        meta = item.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            item["meta"] = meta

        if cat in _CGS_TRELLIS33_SKIP_CATEGORIES:
            if meta.get("replace_with_supplier") is not False:
                meta["replace_with_supplier"] = False
                changed = True
            continue

        if meta.get("replace_with_supplier") is not True:
            meta["replace_with_supplier"] = True
            changed = True

        # Some older code paths check top-level field instead of meta.
        if item.get("replace_with_supplier") is not True:
            item["replace_with_supplier"] = True
            changed = True

    if changed:
        meta_root = payload.setdefault("meta", {})
        if isinstance(meta_root, dict):
            meta_root["trellis33_force_supplier_targets_patch"] = {
                "enabled": True,
                "policy": "all_procedural_items_except_pillow_blanket",
                "skip_categories": sorted(_CGS_TRELLIS33_SKIP_CATEGORIES),
            }
    return payload

try:
    _CGS_TRELLIS33_ORIG_READ_JSON
except NameError:
    try:
        _CGS_TRELLIS33_ORIG_READ_JSON = read_json
        def read_json(*args, **kwargs):  # type: ignore[no-redef]
            data = _CGS_TRELLIS33_ORIG_READ_JSON(*args, **kwargs)
            return _cgs_trellis33_force_supplier_targets_payload(data)
    except NameError:
        pass

try:
    _CGS_TRELLIS33_ORIG_WRITE_JSON
except NameError:
    try:
        _CGS_TRELLIS33_ORIG_WRITE_JSON = write_json
        def write_json(path, payload, *args, **kwargs):  # type: ignore[no-redef]
            payload = _cgs_trellis33_force_supplier_targets_payload(payload)
            return _CGS_TRELLIS33_ORIG_WRITE_JSON(path, payload, *args, **kwargs)
    except NameError:
        pass

try:
    _CGS_TRELLIS33_ORIG_CPE_STR
except NameError:
    _CGS_TRELLIS33_ORIG_CPE_STR = _cgs_subprocess.CalledProcessError.__str__

    def _cgs_trellis33_called_process_error_str(self):
        text = _CGS_TRELLIS33_ORIG_CPE_STR(self)
        output = getattr(self, "output", None)
        stderr = getattr(self, "stderr", None)
        extra = []
        if output:
            extra.append(str(output))
        if stderr:
            extra.append(str(stderr))
        if extra:
            joined = "\n".join(extra)
            # Limit size: enough for CUDA/spconv OOM diagnostics.
            text = text + "\n" + joined[:50000]
        return text

    _cgs_subprocess.CalledProcessError.__str__ = _cgs_trellis33_called_process_error_str
# === CGS TRELLIS33 RETRY14 PATCH END ===

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_imports() -> None:
    root = _repo_root()
    src = root / "src"
    for candidate in (root, src):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


_ensure_imports()

try:
    from src.acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from src.apply_supplier_bindings import apply_supplier_bindings_to_json
    from src.layout_targets import create_layout_selection_stub_artifacts
    from src.pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection
    from src.pipeline.curtain_stage import (
        discover_curtain_models,
        discover_supplier_curtain_models,
        load_curtain_catalog,
        write_json as write_curtain_json,
    )
    from src.pipeline.infinigen_scene_improvers import apply_curtains_to_scene
    from src.pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage
    from src.pipeline.procedural_rooms.room_context import build_room_context
    from src.pipeline.procedural_rooms.validation import validate_placements
    from src.pipeline.wall_stage import apply_wall_material_to_scene_with_catalog, run_wall_selection
    from src.supplier_layout_matcher import _merge_catalog_rows, build_bindings_with_candidates, load_supplier_catalog, load_supplier_catalog_json
    from src.suppliers.room_design_spec_builder import build_room_design_spec
    from src.topview_vlm_orientation_repair import (
        collect_scene_objects as collect_topview_vlm_scene_objects,
        filter_target_objects as filter_topview_vlm_target_objects,
        run_topview_vlm_orientation_repair,
    )
    from src.trellis_supplier_asset_orchestrator import run_orchestration, unload_ollama_model
except ModuleNotFoundError:
    from acquire_supplier_bindings_assets import acquire_assets_for_bindings_json
    from apply_supplier_bindings import apply_supplier_bindings_to_json
    from layout_targets import create_layout_selection_stub_artifacts
    from pipeline.flooring_stage import apply_flooring_to_scene, run_flooring_selection
    from pipeline.curtain_stage import (
        discover_curtain_models,
        discover_supplier_curtain_models,
        load_curtain_catalog,
        write_json as write_curtain_json,
    )
    from pipeline.infinigen_scene_improvers import apply_curtains_to_scene
    from pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage
    from pipeline.procedural_rooms.room_context import build_room_context
    from pipeline.procedural_rooms.validation import validate_placements
    from pipeline.wall_stage import apply_wall_material_to_scene_with_catalog, run_wall_selection
    from supplier_layout_matcher import _merge_catalog_rows, build_bindings_with_candidates, load_supplier_catalog, load_supplier_catalog_json
    from suppliers.room_design_spec_builder import build_room_design_spec
    from topview_vlm_orientation_repair import (
        collect_scene_objects as collect_topview_vlm_scene_objects,
        filter_target_objects as filter_topview_vlm_target_objects,
        run_topview_vlm_orientation_repair,
    )
    from trellis_supplier_asset_orchestrator import run_orchestration, unload_ollama_model


SELECTED_BINDING_STATUSES = {
    "heuristic_top1_selected",
    "heuristic_first_acceptable_selected",
    "llm_reranked_top1_selected",
    "llm_reranked_first_acceptable_selected",
}
SUPPORTED_LOCAL_ASSET_EXTS = {".fbx", ".obj", ".glb", ".gltf"}
TRELLIS_ASSET_STATUS = "trellis_generated_local_asset"


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def candidate_asset_paths(candidate: dict[str, Any] | None) -> list[Path]:
    if not isinstance(candidate, dict):
        return []
    raw_values: list[Any] = []
    for key in (
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
    ):
        value = candidate.get(key)
        if value:
            raw_values.append(value)
    extra = candidate.get("extra")
    if isinstance(extra, dict):
        trellis_asset = extra.get("trellis_generated_asset")
        if isinstance(trellis_asset, dict):
            for key in ("asset_local_path", "mesh_path", "glb_path", "gltf_path"):
                value = trellis_asset.get(key)
                if value:
                    raw_values.append(value)
    out: list[Path] = []
    seen: set[str] = set()
    for value in raw_values:
        values = value if isinstance(value, (list, tuple)) else [value]
        for raw in values:
            p = Path(str(raw)).expanduser()
            if p.is_file() and p.suffix.lower() in SUPPORTED_LOCAL_ASSET_EXTS:
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append(p.resolve())
    return out


def candidate_has_supported_local_asset(candidate: dict[str, Any] | None) -> bool:
    return bool(candidate_asset_paths(candidate))


def apply_trellis_card_to_candidate(candidate: dict[str, Any], patched_card: dict[str, Any]) -> None:
    candidate["asset_status"] = patched_card.get("asset_status") or TRELLIS_ASSET_STATUS
    candidate["asset_format"] = patched_card.get("asset_format") or "glb"
    candidate["asset_local_path"] = patched_card.get("asset_local_path")
    candidate["asset_source_url"] = patched_card.get("asset_source_url")
    extra = candidate.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        candidate["extra"] = extra
    patched_extra = patched_card.get("extra") if isinstance(patched_card.get("extra"), dict) else {}
    trellis_asset = patched_extra.get("trellis_generated_asset")
    if isinstance(trellis_asset, dict):
        extra["trellis_generated_asset"] = trellis_asset


def build_trellis_card_from_binding(binding: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    card = dict(candidate)
    target_id = str(binding.get("target_id") or "").strip()
    if target_id:
        card["target_id"] = target_id
        card["supplier_target_id"] = target_id
        card["layout_target_id"] = target_id
    for src_key, dst_key in [
        ("category", "target_category"),
        ("semantic_group", "target_semantic_group"),
        ("requested_size_m", "target_size_m"),
    ]:
        if binding.get(src_key) is not None:
            card[dst_key] = binding.get(src_key)
    def _collect_pool_fields(source: Any) -> list[Any]:
        if not isinstance(source, dict):
            return []
        out: list[Any] = []
        for key in (
            "top_candidates",
            "candidate_pool",
            "supplier_candidate_pool",
            "candidates",
            "supplier_candidates",
            "alternatives",
        ):
            val = source.get(key)
            if isinstance(val, list):
                out.extend(val)
        return out

    collected_top: list[Any] = []
    # Keep explicit order: binding first, then known wrapper sections.
    for source in (
        binding,
        binding.get("binding"),
        binding.get("meta"),
        binding.get("source"),
        candidate.get("meta") if isinstance(candidate, dict) else None,
    ):
        collected_top.extend(_collect_pool_fields(source))
    for alt_field in ("selected_candidates", "all_candidates"):
        alt_val = binding.get(alt_field)
        if isinstance(alt_val, list):
            collected_top.extend(alt_val)

    top_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in collected_top:
        if not isinstance(item, dict):
            continue
        key = str(
            item.get("unique_key")
            or item.get("product_url")
            or item.get("model_page_url")
            or item.get("source_url")
            or item.get("title")
            or item.get("id")
            or ""
        )
        if key and key in seen:
            continue
        seen.add(key)
        top_candidates.append(item)

    if top_candidates:
        card["top_candidates"] = top_candidates
        card["supplier_candidate_pool"] = top_candidates
        card["candidate_pool"] = top_candidates
        card["alternatives"] = top_candidates
    card["binding"] = {
        "target_id": target_id,
        "category": binding.get("category"),
        "semantic_group": binding.get("semantic_group"),
        "requested_size_m": binding.get("requested_size_m"),
        "top_candidates": top_candidates if isinstance(top_candidates, list) else [],
    }
    return card


_TRELLIS_IMAGE_LIST_KEYS = ("images", "images_json", "image_urls", "photos", "preview_images")
_TRELLIS_IMAGE_SINGLE_KEYS = ("preview_local_path", "preview_path", "image_path", "image_local_path", "thumbnail_local_path")
_TRELLIS_IMAGE_DICT_KEYS = ("url", "src", "path", "local_path", "value")
_TRELLIS_POOL_KEYS = ("top_candidates", "candidate_pool", "supplier_candidate_pool", "candidates", "supplier_candidates", "alternatives")
_TRELLIS_SOURCE_KEYS = (
    "unique_key",
    "source_site",
    "supplier",
    "product_url",
    "model_page_url",
    "source_url",
    "asset_source_url",
    "brand",
    "title",
)


def _trellis_allowed_ikea_mebelru_host(host: str) -> bool:
    host = host.lower().strip().strip(".")
    if not host:
        return False
    return host == "mebel.ru" or host.endswith(".mebel.ru") or "ikea." in host


def _trellis_allowed_ikea_mebelru_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False

    url_like: list[str] = re.findall(r"https?://[^\s\"'<>]+", text)
    if "://" in text and text not in url_like:
        url_like.append(text)
    if not url_like and "." in text and not any(ch.isspace() for ch in text):
        url_like.append("https://" + text)
    for raw_url in url_like:
        host = _trellis_url_host(raw_url)
        if _trellis_allowed_ikea_mebelru_host(host):
            return True

    if re.search(r"(^|[^a-z0-9])ikea([._-]?[a-z0-9]+)?([^a-z0-9]|$)", text):
        return True
    if re.search(r"(^|[^a-z0-9])mebel[._-]?ru([^a-z0-9]|$)", text):
        return True
    return False


def _trellis_url_host(value: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(str(value)).netloc.lower()
    except Exception:
        return ""


def _trellis_candidate_source_allowed_ikea_mebelru(candidate: dict[str, Any]) -> bool:
    for key in _TRELLIS_SOURCE_KEYS:
        if _trellis_allowed_ikea_mebelru_text(candidate.get(key)):
            return True
    extra = candidate.get("extra")
    if isinstance(extra, dict):
        for key in _TRELLIS_SOURCE_KEYS:
            if _trellis_allowed_ikea_mebelru_text(extra.get(key)):
                return True
    return False


def _trellis_image_item_allowed_ikea_mebelru(item: Any, *, candidate_source_allowed: bool) -> bool:
    if candidate_source_allowed:
        return True
    if isinstance(item, str):
        return _trellis_allowed_ikea_mebelru_text(item)
    if isinstance(item, dict):
        return any(_trellis_allowed_ikea_mebelru_text(item.get(key)) for key in _TRELLIS_IMAGE_DICT_KEYS)
    return False


def _trellis_filter_image_list_ikea_mebelru(value: Any, *, candidate_source_allowed: bool) -> tuple[Any, int, int]:
    was_json = False
    raw = value
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = parsed
                was_json = True
        except Exception:
            pass
    if not isinstance(raw, list):
        return value, 0, 0

    kept: list[Any] = []
    for item in raw:
        if _trellis_image_item_allowed_ikea_mebelru(item, candidate_source_allowed=candidate_source_allowed):
            kept.append(item)
    if was_json:
        return json.dumps(kept, ensure_ascii=False), len(kept), len(raw)
    return kept, len(kept), len(raw)


def _trellis_candidate_has_usable_image_source(candidate: dict[str, Any]) -> bool:
    for key in _TRELLIS_IMAGE_SINGLE_KEYS:
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return True
    images = candidate.get("images")
    if isinstance(images, list) and images:
        return True
    icf = candidate.get("image_color_features") if isinstance(candidate.get("image_color_features"), dict) else {}
    source_image = icf.get("source_image") if isinstance(icf.get("source_image"), dict) else {}
    for key in ("path", "value"):
        value = source_image.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _trellis_filter_candidate_ikea_mebelru_images_only(candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    filtered = _cgs_copy.deepcopy(candidate)
    source_allowed = _trellis_candidate_source_allowed_ikea_mebelru(filtered)
    kept_total = 0
    seen_total = 0

    for key in _TRELLIS_IMAGE_SINGLE_KEYS:
        value = filtered.get(key)
        if isinstance(value, str) and value.strip():
            seen_total += 1
            if _trellis_image_item_allowed_ikea_mebelru(value, candidate_source_allowed=source_allowed):
                kept_total += 1
            else:
                filtered.pop(key, None)

    for key in _TRELLIS_IMAGE_LIST_KEYS:
        new_value, kept, seen = _trellis_filter_image_list_ikea_mebelru(
            filtered.get(key),
            candidate_source_allowed=source_allowed,
        )
        if seen:
            filtered[key] = new_value
            kept_total += kept
            seen_total += seen

    extra = filtered.get("extra")
    if isinstance(extra, dict):
        for key in _TRELLIS_IMAGE_SINGLE_KEYS:
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                seen_total += 1
                if _trellis_image_item_allowed_ikea_mebelru(value, candidate_source_allowed=source_allowed):
                    kept_total += 1
                else:
                    extra.pop(key, None)
        for key in _TRELLIS_IMAGE_LIST_KEYS:
            new_value, kept, seen = _trellis_filter_image_list_ikea_mebelru(
                extra.get(key),
                candidate_source_allowed=source_allowed,
            )
            if seen:
                extra[key] = new_value
                kept_total += kept
                seen_total += seen

    icf = filtered.get("image_color_features") if isinstance(filtered.get("image_color_features"), dict) else {}
    source_image = icf.get("source_image") if isinstance(icf.get("source_image"), dict) else {}
    if source_image:
        source_values = [source_image.get("path"), source_image.get("value")]
        if any(str(v or "").strip() for v in source_values):
            seen_total += 1
            if any(_trellis_image_item_allowed_ikea_mebelru(v, candidate_source_allowed=source_allowed) for v in source_values):
                kept_total += 1
            else:
                icf.pop("source_image", None)

    if not isinstance(filtered.get("images"), list) or not filtered.get("images"):
        images_json = filtered.get("images_json")
        if isinstance(images_json, str):
            try:
                parsed_images_json = json.loads(images_json)
                if isinstance(parsed_images_json, list) and parsed_images_json:
                    filtered["images"] = parsed_images_json
            except Exception:
                pass
        for key in ("image_urls", "photos", "preview_images"):
            if isinstance(filtered.get("images"), list) and filtered.get("images"):
                break
            value = filtered.get(key)
            if isinstance(value, list) and value:
                filtered["images"] = value
                break

    info = {
        "source_allowed": source_allowed,
        "seen_image_sources": seen_total,
        "kept_image_sources": kept_total,
        "unique_key": filtered.get("unique_key") or filtered.get("product_url") or filtered.get("title"),
    }
    if not _trellis_candidate_has_usable_image_source(filtered):
        return None, info
    return filtered, info


def _trellis_candidate_identity(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("unique_key")
        or candidate.get("product_url")
        or candidate.get("model_page_url")
        or candidate.get("source_url")
        or candidate.get("title")
        or candidate.get("id")
        or ""
    ).strip()


def _trellis_filter_card_ikea_mebelru_images_only(card: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()

    def add_candidate(value: Any) -> None:
        if not isinstance(value, dict):
            return
        identity = _trellis_candidate_identity(value) or str(id(value))
        if identity in seen_candidates:
            return
        seen_candidates.add(identity)
        candidates.append(value)

    add_candidate(card)
    for key in _TRELLIS_POOL_KEYS:
        value = card.get(key)
        if isinstance(value, list):
            for item in value:
                add_candidate(item)
    binding = card.get("binding")
    if isinstance(binding, dict):
        for key in _TRELLIS_POOL_KEYS:
            value = binding.get(key)
            if isinstance(value, list):
                for item in value:
                    add_candidate(item)

    filtered_candidates: list[dict[str, Any]] = []
    filter_infos: list[dict[str, Any]] = []
    for candidate in candidates:
        filtered, info = _trellis_filter_candidate_ikea_mebelru_images_only(candidate)
        filter_infos.append(info)
        if filtered is not None:
            filtered_candidates.append(filtered)

    root, root_info = _trellis_filter_candidate_ikea_mebelru_images_only(card)
    root_replaced = False
    if root is None:
        if not filtered_candidates:
            return None, {
                "enabled": True,
                "status": "empty_after_filter",
                "candidate_count": len(candidates),
                "candidate_filter_infos": filter_infos,
            }
        root = _cgs_copy.deepcopy(filtered_candidates[0])
        root_replaced = True

    for key in ("target_id", "supplier_target_id", "layout_target_id", "target_category", "target_semantic_group", "target_size_m"):
        if card.get(key) is not None:
            root[key] = card.get(key)

    root_binding = root.get("binding") if isinstance(root.get("binding"), dict) else {}
    if isinstance(binding, dict):
        for key in ("target_id", "category", "semantic_group", "requested_size_m"):
            if binding.get(key) is not None:
                root_binding[key] = binding.get(key)
    root_binding["top_candidates"] = filtered_candidates
    root["binding"] = root_binding
    for key in ("top_candidates", "supplier_candidate_pool", "candidate_pool", "alternatives"):
        root[key] = filtered_candidates

    meta = root.get("meta") if isinstance(root.get("meta"), dict) else {}
    filter_info = {
        "enabled": True,
        "status": "applied",
        "root_replaced": root_replaced,
        "candidate_count": len(candidates),
        "kept_candidate_count": len(filtered_candidates),
        "selected_unique_key": _trellis_candidate_identity(root),
        "root_filter_info": root_info,
    }
    meta["ikea_mebelru_images_only"] = filter_info
    root["meta"] = meta
    return root, filter_info


def parse_trellis_gpu_ids(value: Any) -> list[int]:
    raw = str(value if value is not None else "0").strip()
    if not raw:
        return [0]
    out: list[int] = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out or [0]


def trellis_image_count_attempts(args: argparse.Namespace) -> list[int]:
    values: list[int] = []
    for raw in (args.trellis_max_images, args.trellis_oom_retry_max_images):
        value = int(raw or 0)
        if value > 0 and value not in values:
            values.append(value)
    return values or [2]


def exception_text(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    if isinstance(exc, subprocess.CalledProcessError):
        for attr in ("stdout", "stderr", "output"):
            value = getattr(exc, attr, None)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def is_trellis_oom_error(exc: Exception) -> bool:
    text = exception_text(exc).lower()
    return "outofmemoryerror" in text or "cuda out of memory" in text


def validate_trellis2_only_cli_args(args: argparse.Namespace) -> None:
    problems: list[str] = []

    def value(name: str) -> str:
        return str(getattr(args, name, "") or "").strip()

    root = value("trellis_remote_trellis_root")
    root_l = root.lower().rstrip("/")
    if root_l in {"/workspace/trellis", "/workspace/trellis-box"} or any(
        token in root_l for token in ("trellis-box", "ltrellis", "trellis1", "trellis.1")
    ):
        problems.append(f"--trellis-remote-trellis-root {root!r} is legacy; use /workspace/TRELLIS.2")

    model = value("trellis_remote_model_dir")
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
        problems.append(f"--trellis-remote-model-dir {model!r} is legacy; use /workspace/models/TRELLIS.2-4B")

    text_model = value("trellis_remote_text_model_dir")
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
            f"--trellis-remote-text-model-dir {text_model!r} is legacy; leave it empty or use /workspace/models/TRELLIS.2-4B"
        )

    remote_python = value("trellis_remote_python")
    remote_python_l = remote_python.lower()
    if remote_python_l == "/venv/trellis/bin/python" or any(
        token in remote_python_l for token in ("ltrellis", "trellis1", "trellis.1")
    ):
        problems.append(f"--trellis-remote-python {remote_python!r} is legacy; use /venv/trellis2/bin/python")

    runner = value("trellis_remote_runner_path")
    if runner:
        problems.append("--trellis-remote-runner-path is disabled; TRELLIS.2 uses run_trellis2_persistent_worker.py")

    if problems:
        raise SystemExit("Legacy TRELLIS backend is disabled:\n- " + "\n- ".join(problems))


def build_trellis_args(
    args: argparse.Namespace,
    *,
    card_json: Path,
    job_id: str,
    out_dir: Path,
    gpu_id: int | None = None,
    max_images: int | None = None,
    prepare_only: bool = False,
    prepared_job_dir: Path | None = None,
    vlm_unload_after_filter: bool | None = None,
) -> argparse.Namespace:
    selected_gpu = parse_trellis_gpu_ids(args.trellis_remote_cuda_visible_devices)[0] if gpu_id is None else int(gpu_id)
    return argparse.Namespace(
        card_json=str(card_json),
        catalog_json="",
        unique_key="",
        out_dir=str(out_dir),
        job_id=job_id,
        prepared_job_dir=str(prepared_job_dir) if prepared_job_dir else "",
        prepare_only=bool(prepare_only),
        server_host=args.trellis_server_host,
        server_port=args.trellis_server_port,
        server_user=args.trellis_server_user,
        ssh_key=args.trellis_ssh_key,
        remote_root=args.trellis_remote_root,
        remote_trellis_root=args.trellis_remote_trellis_root,
        remote_model_dir=args.trellis_remote_model_dir,
        remote_text_model_dir=args.trellis_remote_text_model_dir,
        remote_python=args.trellis_remote_python,
        remote_worker_root=args.trellis_remote_worker_root,
        remote_worker_timeout_sec=args.trellis_remote_worker_timeout_sec,
        remote_worker_poll_sec=args.trellis_remote_worker_poll_sec,
        remote_persistent_worker=args.trellis_remote_persistent_worker,
        remote_cuda_visible_devices=selected_gpu,
        mode="multi_image",
        multi_mode=args.trellis_multi_mode,
        max_images=int(max_images if max_images is not None else args.trellis_max_images),
        seed=args.trellis_seed,
        sparse_steps=args.trellis_sparse_steps,
        slat_steps=args.trellis_slat_steps,
        texture_size=args.trellis_texture_size,
        simplify=args.trellis_simplify,
        pipeline_type=args.trellis_pipeline_type,
        ss_guidance_strength=args.trellis_ss_guidance_strength,
        slat_guidance_strength=args.trellis_slat_guidance_strength,
        decimation_target=args.trellis_decimation_target,
        pre_export_simplify_target=args.trellis_pre_export_simplify_target,
        no_remesh=args.trellis_no_remesh,
        remesh_band=args.trellis_remesh_band,
        remesh_project=args.trellis_remesh_project,
        no_webp=args.trellis_no_webp,
        image_size=args.trellis_image_size,
        fill_holes_resolution=args.trellis_fill_holes_resolution,
        fill_holes_num_views=args.trellis_fill_holes_num_views,
        trellis_max_candidate_pool=args.trellis_max_candidate_pool,
        remote_runner_path=args.trellis_remote_runner_path,
        image_source_index=0,
        single_object_crop=False,
        single_object_crop_component="largest",
        single_object_crop_padding=0.16,
        vlm_single_object_filter=args.trellis_vlm_single_object_filter,
        vlm_provider=args.trellis_vlm_provider,
        vlm_ollama_url=args.trellis_vlm_ollama_url,
        vlm_model=args.trellis_vlm_model,
        vlm_timeout=args.trellis_vlm_timeout,
        vlm_unload_after_filter=args.trellis_vlm_unload_after_filter if vlm_unload_after_filter is None else bool(vlm_unload_after_filter),
        text_fallback_if_no_single_image=args.trellis_text_fallback_if_no_single_image,
        orientation_yaw_deg=None,
        allow_proxy_fallback=bool(getattr(args, "trellis_allow_proxy_fallback", False)),
        ikea_mebelru_images_only=bool(getattr(args, "trellis_ikea_mebelru_images_only", False)),
    )


def run_trellis_with_oom_retries(
    args: argparse.Namespace,
    *,
    card_json: Path,
    job_id: str,
    out_dir: Path,
    prepared_job_dir: Path | None = None,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    gpu_ids = parse_trellis_gpu_ids(args.trellis_remote_cuda_visible_devices)
    image_counts = trellis_image_count_attempts(args)

    last_exc: Exception | None = None
    attempt_index = 0
    for max_images in image_counts:
        for gpu_id in gpu_ids:
            attempt_index += 1
            attempt_job_id = job_id if attempt_index == 1 else f"{job_id}_retry{attempt_index}_gpu{gpu_id}_img{max_images}"
            attempt = {
                "job_id": attempt_job_id,
                "gpu_id": gpu_id,
                "max_images": max_images,
                "status": "running",
            }
            attempts.append(attempt)
            try:
                summary = run_orchestration(
                    build_trellis_args(
                        args,
                        card_json=card_json,
                        job_id=attempt_job_id,
                        out_dir=out_dir,
                        gpu_id=gpu_id,
                        max_images=max_images,
                        prepared_job_dir=prepared_job_dir,
                        prepare_only=False,
                    )
                )
                attempt["status"] = "success"
                return summary
            except Exception as exc:
                last_exc = exc
                attempt["status"] = "failed"
                attempt["oom"] = is_trellis_oom_error(exc)
                attempt["error"] = exception_text(exc)[:4000]
                if not attempt["oom"]:
                    raise

    raise RuntimeError("TRELLIS failed after CUDA OOM retries") from last_exc


def enrich_missing_assets_with_trellis(
    *,
    bindings_json_path: Path,
    output_json_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    data = read_json(bindings_json_path)
    bindings = data.get("bindings") or []
    if not isinstance(bindings, list):
        raise RuntimeError("Некорректный supplier bindings JSON: нет bindings")

    trellis_out_dir = out_dir / "trellis_missing_assets"
    trellis_out_dir.mkdir(parents=True, exist_ok=True)
    generated_by_key: dict[str, dict[str, Any]] = {}
    failed_by_key: dict[str, dict[str, Any]] = {}
    disabled_after_oom_reason = ""
    report: dict[str, Any] = {
        "enabled": True,
        "mode": "vlm_prepare_all_then_unload_then_trellis",
        "input_bindings_json": str(bindings_json_path),
        "output_bindings_json": str(output_json_path),
        "prepared_count": 0,
        "generated_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "items": [],
    }

    pending_jobs: list[dict[str, Any]] = []
    pending_by_key: dict[str, dict[str, Any]] = {}
    max_assets = int(args.trellis_max_assets or 0)
    trellis_skip_categories = {
        str(value).strip().lower()
        for raw in str(getattr(args, "trellis_skip_categories", "") or "").split(",")
        for value in (raw,)
        if str(value).strip()
    }
    trellis_ikea_mebelru_images_only = bool(getattr(args, "trellis_ikea_mebelru_images_only", False))
    report["ikea_mebelru_images_only"] = trellis_ikea_mebelru_images_only
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        if str(binding.get("selection_status") or "") not in SELECTED_BINDING_STATUSES:
            continue
        candidate = binding.get("chosen_candidate")
        if not isinstance(candidate, dict):
            continue
        target_id = str(binding.get("target_id") or "").strip()
        unique_key = str(candidate.get("unique_key") or candidate.get("product_url") or candidate.get("title") or target_id).strip()
        target_category = str(binding.get("category") or binding.get("semantic_group") or candidate.get("semantic_group") or candidate.get("category_norm") or "").strip().lower()
        if target_category in trellis_skip_categories:
            notes = list(binding.get("selection_notes") or [])
            notes.append(f"trellis_generation_skipped_for_category:{target_category}")
            binding["selection_notes"] = notes
            report["skipped_count"] += 1
            report["items"].append(
                {
                    "target_id": target_id,
                    "unique_key": unique_key,
                    "status": "skipped_trellis_category",
                    "category": target_category,
                }
            )
            continue
        if candidate_has_supported_local_asset(candidate):
            report["skipped_count"] += 1
            report["items"].append({"target_id": target_id, "unique_key": unique_key, "status": "skipped_existing_local_asset"})
            continue
        if not unique_key:
            report["skipped_count"] += 1
            report["items"].append({"target_id": target_id, "status": "skipped_missing_unique_key"})
            continue

        trellis_card = build_trellis_card_from_binding(binding, candidate)
        filter_info: dict[str, Any] | None = None
        if trellis_ikea_mebelru_images_only:
            filtered_card, filter_info = _trellis_filter_card_ikea_mebelru_images_only(trellis_card)
            if filtered_card is None:
                notes = list(binding.get("selection_notes") or [])
                notes.append("trellis_generation_skipped_no_ikea_mebelru_images")
                binding["selection_notes"] = notes
                report["skipped_count"] += 1
                report["items"].append(
                    {
                        "target_id": target_id,
                        "unique_key": unique_key,
                        "status": "skipped_no_ikea_mebelru_images",
                        "filter": filter_info,
                    }
                )
                continue
            trellis_card = filtered_card
            unique_key = str(trellis_card.get("unique_key") or trellis_card.get("product_url") or trellis_card.get("title") or unique_key).strip()

        if unique_key in pending_by_key:
            pending_by_key[unique_key].setdefault("dependents", []).append(
                {"binding": binding, "candidate": candidate, "target_id": target_id}
            )
            continue
        if max_assets > 0 and len(pending_jobs) >= max_assets:
            report["skipped_count"] += 1
            report["items"].append({"target_id": target_id, "unique_key": unique_key, "status": "skipped_trellis_max_assets"})
            continue

        job_index = len(pending_jobs) + 1
        card_json = trellis_out_dir / f"{job_index:03d}_{target_id or 'target'}_card.json"
        write_json(card_json, trellis_card)
        job_id = f"procedural_missing_{job_index:03d}_{target_id or 'target'}"
        job = {
            "binding": binding,
            "candidate": candidate,
            "target_id": target_id,
            "unique_key": unique_key,
            "card_json": card_json,
            "job_id": job_id,
            "prepared_job_dir": None,
            "prepare_summary": None,
            "dependents": [],
            "filter_info": filter_info,
        }
        pending_jobs.append(job)
        pending_by_key[unique_key] = job
        report["prepared_count"] += 1

    if pending_jobs and args.trellis_vlm_provider == "ollama" and args.trellis_vlm_unload_after_filter:
        unload_report = unload_ollama_model(
            ollama_url=args.trellis_vlm_ollama_url,
            model=args.trellis_vlm_model,
            timeout_sec=max(30, int(args.trellis_vlm_timeout or 120)),
        )
        report["vlm_unload_after_prepare_all"] = unload_report
        if unload_report.get("ok"):
            print(f"[VLM] unloaded Ollama model after preparing all TRELLIS images: {args.trellis_vlm_model}", file=sys.stderr)
        else:
            print(f"[WARN] failed to unload Ollama model after prepare phase: {unload_report.get('error')}", file=sys.stderr)

    for job in pending_jobs:
        binding = job["binding"]
        candidate = job["candidate"]
        target_id = str(job["target_id"])
        unique_key = str(job["unique_key"])
        if disabled_after_oom_reason:
            report["skipped_count"] += 1
            notes = list(binding.get("selection_notes") or [])
            notes.append("trellis_generation_skipped_after_oom")
            binding["selection_notes"] = notes
            report["items"].append(
                {
                    "target_id": target_id,
                    "unique_key": unique_key,
                    "status": "skipped_trellis_disabled_after_oom",
                    "reason": disabled_after_oom_reason,
                }
            )
            for dep in job.get("dependents", []):
                report["skipped_count"] += 1
                dep_binding = dep["binding"]
                dep_notes = list(dep_binding.get("selection_notes") or [])
                dep_notes.append("trellis_generation_skipped_after_oom")
                dep_binding["selection_notes"] = dep_notes
                report["items"].append(
                    {
                        "target_id": dep["target_id"],
                        "unique_key": unique_key,
                        "status": "skipped_trellis_disabled_after_oom",
                        "reason": disabled_after_oom_reason,
                    }
                )
            continue

        trellis_attempts: list[dict[str, Any]] = []
        try:
            summary = run_trellis_with_oom_retries(
                args,
                card_json=job["card_json"],
                job_id=job["job_id"],
                out_dir=trellis_out_dir,
                prepared_job_dir=job.get("prepared_job_dir"),
                attempts=trellis_attempts,
            )
            patched_card_path = Path(summary["card_with_trellis_asset_json"]).expanduser().resolve()
            patched_card = read_json(patched_card_path)
            if not candidate_has_supported_local_asset(patched_card):
                raise RuntimeError("TRELLIS finished but card has no local GLB/GLTF asset path")
            selected_unique_key = str(patched_card.get("unique_key") or unique_key)
            generated_by_key[selected_unique_key] = patched_card
            binding["chosen_candidate"] = patched_card
            apply_trellis_card_to_candidate(binding["chosen_candidate"], patched_card)
            report["generated_count"] += 1
            report["items"].append(
                {
                    "target_id": target_id,
                    "unique_key": selected_unique_key,
                    "status": "generated",
                    "asset_local_path": patched_card.get("asset_local_path"),
                    "summary_json": str(Path(summary["local_job_dir"]) / "summary.json"),
                    "trellis_attempts": trellis_attempts,
                    "filter": job.get("filter_info"),
                }
            )
            for dep in job.get("dependents", []):
                apply_trellis_card_to_candidate(dep["candidate"], patched_card)
                report["generated_count"] += 1
                report["items"].append({"target_id": dep["target_id"], "unique_key": unique_key, "status": "reused_generated_asset"})
        except Exception as exc:
            report["failed_count"] += 1
            oom_attempts = [attempt for attempt in trellis_attempts if attempt.get("oom")]
            failure_record = {
                "error": exception_text(exc)[:4000],
                "oom": bool(oom_attempts) and len(oom_attempts) == len(trellis_attempts),
                "attempt_count": len(trellis_attempts),
            }
            if unique_key:
                failed_by_key[unique_key] = failure_record
            if failure_record["oom"] and args.trellis_disable_after_oom:
                disabled_after_oom_reason = "trellis_cuda_oom_all_attempts"
            notes = list(binding.get("selection_notes") or [])
            notes.append(f"trellis_generation_failed:{type(exc).__name__}")
            binding["selection_notes"] = notes
            report["items"].append(
                {
                    "target_id": target_id,
                    "unique_key": unique_key,
                    "status": "failed",
                    "error": exception_text(exc)[:4000],
                    "trellis_attempts": trellis_attempts,
                }
            )
            for dep in job.get("dependents", []):
                report["skipped_count"] += 1
                dep_binding = dep["binding"]
                dep_notes = list(dep_binding.get("selection_notes") or [])
                dep_notes.append("trellis_generation_skipped_same_unique_key_failed")
                dep_binding["selection_notes"] = dep_notes
                report["items"].append(
                    {
                        "target_id": dep["target_id"],
                        "unique_key": unique_key,
                        "status": "skipped_same_unique_key_failed",
                        "previous_failure": failure_record,
                    }
                )

    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    meta["trellis_missing_asset_generation"] = {
        "mode": report["mode"],
        "prepared_count": report["prepared_count"],
        "generated_count": report["generated_count"],
        "skipped_count": report["skipped_count"],
        "failed_count": report["failed_count"],
        "out_dir": str(trellis_out_dir),
        "disabled_after_oom": bool(disabled_after_oom_reason),
        "disabled_after_oom_reason": disabled_after_oom_reason or None,
        "ikea_mebelru_images_only": trellis_ikea_mebelru_images_only,
    }
    write_json(output_json_path, data)
    write_json(out_dir / "trellis_missing_assets.report.json", report)
    return output_json_path, report


def blender_binary(args: argparse.Namespace) -> str:
    if args.blender:
        return str(args.blender)
    mac_blender = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac_blender.is_file():
        return str(mac_blender)
    return "blender"


def maybe_build_blend_scene(args: argparse.Namespace, *, supplier_scene: Path, out_dir: Path) -> dict[str, Any] | None:
    if not args.build_blend:
        return None
    scene_builder = Path(args.scene_builder_script).expanduser()
    if not scene_builder.is_absolute():
        scene_builder = _repo_root() / scene_builder
    out_blend = Path(args.out_blend).expanduser() if args.out_blend else out_dir / "final_supplier_procedural.blend"
    if not out_blend.is_absolute():
        out_blend = out_dir / out_blend
    out_png = Path(args.out_png).expanduser() if args.out_png else None
    if out_png is not None and not out_png.is_absolute():
        out_png = out_dir / out_png
    build_report = out_dir / "final_supplier_procedural.build_report.json"
    cmd = [
        blender_binary(args),
        "--background",
        "--python",
        str(scene_builder),
        "--",
        "--json",
        str(supplier_scene),
        "--save-blend",
        str(out_blend),
        "--build-report",
        str(build_report),
    ]
    if out_png is not None:
        cmd.extend(["--render", str(out_png)])
    subprocess.run(cmd, check=True)
    return {
        "blend": str(out_blend),
        "png": str(out_png) if out_png else None,
        "build_report": str(build_report),
        "command": cmd,
    }


def _room_point_xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _room_has_door(room: dict[str, Any]) -> bool:
    if isinstance(room.get("doors"), list) and any(isinstance(item, dict) for item in room["doors"]):
        return True
    openings = room.get("openings")
    if isinstance(openings, dict):
        doors = openings.get("doors")
        return isinstance(doors, list) and any(isinstance(item, dict) for item in doors)
    if isinstance(openings, list):
        return any(isinstance(item, dict) and str(item.get("type") or "").strip().lower() == "door" for item in openings)
    return False


def _room_openings_from_anywhere(room: dict[str, Any], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add_many(value: Any) -> None:
        if isinstance(value, list):
            out.extend([dict(item) for item in value if isinstance(item, dict)])

    add_many(room.get(key))
    openings = room.get("openings")
    if isinstance(openings, dict):
        add_many(openings.get(key))
    elif isinstance(openings, list):
        expected_type = "door" if key == "doors" else "window"
        out.extend(
            dict(item)
            for item in openings
            if isinstance(item, dict) and str(item.get("type") or "").strip().lower() == expected_type
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        marker = str(item.get("id") or item.get("wall_id") or "") + "|" + str(item.get("s") or item.get("x") or "")
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _room_wall_points(room: dict[str, Any], wall: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    poly = room.get("floor_polygon") or room.get("floor_polygon_xz") or room.get("vertices") or []
    if "from_vertex" in wall and "to_vertex" in wall and isinstance(poly, list):
        try:
            p0 = _room_point_xy(poly[int(wall["from_vertex"])])
            p1 = _room_point_xy(poly[int(wall["to_vertex"])])
        except Exception:
            return None
        if p0 is not None and p1 is not None:
            return p0, p1
    for from_key, to_key in (("from", "to"), ("start", "end"), ("p0", "p1")):
        p0 = _room_point_xy(wall.get(from_key))
        p1 = _room_point_xy(wall.get(to_key))
        if p0 is not None and p1 is not None:
            return p0, p1
    return None


def ensure_room_has_door(room_json: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _cgs_copy.deepcopy(room_json)
    room = data.get("room") if isinstance(data.get("room"), dict) else data
    if not isinstance(room, dict):
        return data, {"changed": False, "reason": "missing_room"}
    if _room_has_door(room):
        changed = False
        doors = _room_openings_from_anywhere(room, "doors")
        windows = _room_openings_from_anywhere(room, "windows")
        if doors and not (isinstance(room.get("doors"), list) and room.get("doors")):
            room["doors"] = doors
            changed = True
        if windows and not (isinstance(room.get("windows"), list) and room.get("windows")):
            room["windows"] = windows
            changed = True
        if changed:
            meta = data.setdefault("meta", {}) if isinstance(data, dict) else {}
            if isinstance(meta, dict):
                meta["procedural_room_supplier_ensure_door"] = {"changed": True, "reason": "normalized_existing_openings"}
            return data, {"changed": True, "reason": "normalized_existing_openings", "door_count": len(doors)}
        return data, {"changed": False, "reason": "existing_door"}

    walls = room.get("walls")
    poly = room.get("floor_polygon") or room.get("floor_polygon_xz") or []
    if not isinstance(walls, list) or not walls:
        if not isinstance(poly, list) or len(poly) < 3:
            return data, {"changed": False, "reason": "missing_walls_and_floor_polygon"}
        walls = [{"id": f"w{i}", "from_vertex": i, "to_vertex": (i + 1) % len(poly)} for i in range(len(poly))]
        room["walls"] = walls

    candidates: list[tuple[float, float, int, dict[str, Any], float]] = []
    for index, wall in enumerate(walls):
        if not isinstance(wall, dict):
            continue
        pts = _room_wall_points(room, wall)
        if pts is None:
            continue
        (x0, y0), (x1, y1) = pts
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1.05:
            continue
        # Prefer the lower/front wall for a normal room entrance, then longer walls.
        candidates.append(((y0 + y1) * 0.5, -length, index, wall, length))
    if not candidates:
        return data, {"changed": False, "reason": "no_wall_long_enough"}

    _, _, index, wall, length = sorted(candidates)[0]
    wall_id = str(wall.get("id") or f"w{index}")
    wall["id"] = wall_id
    door_width = min(0.9, max(0.75, length * 0.18))
    s0 = max(0.10, min(max(length - door_width - 0.10, 0.0), (length - door_width) * 0.15))
    door = {
        "id": "door_main_0001",
        "type": "door",
        "wall_id": wall_id,
        "s": round(s0, 3),
        "width": round(door_width, 3),
        "height": 2.05,
        "z0": 0.0,
        "swing": "inward",
        "source": "procedural_room_supplier_ensure_door",
    }
    doors_list = room.get("doors")
    if not isinstance(doors_list, list):
        doors_list = []
        room["doors"] = doors_list
    doors_list.append(door)

    openings = room.get("openings")
    if isinstance(openings, dict):
        openings.setdefault("doors", []).append(dict(door))
        if "windows" not in openings:
            windows = _room_openings_from_anywhere(room, "windows")
            if windows:
                openings["windows"] = windows
    elif isinstance(openings, list):
        openings.append(dict(door))
    else:
        windows = _room_openings_from_anywhere(room, "windows")
        room["openings"] = {"doors": [dict(door)], "windows": windows}
    meta = data.setdefault("meta", {}) if isinstance(data, dict) else {}
    if isinstance(meta, dict):
        meta["procedural_room_supplier_ensure_door"] = {"changed": True, "door": door}
    return data, {"changed": True, "door": door}


def _scene_windows(scene: dict[str, Any]) -> list[dict[str, Any]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    windows = _room_openings_from_anywhere(room, "windows") if isinstance(room, dict) else []
    if not windows:
        return []
    return windows


def _scene_has_curtain_items(scene: dict[str, Any]) -> bool:
    items = scene.get("items") if isinstance(scene.get("items"), list) else scene.get("placements")
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            continue
        if _is_curtain_item(item) and not _is_generated_placeholder_curtain(item):
            return True
    return False


def _is_curtain_item(item: dict[str, Any]) -> bool:
    text = " ".join(str(item.get(key) or "") for key in ("id", "name", "category", "semantic_group")).lower()
    source = item.get("source")
    if isinstance(source, dict):
        text += " " + " ".join(str(value or "") for value in source.values()).lower()
    asset = item.get("asset")
    if isinstance(asset, dict):
        text += " " + str(asset.get("kind") or "").lower()
    return any(token in text for token in ("curtain", "shtor", "штор", "занавес", "window_covering"))


def _is_generated_placeholder_curtain(item: dict[str, Any]) -> bool:
    if not _is_curtain_item(item):
        return False
    asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    has_texture = bool(item.get("texture_path") or asset.get("texture_path"))
    has_mesh = bool(asset.get("mesh_path") or asset.get("mesh_local_path") or asset.get("asset_local_path"))
    asset_kind = str(asset.get("kind") or "").strip().lower()
    return bool(
        not has_texture
        and not has_mesh
        and asset_kind in {"", "procedural_placeholder"}
        and (
            bool(meta.get("procedural"))
            or str(source.get("placement_source") or "").startswith("procedural_room_stage")
        )
    )


def _strip_generated_placeholder_curtains(scene: dict[str, Any]) -> tuple[dict[str, Any], int]:
    out = _cgs_copy.deepcopy(scene)
    removed_total = 0
    canonical: list[dict[str, Any]] | None = None
    for key in ("items", "placements"):
        items = out.get(key)
        if not isinstance(items, list):
            continue
        kept = [item for item in items if not (isinstance(item, dict) and _is_generated_placeholder_curtain(item))]
        removed_total += len(items) - len(kept)
        out[key] = kept
        if key == "items":
            canonical = kept
    if canonical is not None and isinstance(out.get("placements"), list):
        out["placements"] = _cgs_copy.deepcopy(canonical)
    return out, removed_total


def _curtains_needed_for_scene(*, scene: dict[str, Any], prompt_text: str, style_profile: dict[str, Any], policy: str) -> tuple[bool, str]:
    if not _scene_windows(scene):
        return False, "missing_windows"
    if _scene_has_curtain_items(scene):
        return False, "existing_curtains"
    if policy == "always":
        return True, "policy_always"

    text_parts = [
        prompt_text,
        str(style_profile.get("expanded_prompt") or ""),
        str(style_profile.get("style_hint") or ""),
        str(style_profile.get("surface_design_brief") or ""),
        str(style_profile.get("chooser_prompt") or ""),
        str(style_profile.get("room_type") or ""),
    ]
    text = " ".join(part for part in text_parts if part).lower()
    if any(
        token in text
        for token in (
            "no curtain",
            "no curtains",
            "without curtain",
            "without curtains",
            "без штор",
            "без занавес",
            "не нужны шторы",
            "шторы не нужны",
            "без жалюзи",
        )
    ):
        return False, "prompt_says_no_curtains"

    for key in ("needs_curtains", "wants_curtains", "curtains", "window_coverings"):
        if bool(style_profile.get(key)):
            return True, f"profile_{key}"

    if any(
        token in text
        for token in (
            "curtain",
            "curtains",
            "drape",
            "drapes",
            "window treatment",
            "window covering",
            "tulle",
            "blind",
            "blinds",
            "штор",
            "занавес",
            "тюль",
            "гардин",
            "портьер",
            "жалюзи",
        )
    ):
        return True, "prompt_mentions_curtains"

    room_type = str(style_profile.get("room_type") or "").strip().lower().replace(" ", "_")
    if room_type in {"bedroom", "livingroom", "living_room", "kids_room", "nursery", "детская", "спальня", "гостиная"}:
        return True, f"default_for_room_type:{room_type}"
    return False, "auto_not_requested"


def maybe_apply_curtains_to_supplier_scene(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    supplier_scene: Path,
    prompt: str,
    room_design_spec: dict[str, Any],
) -> tuple[Path, dict[str, Any] | None]:
    policy = str(getattr(args, "curtains", "auto") or "auto").strip().lower()
    if bool(getattr(args, "no_curtains", False)):
        policy = "never"
    if policy in {"off", "false", "0", "no"}:
        policy = "never"
    if policy in {"on", "true", "1", "yes"}:
        policy = "always"
    if policy not in {"auto", "always", "never"}:
        policy = "auto"
    if policy == "never":
        return supplier_scene, {"added_count": 0, "skipped_reason": "policy_never", "policy": policy}

    scene = read_json(supplier_scene)
    stripped_placeholder_curtains = 0
    needed, needed_reason = _curtains_needed_for_scene(
        scene=scene if isinstance(scene, dict) else {},
        prompt_text=prompt,
        style_profile=room_design_spec if isinstance(room_design_spec, dict) else {},
        policy=policy,
    )
    if not needed:
        return supplier_scene, {"added_count": 0, "skipped_reason": needed_reason, "policy": policy}
    if isinstance(scene, dict):
        scene, stripped_placeholder_curtains = _strip_generated_placeholder_curtains(scene)

    materials_path = Path(str(getattr(args, "curtain_materials", "") or "")).expanduser()
    if not materials_path.is_absolute():
        materials_path = (_repo_root() / materials_path).resolve()
    if not (materials_path.is_file() or materials_path.is_dir()):
        return supplier_scene, {"added_count": 0, "skipped_reason": "missing_curtain_materials", "path": str(materials_path), "policy": policy}

    catalog, catalog_base_dir = load_curtain_catalog(materials_path)
    if not catalog:
        return supplier_scene, {"added_count": 0, "skipped_reason": "empty_curtain_catalog", "path": str(materials_path), "policy": policy}

    models_dir = Path(str(getattr(args, "curtain_models_dir", "") or "")).expanduser()
    if not models_dir.is_absolute():
        models_dir = (_repo_root() / models_dir).resolve()
    curtain_model_paths = discover_curtain_models(models_dir)

    supplier_catalog_path = Path(str(getattr(args, "curtain_supplier_catalog", "") or "")).expanduser()
    if not supplier_catalog_path.is_absolute():
        supplier_catalog_path = (_repo_root() / supplier_catalog_path).resolve()
    supplier_curtain_models = discover_supplier_curtain_models(
        supplier_catalog_path=supplier_catalog_path,
        manual_assets_root=_repo_root() / "data/sourse/suppliers/manual_assets/3ddd",
    )

    seed = int(getattr(args, "curtain_seed", 0) or 0) or int(getattr(args, "seed", 0) or 0)
    scene_with_curtains, info = apply_curtains_to_scene(
        scene,
        catalog=catalog,
        catalog_base_dir=catalog_base_dir,
        curtain_model_paths=curtain_model_paths,
        curtain_models=supplier_curtain_models,
        style_profile=room_design_spec if isinstance(room_design_spec, dict) else {},
        seed=seed,
    )
    if isinstance(scene_with_curtains.get("items"), list) and isinstance(scene_with_curtains.get("placements"), list):
        scene_with_curtains["placements"] = _cgs_copy.deepcopy(scene_with_curtains["items"])
    if int(info.get("added_count", 0) or 0) <= 0:
        info.setdefault("policy", policy)
        info.setdefault("needed_reason", needed_reason)
        info.setdefault("stripped_placeholder_curtain_count", stripped_placeholder_curtains)
        return supplier_scene, info

    scene_out_path = out_dir / f"{supplier_scene.stem}.curtains.v1.json"
    write_curtain_json(scene_out_path, scene_with_curtains)
    first = (info.get("selected") or [{}])[0]
    print(
        "[CURTAINS] selected "
        f"added={info.get('added_count')} first={first.get('sku')} {first.get('name')} texture={first.get('texture_path')}",
        flush=True,
    )
    return scene_out_path, {
        "scene_v1": str(scene_out_path.resolve()),
        "catalog_path": str(materials_path.resolve()),
        "models_dir": str(models_dir.resolve()),
        "supplier_catalog_path": str(supplier_catalog_path.resolve()),
        "policy": policy,
        "needed_reason": needed_reason,
        "stripped_placeholder_curtain_count": stripped_placeholder_curtains,
        **info,
    }


def maybe_apply_topview_vlm_orientation_repair(
    args: argparse.Namespace,
    *,
    supplier_scene: Path,
    out_dir: Path,
    blend_artifacts: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any] | None, dict[str, Any] | None]:
    if not bool(getattr(args, "topview_vlm_orientation_repair", False)):
        return supplier_scene, None, blend_artifacts
    if not args.build_blend or not blend_artifacts or not blend_artifacts.get("blend"):
        return supplier_scene, {"skipped_reason": "requires_build_blend"}, blend_artifacts

    blend_path = Path(str(blend_artifacts["blend"])).expanduser().resolve()
    if not blend_path.is_file():
        return supplier_scene, {"skipped_reason": "blend_not_found", "blend": str(blend_path)}, blend_artifacts

    scene = read_json(supplier_scene)
    if not isinstance(scene, dict):
        return supplier_scene, {"skipped_reason": "invalid_scene_json"}, blend_artifacts

    target_scope = str(getattr(args, "topview_vlm_target_scope", "all") or "all")
    include_armchairs = bool(getattr(args, "topview_vlm_include_armchairs", True))
    refs = collect_topview_vlm_scene_objects(scene, max_objects=int(getattr(args, "topview_vlm_max_objects", 10000) or 10000))
    targets = filter_topview_vlm_target_objects(refs, scope=target_scope, include_armchairs=include_armchairs)
    if not targets:
        return supplier_scene, {"skipped_reason": "no_target_objects", "target_scope": target_scope}, blend_artifacts

    label_map = {f"C{i + 1}": ref.object_id for i, ref in enumerate(targets)}
    target_ids = ",".join(ref.object_id for ref in targets)
    label_map_path = out_dir / f"topview_vlm_orientation.{target_scope}.target_label_map.json"
    write_json(label_map_path, label_map)

    for key in ("blend", "png", "build_report"):
        value = blend_artifacts.get(key)
        if not value:
            continue
        src = Path(str(value)).expanduser()
        if src.is_file():
            dst = src.with_name(src.stem + ".before_topview_orientation" + src.suffix)
            try:
                shutil.copy2(src, dst)
                blend_artifacts[f"before_topview_orientation_{key}"] = str(dst)
            except Exception:
                pass

    render_script = (_repo_root() / "src/tools/render_saved_blend_top_view.py").resolve()
    topview_image = out_dir / f"topview_vlm_orientation.{target_scope}.before.png"
    render_cmd = [
        blender_binary(args),
        str(blend_path),
        "-b",
        "--python",
        str(render_script),
        "--",
        "--out",
        str(topview_image),
        "--azimuth-deg",
        "-90",
        "--elevation-deg",
        str(float(getattr(args, "topview_vlm_elevation_deg", 80.0) or 80.0)),
        "--radius-mult",
        str(float(getattr(args, "topview_vlm_radius_mult", 0.55) or 0.55)),
        "--lens",
        str(float(getattr(args, "topview_vlm_lens", 32.0) or 32.0)),
        "--resolution-x",
        str(int(getattr(args, "topview_vlm_resolution_x", 1400) or 1400)),
        "--resolution-y",
        str(int(getattr(args, "topview_vlm_resolution_y", 1050) or 1050)),
        "--scene-json",
        str(supplier_scene),
        "--target-ids",
        target_ids,
        "--target-label-map",
        str(label_map_path),
        "--target-scope",
        target_scope,
        "--highlight-targets",
        "--highlight-style",
        "label_only",
    ]
    if include_armchairs:
        render_cmd.append("--include-armchairs")
    render_proc = subprocess.run(render_cmd, check=False)
    if render_proc.returncode != 0:
        topview_ready = topview_image.is_file() and topview_image.stat().st_size > 0
        # Blender can abort while tearing down a .blend that contains bad third-party
        # FBX texture records, even after the requested top-view PNG was written.
        # In that case the rendered image is still usable for the VLM review.
        if topview_ready and render_proc.returncode in {-6, 134}:
            print(
                f"[TOPVIEW][render-warning] Blender exited with code {render_proc.returncode} "
                f"after writing {topview_image}; continuing",
                file=sys.stderr,
                flush=True,
            )
        else:
            raise subprocess.CalledProcessError(render_proc.returncode, render_cmd)

    out_scene = out_dir / f"{supplier_scene.stem}.topview_oriented.v1.json"
    out_review = out_dir / f"topview_vlm_orientation.{target_scope}.review.json"
    out_report = out_dir / f"topview_vlm_orientation.{target_scope}.report.json"
    out_prompt = out_dir / f"topview_vlm_orientation.{target_scope}.prompt.txt"
    report = run_topview_vlm_orientation_repair(
        scene_path=supplier_scene,
        image_path=topview_image,
        out_scene_path=out_scene,
        out_review_path=out_review,
        out_report_path=out_report,
        out_prompt_path=out_prompt,
        provider=str(getattr(args, "topview_vlm_provider", "ollama") or "ollama"),
        model=str(getattr(args, "topview_vlm_model", "") or "").strip() or None,
        max_objects=int(getattr(args, "topview_vlm_max_objects", 10000) or 10000),
        target_scope=target_scope,
        include_armchairs=include_armchairs,
        min_confidence=float(getattr(args, "topview_vlm_min_confidence", 0.80) or 0.80),
        max_delta_deg=float(getattr(args, "topview_vlm_max_delta_deg", 180.0) or 180.0),
        snap_step_deg=float(getattr(args, "topview_vlm_snap_step_deg", 90.0) or 90.0),
        target_label_map_path=label_map_path,
        max_repairs_per_object=int(getattr(args, "topview_vlm_max_repairs_per_object", 1) or 1),
        visual_front_offset_deg=float(getattr(args, "topview_vlm_visual_front_offset_deg", 0.0) or 0.0),
        apply=True,
    )

    rebuilt_artifacts = maybe_build_blend_scene(args, supplier_scene=out_scene, out_dir=out_dir)
    return out_scene, {
        "input_scene_json": str(supplier_scene),
        "output_scene_json": str(out_scene),
        "topview_image": str(topview_image),
        "target_label_map": str(label_map_path),
        "review_json": str(out_review),
        "report_json": str(out_report),
        "prompt_txt": str(out_prompt),
        "target_scope": target_scope,
        "target_count": len(targets),
        "report": report,
    }, rebuilt_artifacts or blend_artifacts


def build_scene_from_room(room_json: dict[str, Any]) -> dict[str, Any]:
    room = room_json["room"] if isinstance(room_json.get("room"), dict) else room_json
    return {
        "schema": "scene.v1",
        "room": room,
        "placements": [],
        "meta": {
            "placer": "procedural_room_stage",
            "mode": "procedural_room_supplier",
        },
    }


class StageTimings:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.started_at = time.perf_counter()
        self.records: list[dict[str, Any]] = []
        self.total_seconds: float | None = None

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            seconds = time.perf_counter() - start
            record = {
                "stage": str(name),
                "seconds": round(seconds, 3),
                "status": status,
            }
            self.records.append(record)
            suffix = "" if status == "ok" else " failed"
            print(f"[TIMER] {name}: {seconds:.3f}s{suffix}", file=sys.stderr, flush=True)

    def finish(self) -> None:
        if not self.enabled or self.total_seconds is not None:
            return
        self.total_seconds = time.perf_counter() - self.started_at
        print(f"[TIMER] total: {self.total_seconds:.3f}s", file=sys.stderr, flush=True)

    def report(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        total = self.total_seconds
        if total is None:
            total = time.perf_counter() - self.started_at
        return {
            "schema": "stage_timings/v1",
            "total_seconds": round(float(total), 3),
            "stages": list(self.records),
        }


def _surface_style_from_design_spec(room_design_spec: dict[str, Any]) -> str:
    style = room_design_spec.get("style") if isinstance(room_design_spec.get("style"), dict) else {}
    raw = str(style.get("primary") or "contemporary").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "soft_classic": "classic",
        "soft_traditional": "classic",
        "residential_classic": "classic",
        "modern": "contemporary",
        "minimalism": "contemporary",
        "loft_industrial": "loft",
    }
    return aliases.get(raw, raw or "contemporary")


def _room_type_from_scene(scene_path: Path) -> str | None:
    try:
        data = read_json(scene_path)
    except Exception:
        return None
    room = data.get("room") if isinstance(data, dict) else {}
    if not isinstance(room, dict):
        return None
    return str(room.get("room_type") or room.get("type_hint") or "").strip().lower() or None


def _surface_prompt(prompt: str, room_design_spec: dict[str, Any]) -> str:
    palette = room_design_spec.get("color_palette") if isinstance(room_design_spec.get("color_palette"), dict) else {}
    materials = room_design_spec.get("materials") if isinstance(room_design_spec.get("materials"), dict) else {}
    user_prompt = str(prompt or "").strip()
    parts = [user_prompt]
    if user_prompt:
        parts.append("User prompt is the highest priority for floor and wall material selection; do not override explicit colors, tones, styles, or materials from it.")
    preferred = palette.get("preferred_colors") or palette.get("primary") or []
    forbidden = palette.get("forbidden_colors") or palette.get("forbidden") or []
    if preferred:
        parts.append(f"Preferred surface colors: {', '.join(str(x) for x in preferred)}.")
    if forbidden:
        parts.append(f"Avoid surface colors: {', '.join(str(x) for x in forbidden)}.")
    preferred_materials = materials.get("preferred") if isinstance(materials, dict) else None
    if preferred_materials:
        parts.append(f"Preferred materials: {', '.join(str(x) for x in preferred_materials)}.")
    if not user_prompt and not preferred and not forbidden and not preferred_materials:
        parts.append("Fallback preference: light natural wood flooring and warm beige or cream subtle textured wallpaper/plaster-like wall covering.")
    return "\n".join(part for part in parts if part)


def _llm_settings_from_args(args: argparse.Namespace, prefix: str, *, default_provider: str = "ollama") -> dict[str, Any]:
    return {
        "provider": str(getattr(args, f"{prefix}_llm_provider", default_provider) or default_provider),
        "ollama_url": str(getattr(args, f"{prefix}_ollama_url", "") or "http://127.0.0.1:11434"),
        "ollama_model": str(getattr(args, f"{prefix}_ollama_model", "") or "gpt-oss:20b"),
        "ollama_timeout": int(getattr(args, f"{prefix}_ollama_timeout", None) or 180),
        "ollama_temperature": float(getattr(args, f"{prefix}_ollama_temperature", 0.0) or 0.0),
        "ollama_num_ctx": int(getattr(args, f"{prefix}_ollama_num_ctx", 8192) or 8192),
        "top_n": int(getattr(args, f"{prefix}_llm_top_n", 5) or 5),
    }


def maybe_apply_surface_materials(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    supplier_scene: Path,
    supplier_placement: Path,
    prompt: str,
    room_design_spec: dict[str, Any],
    timings: StageTimings | None = None,
) -> tuple[Path, Path, dict[str, Any] | None]:
    info: dict[str, Any] = {"schema": "procedural_surface_materials/v1"}
    current_scene = supplier_scene
    current_placement = supplier_placement
    style = _surface_style_from_design_spec(room_design_spec)
    room_type = _room_type_from_scene(current_scene)
    selector_prompt = _surface_prompt(prompt, room_design_spec)

    if not bool(getattr(args, "no_flooring", False)):
        materials_path = Path(str(args.flooring_materials or "data/floor_materials")).expanduser()
        style_rules_path = Path(str(args.flooring_style_rules or "config/flooring_style_rules.json")).expanduser()
        if not materials_path.is_absolute():
            materials_path = (_repo_root() / materials_path).resolve()
        if not style_rules_path.is_absolute():
            style_rules_path = (_repo_root() / style_rules_path).resolve()
        with (timings.stage("flooring_selection_apply") if timings else nullcontext()):
            if (materials_path.is_dir() or materials_path.is_file()) and style_rules_path.is_file():
                selection_path = out_dir / "flooring.selection.supplier.v1.json"
                selection = run_flooring_selection(
                    prompt=selector_prompt,
                    style=style,
                    room_type=room_type,
                    room_description=selector_prompt,
                    room_id="room_001",
                    materials_path=materials_path,
                    style_rules_path=style_rules_path,
                    out_path=selection_path,
                    top_k=int(args.flooring_top_k or 10),
                    llm_settings=_llm_settings_from_args(args, "flooring"),
                )
                scene_with_floor = apply_flooring_to_scene(read_json(current_scene), selection)
                placement_with_floor = apply_flooring_to_scene(read_json(current_placement), selection)
                current_scene = out_dir / f"{current_scene.stem}.flooring.v1.json"
                current_placement = out_dir / f"{current_placement.stem}.flooring.v1.json"
                write_json(current_scene, scene_with_floor)
                write_json(current_placement, placement_with_floor)
                selected = selection.get("selected_material") or {}
                texture = selection.get("texture_candidate") or {}
                info["flooring"] = {
                    "selection_json": str(selection_path),
                    "scene_v1": str(current_scene),
                    "placement_v1": str(current_placement),
                    "selected_sku": selected.get("sku"),
                    "selected_name": selected.get("name"),
                    "selected_product_url": selected.get("product_url"),
                    "selected_price": selected.get("price"),
                    "selected_price_currency": selected.get("price_currency"),
                    "package_area_m2": selected.get("package_area_m2"),
                    "texture_path": texture.get("texture_abs_path") or texture.get("texture_path"),
                    "llm_rerank": selection.get("llm_rerank"),
                }
            else:
                info["flooring"] = {"skipped": True, "reason": "materials_or_style_rules_missing"}

    if not bool(getattr(args, "no_wall_material", False)):
        materials_path = Path(str(args.wall_materials or "data/floor_materials")).expanduser()
        if not materials_path.is_absolute():
            materials_path = (_repo_root() / materials_path).resolve()
        with (timings.stage("wall_material_selection_apply") if timings else nullcontext()):
            if materials_path.is_dir() or materials_path.is_file():
                selection_path = out_dir / "wall_material.selection.supplier.v1.json"
                selection = run_wall_selection(
                    prompt=selector_prompt,
                    style=style,
                    room_type=room_type,
                    room_description=selector_prompt,
                    room_id="room_001",
                    materials_path=materials_path,
                    out_path=selection_path,
                    top_k=int(args.wall_top_k or 10),
                    llm_settings=_llm_settings_from_args(args, "wall"),
                )
                scene_with_wall = apply_wall_material_to_scene_with_catalog(read_json(current_scene), selection, materials_path=materials_path)
                placement_with_wall = apply_wall_material_to_scene_with_catalog(read_json(current_placement), selection, materials_path=materials_path)
                current_scene = out_dir / f"{current_scene.stem}.wall_material.v1.json"
                current_placement = out_dir / f"{current_placement.stem}.wall_material.v1.json"
                write_json(current_scene, scene_with_wall)
                write_json(current_placement, placement_with_wall)
                selected = selection.get("selected_material") or {}
                info["wall_material"] = {
                    "selection_json": str(selection_path),
                    "scene_v1": str(current_scene),
                    "placement_v1": str(current_placement),
                    "selected_sku": selected.get("sku"),
                    "selected_name": selected.get("name"),
                    "selected_product_url": selected.get("product_url"),
                    "selected_price": selected.get("price"),
                    "selected_price_currency": selected.get("price_currency"),
                    "roll_width_cm": selected.get("width_cm"),
                    "roll_length_m": selected.get("length_m"),
                    "average_hex": selected.get("average_hex"),
                    "dominant_colors_hex": selected.get("dominant_colors_hex"),
                    "llm_rerank": selection.get("llm_rerank"),
                }
            else:
                info["wall_material"] = {"skipped": True, "reason": "materials_missing"}

    if set(info.keys()) == {"schema"}:
        return current_scene, current_placement, None
    return current_scene, current_placement, info


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip().replace("\xa0", " ")
    if not text:
        return None
    match = re.search(r"-?\d+(?:[\s.,]\d+)*", text)
    if not match:
        return None
    token = match.group(0).replace(" ", "")
    if "," in token and "." in token:
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        value_f = float(token)
    except Exception:
        return None
    return value_f if math.isfinite(value_f) else None


def _money(value: Any) -> float | None:
    price = _num(value)
    if price is None or price <= 0:
        return None
    return round(price, 2)


def _format_price(value: float | None, currency: str | None = "RUB") -> str:
    if value is None:
        return ""
    cur = str(currency or "RUB").strip() or "RUB"
    return f"{value:,.2f} {cur}".replace(",", " ")


def _proxy_like_asset_path(path: Any) -> bool:
    text = str(path or "").replace("\\", "/").lower()
    return text.endswith("/built/proxy.glb") or text.endswith("/proxy.glb")


def _markdown_cell(value: Any) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|").strip()
    return text


def _material_from_selection(info: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    path = str(info.get("selection_json") or "").strip()
    if not path:
        return {}
    try:
        selection = read_json(path)
    except Exception:
        return {}
    material = selection.get("selected_material") if isinstance(selection, dict) else None
    return material if isinstance(material, dict) else {}


def _polygon_area_m2(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    acc = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        acc += x0 * y1 - x1 * y0
    area = abs(acc) * 0.5
    return area if area > 0 else None


def _room_polygon_points(room: dict[str, Any]) -> list[tuple[float, float]]:
    raw = room.get("floor_polygon") or room.get("floor_polygon_xz") or room.get("vertices") or []
    points: list[tuple[float, float]] = []
    if isinstance(raw, list):
        for point in raw:
            xy = _room_point_xy(point)
            if xy is not None:
                points.append(xy)
    if points:
        return points
    width = _num(room.get("width_m") or room.get("width"))
    depth = _num(room.get("depth_m") or room.get("depth"))
    if width and depth:
        return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]
    return []


def _floor_area_m2(scene: dict[str, Any]) -> float | None:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else scene
    explicit = _num(room.get("area_m2")) if isinstance(room, dict) else None
    if explicit and explicit > 0:
        return round(explicit, 3)
    if not isinstance(room, dict):
        return None
    return round(_polygon_area_m2(_room_polygon_points(room)) or 0.0, 3) or None


def _wall_length_m(room: dict[str, Any], wall: dict[str, Any]) -> float | None:
    for key in ("length_m", "length"):
        value = _num(wall.get(key))
        if value and value > 0:
            return value
    points = _room_wall_points(room, wall)
    if points is None:
        return None
    (x0, y0), (x1, y1) = points
    return math.hypot(x1 - x0, y1 - y0)


def _wall_area_m2(scene: dict[str, Any]) -> float | None:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else scene
    if not isinstance(room, dict):
        return None
    height = _num(room.get("ceiling_height_m") or room.get("ceiling_height") or room.get("height_m")) or 2.8
    perimeter = 0.0
    walls = room.get("walls")
    if isinstance(walls, list) and walls:
        for wall in walls:
            if isinstance(wall, dict):
                perimeter += _wall_length_m(room, wall) or 0.0
    else:
        points = _room_polygon_points(room)
        for idx, (x0, y0) in enumerate(points):
            x1, y1 = points[(idx + 1) % len(points)]
            perimeter += math.hypot(x1 - x0, y1 - y0)
    if perimeter <= 0:
        return None
    area = perimeter * height
    for key, default_height in (("doors", 2.05), ("windows", 1.2)):
        for opening in _room_openings_from_anywhere(room, key):
            width = _num(opening.get("width_m") or opening.get("width") or opening.get("w"))
            opening_height = _num(opening.get("height_m") or opening.get("height") or opening.get("h")) or default_height
            if width and opening_height:
                area -= width * opening_height
    return round(max(area, 0.0), 3)


def _coverage_from_name(name: str) -> float | None:
    text = str(name or "").lower().replace(",", ".")
    pair = re.search(r"(\d+(?:\.\d+)?)\s*[xх×]\s*(\d+(?:\.\d+)?)\s*м", text)
    if pair:
        width = _num(pair.group(1))
        length = _num(pair.group(2))
        if width and length:
            return round(width * length, 4)
    values = [_num(x.group(1)) for x in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:м2|м²|m2)", text)]
    values = [v for v in values if v and v > 0]
    return values[-1] if values else None


def _material_coverage_m2(material: dict[str, Any], *, kind: str) -> float | None:
    if not isinstance(material, dict):
        return None
    if kind == "flooring":
        for key in ("package_area_m2", "pack_area_m2", "coverage_m2", "area_m2"):
            value = _num(material.get(key))
            if value and value > 0:
                return value
    if kind == "wall_material":
        width_cm = _num(material.get("width_cm") or material.get("roll_width_cm"))
        length_m = _num(material.get("length_m") or material.get("roll_length_m"))
        if width_cm and length_m:
            width_m = width_cm / 100.0 if width_cm > 5 else width_cm
            if width_m > 0:
                return round(width_m * length_m, 4)
    raw_props = material.get("raw_properties") if isinstance(material.get("raw_properties"), dict) else {}
    for key in (
        "Площадь в упаковке",
        "Количество м² в упаковке",
        "Площадь упаковки",
        "м² в упаковке",
        "Площадь рулона",
    ):
        value = _num(raw_props.get(key))
        if value and value > 0:
            return value
    return _coverage_from_name(str(material.get("name") or ""))


def _surface_cost_line(
    *,
    kind: str,
    label: str,
    area_m2: float | None,
    material: dict[str, Any],
) -> dict[str, Any]:
    price = _money(material.get("price") if isinstance(material, dict) else None)
    currency = str((material or {}).get("price_currency") or "RUB")
    coverage = _material_coverage_m2(material, kind=kind)
    units = math.ceil(area_m2 / coverage) if area_m2 and coverage and coverage > 0 else None
    total = round(units * price, 2) if units is not None and price is not None else None
    return {
        "kind": kind,
        "label": label,
        "sku": (material or {}).get("sku"),
        "name": (material or {}).get("name"),
        "product_url": (material or {}).get("product_url"),
        "area_m2": area_m2,
        "coverage_per_unit_m2": coverage,
        "units_needed": units,
        "unit_price": price,
        "price_currency": currency,
        "estimated_total_price": total,
        "price_status": "computed" if total is not None else "missing_price_or_coverage",
    }


def build_cost_report(
    *,
    out_dir: Path,
    bindings_data: dict[str, Any],
    supplier_scene_data: dict[str, Any],
    surface_materials_info: dict[str, Any] | None,
) -> dict[str, Any]:
    placements = supplier_scene_data.get("placements") or supplier_scene_data.get("items") or []
    by_id = {
        str(item.get("id") or "").strip(): item
        for item in placements
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    items: list[dict[str, Any]] = []
    for binding in bindings_data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        if str(binding.get("selection_status") or "") not in SELECTED_BINDING_STATUSES:
            continue
        candidate = binding.get("chosen_candidate") if isinstance(binding.get("chosen_candidate"), dict) else {}
        if not candidate:
            continue
        target_id = str(binding.get("target_id") or "")
        placement = by_id.get(target_id) or {}
        source = placement.get("source") if isinstance(placement.get("source"), dict) else {}
        asset = placement.get("asset") if isinstance(placement.get("asset"), dict) else {}
        asset_path = asset.get("mesh_path") or candidate.get("asset_local_path")
        price = _money(candidate.get("price_value"))
        currency = str(candidate.get("price_currency") or "RUB")
        items.append(
            {
                "target_id": target_id,
                "category": binding.get("category"),
                "semantic_group": binding.get("semantic_group"),
                "title": candidate.get("title"),
                "source_site": candidate.get("source_site"),
                "product_url": candidate.get("product_url") or candidate.get("model_page_url"),
                "model_download_url": candidate.get("model_download_url"),
                "unique_key": candidate.get("unique_key"),
                "quantity": 1,
                "unit_price": price,
                "price_currency": currency,
                "estimated_total_price": price,
                "requested_size_m": binding.get("requested_size_m"),
                "selected_dimensions_m": _candidate_dimensions_m_for_report(candidate),
                "asset_source": source.get("asset_source"),
                "asset_path": asset_path,
                "asset_quality": "proxy_like_asset" if _proxy_like_asset_path(asset_path) else "real_or_generated_asset",
                "selection_mode": binding.get("supplier_selection_mode"),
                "selection_policy": binding.get("selection_policy"),
            }
        )

    floor_material = _material_from_selection((surface_materials_info or {}).get("flooring") if surface_materials_info else None)
    wall_material = _material_from_selection((surface_materials_info or {}).get("wall_material") if surface_materials_info else None)
    surfaces = [
        _surface_cost_line(kind="flooring", label="Floor covering", area_m2=_floor_area_m2(supplier_scene_data), material=floor_material),
        _surface_cost_line(kind="wall_material", label="Wall covering", area_m2=_wall_area_m2(supplier_scene_data), material=wall_material),
    ]
    currency_totals: dict[str, float] = {}
    for item in items:
        total = _money(item.get("estimated_total_price"))
        if total is not None:
            currency = str(item.get("price_currency") or "RUB")
            currency_totals[currency] = round(currency_totals.get(currency, 0.0) + total, 2)
    for surface in surfaces:
        total = _money(surface.get("estimated_total_price"))
        if total is not None:
            currency = str(surface.get("price_currency") or "RUB")
            currency_totals[currency] = round(currency_totals.get(currency, 0.0) + total, 2)

    asset_source_counts: dict[str, int] = {}
    for item in placements:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
        key = "proxy_like_asset" if _proxy_like_asset_path(asset.get("mesh_path")) else str(source.get("asset_source") or "generated").strip() or "generated"
        asset_source_counts[key] = int(asset_source_counts.get(key, 0)) + 1

    report = {
        "schema": "supplier_cost_report/v1",
        "items": items,
        "surface_materials": surfaces,
        "totals_by_currency": currency_totals,
        "asset_source_counts": dict(sorted(asset_source_counts.items())),
        "notes": [
            "Surface quantities are estimated from room geometry; wallpaper/flooring totals use selected package or roll coverage when available.",
            "Supplier item totals use quantity=1 per selected replacement target.",
        ],
    }
    json_path = out_dir / "supplier_cost_report.json"
    md_path = out_dir / "supplier_cost_report.md"
    write_json(json_path, report)
    md_path.write_text(_cost_report_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "totals_by_currency": currency_totals,
        "item_count": len(items),
        "surface_count": len(surfaces),
        "asset_source_counts": report["asset_source_counts"],
    }


def _candidate_dimensions_m_for_report(candidate: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for src, dst in (("width_cm", "width"), ("depth_cm", "depth"), ("height_cm", "height")):
        value = _num(candidate.get(src))
        out[dst] = round(value / 100.0, 4) if value is not None else None
    return out


def _cost_report_markdown(report: dict[str, Any]) -> str:
    lines = ["# Supplier Cost Report", ""]
    totals = report.get("totals_by_currency") if isinstance(report.get("totals_by_currency"), dict) else {}
    if totals:
        lines.append("## Totals")
        for currency, value in totals.items():
            lines.append(f"- {_format_price(_money(value), str(currency))}")
        lines.append("")

    lines.extend(["## Items", "| Target | Item | Link | Unit price | Asset source | Asset quality |", "|---|---|---|---:|---|---|"])
    for item in report.get("items") or []:
        if not isinstance(item, dict):
            continue
        title = _markdown_cell(item.get("title") or item.get("unique_key"))
        url = str(item.get("product_url") or "").strip()
        link = f"[link]({url})" if url else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(item.get("target_id")),
                    title,
                    link,
                    _format_price(_money(item.get("unit_price")), str(item.get("price_currency") or "RUB")),
                    _markdown_cell(item.get("asset_source")),
                    _markdown_cell(item.get("asset_quality")),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Surface Materials", "| Surface | Material | Link | Area m2 | Coverage/unit m2 | Units | Unit price | Estimated total |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for surface in report.get("surface_materials") or []:
        if not isinstance(surface, dict):
            continue
        url = str(surface.get("product_url") or "").strip()
        link = f"[link]({url})" if url else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(surface.get("label") or surface.get("kind")),
                    _markdown_cell(surface.get("name") or surface.get("sku")),
                    link,
                    str(surface.get("area_m2") or ""),
                    str(surface.get("coverage_per_unit_m2") or ""),
                    str(surface.get("units_needed") or ""),
                    _format_price(_money(surface.get("unit_price")), str(surface.get("price_currency") or "RUB")),
                    _format_price(_money(surface.get("estimated_total_price")), str(surface.get("price_currency") or "RUB")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Procedural-only room generation with supplier replacement. Does not call Infinigen.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--room", help="Input room.json")
    src.add_argument("--scene", help="Input scene.v1.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--density", default="very_high", choices=["normal", "high", "very_high"])
    parser.add_argument("--policy", default="always", choices=["auto", "always", "never"])
    parser.add_argument("--replace-existing", action="store_true", default=True)
    parser.add_argument("--no-replace-existing", action="store_false", dest="replace_existing")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--supplier-catalog-json", action="append", default=[])
    parser.add_argument("--supplier-assets-db", action="append", default=[])
    parser.add_argument("--supplier-site", action="append", default=[])
    parser.add_argument(
        "--supplier-selection-mode",
        default="best_visual_reference",
        choices=[
            "cheapest",
            "min_price",
            "lowest_price",
            "cheapest_top20",
            "cheap_top20",
            "optimal",
            "best_match",
            "best_match_v1",
            "best_match_v2",
            "best_visual_reference",
            "best_suitable",
            "most_suitable",
            "legacy_asset_priority",
        ],
    )
    parser.add_argument("--supplier-selection-strategy", default="balanced")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--supplier-asset-fallback-mode",
        choices=["none", "fbx_obj_proxy", "fbx_obj_trellis_proxy"],
        default="none",
        help="Fallback policy for missing local FBX/OBJ assets when applying supplier replacements.",
    )
    parser.add_argument("--require-real-asset", action="store_true")
    parser.add_argument("--acquire-assets", action="store_true")
    parser.add_argument("--supplier-llm-provider", default="none", choices=["none", "ollama"])
    parser.add_argument("--supplier-ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--supplier-ollama-model", default="gpt-oss:20b")
    parser.add_argument("--supplier-ollama-timeout", type=int, default=180)
    parser.add_argument("--supplier-ollama-temperature", type=float, default=0.0)
    parser.add_argument("--supplier-llm-top-n", type=int, default=5)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--no-stage-timings", action="store_true", help="Disable [TIMER] stage logs and report timings.")

    surfaces = parser.add_argument_group("Surface materials")
    surfaces.add_argument("--no-flooring", action="store_true", help="Do not select/apply supplier flooring.")
    surfaces.add_argument("--flooring-materials", default="data/floor_materials")
    surfaces.add_argument("--flooring-style-rules", default="config/flooring_style_rules.json")
    surfaces.add_argument("--flooring-top-k", type=int, default=10)
    surfaces.add_argument("--flooring-llm-provider", default="ollama", choices=["none", "ollama"])
    surfaces.add_argument("--flooring-ollama-url", default="http://127.0.0.1:11434")
    surfaces.add_argument("--flooring-ollama-model", default="gpt-oss:20b")
    surfaces.add_argument("--flooring-ollama-timeout", type=int, default=180)
    surfaces.add_argument("--flooring-ollama-temperature", type=float, default=0.0)
    surfaces.add_argument("--flooring-ollama-num-ctx", type=int, default=8192)
    surfaces.add_argument("--flooring-llm-top-n", type=int, default=5)
    surfaces.add_argument("--no-wall-material", action="store_true", help="Do not select/apply supplier wall material.")
    surfaces.add_argument("--wall-materials", default="data/floor_materials")
    surfaces.add_argument("--wall-top-k", type=int, default=10)
    surfaces.add_argument("--wall-llm-provider", default="ollama", choices=["none", "ollama"])
    surfaces.add_argument("--wall-ollama-url", default="http://127.0.0.1:11434")
    surfaces.add_argument("--wall-ollama-model", default="gpt-oss:20b")
    surfaces.add_argument("--wall-ollama-timeout", type=int, default=180)
    surfaces.add_argument("--wall-ollama-temperature", type=float, default=0.0)
    surfaces.add_argument("--wall-ollama-num-ctx", type=int, default=8192)
    surfaces.add_argument("--wall-llm-top-n", type=int, default=5)

    room_post = parser.add_argument_group("Room/postprocessing")
    room_post.add_argument("--ensure-door", action=argparse.BooleanOptionalAction, default=True, help="Add one wall door to room JSON when it has no door.")
    room_post.add_argument("--curtains", choices=["auto", "always", "never"], default="auto", help="Shtorystore curtain postprocess policy.")
    room_post.add_argument("--no-curtains", action="store_true", help="Disable curtain postprocess for room windows.")
    room_post.add_argument("--curtain-materials", default="data/floor_materials/shtorystore_curtains")
    room_post.add_argument("--curtain-models-dir", default="data/sourse/curtains_3d")
    room_post.add_argument("--curtain-supplier-catalog", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    room_post.add_argument("--curtain-seed", type=int, default=0)

    trellis = parser.add_argument_group("TRELLIS fallback for missing supplier assets")
    trellis.add_argument(
        "--trellis-generate-missing-assets",
        action="store_true",
        help="Generate TRELLIS GLB assets for selected supplier candidates that still have no FBX/OBJ/GLB after acquisition.",
    )
    trellis.add_argument("--trellis-max-assets", type=int, default=0, help="Max TRELLIS assets to generate; 0 means no limit.")
    trellis.add_argument(
        "--trellis-skip-categories",
        default="",
        help="Comma-separated target categories that must not use TRELLIS image-to-3D fallback, e.g. bed,sofa.",
    )
    trellis.add_argument(
        "--trellis-ikea-mebelru-images-only",
        "--fast-ikea-mebelru-images-only",
        dest="trellis_ikea_mebelru_images_only",
        action="store_true",
        help="For TRELLIS fallback cards, keep only IKEA/mebel.ru candidates and image sources.",
    )
    trellis.add_argument("--trellis-server-host", default="")

    parser.add_argument(
        "--trellis-max-failures-per-candidate",
        type=int,
        default=2,
        help="After this many TRELLIS failures for one supplier candidate, blacklist it and try another candidate for the same target.",
    )
    parser.add_argument(
        "--trellis-progress-log",
        action="store_true",
        default=True,
        help="Print detailed TRELLIS progress with elapsed time and ETA.",
    )
    trellis.add_argument(
        "--trellis-allow-proxy-fallback",
        action="store_true",
        help="Debug only: allow a simple local proxy GLB if all direct/TRELLIS candidates fail. Leave disabled for real-asset runs.",
    )
    trellis.add_argument("--trellis-server-port", type=int, default=28553)
    trellis.add_argument("--trellis-server-user", default="root")
    trellis.add_argument("--trellis-ssh-key", default="")
    trellis.add_argument("--trellis-remote-root", default="/workspace/trellis2_supplier_jobs")
    trellis.add_argument("--trellis-remote-trellis-root", default="/workspace/TRELLIS.2")
    trellis.add_argument("--trellis-remote-model-dir", default="/workspace/models/TRELLIS.2-4B")
    trellis.add_argument("--trellis-remote-python", default="/venv/trellis2/bin/python")
    trellis.add_argument("--trellis-remote-worker-root", default="/workspace/trellis2_worker")
    trellis.add_argument("--trellis-remote-worker-timeout-sec", type=float, default=1800.0)
    trellis.add_argument("--trellis-remote-worker-poll-sec", type=float, default=2.0)
    trellis.add_argument("--trellis-remote-persistent-worker", action=argparse.BooleanOptionalAction, default=True)
    trellis.add_argument(
        "--trellis-remote-text-model-dir",
        default="",
        help="Deprecated. Legacy text fallback is disabled in TRELLIS.2-only mode.",
    )
    trellis.add_argument(
        "--trellis-remote-cuda-visible-devices",
        default="0",
        help="Comma-separated physical GPU ids to try for TRELLIS, for example '0,1'. CUDA_VISIBLE_DEVICES is set to one id per attempt.",
    )
    trellis.add_argument("--trellis-multi-mode", default="stochastic", choices=["stochastic", "multidiffusion"])
    trellis.add_argument("--trellis-max-images", type=int, default=2)
    trellis.add_argument(
        "--trellis-oom-retry-max-images",
        type=int,
        default=1,
        help="If TRELLIS hits CUDA OOM, retry multi-image generation with this smaller image count; 0 disables image-count retry.",
    )
    trellis.add_argument(
        "--trellis-max-candidate-pool",
        type=int,
        default=0,
        help="Limit number of TRELLIS candidates to try per target in fallback mode; 0 means no hard limit (use all available candidates).",
    )
    trellis.add_argument(
        "--trellis-disable-after-oom",
        action="store_true",
        default=True,
        help="After all TRELLIS attempts for one asset fail with CUDA OOM, skip later missing assets and use proxy fallback.",
    )
    trellis.add_argument("--no-trellis-disable-after-oom", action="store_false", dest="trellis_disable_after_oom")
    trellis.add_argument("--trellis-stop-after-oom", action="store_true", dest="trellis_disable_after_oom", help=argparse.SUPPRESS)
    trellis.add_argument("--no-trellis-stop-after-oom", action="store_false", dest="trellis_disable_after_oom", help=argparse.SUPPRESS)
    trellis.add_argument("--trellis-seed", type=int, default=1)
    trellis.add_argument("--trellis-sparse-steps", type=int, default=4)
    trellis.add_argument("--trellis-slat-steps", type=int, default=4)
    trellis.add_argument("--trellis-texture-size", type=int, default=256)
    trellis.add_argument("--trellis-simplify", type=float, default=0.98)
    trellis.add_argument("--trellis-pipeline-type", type=int, default=512)
    trellis.add_argument("--trellis-ss-guidance-strength", type=float, default=7.5)
    trellis.add_argument("--trellis-slat-guidance-strength", type=float, default=3.0)
    trellis.add_argument("--trellis-decimation-target", type=int, default=50000)
    trellis.add_argument("--trellis-pre-export-simplify-target", type=int, default=0)
    trellis.add_argument("--trellis-no-remesh", action=argparse.BooleanOptionalAction, default=False)
    trellis.add_argument("--trellis-remesh-band", type=int, default=1)
    trellis.add_argument("--trellis-remesh-project", type=float, default=0.0)
    trellis.add_argument("--trellis-no-webp", action=argparse.BooleanOptionalAction, default=True)
    trellis.add_argument("--trellis-image-size", type=int, default=336)
    trellis.add_argument("--trellis-fill-holes-resolution", type=int, default=256)
    trellis.add_argument("--trellis-fill-holes-num-views", type=int, default=120)
    trellis.add_argument("--trellis-remote-runner-path", default="", help="Deprecated; disabled in TRELLIS.2-only mode.")
    trellis.add_argument("--trellis-vlm-single-object-filter", action="store_true", default=True)
    trellis.add_argument("--no-trellis-vlm-single-object-filter", action="store_false", dest="trellis_vlm_single_object_filter")
    trellis.add_argument("--trellis-vlm-provider", default="ollama", choices=["ollama", "openai", "openrouter"])
    trellis.add_argument("--trellis-vlm-ollama-url", default="http://127.0.0.1:11435")
    trellis.add_argument("--trellis-vlm-model", default="llama3.2-vision:11b")
    trellis.add_argument("--trellis-vlm-timeout", type=int, default=120)
    trellis.add_argument("--trellis-vlm-unload-after-filter", action="store_true", default=True)
    trellis.add_argument("--no-trellis-vlm-unload-after-filter", action="store_false", dest="trellis_vlm_unload_after_filter")
    trellis.add_argument(
        "--trellis-text-fallback-if-no-single-image",
        action="store_true",
        default=True,
        help="Deprecated; legacy text fallback is disabled in TRELLIS.2-only mode.",
    )

    blend = parser.add_argument_group("Blender scene build")
    blend.add_argument("--build-blend", action="store_true", help="Build final .blend from supplier procedural scene.")
    blend.add_argument("--scene-builder-script", default="src/Plasement/blender_scene_builder.py")
    blend.add_argument("--out-blend", default="")
    blend.add_argument("--out-png", default="")

    topview = parser.add_argument_group("Top-view VLM orientation repair")
    topview.add_argument("--topview-vlm-orientation-repair", action="store_true", help="After the first Blender build, ask VLM to repair object yaw from a labeled top view and rebuild final outputs.")
    topview.add_argument("--topview-vlm-provider", choices=["none", "openai", "openrouter", "ollama"], default="ollama")
    topview.add_argument("--topview-vlm-model", default="llama3.2-vision:11b")
    topview.add_argument("--topview-vlm-target-scope", choices=["chairs", "all"], default="all")
    topview.add_argument("--topview-vlm-include-armchairs", action=argparse.BooleanOptionalAction, default=True)
    topview.add_argument("--topview-vlm-min-confidence", type=float, default=0.80)
    topview.add_argument("--topview-vlm-max-delta-deg", type=float, default=180.0)
    topview.add_argument("--topview-vlm-snap-step-deg", type=float, default=90.0)
    topview.add_argument("--topview-vlm-max-objects", type=int, default=10000)
    topview.add_argument("--topview-vlm-max-repairs-per-object", type=int, default=1)
    topview.add_argument("--topview-vlm-visual-front-offset-deg", type=float, default=0.0)
    topview.add_argument("--topview-vlm-resolution-x", type=int, default=1400)
    topview.add_argument("--topview-vlm-resolution-y", type=int, default=1050)
    topview.add_argument("--topview-vlm-elevation-deg", type=float, default=80.0)
    topview.add_argument("--topview-vlm-radius-mult", type=float, default=0.55)
    topview.add_argument("--topview-vlm-lens", type=float, default=32.0)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    timings = StageTimings(enabled=not bool(args.no_stage_timings))

    with timings.stage("argument_setup"):
        validate_trellis2_only_cli_args(args)

    with timings.stage("output_setup"):
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.supplier_catalog_json and not args.supplier_assets_db:
            default_catalog = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
            if default_catalog.is_file():
                args.supplier_catalog_json = [str(default_catalog)]

    with timings.stage("input_scene_prepare"):
        room_postprocess_info: dict[str, Any] | None = None
        if args.scene:
            input_scene = Path(args.scene).expanduser().resolve()
        else:
            room_payload = read_json(args.room)
            if bool(getattr(args, "ensure_door", True)):
                room_payload, room_postprocess_info = ensure_room_has_door(room_payload)
                if room_postprocess_info and room_postprocess_info.get("changed"):
                    write_json(out_dir / "input_room.with_door.v1.json", room_payload)
            input_scene = out_dir / "input_scene_from_room.v1.json"
            write_json(input_scene, build_scene_from_room(room_payload))

    with timings.stage("procedural_room_stage"):
        procedural_report = apply_procedural_room_stage(
            scene_json_path=input_scene,
            out_dir=out_dir,
            prompt=args.prompt,
            policy=args.policy,
            density=args.density,
            replace_existing=args.replace_existing,
            seed=args.seed,
            tag="standalone",
        )
    if procedural_report.get("skipped"):
        timings.finish()
        skipped_report = {
            "procedural_room_stage": procedural_report,
            "stage_timings": timings.report(),
        }
        write_json(out_dir / "procedural_room_supplier_report.json", skipped_report)
        print(json.dumps(skipped_report, ensure_ascii=False, indent=2))
        return

    procedural_scene = Path(procedural_report["output_scene_json"]).expanduser().resolve()
    procedural_placement = Path(procedural_report["output_placement_json"]).expanduser().resolve()

    with timings.stage("layout_targets"):
        layout_artifacts = create_layout_selection_stub_artifacts(
            source_json_path=procedural_scene,
            run_dir=out_dir,
            prefix="procedural",
        )
        layout_targets_path = Path(layout_artifacts["layout_targets_json"])
        layout_targets = read_json(layout_targets_path)

    with timings.stage("room_design_spec"):
        room_design_spec = build_room_design_spec(
            user_prompt=args.prompt,
            layout_targets=layout_targets,
            style_profile=None,
        )
        room_design_spec_path = out_dir / "room_design_spec.json"
        write_json(room_design_spec_path, room_design_spec)

    with timings.stage("supplier_catalog_load"):
        sites = {str(x).strip() for x in args.supplier_site if str(x).strip()} or None
        db_paths = [Path(x).expanduser().resolve() for x in args.supplier_assets_db if str(x).strip()]
        json_paths = [Path(x).expanduser().resolve() for x in args.supplier_catalog_json if str(x).strip()]
        catalog_rows = []
        if db_paths:
            catalog_rows.extend(load_supplier_catalog(db_paths, sites=sites, rich_only=False))
        if json_paths:
            catalog_rows.extend(load_supplier_catalog_json(json_paths, sites=sites, rich_only=False))
        catalog_rows = _merge_catalog_rows(catalog_rows)
        if not catalog_rows:
            raise RuntimeError("Pass at least one --supplier-assets-db or --supplier-catalog-json with supplier records.")

    user_preferences = {"global": {"require_real_asset": True}} if args.require_real_asset else None
    supplier_llm_settings = {
        "provider": str(args.supplier_llm_provider or "none").strip().lower(),
        "ollama_url": str(args.supplier_ollama_url or "http://127.0.0.1:11434"),
        "ollama_model": str(args.supplier_ollama_model or "gpt-oss:20b"),
        "ollama_timeout": int(args.supplier_ollama_timeout or 180),
        "ollama_temperature": float(args.supplier_ollama_temperature or 0.0),
        "top_n": int(args.supplier_llm_top_n or 5),
    }
    with timings.stage("supplier_matching"):
        bindings = build_bindings_with_candidates(
            targets_json_path=layout_targets_path,
            catalog_rows=catalog_rows,
            top_k=args.top_k,
            selection_strategy=args.supplier_selection_strategy,
            user_preferences=user_preferences,
            llm_settings=supplier_llm_settings,
            room_design_spec=room_design_spec,
            selection_mode=args.supplier_selection_mode,
        )
    bindings_path = out_dir / f"supplier_bindings.{args.supplier_selection_mode}.json"
    with timings.stage("write_supplier_bindings"):
        write_json(bindings_path, bindings)

    bindings_for_apply = bindings_path
    uses_mesh_or_proxy_fallback = args.supplier_asset_fallback_mode in {"fbx_obj_proxy", "fbx_obj_trellis_proxy"}
    should_acquire_assets = bool(args.acquire_assets or uses_mesh_or_proxy_fallback)
    if should_acquire_assets:
        if not db_paths and not json_paths:
            raise RuntimeError("--acquire-assets or mesh/proxy supplier fallback requires at least one --supplier-catalog-json")
        acquisition_db_path = db_paths[0] if db_paths else (out_dir / "supplier_assets" / "supplier_assets_cache.db")
        with timings.stage("supplier_asset_acquisition"):
            bindings_for_apply = acquire_assets_for_bindings_json(
                bindings_json_path=bindings_path,
                output_json_path=out_dir / f"supplier_bindings.{args.supplier_selection_mode}.with_assets.json",
                db_path=acquisition_db_path,
                out_dir=out_dir / "supplier_assets",
                blender_bin=args.blender,
                catalog_json_paths=json_paths,
                keep_unresolved_candidates=bool((uses_mesh_or_proxy_fallback or args.trellis_generate_missing_assets) and not args.require_real_asset),
            )

    trellis_generation_report: dict[str, Any] | None = None
    if args.trellis_generate_missing_assets:
        if not str(args.trellis_server_host or "").strip():
            raise RuntimeError("--trellis-generate-missing-assets requires --trellis-server-host")
        with timings.stage("trellis_missing_assets"):
            bindings_for_apply, trellis_generation_report = enrich_missing_assets_with_trellis(
                bindings_json_path=Path(bindings_for_apply).expanduser().resolve(),
                output_json_path=out_dir / f"supplier_bindings.{args.supplier_selection_mode}.with_trellis_assets.json",
                out_dir=out_dir,
                args=args,
            )

    supplier_scene = out_dir / f"scene_supplier_procedural.{args.supplier_selection_mode}.v1.json"
    supplier_placement = out_dir / f"placement_supplier_procedural.{args.supplier_selection_mode}.v1.json"
    with timings.stage("apply_supplier_bindings"):
        apply_supplier_bindings_to_json(
            input_json_path=procedural_scene,
            bindings_json_path=bindings_for_apply,
            output_json_path=supplier_scene,
            require_local_asset=args.require_real_asset,
            fallback_mode=args.supplier_asset_fallback_mode,
        )
        apply_supplier_bindings_to_json(
            input_json_path=procedural_placement,
            bindings_json_path=bindings_for_apply,
            output_json_path=supplier_placement,
            require_local_asset=args.require_real_asset,
            fallback_mode=args.supplier_asset_fallback_mode,
        )

    supplier_scene, supplier_placement, surface_materials_info = maybe_apply_surface_materials(
        args,
        out_dir=out_dir,
        supplier_scene=supplier_scene,
        supplier_placement=supplier_placement,
        prompt=args.prompt,
        room_design_spec=room_design_spec,
        timings=timings,
    )

    curtain_postprocess_info: dict[str, Any] | None = None
    with timings.stage("curtain_postprocess"):
        supplier_scene, curtain_postprocess_info = maybe_apply_curtains_to_supplier_scene(
            args,
            out_dir=out_dir,
            supplier_scene=supplier_scene,
            prompt=args.prompt,
            room_design_spec=room_design_spec,
        )

    with timings.stage("blender_build"):
        blend_artifacts = maybe_build_blend_scene(args, supplier_scene=supplier_scene, out_dir=out_dir)

    topview_orientation_info: dict[str, Any] | None = None
    with timings.stage("topview_vlm_orientation_repair"):
        supplier_scene, topview_orientation_info, blend_artifacts = maybe_apply_topview_vlm_orientation_repair(
            args,
            supplier_scene=supplier_scene,
            out_dir=out_dir,
            blend_artifacts=blend_artifacts,
        )

    with timings.stage("validation_and_summary"):
        supplier_scene_data = read_json(supplier_scene)
        validation = validate_placements(
            build_room_context(supplier_scene_data, prompt=args.prompt),
            supplier_scene_data.get("placements") or [],
        )
        replaced_count = sum(
            1
            for item in supplier_scene_data.get("placements", [])
            if isinstance(item, dict) and isinstance(item.get("meta"), dict) and item["meta"].get("supplier_binding_applied")
        )
        local_asset_replaced = sum(
            1
            for item in supplier_scene_data.get("placements", [])
            if isinstance(item, dict)
            and isinstance(item.get("source"), dict)
            and item["source"].get("asset_source") in {"supplier_catalog_local_asset", "trellis_generated_local_asset"}
        )
        trellis_asset_replaced = sum(
            1
            for item in supplier_scene_data.get("placements", [])
            if isinstance(item, dict)
            and isinstance(item.get("source"), dict)
            and item["source"].get("asset_source") == "trellis_generated_local_asset"
        )
        proxy_asset_replaced = sum(
            1
            for item in supplier_scene_data.get("placements", [])
            if isinstance(item, dict)
            and isinstance(item.get("source"), dict)
            and item["source"].get("asset_source") == "supplier_catalog_procedural_proxy"
        )
        cost_report_info = build_cost_report(
            out_dir=out_dir,
            bindings_data=read_json(bindings_for_apply),
            supplier_scene_data=supplier_scene_data,
            surface_materials_info=surface_materials_info,
        )
    report = {
        "schema": "procedural_room_supplier_report/v1",
        "input_scene_json": str(input_scene),
        "room_postprocess": room_postprocess_info,
        "procedural_room_stage": procedural_report,
        "layout_targets_json": str(layout_targets_path),
        "room_design_spec_json": str(room_design_spec_path),
        "supplier_bindings_json": str(bindings_path),
        "supplier_bindings_for_apply_json": str(bindings_for_apply),
        "trellis_missing_asset_generation": trellis_generation_report,
        "surface_materials": surface_materials_info,
        "curtain_postprocess": curtain_postprocess_info,
        "topview_vlm_orientation_repair": topview_orientation_info,
        "supplier_scene_json": str(supplier_scene),
        "supplier_placement_json": str(supplier_placement),
        "blend_artifacts": blend_artifacts,
        "cost_report": cost_report_info,
        "summary": {
            "layout_source": Path(layout_targets.get("source_json", "")).name,
            "target_count": len(layout_targets.get("targets") or []),
            "matched_target_count": (bindings.get("meta") or {}).get("matched_target_count"),
            "replaced": replaced_count,
            "local_asset_replaced": local_asset_replaced,
            "trellis_asset_replaced": trellis_asset_replaced,
            "proxy_asset_replaced": proxy_asset_replaced,
            "collisions": len(validation.get("collisions") or []),
            "accessibility_ok": validation.get("accessibility_ok"),
        },
        "validation": validation,
        "stage_timings": timings.report(),
    }
    report_path = out_dir / "procedural_room_supplier_report.json"
    with timings.stage("write_report"):
        write_json(report_path, report)
    timings.finish()
    report["stage_timings"] = timings.report()
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
