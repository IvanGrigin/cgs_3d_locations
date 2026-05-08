#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .supplier_identity_gates import candidate_identity_gate
except ImportError:
    from supplier_identity_gates import candidate_identity_gate


SELECTED_BINDING_STATUSES = {
    "heuristic_top1_selected",
    "heuristic_first_acceptable_selected",
    "llm_reranked_top1_selected",
    "llm_reranked_first_acceptable_selected",
}
MODES = ("cheapest", "optimal", "best_match")


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, "root JSON is not an object"
    return data, None


def _mode_from_path(path: Path) -> str | None:
    name = path.name.lower()
    for mode in MODES:
        if re.search(rf"(^|[._-]){re.escape(mode)}([._-]|$)", name):
            return mode
    return None


def _mode_from_data(data: dict[str, Any], path: Path) -> str:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    for key in ("selection_mode", "supplier_selection_mode"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    for binding in data.get("bindings") or []:
        if not isinstance(binding, dict):
            continue
        value = str(binding.get("supplier_selection_mode") or binding.get("selection_mode") or "").strip()
        if value:
            return value
    return _mode_from_path(path) or "unknown"


def _candidate_id(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("unique_key", "supplier_id", "sku", "external_id", "id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return str(candidate.get("title") or candidate.get("name") or "").strip()


def _has_local_asset(candidate: dict[str, Any] | None) -> bool:
    return isinstance(candidate, dict) and bool(str(candidate.get("asset_local_path") or "").strip())


def _validate_binding_file(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = {
        "path": str(path.resolve()),
        "mode": _mode_from_path(path) or "unknown",
        "total_bindings": 0,
        "selected_count": 0,
        "warnings": [],
        "errors": [],
    }
    data, error = _read_json(path)
    if error or data is None:
        result["errors"].append(f"cannot read JSON: {error}")
        return result, None

    result["mode"] = _mode_from_data(data, path)
    bindings = data.get("bindings")
    if not isinstance(bindings, list):
        result["errors"].append("bindings is missing or not a list")
        return result, data

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    design_aware = bool(meta.get("room_design_spec_enabled"))
    after_acquisition = bool(meta.get("asset_acquisition")) or ".assets." in path.name or path.name.endswith(".assets.json")

    seen_target_ids: set[str] = set()
    consistency_groups: dict[str, set[str]] = defaultdict(set)
    result["total_bindings"] = len(bindings)

    for idx, binding in enumerate(bindings, start=1):
        if not isinstance(binding, dict):
            result["errors"].append(f"binding #{idx} is not an object")
            continue
        target_id = str(binding.get("target_id") or "").strip()
        if not target_id:
            result["errors"].append(f"binding #{idx} has empty target_id")
        elif target_id in seen_target_ids:
            result["errors"].append(f"duplicate target_id: {target_id}")
        seen_target_ids.add(target_id)

        status = str(binding.get("selection_status") or "").strip()
        chosen = binding.get("chosen_candidate") if isinstance(binding.get("chosen_candidate"), dict) else None
        if status in SELECTED_BINDING_STATUSES:
            result["selected_count"] += 1
            if chosen is None:
                result["errors"].append(f"{target_id}: selected target has no chosen_candidate")
            elif after_acquisition and not _has_local_asset(chosen):
                result["errors"].append(f"{target_id}: selected target after acquisition has no asset_local_path")
            if chosen is not None:
                identity_ok, identity_info = candidate_identity_gate(binding, chosen)
                group = str(identity_info.get("identity_target_group") or binding.get("semantic_group") or "").strip()
                if group in {"bed", "tv", "computer"} and not identity_ok:
                    result["errors"].append(
                        f"{target_id}: identity gate failed for {group}: {identity_info.get('identity_reject_reason')}"
                    )
        elif after_acquisition and status == "no_real_asset_after_acquisition" and chosen is not None:
            result["warnings"].append(f"{target_id}: no_real_asset_after_acquisition still has chosen_candidate")

        for rank, candidate in enumerate(binding.get("top_candidates") or [], start=1):
            if not isinstance(candidate, dict):
                result["errors"].append(f"{target_id}: top_candidate #{rank} is not an object")
                continue
            if design_aware and candidate.get("final_score") is None:
                result["warnings"].append(f"{target_id}: top_candidate #{rank} has no final_score")
            breakdown = candidate.get("score_breakdown")
            if breakdown is not None and not isinstance(breakdown, dict):
                result["errors"].append(f"{target_id}: top_candidate #{rank} score_breakdown is not a dict")

        if chosen is not None:
            breakdown = chosen.get("score_breakdown")
            if breakdown is not None and not isinstance(breakdown, dict):
                result["errors"].append(f"{target_id}: chosen score_breakdown is not a dict")

        notes = [str(x) for x in (binding.get("selection_notes") or [])]
        shared_candidate = ""
        for note in notes:
            if note.startswith("scene_consistency_shared_candidate:"):
                shared_candidate = note.split(":", 1)[1].strip()
                break
        group_id = str(binding.get("consistency_group_id") or "").strip()
        if not group_id and shared_candidate:
            group_id = f"implicit:{binding.get('semantic_group') or binding.get('category') or 'group'}"
        if group_id:
            cid = shared_candidate or _candidate_id(chosen)
            if cid:
                consistency_groups[group_id].add(cid)

    for group_id, ids in sorted(consistency_groups.items()):
        if len(ids) > 1:
            result["errors"].append(f"consistency group {group_id} has different candidate_ids: {sorted(ids)}")

    return result, data


def _cross_mode_summary(items: list[dict[str, Any]], data_by_path: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids_by_mode: dict[str, set[str]] = {}
    candidates_by_target: dict[str, set[str]] = defaultdict(set)
    for item in items:
        data = data_by_path.get(item["path"])
        if not data:
            continue
        mode = str(item.get("mode") or "unknown")
        target_ids: set[str] = set()
        for binding in data.get("bindings") or []:
            if not isinstance(binding, dict):
                continue
            target_id = str(binding.get("target_id") or "").strip()
            if not target_id:
                continue
            target_ids.add(target_id)
            chosen = binding.get("chosen_candidate") if isinstance(binding.get("chosen_candidate"), dict) else None
            candidates_by_target[target_id].add(_candidate_id(chosen))
        ids_by_mode[mode] = target_ids

    target_sets = list(ids_by_mode.values())
    target_ids_same = all(s == target_sets[0] for s in target_sets[1:]) if target_sets else True
    different = 0
    same = 0
    for ids in candidates_by_target.values():
        normalized = {x for x in ids if x}
        if len(normalized) > 1:
            different += 1
        elif len(normalized) == 1:
            same += 1
    return {
        "target_ids_same": target_ids_same,
        "different_candidate_count": different,
        "same_candidate_count": same,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate supplier binding variants and cross-mode coverage.")
    ap.add_argument("--bindings", action="append", required=True, help="supplier_bindings JSON path. Can be repeated.")
    ap.add_argument("--out", required=True, help="Output validation JSON path.")
    args = ap.parse_args(argv)

    items: list[dict[str, Any]] = []
    data_by_path: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for raw_path in args.bindings:
        path = Path(raw_path).expanduser().resolve()
        item, data = _validate_binding_file(path)
        items.append(item)
        errors.extend(f"{path.name}: {x}" for x in item["errors"])
        warnings.extend(f"{path.name}: {x}" for x in item["warnings"])
        if data is not None:
            data_by_path[item["path"]] = data

    cross_mode = _cross_mode_summary(items, data_by_path)
    if len(items) > 1 and not cross_mode["target_ids_same"]:
        warnings.append("Target coverage differs across modes.")

    out = {
        "ok": not errors,
        "bindings": items,
        "cross_mode": cross_mode,
        "errors": errors,
        "warnings": warnings,
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
