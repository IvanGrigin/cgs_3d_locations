#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


SELECTED_BINDING_STATUSES = {
    "heuristic_top1_selected",
    "heuristic_first_acceptable_selected",
    "llm_reranked_top1_selected",
    "llm_reranked_first_acceptable_selected",
}
ELIGIBLE_REPEAT_GROUPS = {"nightstand", "chair", "armchair", "lamp_table", "lamp_floor"}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _size_tuple(binding: dict[str, Any]) -> tuple[float, float, float] | None:
    raw = binding.get("requested_size_m")
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    try:
        return tuple(max(float(x), 1e-6) for x in raw[:3])
    except Exception:
        return None


def _sizes_similar(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> bool:
    if a is None or b is None:
        return True
    ratios = [max(x, y) / max(min(x, y), 1e-6) for x, y in zip(a, b)]
    return max(ratios) <= 1.22


def _candidate_key(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("unique_key") or "").strip()


def _candidate_asset_rank(candidate: dict[str, Any] | None) -> int:
    if not isinstance(candidate, dict):
        return 3
    if candidate.get("asset_local_path"):
        return 0
    if candidate.get("model_download_url") or candidate.get("model_page_url") or candidate.get("model_download_landing_url"):
        return 1
    return 2


def _candidate_score(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    for key in ("final_score",):
        try:
            return float(candidate.get(key) or 0.0)
        except Exception:
            pass
    breakdown = candidate.get("score_breakdown") if isinstance(candidate.get("score_breakdown"), dict) else {}
    try:
        return float(breakdown.get("final_score") or breakdown.get("design_score") or 0.0)
    except Exception:
        return 0.0


def _selected(binding: dict[str, Any]) -> bool:
    return str(binding.get("selection_status") or "") in SELECTED_BINDING_STATUSES and isinstance(binding.get("chosen_candidate"), dict)


def _group_key(binding: dict[str, Any]) -> str | None:
    group = str(binding.get("semantic_group") or "").strip()
    if group not in ELIGIBLE_REPEAT_GROUPS:
        return None
    return group


def _choose_shared_candidate(bindings: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for binding in bindings:
        pool = []
        if isinstance(binding.get("chosen_candidate"), dict):
            pool.append(binding["chosen_candidate"])
        pool.extend([x for x in (binding.get("top_candidates") or []) if isinstance(x, dict)])
        for candidate in pool:
            key = _candidate_key(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (_candidate_asset_rank(c), -_candidate_score(c), str(c.get("unique_key") or "")))
    return deepcopy(candidates[0])


def apply_supplier_scene_consistency(bindings_data: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(bindings_data)
    bindings = [x for x in (out.get("bindings") or []) if isinstance(x, dict)]
    buckets: dict[str, list[dict[str, Any]]] = {}
    for binding in bindings:
        key = _group_key(binding)
        if not key or not _selected(binding):
            continue
        buckets.setdefault(key, []).append(binding)

    applied_groups: list[dict[str, Any]] = []
    for key, group_bindings in buckets.items():
        if len(group_bindings) < 2:
            continue
        clusters: list[list[dict[str, Any]]] = []
        for binding in group_bindings:
            size = _size_tuple(binding)
            for cluster in clusters:
                if _sizes_similar(size, _size_tuple(cluster[0])):
                    cluster.append(binding)
                    break
            else:
                clusters.append([binding])

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            shared = _choose_shared_candidate(cluster)
            shared_key = _candidate_key(shared)
            if not shared or not shared_key:
                continue
            changed_ids: list[str] = []
            for binding in cluster:
                old_key = _candidate_key(binding.get("chosen_candidate"))
                if old_key == shared_key:
                    continue
                binding["chosen_candidate"] = deepcopy(shared)
                notes = list(binding.get("selection_notes") or [])
                notes.append(f"scene_consistency_shared_candidate:{shared_key}")
                binding["selection_notes"] = notes
                changed_ids.append(str(binding.get("target_id") or ""))
            if changed_ids:
                applied_groups.append({"semantic_group": key, "shared_unique_key": shared_key, "target_ids": changed_ids})

    meta = deepcopy(out.get("meta") or {})
    meta["scene_consistency"] = {
        "enabled": True,
        "policy": "repeat_same_semantic_group_similar_size_share_candidate",
        "eligible_groups": sorted(ELIGIBLE_REPEAT_GROUPS),
        "applied_group_count": len(applied_groups),
        "applied_groups": applied_groups,
    }
    out["meta"] = meta
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply scene-level supplier consistency to supplier_bindings JSON.")
    ap.add_argument("--bindings-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = apply_supplier_scene_consistency(read_json(args.bindings_json))
    write_json(args.out, out)
    summary = (out.get("meta") or {}).get("scene_consistency") or {}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
