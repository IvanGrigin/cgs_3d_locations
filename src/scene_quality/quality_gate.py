from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.prompt_compiler.inventory_mapping import factory_to_semantic
from src.prompt_compiler.schemas import CompiledPolicy, GateResult


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_counts(candidate_dir: Path) -> tuple[Counter[str], Counter[str], dict[str, Any]]:
    inventory_items = _load_json(candidate_dir / "inventory.json")
    summary = _load_json(candidate_dir / "inventory_summary.json")
    semantic_counts: Counter[str] = Counter()
    factory_counts: Counter[str] = Counter()
    core_semantic_counts = Counter({str(k): int(v) for k, v in (summary.get("core_semantic_counts") or {}).items()})
    core_factory_counts = Counter({str(k): int(v) for k, v in (summary.get("core_factory_counts") or {}).items()})
    for item in inventory_items:
        factory_name = str(item.get("factory_name") or "").strip()
        semantic = factory_to_semantic(factory_name) or str(item.get("semantic") or factory_name).strip()
        if semantic and semantic not in core_semantic_counts:
            semantic_counts[semantic] += 1
        if factory_name and factory_name not in core_factory_counts:
            factory_counts[factory_name] += 1
    semantic_counts.update(core_semantic_counts)
    factory_counts.update(core_factory_counts)
    return semantic_counts, factory_counts, summary


def evaluate_candidate(compiled_policy: CompiledPolicy, candidate_dir: str | Path) -> GateResult:
    candidate_path = Path(candidate_dir).expanduser().resolve()
    semantic_counts, factory_counts, inventory_summary = _normalized_counts(candidate_path)
    solver_summary = _load_json(candidate_path / "solver_summary.json") if (candidate_path / "solver_summary.json").is_file() else {}
    hard_failures: list[str] = []
    soft_failures: list[str] = []
    score = 10.0
    real_object_count = int(inventory_summary.get("real_object_count", 0))

    for semantic in compiled_policy.acceptance_policy.required_semantics:
        if semantic == "Bed" and semantic_counts.get("Bed", 0) <= 0:
            hard_failures.append("missing_required_bed")
        elif semantic_counts.get(semantic, 0) <= 0 and semantic not in {"Lighting", "CeilingLight"}:
            soft_failures.append(f"missing_preferred_{semantic.lower()}")
            score -= 0.5

    if compiled_policy.geometry.room_type.value == "Bedroom" and semantic_counts.get("Bed", 0) <= 0:
        hard_failures.append("bedroom_without_bed")

    for factory_name in compiled_policy.acceptance_policy.factory_blacklist:
        if factory_counts.get(factory_name, 0) > 0:
            hard_failures.append(f"forbidden_factory:{factory_name}")

    for semantic, max_count in compiled_policy.acceptance_policy.max_counts.items():
        current = semantic_counts.get(semantic, 0)
        if current > int(max_count):
            if compiled_policy.geometry.area_bucket.value == "micro" and semantic in {"Storage", "LowStorage", "TallStorage"}:
                hard_failures.append("storage_overflow_small_room")
            else:
                soft_failures.append(f"count_overflow:{semantic}>{max_count}")
                score -= min(2.0, 0.5 * (current - int(max_count)))

    if real_object_count < compiled_policy.acceptance_policy.min_real_objects:
        hard_failures.append("too_few_real_objects")
    if real_object_count == 0:
        hard_failures.append("empty_scene_generated")

    repeated = max(factory_counts.values(), default=0)
    if repeated > compiled_policy.acceptance_policy.max_repeated_factory_count:
        soft_failures.append("repeated_factory_family")
        score -= min(2.0, 0.5 * (repeated - compiled_policy.acceptance_policy.max_repeated_factory_count))

    violations = solver_summary.get("violations") or {}
    if violations and compiled_policy.acceptance_policy.reject_on_solver_violation:
        forbidden_violations = {
            str(key): value
            for key, value in violations.items()
            if str(key) not in compiled_policy.acceptance_policy.allowed_solver_violations and value
        }
        if forbidden_violations:
            hard_failures.append("solver_constraint_violation")

    if solver_summary.get("empty_candidate_pool_detected"):
        hard_failures.append("empty_candidate_pool_before_solve")

    if compiled_policy.geometry.area_bucket.value == "micro" and semantic_counts.get("Storage", 0) > 1:
        hard_failures.append("storage_overflow_small_room")

    whitelist = set(compiled_policy.acceptance_policy.factory_whitelist)
    if whitelist:
        matched_whitelist = sum(factory_counts.get(name, 0) for name in whitelist)
        if matched_whitelist <= 0:
            soft_failures.append("style_whitelist_not_observed")
            score -= 1.0

    hard_failures = list(dict.fromkeys(hard_failures))
    soft_failures = list(dict.fromkeys(soft_failures))
    if real_object_count == 0:
        score = 0.0
    elif hard_failures:
        score = min(score, max(0.5, 3.0 - 0.5 * max(0, len(hard_failures) - 1)))
    score = max(0.0, min(10.0, score))
    passed = not hard_failures and score >= compiled_policy.acceptance_policy.min_rule_score
    return GateResult(
        passed=passed,
        rule_score=score,
        hard_failures=hard_failures,
        soft_failures=soft_failures,
        inventory_summary=dict(inventory_summary),
        solver_summary=dict(solver_summary),
        candidate_dir=str(candidate_path),
    )


def write_gate_result(gate_result: GateResult, out_path: str | Path) -> None:
    gate_result.save(out_path)
