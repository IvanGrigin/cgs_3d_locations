from __future__ import annotations

from pathlib import Path

from src.prompt_compiler.compile_to_infinigen import (
    _build_preflight,
    _normalize_factory_lists,
    build_gin_overrides,
)
from src.prompt_compiler.schemas import CompiledPolicy, GateResult, JudgeResult, RepairPlan, RepairReason


def build_repair_plan(compiled_policy: CompiledPolicy, gate_result: GateResult, judge_result: JudgeResult | None) -> RepairPlan:
    reasons: list[RepairReason] = []
    actions: list[str] = []
    updated_monkeypatch_params: dict[str, float] = {}
    updated_max_counts: dict[str, int] = {}
    added_factory_blacklist: list[str] = []
    added_factory_whitelist: list[str] = []
    added_required_semantics: list[str] = []
    removed_optional_semantics: list[str] = []
    added_gin_overrides: list[str] = []

    if "missing_required_bed" in gate_result.hard_failures or "bedroom_without_bed" in gate_result.hard_failures:
        reasons.append(RepairReason.MISSING_REQUIRED_BED)
        actions.append("strengthen bed requirement and reduce storage competition")
        added_required_semantics.append("Bed")
        updated_max_counts["Storage"] = min(1, compiled_policy.program.max_counts.get("Storage", 1))
        added_factory_blacklist.extend(["LargeShelfFactory", "SimpleBookcaseFactory", "CellShelfFactory"])
        updated_monkeypatch_params["furniture_fullness_pct"] = max(
            0.38,
            float(compiled_policy.infinigen_policy.monkeypatch_params.get("furniture_fullness_pct", 0.5)) - 0.06,
        )

    if "storage_overflow_small_room" in gate_result.hard_failures:
        reasons.append(RepairReason.STORAGE_OVERFLOW_SMALL_ROOM)
        actions.append("tighten storage count and blacklist tall storage families")
        updated_max_counts["Storage"] = 1
        updated_max_counts["TallStorage"] = 0
        added_factory_blacklist.extend(["LargeShelfFactory", "SimpleBookcaseFactory", "CellShelfFactory", "KitchenCabinetFactory"])
        updated_monkeypatch_params["obj_interior_obj_pct"] = max(
            0.12,
            float(compiled_policy.infinigen_policy.monkeypatch_params.get("obj_interior_obj_pct", 0.2)) - 0.08,
        )
        updated_monkeypatch_params["obj_on_storage_pct"] = max(
            0.05,
            float(compiled_policy.infinigen_policy.monkeypatch_params.get("obj_on_storage_pct", 0.1)) - 0.05,
        )

    if "repeated_factory_family" in gate_result.soft_failures:
        reasons.append(RepairReason.REPEATED_FACTORY_FAMILY)
        actions.append("reduce repeated family pressure and enable broader screening")
        updated_monkeypatch_params["obj_interior_obj_pct"] = max(
            0.12,
            float(compiled_policy.infinigen_policy.monkeypatch_params.get("obj_interior_obj_pct", 0.2)) - 0.04,
        )

    if gate_result.rule_score < compiled_policy.acceptance_policy.min_rule_score:
        if gate_result.inventory_summary.get("real_object_count", 0) < compiled_policy.acceptance_policy.min_real_objects:
            reasons.append(RepairReason.TOO_EMPTY)
            actions.append("slightly raise density and enable medium solve if available")
            updated_monkeypatch_params["furniture_fullness_pct"] = min(
                0.68,
                float(compiled_policy.infinigen_policy.monkeypatch_params.get("furniture_fullness_pct", 0.5)) + 0.04,
            )
            added_gin_overrides.append("compose_indoors.solve_medium_enabled=True")
        else:
            reasons.append(RepairReason.TOO_CLUTTERED)
            actions.append("lower clutter metrics and drop optional semantics")
            updated_monkeypatch_params["obj_on_nonstorage_pct"] = max(
                0.04,
                float(compiled_policy.infinigen_policy.monkeypatch_params.get("obj_on_nonstorage_pct", 0.1)) - 0.04,
            )
            removed_optional_semantics.extend(["LargePlant", "Decor"])

    if judge_result is not None and judge_result.style_match_score < compiled_policy.acceptance_policy.min_judge_score:
        reasons.append(RepairReason.STYLE_NOT_READABLE)
        actions.append("strengthen style whitelist and medium-stage solving")
        added_factory_whitelist.extend(compiled_policy.style_policy.factory_whitelist[:4])
        added_gin_overrides.append("compose_indoors.solve_medium_enabled=True")

    if gate_result.solver_summary.get("violations"):
        reasons.append(RepairReason.SOLVER_CONSTRAINT_VIOLATION)
        actions.append("keep solver stages but rerun more seeds under stricter program")

    if "empty_scene_generated" in gate_result.hard_failures or "empty_candidate_pool_before_solve" in gate_result.hard_failures:
        reasons.append(RepairReason.BAD_AREA_PROGRAM_FIT)
        actions.append("remove aggressive semantic pruning and widen effective pool for required bedroom objects")
        updated_monkeypatch_params["furniture_fullness_pct"] = max(
            0.44,
            float(compiled_policy.infinigen_policy.monkeypatch_params.get("furniture_fullness_pct", 0.5)) - 0.04,
        )
        added_factory_whitelist.extend(["BedFactory", "SingleCabinetFactory", "CeilingLightFactory", "LampFactory"])
        added_factory_blacklist.extend(["LargeShelfFactory", "SimpleBookcaseFactory", "CellShelfFactory"])

    return RepairPlan(
        reasons=reasons,
        actions=actions,
        updated_monkeypatch_params=updated_monkeypatch_params,
        added_factory_blacklist=sorted(set(added_factory_blacklist)),
        added_factory_whitelist=sorted(set(added_factory_whitelist)),
        updated_max_counts=updated_max_counts,
        added_required_semantics=sorted(set(added_required_semantics)),
        removed_optional_semantics=sorted(set(removed_optional_semantics)),
        added_gin_overrides=sorted(set(added_gin_overrides)),
    )


def apply_repair_plan(compiled_policy: CompiledPolicy, repair_plan: RepairPlan) -> CompiledPolicy:
    repaired = CompiledPolicy.model_validate(compiled_policy.model_dump(mode="json"))
    repaired.repair_round += 1
    repaired.parent_policy_path = repaired.artifacts.get("compiled_policy") or repaired.parent_policy_path

    repaired.infinigen_policy.monkeypatch_params.update(repair_plan.updated_monkeypatch_params)

    merged_required = sorted(
        set(repaired.program.required_semantics)
        | set(repaired.acceptance_policy.required_semantics)
        | set(repair_plan.added_required_semantics)
    )
    merged_forbidden = sorted(
        set(repaired.program.forbidden_semantics)
        | set(repaired.acceptance_policy.forbidden_semantics)
    )

    base_blacklist = set(repaired.infinigen_policy.factory_blacklist) | set(repaired.program.factory_blacklist) | set(repair_plan.added_factory_blacklist)
    whitelist_inputs = [
        set(repaired.infinigen_policy.factory_whitelist),
        set(repaired.program.factory_whitelist),
        set(repaired.style_policy.factory_whitelist),
    ]
    if repair_plan.added_factory_whitelist:
        whitelist_inputs.append(set(repair_plan.added_factory_whitelist))
    non_empty_whitelists = [items for items in whitelist_inputs if items]
    if non_empty_whitelists:
        merged_whitelist = set.intersection(*non_empty_whitelists)
    else:
        merged_whitelist = set()
    normalized_whitelist, normalized_blacklist = _normalize_factory_lists(sorted(merged_whitelist), sorted(base_blacklist))

    merged_max_counts = dict(repaired.program.max_counts)
    for source in (repaired.acceptance_policy.max_counts, repaired.infinigen_policy.max_counts, repair_plan.updated_max_counts):
        for key, value in source.items():
            if key in merged_max_counts:
                merged_max_counts[key] = min(int(merged_max_counts[key]), int(value))
            else:
                merged_max_counts[key] = int(value)

    repaired.program.required_semantics = list(merged_required)
    repaired.acceptance_policy.required_semantics = list(merged_required)
    repaired.style_policy.required_semantics = list(merged_required)
    repaired.program.forbidden_semantics = list(merged_forbidden)
    repaired.acceptance_policy.forbidden_semantics = list(merged_forbidden)
    repaired.style_policy.forbidden_semantics = list(merged_forbidden)
    repaired.infinigen_policy.factory_blacklist = list(normalized_blacklist)
    repaired.program.factory_blacklist = list(normalized_blacklist)
    repaired.acceptance_policy.factory_blacklist = list(normalized_blacklist)
    repaired.style_policy.factory_blacklist = list(normalized_blacklist)
    repaired.infinigen_policy.factory_whitelist = list(normalized_whitelist)
    repaired.program.factory_whitelist = list(normalized_whitelist)
    repaired.acceptance_policy.factory_whitelist = list(normalized_whitelist)
    repaired.style_policy.factory_whitelist = list(normalized_whitelist)
    repaired.infinigen_policy.max_counts = dict(merged_max_counts)
    repaired.program.max_counts = dict(merged_max_counts)
    repaired.acceptance_policy.max_counts = dict(merged_max_counts)

    for semantic in repair_plan.added_required_semantics:
        if semantic not in repaired.program.required_semantics:
            repaired.program.required_semantics.append(semantic)
        if semantic not in repaired.acceptance_policy.required_semantics:
            repaired.acceptance_policy.required_semantics.append(semantic)

    if repair_plan.removed_optional_semantics:
        repaired.program.optional_semantics = [
            item for item in repaired.program.optional_semantics if item not in set(repair_plan.removed_optional_semantics)
        ]

    repaired.preflight = _build_preflight(
        repaired.program.required_semantics,
        repaired.program.factory_whitelist,
        repaired.program.factory_blacklist,
    )
    if repaired.program.allowed_primary:
        repaired.preflight["apply_child_restrictions"] = True
        repaired.preflight["final_restrict_child_primary"] = list(repaired.program.allowed_primary)
        repaired.preflight["final_restrict_child_secondary"] = list(repaired.program.allowed_secondary)
    repaired.preflight["max_counts"] = dict(repaired.program.max_counts)
    repaired.preflight["stage_flags"] = dict(repaired.infinigen_policy.stage_flags)
    repaired.preflight["solver_steps"] = dict(repaired.infinigen_policy.solver_steps)

    repaired.infinigen_policy.gin_overrides = sorted(
        set(build_gin_overrides(repaired) + repair_plan.added_gin_overrides)
    )
    return repaired
