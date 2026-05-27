import pytest

pytest.skip("legacy test for archived scene_quality/prompt_compiler modules", allow_module_level=True)

from src.prompt_compiler.compile_to_infinigen import compile_prompt_intent
from src.prompt_compiler.llm_client import StubLLMClient
from src.prompt_compiler.policies import load_policies
from src.prompt_compiler.prompt_to_intent import extract_intent
from src.prompt_compiler.schemas import GateResult, JudgeResult
from src.scene_quality.repair_loop import apply_repair_plan, build_repair_plan


def _compiled_policy():
    policies = load_policies("config/scene_policies.yaml")
    intent = extract_intent(
        "small japandi bedroom, 5 sqm, one bed",
        StubLLMClient(
            {
                "room_type": "Bedroom",
                "style_label": "japandi",
                "target_area_sqm": 5.0,
                "required_objects": ["bed"],
            }
        ),
    )
    return compile_prompt_intent(intent, policies)


def test_missing_required_bed_repair_plan() -> None:
    compiled = _compiled_policy()
    gate = GateResult(
        passed=False,
        rule_score=3.0,
        hard_failures=["missing_required_bed"],
        soft_failures=[],
        inventory_summary={"real_object_count": 1},
        solver_summary={},
    )
    judge = JudgeResult(
        passed=False,
        total_score=3.5,
        functionality_score=2.0,
        prompt_match_score=4.0,
        style_match_score=4.0,
        composition_score=4.0,
    )
    plan = build_repair_plan(compiled, gate, judge)
    assert "Bed" in plan.added_required_semantics
    assert "LargeShelfFactory" in plan.added_factory_blacklist


def test_storage_overflow_repair_tightens_policy() -> None:
    compiled = _compiled_policy()
    gate = GateResult(
        passed=False,
        rule_score=4.0,
        hard_failures=["storage_overflow_small_room"],
        soft_failures=[],
        inventory_summary={"real_object_count": 4},
        solver_summary={},
    )
    plan = build_repair_plan(compiled, gate, None)
    repaired = apply_repair_plan(compiled, plan)
    assert repaired.program.max_counts["Storage"] == 1
    assert "SimpleBookcaseFactory" in repaired.program.factory_blacklist


def test_empty_scene_repair_recomputes_restrictive_runtime_policy() -> None:
    compiled = _compiled_policy()
    gate = GateResult(
        passed=False,
        rule_score=0.0,
        hard_failures=["empty_scene_generated", "missing_required_bed"],
        soft_failures=[],
        inventory_summary={"real_object_count": 0},
        solver_summary={"empty_candidate_pool_detected": True},
    )
    plan = build_repair_plan(compiled, gate, None)
    repaired = apply_repair_plan(compiled, plan)
    assert "bad_area_program_fit" in [reason.value for reason in plan.reasons]
    assert repaired.preflight["apply_child_restrictions"] is True
    assert repaired.program.max_counts["Storage"] == 1
    assert repaired.program.factory_blacklist
    assert all(name not in repaired.program.factory_whitelist for name in repaired.program.factory_blacklist)
