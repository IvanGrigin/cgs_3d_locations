import json
from pathlib import Path

import pytest

pytest.skip("legacy test for archived scene_quality/prompt_compiler modules", allow_module_level=True)

from src.prompt_compiler.compile_to_infinigen import compile_prompt_intent
from src.prompt_compiler.llm_client import StubLLMClient
from src.prompt_compiler.policies import load_policies
from src.prompt_compiler.prompt_to_intent import extract_intent
from src.scene_quality.quality_gate import evaluate_candidate


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


def _write_candidate(tmp_path: Path, inventory_items: list[dict], summary: dict, solver_summary: dict) -> Path:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "inventory.json").write_text(json.dumps(inventory_items, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "solver_summary.json").write_text(json.dumps(solver_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return candidate_dir


def test_bedroom_without_bed_hard_fail(tmp_path: Path) -> None:
    compiled = _compiled_policy()
    candidate_dir = _write_candidate(
        tmp_path,
        [{"factory_name": "SingleCabinetFactory", "semantic": "LowStorage"}],
        {"real_object_count": 1, "factory_counts": {"SingleCabinetFactory": 1}, "semantic_counts": {"LowStorage": 1}},
        {"violations": {"bedroom": 1}},
    )
    result = evaluate_candidate(compiled, candidate_dir)
    assert not result.passed
    assert "bedroom_without_bed" in result.hard_failures


def test_micro_room_storage_overflow_hard_fail(tmp_path: Path) -> None:
    compiled = _compiled_policy()
    candidate_dir = _write_candidate(
        tmp_path,
        [
            {"factory_name": "BedFactory", "semantic": "Bed"},
            {"factory_name": "SingleCabinetFactory", "semantic": "LowStorage"},
            {"factory_name": "SimpleBookcaseFactory", "semantic": "TallStorage"},
            {"factory_name": "CellShelfFactory", "semantic": "TallStorage"},
        ],
        {
            "real_object_count": 4,
            "factory_counts": {"BedFactory": 1, "SingleCabinetFactory": 1, "SimpleBookcaseFactory": 1, "CellShelfFactory": 1},
            "semantic_counts": {"Bed": 1, "Storage": 3},
        },
        {"violations": {}},
    )
    result = evaluate_candidate(compiled, candidate_dir)
    assert "storage_overflow_small_room" in result.hard_failures


def test_good_candidate_passes(tmp_path: Path) -> None:
    compiled = _compiled_policy()
    candidate_dir = _write_candidate(
        tmp_path,
        [
            {"factory_name": "BedFactory", "semantic": "Bed"},
            {"factory_name": "CeilingLightFactory", "semantic": "CeilingLight"},
        ],
        {
            "real_object_count": 2,
            "factory_counts": {"BedFactory": 1, "CeilingLightFactory": 1},
            "semantic_counts": {"Bed": 1, "CeilingLight": 1},
        },
        {"violations": {}},
    )
    result = evaluate_candidate(compiled, candidate_dir)
    assert result.passed
