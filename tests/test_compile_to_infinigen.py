import pytest

pytest.skip("legacy test for archived prompt compiler modules", allow_module_level=True)

from src.prompt_compiler.compile_to_infinigen import compile_prompt_intent
from src.prompt_compiler.policies import load_policies
from src.prompt_compiler.prompt_to_intent import extract_intent
from src.prompt_compiler.llm_client import StubLLMClient


def test_micro_japandi_bedroom_policy() -> None:
    policies = load_policies("config/scene_policies.yaml")
    intent = extract_intent(
        "small japandi bedroom, 5 sqm, one bed, minimal, calm, light wood",
        StubLLMClient(
            {
                "room_type": "Bedroom",
                "style_label": "japandi",
                "target_area_sqm": 5.0,
                "required_objects": ["bed"],
                "favorite_colors": ["light wood"],
            }
        ),
    )
    compiled = compile_prompt_intent(intent, policies)
    assert "Bed" in compiled.program.required_semantics
    assert compiled.program.max_counts["Storage"] == 1
    assert "FloorLampFactory" in compiled.program.factory_blacklist
    assert compiled.infinigen_policy.stage_flags["solve_medium_enabled"] is True
    assert compiled.preflight["apply_child_restrictions"] is True
    assert compiled.program.allowed_primary == ["Bed", "Lighting", "Storage", "SideTable", "CeilingLight"]
    assert compiled.program.allowed_secondary == []
    assert "restrict_solving.restrict_child_primary=['Bed', 'Lighting', 'Storage', 'SideTable', 'CeilingLight']" in compiled.infinigen_policy.gin_overrides


def test_standard_baroque_bedroom_core_pass_is_bed_first() -> None:
    policies = load_policies("config/scene_policies.yaml")
    intent = extract_intent(
        "baroque inspired bedroom, 2.5 x 3.5 m, ornate but realistic, one bed, warm light, dark wood",
        StubLLMClient(
            {
                "room_type": "Bedroom",
                "style_label": "baroque_inspired",
                "target_area_sqm": 8.75,
                "required_objects": ["bed"],
            }
        ),
    )
    compiled = compile_prompt_intent(intent, policies)
    assert compiled.program.required_primary == ["Bed", "Lighting"]
    assert compiled.program.allowed_primary == ["Bed", "Lighting", "Storage", "SideTable", "CeilingLight"]
    assert compiled.program.allowed_secondary == []
    assert "MirrorFactory" not in compiled.program.factory_whitelist
    assert "FloorLampFactory" not in compiled.program.factory_whitelist
    assert "KitchenCabinetFactory" not in compiled.program.factory_whitelist
