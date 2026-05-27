import pytest

from src.prompt_compiler.policies import (
    ScenePolicies,
    build_policy_key,
    get_acceptance_profile,
    get_room_program,
    get_solver_profile,
    get_style_policy,
    load_policies,
    resolve_area_bucket,
)
from src.prompt_compiler.schemas import GeometryHint, PromptIntent, RoomType, StyleIntent, StyleLabel


def test_area_bucket_resolution() -> None:
    policies = load_policies("config/scene_policies.yaml")
    assert resolve_area_bucket(5.0, policies) == "micro"
    assert resolve_area_bucket(7.0, policies) == "compact"
    assert resolve_area_bucket(9.0, policies) == "standard"


def test_room_program_lookup() -> None:
    policies = load_policies("config/scene_policies.yaml")
    program = get_room_program("Bedroom", "micro", policies)
    assert "Bed" in program["required_semantics"]
    assert program["max_counts"]["Storage"] == 1


def test_style_policy_lookup() -> None:
    policies = load_policies("config/scene_policies.yaml")
    style = get_style_policy("japandi", policies)
    assert "FloorLampFactory" in style["factory_blacklist"]


def test_policy_errors_profiles_and_key_builder() -> None:
    policies = load_policies("config/scene_policies.yaml")

    with pytest.raises(KeyError, match="area bucket"):
        resolve_area_bucket(9999.0, policies)
    with pytest.raises(KeyError, match="room_type"):
        get_room_program("Garage", "standard", policies)
    minimal = ScenePolicies(schema_version="unit", room_programs={"Bedroom": {"micro": {}}})
    with pytest.raises(KeyError, match="room program"):
        get_room_program("Bedroom", "missing_bucket", minimal)
    with pytest.raises(KeyError, match="style policy"):
        get_style_policy("unknown", policies)
    with pytest.raises(KeyError, match="solver profile"):
        get_solver_profile("missing", policies)
    with pytest.raises(KeyError, match="acceptance profile"):
        get_acceptance_profile("missing", policies)

    assert get_solver_profile("bedroom_micro_low_density", policies)
    assert get_acceptance_profile("bedroom_balanced", policies)

    cases = [
        (5.0, "micro"),
        (7.0, "compact"),
        (10.0, "standard"),
        (13.0, "large"),
    ]
    for area, bucket in cases:
        intent = PromptIntent(
            prompt_text="x",
            room_type=RoomType.BEDROOM,
            geometry=GeometryHint(target_area_sqm=area),
            style=StyleIntent(style_label=StyleLabel.JAPANDI),
        )
        assert build_policy_key(intent) == f"bedroom_{bucket}_japandi"

    unstyled = PromptIntent(prompt_text="x", room_type=RoomType.KITCHEN, geometry=GeometryHint(target_area_sqm=None))
    assert build_policy_key(unstyled) == "kitchen_micro_unstyled"
