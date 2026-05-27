from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prompt_compiler import compile_to_infinigen as cti
from src.prompt_compiler import inventory_mapping as inv
from src.prompt_compiler.policies import get_room_program, load_policies
from src.prompt_compiler.schemas import (
    GeometryHint,
    ObjectsIntent,
    OpeningsIntent,
    PreferencesIntent,
    PromptIntent,
    RoomType,
    StyleIntent,
    StyleLabel,
)


def _policies():
    return load_policies(Path("config/scene_policies.yaml"))


def test_compile_prompt_intent_bedroom_core_and_artifacts(tmp_path):
    intent = PromptIntent(
        prompt_text="minimal japandi bedroom with bed and storage",
        room_type=RoomType.BEDROOM,
        geometry=GeometryHint(target_area_sqm=12.0, width_m=None, depth_m=None, height_m=2.8),
        style=StyleIntent(style_label=StyleLabel.JAPANDI, palette_hint=["warm white"], material_family=["wood"]),
        openings=OpeningsIntent(wants_door=True, wants_window=True),
        objects=ObjectsIntent(required=["Bed"], desired=["Storage", "SideTable"], forbidden=["LargePlant"]),
        preferences=PreferencesIntent(favorite_colors=["oak"], avoid_colors=["red"], notes="quiet"),
    )

    compiled = cti.compile_prompt_intent(intent, _policies())

    assert compiled.geometry.room_type == RoomType.BEDROOM
    assert compiled.geometry.area_sqm == 12.0
    assert compiled.geometry.width_m > 0
    assert compiled.geometry.depth_m > 0
    assert [wall.id for wall in compiled.geometry.walls] == ["w0", "w1", "w2", "w3"]
    assert compiled.geometry.doors and compiled.geometry.windows
    assert compiled.program.required_semantics == cti.BEDROOM_CORE_REQUIRED
    assert "BedFactory" in compiled.program.factory_whitelist
    assert "LargePlantContainerFactory" in compiled.program.factory_blacklist
    assert compiled.preflight["apply_child_restrictions"] is True
    assert any(item.startswith("compose_indoors.solve_steps_large=") for item in compiled.infinigen_policy.gin_overrides)
    assert any("restrict_parent_rooms" in item for item in compiled.infinigen_policy.gin_overrides)

    room_json = cti.build_room_json(compiled)
    assert room_json["room"]["room_type"] == "Bedroom"
    style_profile = cti.build_style_profile(compiled)
    assert style_profile["style_label"] == "japandi"

    cti.write_compiled_artifacts(compiled, tmp_path)
    assert json.loads((tmp_path / "room.json").read_text(encoding="utf-8"))["room"]["style_hint"] == "japandi"
    assert json.loads((tmp_path / "gin_overrides.json").read_text(encoding="utf-8"))
    assert compiled.artifacts["compiled_policy"].endswith("compiled_policy.original.json")


def test_compile_prompt_intent_helpers_and_error_paths(monkeypatch):
    assert cti._canonical_dimensions(10.0, "Bedroom", {"canonical_dimensions_m": [3.0, 4.0]}) == (3.0, 4.0)
    assert cti._build_floor_polygon(2.0, 3.0)[2] == {"x": 2.0, "y": 3.0}
    assert cti._build_walls()[0]["id"] == "w0"
    doors, windows = cti._build_openings(2.0, 2.0, False, True)
    assert doors == []
    assert windows[0]["wall_id"] == "w2"
    assert cti._unique(["a", "b", "a"]) == ["a", "b"]
    assert cti._merge_semantics(["Bed"], ["Lighting", "Bed"]) == ["Bed", "Lighting"]

    primary, secondary = cti._allowed_semantics(["Bed"], ["Decor"], ["Lighting"], ["Decor"])
    assert "Bed" in primary
    assert "Decor" not in secondary

    whitelist, blacklist = cti._normalize_factory_lists(["A", "B", "A"], ["B", "C", ""])
    assert whitelist == ["A"]
    assert blacklist == ["B", "C"]
    coverage = cti._required_factory_coverage(["Bed"], ["BedFactory"], [])
    assert coverage["Bed"] == ["BedFactory"]
    preflight = cti._build_preflight(["Bed", "MissingSemantic"], ["BedFactory"], [])
    assert "MissingSemantic" in preflight["empty_required_semantics"]

    required, optional, preferred, max_counts, blacklist, flags, steps = cti._apply_bedroom_core_screening(
        required_semantics=["Bed", "Decor"],
        optional_semantics=["Storage", "Decor"],
        preferred_semantics=["SideTable"],
        max_counts={"Storage": 3, "Decor": 5},
        factory_blacklist=["OldFactory"],
        stage_flags={"solve_small_enabled": True},
        solver_steps={"solve_steps_large": 5, "solve_steps_small": 2},
    )
    assert required == cti.BEDROOM_CORE_REQUIRED
    assert "Storage" in optional
    assert max_counts["Decor"] == 0
    assert "OldFactory" in blacklist
    assert flags["solve_small_enabled"] is False
    assert steps["solve_steps_small"] == 0

    assert cti._apply_bedroom_core_factory_whitelist(["BedFactory", "Other"]) == ["BedFactory"]
    assert cti._solver_profile_key("Bedroom", "standard", {"density": "medium"}) == "bedroom_standard_medium_density"
    assert cti._stage_overrides({"solve_medium_enabled": True}, {"solve_steps_medium": 10}) == [
        "compose_indoors.solve_medium_enabled=True",
        "compose_indoors.solve_steps_medium=10",
    ]
    assert cti._solver_steps_from_profile({"solve_steps_large": "7", "other": 1}) == {"solve_steps_large": 7}

    bad_intent = PromptIntent(prompt_text="impossible", room_type=RoomType.BEDROOM, style=StyleIntent(style_label=StyleLabel.JAPANDI))
    monkeypatch.setattr(cti, "_build_preflight", lambda *_: {"empty_required_semantics": ["Bed"]})
    with pytest.raises(RuntimeError, match="empty_candidate_pool_before_solve"):
        cti.compile_prompt_intent(bad_intent, _policies())


def test_policy_standard_room_program_fallback() -> None:
    policies = _policies()
    standard = get_room_program("Bedroom", "missing_bucket", policies)
    assert standard == policies.room_programs["Bedroom"]["standard"]


def test_inventory_mapping_direct_helpers() -> None:
    assert inv._normalized_tokens("Queen-Bed") == ["queen bed", "queenbed"]
    assert inv.normalize_prompt_object("queen bed") == "Bed"
    assert inv.normalize_prompt_object("not in mapping") is None
    assert inv.factory_to_semantic("BedFactory") == "Bed"
    assert inv.factory_to_semantic("missing") is None
    assert "BedFactory" in inv.semantic_to_factory_family("Bed")
    assert inv.semantic_to_factory_family("missing") == []
    assert "BedFactory" in inv.expand_semantics_to_factories(["Bed", "Bed"])
    assert inv.is_technical_factory_name("hoof_parent_temp.001")
    assert inv.is_core_furniture_factory("hoof_parent_temp.001") is False
    assert inv.is_core_furniture_factory("BookStackFactory") is False
    assert inv.is_core_furniture_factory("WallArtFactory") is True
    assert inv.is_core_furniture_factory("BedFactory") is True
    assert inv.policy_semantic_to_infinigen("Desk") == ["Table"]
    assert inv.policy_semantic_to_infinigen("Rug") == []
    assert inv.expand_policy_semantics_to_infinigen(["Desk", "FloorLamp"]) == ["Lighting", "Table"]
