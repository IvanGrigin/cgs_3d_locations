from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.pipeline.procedural_rooms import bathroom_generator, corridor_generator, living_room_generator, procedural_room_stage, toilet_generator
from src.pipeline.procedural_rooms.room_context import build_room_context, normalize_room_type


def room_scene(room_type: str, width: float, depth: float, *, doors: bool = True, windows: bool = False) -> dict:
    room = {
        "id": f"{room_type}_room",
        "room_type": room_type,
        "width_m": width,
        "depth_m": depth,
        "area_m2": width * depth,
        "ceiling_height_m": 2.8,
        "floor_polygon": [[0, 0], [width, 0], [width, depth], [0, depth]],
        "walls": [
            {"id": "w0", "name": "south"},
            {"id": "w1", "name": "east"},
            {"id": "w2", "name": "north"},
            {"id": "w3", "name": "west"},
        ],
    }
    if doors:
        room["doors"] = [{"id": "door", "wall_id": "w0", "s": max(0.1, width * 0.5 - 0.4), "width": 0.8}]
    if windows:
        room["windows"] = [{"id": "window", "wall_id": "w3", "s": 1.0, "width": 1.2}]
    return {"schema": "scene.v1", "room": room, "placements": []}


def categories(items: list[dict]) -> set[str]:
    return {str(item.get("category")) for item in items}


def test_living_room_and_corridor_generators_cover_dense_paths() -> None:
    living_ctx = build_room_context(room_scene("living_room", 6.0, 5.0, windows=True))
    living, living_report = living_room_generator.generate_living_room(living_ctx, density="very_high", seed=3)
    living_categories = categories(living)
    assert {"sofa", "tv_stand", "ceiling_light"} <= living_categories
    assert living_report["generator"] == "living_room_generator"
    assert living_report["tv_wall_id"]
    assert any(item.get("meta", {}).get("front_target") for item in living)

    tiny_ctx = build_room_context(room_scene("living_room", 0.2, 0.2, doors=False))
    tiny, tiny_report = living_room_generator.generate_living_room(tiny_ctx, density="normal", seed=1)
    assert isinstance(tiny, list)
    assert tiny_report["generator"] == "living_room_generator"

    corridor_ctx = build_room_context(room_scene("corridor", 5.5, 1.7, windows=True))
    corridor, corridor_report = corridor_generator.generate_corridor(corridor_ctx, density="very_high", seed=4)
    corridor_categories = categories(corridor)
    assert {"mirror", "ceiling_light"} <= corridor_categories
    assert corridor_report["corridor_width_m"] == corridor_ctx.min_side_m
    assert sum(1 for item in corridor if item.get("category") == "ceiling_light") >= 2


def test_bathroom_and_toilet_fallback_generators_with_sanitary_solver_disabled(monkeypatch) -> None:
    monkeypatch.setattr(bathroom_generator, "generate_sanitary_bathroom", lambda *args, **kwargs: None)
    monkeypatch.setattr(toilet_generator, "generate_sanitary_toilet", lambda *args, **kwargs: None)

    bathroom_ctx = build_room_context(room_scene("bathroom", 2.5, 2.4))
    bathroom, bathroom_report = bathroom_generator.generate_bathroom(bathroom_ctx, density="very_high", seed=5)
    bathroom_categories = categories(bathroom)
    assert "sink" in bathroom_categories
    assert {"bathtub", "shower"} & bathroom_categories
    assert "ceiling_light" in bathroom_categories
    assert bathroom_report["required"]["sink"] is True

    toilet_ctx = build_room_context(room_scene("toilet", 1.5, 2.0))
    toilet, toilet_report = toilet_generator.generate_toilet(toilet_ctx, density="very_high", seed=6)
    toilet_categories = categories(toilet)
    assert "toilet" in toilet_categories
    assert "ceiling_light" in toilet_categories
    assert toilet_report["required"]["toilet"] is True


def test_bathroom_generator_helper_fallback_edges(monkeypatch) -> None:
    bathroom_ctx = build_room_context(room_scene("bathroom", 2.0, 1.8))
    solved = ([{"id": "solved"}], {"generator": "sanitary"})
    monkeypatch.setattr(bathroom_generator, "generate_sanitary_bathroom", lambda *args, **kwargs: solved)
    assert bathroom_generator.generate_bathroom(bathroom_ctx, density="normal", seed=1) == solved

    monkeypatch.setattr(bathroom_generator, "generate_sanitary_bathroom", lambda *args, **kwargs: None)
    monkeypatch.setattr(bathroom_generator, "_try_wall_specs", lambda *args, **kwargs: None)
    fallback_items, fallback_report = bathroom_generator.generate_bathroom(bathroom_ctx, density="normal", seed=2)
    assert fallback_report["required"] == {"sink": True, "bathing_fixture": True}
    assert {"sink", "shower"} <= categories(fallback_items)

    wall = bathroom_ctx.walls[1]
    only_window_ctx = SimpleNamespace(walls=[SimpleNamespace(id="w", length=2.0, has_door=False, has_window=True)])
    only_door_ctx = SimpleNamespace(walls=[SimpleNamespace(id="d", length=2.0, has_door=True, has_window=False)])
    assert bathroom_generator._candidate_walls(only_window_ctx, random.Random(1), avoid_windows=True)[0].id == "w"
    assert bathroom_generator._candidate_walls(only_door_ctx, random.Random(1))[0].id == "d"
    assert bathroom_generator._door_wall(SimpleNamespace(walls=[wall], doors=[{"wall_id": wall.id}])) == wall
    assert bathroom_generator._door_wall(SimpleNamespace(walls=[], doors=[])) is None
    assert bathroom_generator._far_wall_from_door(SimpleNamespace(walls=bathroom_ctx.walls, doors=[], polygon=bathroom_ctx.polygon)) is not None
    assert bathroom_generator._wall_id_and_along(None) == ("", None)

    class FakeEngine:
        def __init__(self) -> None:
            self.rng = random.Random(1)
            self.rejected: list[dict] = []

        def add_wall_aligned(self, *args, **kwargs):
            return None

        def add_item(self, spec, center, yaw, **kwargs):
            return {"id": "fallback", "category": kwargs.get("category"), "position_m": [center.x, center.y, 0.0], "size_m": list(spec.size_m), "yaw_deg": yaw}

        def add_wall_art(self, wall_id, along, spec, **kwargs):
            return {"id": "wall_art", "wall_id": wall_id, "along": along, "category": kwargs.get("category")}

        def can_place(self, *args, **kwargs):
            return False, "blocked"

    fake_engine = FakeEngine()
    assert bathroom_generator._try_wall_specs(fake_engine, bathroom_ctx, ["compact_sink"], category="sink", layer="primary", front_target="door") is None
    fallback = bathroom_generator._fallback_center_fixture(
        fake_engine,
        bathroom_ctx,
        bathroom_generator.BATHROOM_SPECS["compact_sink"],
        category="sink",
        front_target="door",
    )
    assert fallback["category"] == "sink"
    assert bathroom_generator._add_wall_near(fake_engine, bathroom_ctx, None, "mirror", z_center=1.4)["category"] == "mirror"
    assert bathroom_generator._add_wall_near(fake_engine, SimpleNamespace(walls=[]), None, "mirror", z_center=1.4) is None

    assert bathroom_generator._add_bath_mat_in_front(fake_engine, None) is None
    assert bathroom_generator._add_bath_mat_in_front(
        fake_engine,
        {"id": "fixture", "position_m": [1, 1, 0], "size_m": [1, 1, 1], "yaw_deg": 0},
    ) is None
    assert fake_engine.rejected[-1]["reason"] == "bath_mat_no_front_clearance"


def test_room_context_and_stage_small_branch_edges(tmp_path: Path, monkeypatch) -> None:
    assert normalize_room_type("room", prompt="living room with sofa") == "living_room"
    assert normalize_room_type("room", area_m2=20.0) == "living_room"
    assert normalize_room_type("studio", prompt="bedroom with bed") == "bedroom"
    assert normalize_room_type("studio") == "living_room"
    assert normalize_room_type("office") == "office"

    nested_scene = {
        "data": {
            "room": {
                "id": "nested",
                "room_type": "room",
                "source_room_type": "wc",
                "width_m": 0,
                "depth_m": 2,
                "area_m2": 0,
                "windows": [{"wall_id": "missing"}],
            }
        }
    }
    nested_ctx = build_room_context(nested_scene)
    assert nested_ctx.room_type == "toilet"
    assert nested_ctx.aspect_ratio == 1.0
    assert nested_ctx.area_m2 == 0.0
    assert nested_ctx.size_class == "tiny"
    assert build_room_context({"room": {"room_type": "bedroom", "width_m": 3, "depth_m": 3, "area_m2": 8}}).size_class == "small"
    assert build_room_context({"room": {"room_type": "bedroom", "width_m": 4, "depth_m": 3, "area_m2": 12}}).size_class == "medium"
    assert build_room_context({"room": {"room_type": "bedroom", "width_m": 7, "depth_m": 5, "area_m2": 35}}).size_class == "xlarge"
    assert build_room_context({}).room_id == "room"

    assert procedural_room_stage._as_bool(None, True) is True
    assert procedural_room_stage._as_bool(True) is True
    assert procedural_room_stage._as_bool("yes") is True
    assert procedural_room_stage._as_bool("off", True) is False
    assert procedural_room_stage._as_bool("maybe", True) is True
    assert procedural_room_stage._extract_placements({"items": [{"id": "item"}]}) == [{"id": "item"}]
    assert procedural_room_stage._extract_placements({}) == []
    assert procedural_room_stage._is_removable_item({"category": "floor"}) is False
    assert procedural_room_stage._is_removable_item({"category": "custom", "source": {"placement_source": "procedural_room_stage/unit"}}) is True
    kept, removed = procedural_room_stage._filter_existing_placements([{"category": "bed"}], replace_existing=False)
    assert kept == [{"category": "bed"}]
    assert removed == 0

    calls: list[str] = []
    monkeypatch.setattr(procedural_room_stage, "generate_bedroom", lambda *args, **kwargs: (calls.append("bedroom") or [], {}))
    monkeypatch.setattr(procedural_room_stage, "generate_living_room", lambda *args, **kwargs: (calls.append("living_room") or [], {}))
    monkeypatch.setattr(procedural_room_stage, "generate_corridor", lambda *args, **kwargs: (calls.append("corridor") or [], {}))
    monkeypatch.setattr(procedural_room_stage, "generate_bathroom", lambda *args, **kwargs: (calls.append("bathroom") or [], {}))
    monkeypatch.setattr(procedural_room_stage, "generate_toilet", lambda *args, **kwargs: (calls.append("toilet") or [], {}))
    for room_type in ["bedroom", "living_room", "corridor", "bathroom", "toilet"]:
        procedural_room_stage._dispatch_generator(SimpleNamespace(room_type=room_type), "normal", 1)
    unsupported_items, unsupported_report = procedural_room_stage._dispatch_generator(SimpleNamespace(room_type="office"), "normal", 1)
    assert calls == ["bedroom", "living_room", "corridor", "bathroom", "toilet"]
    assert unsupported_items == []
    assert unsupported_report["status"] == "unsupported_room_type"

    scene_path = tmp_path / "empty.json"
    scene_path.write_text(json.dumps(room_scene("bedroom", 3, 3)), encoding="utf-8")
    monkeypatch.setattr(procedural_room_stage, "_dispatch_generator", lambda *args, **kwargs: ([], {"generator": "empty"}))
    empty_report = procedural_room_stage.apply_procedural_room_stage(
        scene_json_path=scene_path,
        out_dir=tmp_path,
        policy="always",
        tag="empty",
    )
    assert empty_report["warnings"] == ["No procedural objects were generated."]

    @dataclass(frozen=True)
    class ArtifactData:
        scene_path: Path
        placement_path: Path | None = None

    updated_data = procedural_room_stage._updated_artifacts_copy(ArtifactData(scene_path), scene_path="scene.new.json", placement_path="placement.new.json")
    assert updated_data.scene_path == Path("scene.new.json")
    assert updated_data.placement_path == Path("placement.new.json")
    obj_artifacts = SimpleNamespace(scene_json="old.json", placement_json="old-placement.json")
    updated_obj = procedural_room_stage._updated_artifacts_copy(obj_artifacts, scene_path="scene.obj.json", placement_path="placement.obj.json")
    assert updated_obj.scene_json == "scene.obj.json"
    assert updated_obj.placement_json == "placement.obj.json"
    assert procedural_room_stage._get_attr(SimpleNamespace(scene_v1="scene"), ["scene_v1"]) == "scene"

    skipped_artifacts = {"scene_path": str(scene_path), "placement_path": ""}
    returned, skipped_report = procedural_room_stage.apply_procedural_room_stage_to_artifacts(
        artifacts=skipped_artifacts,
        run_dir=tmp_path,
        policy="never",
    )
    assert returned is skipped_artifacts
    assert skipped_report["reason"] == "policy_never"


def test_procedural_room_stage_reports_artifacts_and_wrapper_paths(tmp_path: Path, monkeypatch) -> None:
    scene_path = tmp_path / "living.json"
    scene = room_scene("living_room", 4.5, 4.0, windows=True)
    scene["placements"] = [
        {"id": "old_bed", "category": "bed", "meta": {"procedural": True}},
        {"id": "wall", "category": "wall", "meta": {}},
    ]
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    never = procedural_room_stage.apply_procedural_room_stage(
        scene_json_path=scene_path,
        out_dir=tmp_path,
        policy="never",
        tag="never",
    )
    assert never["skipped"] is True
    assert never["reason"] == "policy_never"

    unsupported = procedural_room_stage.apply_procedural_room_stage(
        scene_json_path=scene_path,
        out_dir=tmp_path,
        policy="auto",
        enabled_room_types={"bedroom"},
        tag="unsupported",
    )
    assert unsupported["reason"] == "unsupported_room_type"

    def fake_dispatch(ctx, density, seed):
        return (
            [
                {
                    "id": "generated_sofa",
                    "category": "sofa",
                    "name": "Generated sofa",
                    "position_m": [1.5, 1.5, 0.4],
                    "size_m": [1.8, 0.8, 0.8],
                    "yaw_deg": 0.0,
                    "meta": {"procedural": True, "density_layer": "primary", "physical_role": "solid_floor"},
                    "source": {"placement_source": "procedural_room_stage"},
                }
            ],
            {
                "generator": "unit_generator",
                "rejected": [
                    {"category": "chair", "reason": "collision"},
                    {"category": "chair", "reason": "collision"},
                ],
            },
        )

    monkeypatch.setattr(procedural_room_stage, "_dispatch_generator", fake_dispatch)
    report = procedural_room_stage.apply_procedural_room_stage(
        scene_json_path=scene_path,
        out_dir=tmp_path,
        policy="always",
        replace_existing=True,
        tag="unit",
        seed=7,
    )

    assert report["skipped"] is False
    assert report["removed_existing_count"] == 1
    assert report["kept_existing_count"] == 1
    assert report["generator"]["rejected_count"] == 2
    assert Path(report["output_scene_json"]).is_file()
    assert Path(report["output_placement_json"]).is_file()
    assert Path(report["report_json"]).is_file()

    updated_dict, dict_report = procedural_room_stage.apply_procedural_room_stage_to_artifacts(
        artifacts={"scene_path": str(scene_path), "placement_path": ""},
        run_dir=tmp_path,
        policy="always",
        tag="dict",
    )
    assert dict_report["skipped"] is False
    assert updated_dict["scene_path"].endswith("scene_procedural_room.dict.v1.json")
    assert updated_dict["placement_path"].endswith("placement_procedural_room.dict.v1.json")

    no_scene, no_scene_report = procedural_room_stage.apply_procedural_room_stage_to_artifacts(
        artifacts=SimpleNamespace(other="x"),
        run_dir=tmp_path,
    )
    assert no_scene.other == "x"
    assert no_scene_report["reason"] == "artifact_has_no_scene_path"
