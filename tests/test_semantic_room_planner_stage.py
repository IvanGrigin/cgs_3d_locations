from __future__ import annotations

from argparse import Namespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.pipeline.semantic_room_planner_stage import (
    _llm_settings,
    _normalize_zones_payload,
    maybe_run_semantic_room_planner_stage,
    run_semantic_room_planner,
)  # noqa: E402


def test_normalize_zones_payload_defaults_and_uniq_ids() -> None:
    payload = {
        "zones": [
            {"type": "sleeping_zone", "id": "zone", "priority": 2},
            {"type": "sleeping_zone", "id": "zone", "priority": 1},
            {"type": "unknown_zone", "id": "bad"},
            {"type": "work_zone", "name": "Рабочая"},
        ]
    }
    zones = _normalize_zones_payload(payload)
    assert len(zones) == 3
    assert zones[0]["id"] == "zone"
    assert zones[1]["id"] == "zone_2"
    assert zones[2]["id"] == "zone_work"
    assert zones[2]["placement_preferences"]["against_wall"] is True


def test_normalize_zones_payload_default_when_empty() -> None:
    zones = _normalize_zones_payload({})
    assert {row["id"] for row in zones} == {"zone_sleeping", "zone_work", "zone_storage"}


def test_llm_settings_with_debug_and_custom_values(tmp_path: Path) -> None:
    args = Namespace(
        semantic_room_planner_provider="ollama",
        semantic_room_planner_model="m",
        semantic_room_planner_ollama_url="http://localhost:11435",
        semantic_room_planner_openrouter_model="openrouter-model",
        semantic_room_planner_timeout=90,
        semantic_room_planner_temperature=0.3,
        semantic_room_planner_max_attempts=5,
        semantic_room_planner_debug=True,
        semantic_room_planner_llm_catalog_queries=True,
        semantic_room_planner_llm_catalog_max_objects=12,
    )
    settings = _llm_settings(args, tmp_path)
    assert settings["provider"] == "ollama"
    assert settings["model"] == "m"
    assert settings["ollama_url"] == "http://localhost:11435"
    assert settings["debug_dir"] == str((tmp_path / "llm_debug").resolve())
    assert settings["use_llm_catalog_queries"] is True
    assert settings["llm_catalog_max_objects"] == 12


def test_run_semantic_room_planner_with_stubbed_steps(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.normalize_room_input", lambda room, prompt: {"prompt": prompt, "room": room})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.analyze_room_geometry", lambda room: {"assumptions": ["ok"], "room_polygon": room})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.extract_theme_spec", lambda prompt: {"palette": "warm"})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.run_room_intent_step", lambda state, settings: {"intent": "ok"})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.run_zones_step", lambda state, settings: {"zones": [{"id": "z1", "type": "sleeping_zone"}]})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.run_zone_items_step", lambda state, zone, settings: {"zone_items": zone["id"]})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.run_zone_relations_step", lambda state, zone, items, settings: {"zone_relations": zone["id"]})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.normalize_objects", lambda all_zone_items, zones: {"objects": [{"id": "obj1"}], "warnings": []})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.repair_semantic_objects", lambda objects, zones, theme_spec, max_total_objects=None: {"objects": objects, "warnings": ["w1"]})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.resolve_relations_by_subclass", lambda all_zone_relations, objects: [{"from": "a", "to": "b"}])
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.augment_relations_with_rules", lambda objects, relations: relations)
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.validate_relation_targets_exist", lambda edges, objects, zones: {"is_valid": True})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.build_relationship_graph", lambda objects, edges: {"nodes": objects, "edges": edges})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.generate_catalog_queries", lambda objects, settings: {"queries": []})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.generate_anchors", lambda objects: {"nodes": objects})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.solve_placements", lambda geometry, objects, graph, anchors, cfg: {"placements": [], "schema": "placements_generated/v1"})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.validate_geometry", lambda geometry, objects, graph, placements: {"schema": "geometry_validation/v1", "is_valid": True, "score": 1.0, "hard_errors": [], "soft_warnings": []})
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.repair_scene", lambda geometry, objects, graph, placements, geom_validation, max_repair_iterations: {
        "placements": placements,
        "objects": objects,
        "geometry_validation": geom_validation,
        "schema": "repair_report/v1",
        "status": "ok",
        "moved_items": [],
    })
    monkeypatch.setattr("src.pipeline.semantic_room_planner_stage.export_scene_plan", lambda state, root: {
        "plan": {"status": "success", "warnings": ["ok"]},
    })

    room = {"room": {"id": "r1"}}
    info = run_semantic_room_planner(
        input_json=room,
        prompt="cozy living room",
        out_dir=tmp_path / "out",
        llm_settings={
            "provider": "none",
            "max_attempts": 1,
        },
        apply_placement=True,
        max_repair_iterations=1,
        skip_catalog_queries=True,
    )
    assert info["status"] == "success"
    assert info["warnings"] == ["ok"]
    assert "scene_v1" in info and "placement_v1" in info
 


def test_maybe_run_semantic_room_planner_stage_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    room_path = tmp_path / "room.json"
    room_path.write_text("{\"room\": {\"id\": \"r1\"}}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    args = Namespace(
        semantic_room_planner=True,
        semantic_room_planner_provider="none",
        semantic_room_planner_model=None,
        semantic_room_planner_ollama_url="http://127.0.0.1:11434",
        semantic_room_planner_openrouter_model=None,
        semantic_room_planner_timeout=10,
        semantic_room_planner_temperature=0.2,
        semantic_room_planner_max_attempts=1,
        semantic_room_planner_debug=False,
        semantic_room_planner_apply_placement=False,
        semantic_room_planner_max_repair_iterations=1,
        semantic_room_planner_skip_catalog_queries=True,
        semantic_room_planner_llm_catalog_queries=False,
        semantic_room_planner_llm_catalog_max_objects=5,
        semantic_room_planner_no_fail=False,
        semantic_room_planner_out_dir=None,
    )

    def fake_run(*, input_json, prompt, out_dir, llm_settings, apply_placement, max_repair_iterations, skip_catalog_queries):
        return {
            "enabled": True,
            "status": "success",
            "warnings": [],
            "hard_errors": [],
            "out_dir": str(out_dir),
            "final_room_scene_plan": str((Path(out_dir) / "final_room_scene_plan.json").resolve()),
            "scene_v1": str((Path(out_dir) / "scene.semantic.v1.json").resolve()),
            "placement_v1": str((Path(out_dir) / "placement.semantic.v1.json").resolve()),
        }

    import src.pipeline.semantic_room_planner_stage as s
    monkeypatch.setattr(s, "run_semantic_room_planner", fake_run)
    result = maybe_run_semantic_room_planner_stage(
        args=args,
        room_path=room_path,
        prompt_text="warm style",
        run_dir=tmp_path,
        manifest_path=manifest,
    )

    assert result is not None
    assert manifest.read_text(encoding="utf-8").strip().startswith("{")
    manifest_data = manifest.read_text(encoding="utf-8")
    assert "semantic_room_planner" in manifest_data
