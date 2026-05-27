from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .semantic_room_planner.schemas import read_json, write_json
    from .semantic_room_planner.geometry_analyzer import normalize_room_input, analyze_room_geometry
    from .semantic_room_planner.llm_steps import run_room_intent_step, run_zones_step, run_zone_items_step, run_zone_relations_step
    from .semantic_room_planner.semantic_sanitizer import extract_theme_spec, repair_semantic_objects
    from .semantic_room_planner.normalizer import normalize_objects
    from .semantic_room_planner.relation_rules import resolve_relations_by_subclass, augment_relations_with_rules, validate_relation_targets_exist, build_relationship_graph
    from .semantic_room_planner.anchors import generate_anchors
    from .semantic_room_planner.placement_solver import solve_placements
    from .semantic_room_planner.validation import validate_geometry
    from .semantic_room_planner.repair import repair_scene
    from .semantic_room_planner.catalog_queries import generate_catalog_queries
    from .semantic_room_planner.scene_export import export_scene_plan
    from .semantic_room_planner.zone_templates import ZONE_TYPES
except ImportError:  # pragma: no cover
    from pipeline.semantic_room_planner.schemas import read_json, write_json  # pragma: no cover
    from pipeline.semantic_room_planner.geometry_analyzer import normalize_room_input, analyze_room_geometry  # pragma: no cover
    from pipeline.semantic_room_planner.llm_steps import run_room_intent_step, run_zones_step, run_zone_items_step, run_zone_relations_step  # pragma: no cover
    from pipeline.semantic_room_planner.semantic_sanitizer import extract_theme_spec, repair_semantic_objects  # pragma: no cover
    from pipeline.semantic_room_planner.normalizer import normalize_objects  # pragma: no cover
    from pipeline.semantic_room_planner.relation_rules import resolve_relations_by_subclass, augment_relations_with_rules, validate_relation_targets_exist, build_relationship_graph  # pragma: no cover
    from pipeline.semantic_room_planner.anchors import generate_anchors  # pragma: no cover
    from pipeline.semantic_room_planner.placement_solver import solve_placements  # pragma: no cover
    from pipeline.semantic_room_planner.validation import validate_geometry  # pragma: no cover
    from pipeline.semantic_room_planner.repair import repair_scene  # pragma: no cover
    from pipeline.semantic_room_planner.catalog_queries import generate_catalog_queries  # pragma: no cover
    from pipeline.semantic_room_planner.scene_export import export_scene_plan  # pragma: no cover
    from pipeline.semantic_room_planner.zone_templates import ZONE_TYPES  # pragma: no cover


def add_semantic_room_planner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--semantic-room-planner", action="store_true")
    parser.add_argument("--semantic-room-planner-provider", choices=["none", "ollama", "openrouter"], default="ollama")
    parser.add_argument("--semantic-room-planner-ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--semantic-room-planner-model", default=None)
    parser.add_argument("--semantic-room-planner-openrouter-model", default=None)
    parser.add_argument("--semantic-room-planner-timeout", type=int, default=180)
    parser.add_argument("--semantic-room-planner-temperature", type=float, default=0.1)
    parser.add_argument("--semantic-room-planner-max-attempts", type=int, default=3)
    parser.add_argument("--semantic-room-planner-debug", action="store_true")
    parser.add_argument("--semantic-room-planner-apply-placement", action="store_true")
    parser.add_argument("--semantic-room-planner-max-repair-iterations", type=int, default=3)
    parser.add_argument("--semantic-room-planner-skip-catalog-queries", action="store_true")
    parser.add_argument("--semantic-room-planner-llm-catalog-queries", action="store_true")
    parser.add_argument("--semantic-room-planner-llm-catalog-max-objects", type=int, default=8)
    parser.add_argument("--semantic-room-planner-no-fail", action="store_true")
    parser.add_argument("--semantic-room-planner-out-dir", default=None)


def _llm_settings(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    return {
        "provider": getattr(args, "semantic_room_planner_provider", "none"),
        "model": getattr(args, "semantic_room_planner_model", None),
        "ollama_url": getattr(args, "semantic_room_planner_ollama_url", "http://127.0.0.1:11434"),
        "openrouter_model": getattr(args, "semantic_room_planner_openrouter_model", None),
        "timeout": int(getattr(args, "semantic_room_planner_timeout", 180)),
        "temperature": float(getattr(args, "semantic_room_planner_temperature", 0.1)),
        "max_attempts": int(getattr(args, "semantic_room_planner_max_attempts", 3)),
        "debug_dir": str(out_dir / "llm_debug") if getattr(args, "semantic_room_planner_debug", False) else None,
        "use_llm_catalog_queries": bool(getattr(args, "semantic_room_planner_llm_catalog_queries", False)),
        "llm_catalog_max_objects": int(getattr(args, "semantic_room_planner_llm_catalog_max_objects", 8)),
    }


def _normalize_zones_payload(zones_payload: dict[str, Any]) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    used: set[str] = set()
    for idx, raw in enumerate(list(zones_payload.get("zones") or []), start=1):
        if not isinstance(raw, dict):
            continue  # pragma: no cover
        ztype = str(raw.get("type") or raw.get("zone_type") or "").strip()
        if ztype not in ZONE_TYPES:
            continue
        base_id = str(raw.get("id") or f"zone_{ztype.replace('_zone', '')}").strip()
        zid = base_id
        suffix = 2
        while zid in used:
            zid = f"{base_id}_{suffix}"
            suffix += 1
        used.add(zid)
        prefs = raw.get("placement_preferences")
        if not isinstance(prefs, dict):
            prefs = {"notes": str(prefs or ""), "against_wall": True, "corner_allowed": True, "avoid_door": True}
        zones.append({
            "id": zid,
            "type": ztype,
            "name_ru": str(raw.get("name_ru") or raw.get("name") or ztype),
            "name_en": str(raw.get("name_en") or raw.get("name") or ztype.replace("_", " ")),
            "priority": int(raw.get("priority") or idx),
            "purpose": str(raw.get("purpose") or ""),
            "desired_area_share": float(raw.get("desired_area_share") or 0.2),
            "placement_preferences": prefs,
        })
    if not zones:
        zones = [
            {"id": "zone_sleeping", "type": "sleeping_zone", "name_ru": "Спальная зона", "name_en": "Sleeping zone", "priority": 1, "purpose": "sleeping", "desired_area_share": 0.45, "placement_preferences": {"against_wall": True, "corner_allowed": True, "avoid_door": True}},
            {"id": "zone_work", "type": "work_zone", "name_ru": "Рабочая зона", "name_en": "Work zone", "priority": 2, "purpose": "working", "desired_area_share": 0.25, "placement_preferences": {"against_wall": True, "corner_allowed": True, "avoid_door": True}},
            {"id": "zone_storage", "type": "storage_zone", "name_ru": "Хранение", "name_en": "Storage zone", "priority": 3, "purpose": "storage", "desired_area_share": 0.2, "placement_preferences": {"against_wall": True, "corner_allowed": True, "avoid_door": True}},
        ]
    return zones


def run_semantic_room_planner(
    *,
    input_json: dict[str, Any],
    prompt: str | None,
    out_dir: str | Path,
    llm_settings: dict[str, Any],
    apply_placement: bool = True,
    max_repair_iterations: int = 3,
    skip_catalog_queries: bool = False,
) -> dict[str, Any]:
    def log(stage: str) -> None:
        print(f"[semantic] {stage}", flush=True)

    llm_settings = dict(llm_settings or {})
    llm_settings["provider"] = str(llm_settings.get("provider") or "none").strip().lower()
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_norm = normalize_room_input(input_json, prompt)
    write_json(root / "00_input.json", input_norm)

    log("01 geometry")
    geometry = analyze_room_geometry(input_norm)
    write_json(root / "01_geometry_analysis.json", geometry)
    theme_spec = extract_theme_spec(input_norm.get("prompt") or "")
    write_json(root / "01_theme_spec.json", theme_spec)
    state: dict[str, Any] = {"input": input_norm, "prompt": input_norm.get("prompt"), "room_geometry": geometry, "assumptions": geometry.get("assumptions", []), "theme_spec": theme_spec}

    log("02 intent")
    intent = run_room_intent_step(state, llm_settings)
    state["room_intent"] = intent
    write_json(root / "02_room_intent.json", intent)

    log("03 zones")
    zones_payload = run_zones_step(state, llm_settings)
    zones = _normalize_zones_payload(zones_payload)
    zones_payload["zones"] = zones
    state["zones"] = zones
    write_json(root / "03_zones.json", zones_payload)

    log("04 zone items")
    zone_items_dir = root / "04_zone_items"
    zone_rels_dir = root / "05_zone_relations"
    all_zone_items: list[dict[str, Any]] = []
    all_zone_relations: list[dict[str, Any]] = []
    for zone in zones:
        items = run_zone_items_step(state, zone, llm_settings)
        items["zone_id"] = zone["id"]
        all_zone_items.append(items)
        write_json(zone_items_dir / f"{zone.get('id', 'zone')}.json", items)

    log("05 relations")
    for zone, items in zip(zones, all_zone_items):
        rels = run_zone_relations_step(state, zone, items, llm_settings)
        rels["zone_id"] = zone["id"]
        all_zone_relations.append(rels)
        write_json(zone_rels_dir / f"{zone.get('id', 'zone')}.json", rels)

    log("06 normalize")
    normalized = normalize_objects(all_zone_items, zones)
    semantic_repair = repair_semantic_objects(normalized["objects"], zones, theme_spec, max_total_objects=32)
    objects = semantic_repair["objects"]
    normalized["objects"] = objects
    normalized["semantic_repair_warnings"] = semantic_repair.get("warnings", [])
    write_json(root / "06_objects_normalized.json", normalized)

    llm_edges = resolve_relations_by_subclass(all_zone_relations, objects)
    edges = augment_relations_with_rules(objects, llm_edges)
    relationship_validation = validate_relation_targets_exist(edges, objects, zones)
    graph = build_relationship_graph(objects, edges)
    write_json(root / "07_relationship_graph.json", graph)

    if not relationship_validation.get("is_valid"):
        log("07 placement skipped relationship invalid")  # pragma: no cover
        placements = {"schema": "placements_generated/v1", "placements": [], "solver_limits": {"strategy": "skipped_failed_before_placement"}}  # pragma: no cover
        geom_validation = {  # pragma: no cover
            "schema": "geometry_validation/v1",
            "is_valid": False,
            "score": 0.0,
            "hard_errors": ["failed_before_placement: relationship graph invalid"],
            "soft_warnings": relationship_validation.get("errors", []),
            "relation_scores": {"support": 0.0, "orientation": 0.0, "proximity": 0.0, "wall": 0.0},
        }
        write_json(root / "09_placements.json", placements)  # pragma: no cover
        write_json(root / "10_geometry_validation.json", geom_validation)  # pragma: no cover
        repair = {"schema": "repair_report/v1", "iterations": [], "removed_objects": [], "modified_objects": [], "final_status": "failed_before_placement"}  # pragma: no cover
        write_json(root / "11_repair_report.json", repair)  # pragma: no cover
        skip_placement_tail = True  # pragma: no cover
    else:
        skip_placement_tail = False

    anchors = generate_anchors(objects)
    write_json(root / "08_anchors.json", anchors)

    if not skip_placement_tail:
        log("07 placement")
        placements = solve_placements(geometry, objects, graph, anchors, {"apply_placement": apply_placement, "max_candidates_per_object": 16, "max_total_candidate_combinations": 512})
        write_json(root / "09_placements.json", placements)

        log("08 validation")
        geom_validation = validate_geometry(geometry, objects, graph, placements)
        write_json(root / "10_geometry_validation.json", geom_validation)

        log("09 repair")
        repair = repair_scene(geometry, objects, graph, placements, geom_validation, max_repair_iterations)
        write_json(root / "11_repair_report.json", repair)
        if repair.get("placements"):
            placements = repair["placements"]
            objects = repair.get("objects") or objects
            geom_validation = repair.get("geometry_validation") or geom_validation

    if skip_catalog_queries:
        log("12 catalog queries skipped")
        catalog = {"schema": "catalog_queries/v1", "items": []}
    else:
        if llm_settings.get("provider") != "none" and llm_settings.get("use_llm_catalog_queries"):  # pragma: no cover
            log("12 catalog queries llm batch")  # pragma: no cover
        else:
            log("12 catalog queries fallback")  # pragma: no cover
        catalog = generate_catalog_queries(objects, llm_settings if llm_settings.get("provider") != "none" else {"provider": "none"})  # pragma: no cover
    write_json(root / "12_catalog_queries.json", catalog)

    log("10 export")
    final_state = {
        **state,
        "zones": zones,
        "objects": objects,
        "relationship_graph": graph,
        "relationship_validation": relationship_validation,
        "anchors": anchors,
        "placements": placements,
        "geometry_validation": geom_validation,
        "repair_report": repair,
        "catalog_queries": catalog,
        "warnings": list(geometry.get("assumptions") or []) + list(semantic_repair.get("warnings") or []),
        "status": "failed_before_placement" if skip_placement_tail else None,
    }
    exported = export_scene_plan(final_state, root)
    plan = exported["plan"]
    status = plan.get("status") or "partial_success"
    return {
        "enabled": True,
        "status": status,
        "out_dir": str(root),
        "final_room_scene_plan": str((root / "final_room_scene_plan.json").resolve()),
        "scene_v1": str((root / "scene.semantic.v1.json").resolve()),
        "placement_v1": str((root / "placement.semantic.v1.json").resolve()),
        "layout_targets_json": str((root / "layout_targets.semantic.json").resolve()),
        "geometry_validation": str((root / "10_geometry_validation.json").resolve()),
        "relationship_graph": str((root / "07_relationship_graph.json").resolve()),
        "warnings": plan.get("warnings", []),
        "hard_errors": geom_validation.get("hard_errors", []),
        "validation_score": geom_validation.get("score"),
    }


def maybe_run_semantic_room_planner_stage(
    *,
    args: argparse.Namespace,
    room_path: str | Path,
    prompt_text: str,
    run_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "semantic_room_planner", False)):
        return None  # pragma: no cover
    out_dir = Path(getattr(args, "semantic_room_planner_out_dir", None) or Path(run_dir) / "semantic_room_planner").expanduser().resolve()
    try:
        info = run_semantic_room_planner(
            input_json=read_json(room_path),
            prompt=prompt_text,
            out_dir=out_dir,
            llm_settings=_llm_settings(args, out_dir),
            apply_placement=bool(getattr(args, "semantic_room_planner_apply_placement", False)),
            max_repair_iterations=int(getattr(args, "semantic_room_planner_max_repair_iterations", 3)),
            skip_catalog_queries=bool(getattr(args, "semantic_room_planner_skip_catalog_queries", False)),
        )
    except Exception as exc:  # pragma: no cover
        if not bool(getattr(args, "semantic_room_planner_no_fail", False)):  # pragma: no cover
            raise  # pragma: no cover
        info = {"enabled": True, "status": "failed", "out_dir": str(out_dir), "warnings": [str(exc)]}  # pragma: no cover
    if manifest_path:
        manifest_p = Path(manifest_path)
        manifest = read_json(manifest_p) if manifest_p.is_file() else {}
        manifest["semantic_room_planner"] = info
        write_json(manifest_p, manifest)
    return info
