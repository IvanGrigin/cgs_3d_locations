from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.style_profiles import build_style_profile_from_compiled_policy

from .inventory_mapping import (
    PRIMARY_SEMANTICS,
    SECONDARY_SEMANTICS,
    expand_semantics_to_factories,
)
from .policies import (
    ScenePolicies,
    get_acceptance_profile,
    get_room_program,
    get_solver_profile,
    get_style_policy,
    resolve_area_bucket,
)
from .schemas import (
    AcceptancePolicy,
    AreaBucket,
    CompiledGeometry,
    CompiledInfinigenPolicy,
    CompiledPolicy,
    CompiledProgram,
    CompiledStylePolicy,
    PromptIntent,
)


BEDROOM_CORE_PRIMARY = ["Bed", "Lighting", "Storage", "SideTable", "CeilingLight"]
BEDROOM_CORE_REQUIRED = ["Bed", "Lighting"]
BEDROOM_CORE_PREFERRED = ["Storage", "SideTable", "CeilingLight"]
BEDROOM_CORE_FACTORY_WHITELIST = [
    "BedFactory",
    "CeilingLightFactory",
    "LampFactory",
    "SideTableFactory",
    "SingleCabinetFactory",
]
BEDROOM_CORE_BLACKLIST = [
    "LargePlantContainerFactory",
    "LargeShelfFactory",
    "SimpleBookcaseFactory",
    "CellShelfFactory",
    "BookStackFactory",
    "BookColumnFactory",
    "NatureShelfTrinketsFactory",
]


def _canonical_dimensions(area_sqm: float, room_type: str, room_program: dict[str, Any]) -> tuple[float, float]:
    dims = room_program.get("canonical_dimensions_m")
    if isinstance(dims, list) and len(dims) == 2:
        return float(dims[0]), float(dims[1])
    aspect = 1.18 if room_type == "Bedroom" else 1.25  # pragma: no cover
    width = math.sqrt(area_sqm / aspect)  # pragma: no cover
    depth = area_sqm / width  # pragma: no cover
    return round(width, 3), round(depth, 3)  # pragma: no cover


def _build_floor_polygon(width_m: float, depth_m: float) -> list[dict[str, float]]:
    return [
        {"x": 0.0, "y": 0.0},
        {"x": width_m, "y": 0.0},
        {"x": width_m, "y": depth_m},
        {"x": 0.0, "y": depth_m},
    ]


def _build_walls() -> list[dict[str, int | str]]:
    return [
        {"id": "w0", "from_vertex": 0, "to_vertex": 1},
        {"id": "w1", "from_vertex": 1, "to_vertex": 2},
        {"id": "w2", "from_vertex": 2, "to_vertex": 3},
        {"id": "w3", "from_vertex": 3, "to_vertex": 0},
    ]


def _build_openings(width_m: float, depth_m: float, wants_door: bool, wants_window: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doors: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    if wants_door:
        door_width = min(0.9, max(0.75, width_m * 0.28))
        doors.append({"wall_id": "w0", "s": round((width_m - door_width) * 0.15, 3), "width": round(door_width, 3)})
    if wants_window:
        window_width = min(1.4, max(0.9, width_m * 0.45))
        windows.append({"wall_id": "w2", "s": round((width_m - window_width) * 0.5, 3), "width": round(window_width, 3)})
    return doors, windows


def _unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _merge_semantics(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        out.extend(group)
    return _unique(out)


def _allowed_semantics(required: list[str], optional: list[str], preferred: list[str], forbidden: list[str]) -> tuple[list[str], list[str]]:
    allowed = [item for item in _merge_semantics(required, optional, preferred) if item not in forbidden]
    primary = [item for item in allowed if item in PRIMARY_SEMANTICS]
    secondary = [item for item in allowed if item in SECONDARY_SEMANTICS]
    return _unique(primary), _unique(secondary)


def _normalize_factory_lists(whitelist: list[str], blacklist: list[str]) -> tuple[list[str], list[str]]:
    normalized_blacklist = _unique([item for item in blacklist if item])
    blacklist_set = set(normalized_blacklist)
    normalized_whitelist = [item for item in _unique(whitelist) if item not in blacklist_set]
    return normalized_whitelist, normalized_blacklist


def _required_factory_coverage(required_semantics: list[str], whitelist: list[str], blacklist: list[str]) -> dict[str, list[str]]:
    whitelist_set = set(whitelist)
    blacklist_set = set(blacklist)
    coverage: dict[str, list[str]] = {}
    for semantic in required_semantics:
        factories = [name for name in expand_semantics_to_factories([semantic]) if name not in blacklist_set]
        if whitelist_set:
            factories = [name for name in factories if name in whitelist_set]
        coverage[semantic] = factories
    return coverage


def _build_preflight(required_semantics: list[str], whitelist: list[str], blacklist: list[str]) -> dict[str, Any]:
    coverage = _required_factory_coverage(required_semantics, whitelist, blacklist)
    empty_required = sorted([semantic for semantic, families in coverage.items() if not families])
    return {
        "required_semantic_factory_coverage": coverage,
        "empty_required_semantics": empty_required,
        "effective_factory_whitelist_count": len(whitelist),
        "effective_factory_blacklist_count": len(blacklist),
        "apply_child_restrictions": False,
    }


def _apply_bedroom_core_screening(
    *,
    required_semantics: list[str],
    optional_semantics: list[str],
    preferred_semantics: list[str],
    max_counts: dict[str, int],
    factory_blacklist: list[str],
    stage_flags: dict[str, bool],
    solver_steps: dict[str, int],
) -> tuple[list[str], list[str], list[str], dict[str, int], list[str], dict[str, bool], dict[str, int]]:
    required = list(BEDROOM_CORE_REQUIRED)
    optional = [item for item in _unique(optional_semantics + ["Storage", "SideTable", "CeilingLight"]) if item in {"Storage", "SideTable", "CeilingLight"}]
    preferred = _unique(BEDROOM_CORE_PREFERRED + preferred_semantics)
    tightened_counts = dict(max_counts)
    tightened_counts["Storage"] = min(int(tightened_counts.get("Storage", 1)), 1)
    tightened_counts["Decor"] = 0
    tightened_counts["LargePlant"] = 0
    tightened_counts["FloorLamp"] = 0
    tightened_blacklist = _unique(factory_blacklist + BEDROOM_CORE_BLACKLIST)
    tightened_stage_flags = dict(stage_flags)
    tightened_stage_flags["solve_medium_enabled"] = True
    tightened_stage_flags["solve_small_enabled"] = False
    tightened_solver_steps = dict(solver_steps)
    tightened_solver_steps["solve_steps_large"] = 150
    tightened_solver_steps["solve_steps_medium"] = 30
    tightened_solver_steps["solve_steps_small"] = 0
    return required, optional, preferred, tightened_counts, tightened_blacklist, tightened_stage_flags, tightened_solver_steps


def _apply_bedroom_core_factory_whitelist(factory_whitelist: list[str]) -> list[str]:
    if not factory_whitelist:
        return list(BEDROOM_CORE_FACTORY_WHITELIST)  # pragma: no cover
    whitelist = [item for item in factory_whitelist if item in BEDROOM_CORE_FACTORY_WHITELIST]
    return whitelist or list(BEDROOM_CORE_FACTORY_WHITELIST)


def _solver_profile_key(room_type: str, area_bucket: str, style_policy: dict[str, Any]) -> str:
    density = str(style_policy.get("density") or "low")
    density_suffix = "medium_density" if density == "medium" else "low_density"
    return f"{room_type.lower()}_{area_bucket}_{density_suffix}"


def _stage_overrides(stage_flags: dict[str, bool], solver_profile: dict[str, Any]) -> list[str]:
    overrides: list[str] = []
    for key, value in stage_flags.items():
        overrides.append(f"compose_indoors.{key}={'True' if value else 'False'}")
    for key in ("solve_steps_large", "solve_steps_medium", "solve_steps_small"):
        if key in solver_profile:
            overrides.append(f"compose_indoors.{key}={int(solver_profile[key])}")
    return overrides


def _solver_steps_from_profile(solver_profile: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in ("solve_steps_large", "solve_steps_medium", "solve_steps_small"):
        if key in solver_profile:
            out[key] = int(solver_profile[key])
    return out


def compile_prompt_intent(intent: PromptIntent, policies: ScenePolicies) -> CompiledPolicy:
    area_sqm = float(intent.geometry.target_area_sqm or 0.0)
    if area_sqm <= 0.0 and intent.geometry.width_m and intent.geometry.depth_m:
        area_sqm = round(float(intent.geometry.width_m) * float(intent.geometry.depth_m), 3)  # pragma: no cover
    if area_sqm <= 0.0:
        area_sqm = 8.0
    area_bucket = resolve_area_bucket(area_sqm, policies)
    room_program = get_room_program(intent.room_type.value, area_bucket, policies)
    style_policy = get_style_policy((intent.style.style_label or "").value if intent.style.style_label else "japandi", policies)
    solver_profile_key = _solver_profile_key(intent.room_type.value, area_bucket, style_policy)
    solver_profile = get_solver_profile(solver_profile_key, policies)
    acceptance_profile_name = str(style_policy.get("acceptance_profile") or "bedroom_balanced")
    acceptance_profile = get_acceptance_profile(acceptance_profile_name, policies)

    width_m = float(intent.geometry.width_m or 0.0)
    depth_m = float(intent.geometry.depth_m or 0.0)
    if width_m <= 0.0 or depth_m <= 0.0:
        width_m, depth_m = _canonical_dimensions(area_sqm, intent.room_type.value, room_program)
    floor_polygon = _build_floor_polygon(width_m, depth_m)
    doors, windows = _build_openings(width_m, depth_m, intent.openings.wants_door, intent.openings.wants_window)

    required_semantics = _unique(room_program.get("required_semantics", []) + intent.objects.required)
    optional_semantics = _unique(room_program.get("optional_semantics", []) + intent.objects.desired)
    forbidden_semantics = _unique(
        room_program.get("forbidden_semantics", [])
        + intent.objects.forbidden
        + style_policy.get("forbidden_semantics", [])
    )
    preferred_semantics = _unique(room_program.get("preferred_semantics", []) + style_policy.get("preferred_semantics", []))
    stage_flags = dict(solver_profile.get("stage_flags", {}))
    solver_steps = _solver_steps_from_profile(solver_profile)
    max_counts = dict(room_program.get("max_counts", {}))

    raw_factory_whitelist = _unique(
        list(style_policy.get("factory_whitelist", []))
        + expand_semantics_to_factories(required_semantics)
    )
    raw_factory_blacklist = _unique(
        list(style_policy.get("factory_blacklist", []))
        + expand_semantics_to_factories([semantic for semantic in forbidden_semantics if semantic in SECONDARY_SEMANTICS or semantic in PRIMARY_SEMANTICS])
    )
    if intent.room_type.value == "Bedroom":
        required_semantics, optional_semantics, preferred_semantics, max_counts, raw_factory_blacklist, stage_flags, solver_steps = _apply_bedroom_core_screening(
            required_semantics=required_semantics,
            optional_semantics=optional_semantics,
            preferred_semantics=preferred_semantics,
            max_counts=max_counts,
            factory_blacklist=raw_factory_blacklist,
            stage_flags=stage_flags,
            solver_steps=solver_steps,
        )
        raw_factory_whitelist = _apply_bedroom_core_factory_whitelist(raw_factory_whitelist)
    allowed_primary, allowed_secondary = _allowed_semantics(required_semantics, optional_semantics, preferred_semantics, forbidden_semantics)
    if intent.room_type.value == "Bedroom":
        allowed_primary = [item for item in BEDROOM_CORE_PRIMARY if item in _unique(required_semantics + optional_semantics + preferred_semantics)]
        allowed_secondary = []
    factory_whitelist, factory_blacklist = _normalize_factory_lists(raw_factory_whitelist, raw_factory_blacklist)
    preflight = _build_preflight(required_semantics, factory_whitelist, factory_blacklist)
    if allowed_primary:
        preflight["apply_child_restrictions"] = True
        preflight["final_restrict_child_primary"] = list(allowed_primary)
        preflight["final_restrict_child_secondary"] = list(allowed_secondary)
    preflight["max_counts"] = dict(max_counts)
    preflight["stage_flags"] = dict(stage_flags)
    preflight["solver_steps"] = dict(solver_steps)
    if preflight["empty_required_semantics"]:
        missing = ", ".join(preflight["empty_required_semantics"])
        raise RuntimeError(f"empty_candidate_pool_before_solve: no allowed factories for required semantics: {missing}")

    compiled = CompiledPolicy(
        scene_id="prompt_scene",
        prompt_text=intent.prompt_text,
        intent=intent,
        geometry=CompiledGeometry(
            room_type=intent.room_type,
            area_sqm=round(area_sqm, 3),
            width_m=round(width_m, 3),
            depth_m=round(depth_m, 3),
            height_m=float(intent.geometry.height_m or 2.7),
            area_bucket=AreaBucket(area_bucket),
            floor_polygon=floor_polygon,
            walls=_build_walls(),
            doors=doors,
            windows=windows,
        ),
        program=CompiledProgram(
            required_semantics=required_semantics,
            optional_semantics=optional_semantics,
            forbidden_semantics=forbidden_semantics,
            preferred_semantics=preferred_semantics,
            required_primary=[item for item in required_semantics if item in PRIMARY_SEMANTICS],
            allowed_primary=allowed_primary,
            allowed_secondary=allowed_secondary,
            factory_whitelist=factory_whitelist,
            factory_blacklist=factory_blacklist,
            max_counts=max_counts,
            notes=str(room_program.get("notes") or ""),
        ),
        style_policy=CompiledStylePolicy(
            style_label=(intent.style.style_label or "").value if intent.style.style_label else "japandi",
            style_strength=float(style_policy.get("style_strength", 1.0)),
            density=style_policy.get("density", "low"),
            decor_richness=style_policy.get("decor_richness", "low"),
            palette_hint=_unique(intent.style.palette_hint + style_policy.get("palette_hint", []) + intent.preferences.favorite_colors),
            material_family=_unique(intent.style.material_family + style_policy.get("material_family", [])),
            required_semantics=required_semantics,
            forbidden_semantics=forbidden_semantics,
            factory_whitelist=factory_whitelist,
            factory_blacklist=factory_blacklist,
            preferred_colors=_unique(intent.preferences.favorite_colors + style_policy.get("palette_hint", [])),
            avoid_colors=intent.preferences.avoid_colors,
            notes=str(intent.preferences.notes or room_program.get("notes") or ""),
        ),
        infinigen_policy=CompiledInfinigenPolicy(
            solver_profile_key=solver_profile_key,
            gin_overrides=[],
            monkeypatch_params=dict(solver_profile.get("monkeypatch_params", {})),
            stage_flags=stage_flags,
            solver_steps=solver_steps,
            screening_seeds=[int(x) for x in solver_profile.get("screening_seeds", [0, 1, 2, 3])],
            final_seeds=[int(x) for x in solver_profile.get("final_seeds", [11, 12])],
            required_semantics=required_semantics,
            forbidden_semantics=forbidden_semantics,
            factory_whitelist=factory_whitelist,
            factory_blacklist=factory_blacklist,
            max_counts=max_counts,
        ),
        acceptance_policy=AcceptancePolicy(
            profile_key=acceptance_profile_name,
            required_semantics=required_semantics,
            forbidden_semantics=forbidden_semantics,
            factory_whitelist=factory_whitelist,
            factory_blacklist=factory_blacklist,
            max_counts=max_counts,
            min_real_objects=int(acceptance_profile.get("min_real_objects", 0)),
            max_repeated_factory_count=int(acceptance_profile.get("max_repeated_factory_count", 999)),
            reject_on_solver_violation=bool(acceptance_profile.get("reject_on_solver_violation", False)),
            allowed_solver_violations=list(acceptance_profile.get("allowed_solver_violations", [])),
            min_rule_score=float(acceptance_profile.get("min_rule_score", 0.0)),
            min_judge_score=float(acceptance_profile.get("min_judge_score", 0.0)),
        ),
        preflight=preflight,
    )
    compiled.infinigen_policy.gin_overrides = build_gin_overrides(compiled)
    return compiled


def build_room_json(compiled_policy: CompiledPolicy) -> dict[str, Any]:
    return {
        "room": {
            "name": compiled_policy.geometry.room_type.value,
            "room_type": compiled_policy.geometry.room_type.value,
            "floor_polygon": compiled_policy.geometry.floor_polygon,
            "walls": [wall.model_dump(mode="json") for wall in compiled_policy.geometry.walls],
            "doors": [door.model_dump(mode="json") for door in compiled_policy.geometry.doors],
            "windows": [window.model_dump(mode="json") for window in compiled_policy.geometry.windows],
            "style_hint": compiled_policy.style_policy.style_label,
        }
    }


def build_style_profile(compiled_policy: CompiledPolicy) -> dict[str, Any]:
    return build_style_profile_from_compiled_policy(compiled_policy)


def build_gin_overrides(compiled_policy: CompiledPolicy) -> list[str]:
    overrides = _stage_overrides(
        compiled_policy.infinigen_policy.stage_flags,
        compiled_policy.infinigen_policy.solver_steps,
    )
    overrides.extend(_build_restrict_overrides(compiled_policy))
    return _unique(overrides)


def _build_restrict_overrides(compiled_policy: CompiledPolicy) -> list[str]:
    overrides = [
        f"restrict_solving.restrict_parent_rooms=['{compiled_policy.geometry.room_type.value}']",
    ]
    if compiled_policy.preflight.get("apply_child_restrictions"):
        primary = compiled_policy.preflight.get("final_restrict_child_primary") or compiled_policy.program.allowed_primary
        secondary = compiled_policy.preflight.get("final_restrict_child_secondary") or compiled_policy.program.allowed_secondary
        quoted_primary = ", ".join(f"'{item}'" for item in primary)
        overrides.append(f"restrict_solving.restrict_child_primary=[{quoted_primary}]")
        if secondary:
            quoted_secondary = ", ".join(f"'{item}'" for item in secondary)  # pragma: no cover
            overrides.append(f"restrict_solving.restrict_child_secondary=[{quoted_secondary}]")  # pragma: no cover
    overrides.append("restrict_solving.solve_max_rooms=1")
    return overrides


def write_compiled_artifacts(compiled_policy: CompiledPolicy, out_dir: str | Path) -> None:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    room_json = build_room_json(compiled_policy)
    style_profile = build_style_profile(compiled_policy)
    gin_overrides = build_gin_overrides(compiled_policy)
    compiled_policy.artifacts.update(
        {
            "room_json": str((out_path / "room.json").resolve()),
            "style_profile": str((out_path / "style_profile.json").resolve()),
            "gin_overrides": str((out_path / "gin_overrides.json").resolve()),
            "compiled_policy": str((out_path / "compiled_policy.original.json").resolve()),
        }
    )
    (out_path / "compiled_policy.original.json").write_text(compiled_policy.model_dump_json_pretty(), encoding="utf-8")
    (out_path / "compiled_policy.active.json").write_text(compiled_policy.model_dump_json_pretty(), encoding="utf-8")
    (out_path / "room.json").write_text(json.dumps(room_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "style_profile.json").write_text(json.dumps(style_profile, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "gin_overrides.json").write_text(json.dumps(gin_overrides, ensure_ascii=False, indent=2), encoding="utf-8")
