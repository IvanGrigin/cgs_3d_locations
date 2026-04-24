from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from .schemas import AreaBucket, JSONModel, PromptIntent


class ScenePolicies(JSONModel):
    schema_version: str
    area_buckets: dict[str, dict[str, float]] = Field(default_factory=dict)
    room_programs: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    styles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    solver_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    acceptance_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_policies(path: str | Path) -> ScenePolicies:
    data = yaml.safe_load(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return ScenePolicies.model_validate(data)


def resolve_area_bucket(area_sqm: float, policies: ScenePolicies) -> str:
    area = float(area_sqm)
    for key, spec in policies.area_buckets.items():
        min_sqm = float(spec.get("min_sqm", float("-inf")))
        max_sqm = float(spec.get("max_sqm", float("inf")))
        if min_sqm <= area <= max_sqm:
            return key
    raise KeyError(f"no area bucket configured for {area_sqm} sqm")


def get_room_program(room_type: str, area_bucket: str, policies: ScenePolicies) -> dict[str, Any]:
    room_programs = policies.room_programs.get(str(room_type))
    if not room_programs:
        raise KeyError(f"room_type policy missing: {room_type}")
    if area_bucket in room_programs:
        return room_programs[area_bucket]
    if "standard" in room_programs:
        return room_programs["standard"]
    raise KeyError(f"room program missing for {room_type}/{area_bucket}")


def get_style_policy(style_label: str, policies: ScenePolicies) -> dict[str, Any]:
    try:
        return policies.styles[str(style_label)]
    except KeyError as exc:
        raise KeyError(f"style policy missing: {style_label}") from exc


def get_solver_profile(key: str, policies: ScenePolicies) -> dict[str, Any]:
    try:
        return policies.solver_profiles[key]
    except KeyError as exc:
        raise KeyError(f"solver profile missing: {key}") from exc


def get_acceptance_profile(key: str, policies: ScenePolicies) -> dict[str, Any]:
    try:
        return policies.acceptance_profiles[key]
    except KeyError as exc:
        raise KeyError(f"acceptance profile missing: {key}") from exc


def build_policy_key(intent: PromptIntent) -> str:
    area_sqm = intent.geometry.target_area_sqm or 0.0
    if area_sqm <= 6.0:
        bucket = AreaBucket.MICRO.value
    elif area_sqm <= 8.5:
        bucket = AreaBucket.COMPACT.value
    elif area_sqm <= 11.0:
        bucket = AreaBucket.STANDARD.value
    else:
        bucket = AreaBucket.LARGE.value
    style_label = (intent.style.style_label.value if intent.style.style_label else "unstyled").lower()
    return f"{intent.room_type.value.lower()}_{bucket}_{style_label}"

