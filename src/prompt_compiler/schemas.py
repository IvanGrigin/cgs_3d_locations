from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JSONModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def save(self, path: str | Path) -> None:
        out_path = Path(path).expanduser().resolve()  # pragma: no cover
        out_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no cover
        out_path.write_text(self.model_dump_json_pretty(), encoding="utf-8")  # pragma: no cover

    @classmethod
    def load(cls, path: str | Path) -> "JSONModel":
        data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))  # pragma: no cover
        return cls.model_validate(data)  # pragma: no cover

    def model_dump_json_pretty(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2)


class RoomType(str, Enum):
    BEDROOM = "Bedroom"
    LIVING_ROOM = "LivingRoom"
    KITCHEN = "Kitchen"
    BATHROOM = "Bathroom"
    DINING_ROOM = "DiningRoom"


class AreaBucket(str, Enum):
    MICRO = "micro"
    COMPACT = "compact"
    STANDARD = "standard"
    LARGE = "large"


class DensityLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecorRichness(str, Enum):
    NONE = "none"
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StyleLabel(str, Enum):
    JAPANDI = "japandi"
    SCANDINAVIAN = "scandinavian"
    CONTEMPORARY = "contemporary"
    BAROQUE_INSPIRED = "baroque_inspired"


class RepairReason(str, Enum):
    MISSING_REQUIRED_BED = "missing_required_bed"
    STORAGE_OVERFLOW_SMALL_ROOM = "storage_overflow_small_room"
    STYLE_NOT_READABLE = "style_not_readable"
    TOO_EMPTY = "too_empty"
    TOO_CLUTTERED = "too_cluttered"
    REPEATED_FACTORY_FAMILY = "repeated_factory_family"
    SOLVER_CONSTRAINT_VIOLATION = "solver_constraint_violation"
    BAD_AREA_PROGRAM_FIT = "bad_area_program_fit"


class GeometryHint(JSONModel):
    target_area_sqm: float | None = None
    width_m: float | None = None
    depth_m: float | None = None
    height_m: float | None = 2.7


class StyleIntent(JSONModel):
    style_label: StyleLabel | None = None
    style_raw: str | None = None
    density: DensityLevel | None = None
    decor_richness: DecorRichness | None = None
    palette_hint: list[str] = Field(default_factory=list)
    material_family: list[str] = Field(default_factory=list)


class OpeningsIntent(JSONModel):
    wants_door: bool = True
    wants_window: bool = True
    preferred_window_wall: str | None = None
    preferred_door_wall: str | None = None
    window_count: int | None = None
    door_count: int | None = None


class ObjectsIntent(JSONModel):
    required: list[str] = Field(default_factory=list)
    desired: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class PreferencesIntent(JSONModel):
    favorite_colors: list[str] = Field(default_factory=list)
    avoid_colors: list[str] = Field(default_factory=list)
    notes: str = ""


class PromptIntent(JSONModel):
    prompt_text: str
    room_type: RoomType = RoomType.BEDROOM
    geometry: GeometryHint = Field(default_factory=GeometryHint)
    style: StyleIntent = Field(default_factory=StyleIntent)
    openings: OpeningsIntent = Field(default_factory=OpeningsIntent)
    objects: ObjectsIntent = Field(default_factory=ObjectsIntent)
    preferences: PreferencesIntent = Field(default_factory=PreferencesIntent)


class WallSegment(JSONModel):
    id: str
    from_vertex: int
    to_vertex: int


class OpeningPlacement(JSONModel):
    wall_id: str
    s: float
    width: float


class CompiledGeometry(JSONModel):
    room_type: RoomType
    area_sqm: float
    width_m: float
    depth_m: float
    height_m: float = 2.7
    area_bucket: AreaBucket
    floor_polygon: list[dict[str, float]]
    walls: list[WallSegment]
    doors: list[OpeningPlacement] = Field(default_factory=list)
    windows: list[OpeningPlacement] = Field(default_factory=list)


class CompiledProgram(JSONModel):
    required_semantics: list[str] = Field(default_factory=list)
    optional_semantics: list[str] = Field(default_factory=list)
    forbidden_semantics: list[str] = Field(default_factory=list)
    preferred_semantics: list[str] = Field(default_factory=list)
    required_primary: list[str] = Field(default_factory=list)
    allowed_primary: list[str] = Field(default_factory=list)
    allowed_secondary: list[str] = Field(default_factory=list)
    factory_whitelist: list[str] = Field(default_factory=list)
    factory_blacklist: list[str] = Field(default_factory=list)
    max_counts: dict[str, int] = Field(default_factory=dict)
    notes: str = ""


class CompiledStylePolicy(JSONModel):
    style_label: str
    style_strength: float = 1.0
    density: DensityLevel = DensityLevel.LOW
    decor_richness: DecorRichness = DecorRichness.LOW
    palette_hint: list[str] = Field(default_factory=list)
    material_family: list[str] = Field(default_factory=list)
    required_semantics: list[str] = Field(default_factory=list)
    forbidden_semantics: list[str] = Field(default_factory=list)
    factory_whitelist: list[str] = Field(default_factory=list)
    factory_blacklist: list[str] = Field(default_factory=list)
    preferred_colors: list[str] = Field(default_factory=list)
    avoid_colors: list[str] = Field(default_factory=list)
    notes: str = ""


class CompiledInfinigenPolicy(JSONModel):
    solver_profile_key: str
    gin_overrides: list[str] = Field(default_factory=list)
    monkeypatch_params: dict[str, Any] = Field(default_factory=dict)
    stage_flags: dict[str, bool] = Field(default_factory=dict)
    solver_steps: dict[str, int] = Field(default_factory=dict)
    screening_seeds: list[int] = Field(default_factory=list)
    final_seeds: list[int] = Field(default_factory=list)
    required_semantics: list[str] = Field(default_factory=list)
    forbidden_semantics: list[str] = Field(default_factory=list)
    factory_whitelist: list[str] = Field(default_factory=list)
    factory_blacklist: list[str] = Field(default_factory=list)
    max_counts: dict[str, int] = Field(default_factory=dict)


class AcceptancePolicy(JSONModel):
    profile_key: str
    required_semantics: list[str] = Field(default_factory=list)
    forbidden_semantics: list[str] = Field(default_factory=list)
    factory_whitelist: list[str] = Field(default_factory=list)
    factory_blacklist: list[str] = Field(default_factory=list)
    max_counts: dict[str, int] = Field(default_factory=dict)
    min_real_objects: int = 0
    max_repeated_factory_count: int = 999
    reject_on_solver_violation: bool = False
    allowed_solver_violations: list[str] = Field(default_factory=list)
    min_rule_score: float = 0.0
    min_judge_score: float = 0.0


class CompiledPolicy(JSONModel):
    schema_version: str = "compiled_policy/v1"
    scene_id: str
    prompt_text: str
    intent: PromptIntent
    geometry: CompiledGeometry
    program: CompiledProgram
    style_policy: CompiledStylePolicy
    infinigen_policy: CompiledInfinigenPolicy
    acceptance_policy: AcceptancePolicy
    preflight: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    repair_round: int = 0
    parent_policy_path: str | None = None


class GateResult(JSONModel):
    passed: bool
    rule_score: float
    hard_failures: list[str] = Field(default_factory=list)
    soft_failures: list[str] = Field(default_factory=list)
    inventory_summary: dict[str, Any] = Field(default_factory=dict)
    solver_summary: dict[str, Any] = Field(default_factory=dict)
    candidate_dir: str | None = None


class JudgeResult(JSONModel):
    passed: bool
    total_score: float
    functionality_score: float = 0.0
    prompt_match_score: float = 0.0
    style_match_score: float = 0.0
    composition_score: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    notes: str = ""
    candidate_dir: str | None = None
    diagnostic_only: bool = False


class RepairPlan(JSONModel):
    reasons: list[RepairReason] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    updated_monkeypatch_params: dict[str, Any] = Field(default_factory=dict)
    added_factory_blacklist: list[str] = Field(default_factory=list)
    added_factory_whitelist: list[str] = Field(default_factory=list)
    updated_max_counts: dict[str, int] = Field(default_factory=dict)
    added_required_semantics: list[str] = Field(default_factory=list)
    removed_optional_semantics: list[str] = Field(default_factory=list)
    added_gin_overrides: list[str] = Field(default_factory=list)
