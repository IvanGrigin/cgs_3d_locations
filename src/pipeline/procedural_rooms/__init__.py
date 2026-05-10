"""
Procedural room generation package.

The package adds deterministic, rule-based dense furnishing for room types that
are hard to control with generic placers: bedrooms, living rooms and corridors.

Public entry points:
    - apply_procedural_room_stage
    - apply_procedural_room_stage_to_artifacts
"""

from .procedural_room_stage import (
    apply_procedural_room_stage,
    apply_procedural_room_stage_to_artifacts,
)

__all__ = [
    "apply_procedural_room_stage",
    "apply_procedural_room_stage_to_artifacts",
]
