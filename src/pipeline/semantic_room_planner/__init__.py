"""Sequential semantic room planner.

LLM stages only add semantic intent, zones, items, relations, and catalog text.
Geometry, placement, validation, and repair are deterministic code stages.
"""

from .geometry_analyzer import analyze_room_geometry, normalize_room_input

__all__ = ["analyze_room_geometry", "normalize_room_input"]
