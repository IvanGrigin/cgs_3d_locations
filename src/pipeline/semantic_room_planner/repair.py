from __future__ import annotations

from typing import Any

from .placement_solver import solve_placements
from .validation import validate_geometry


def repair_scene(room_geometry: dict[str, Any], objects: list[dict[str, Any]], relations: dict[str, Any], placements: dict[str, Any], validation: dict[str, Any], max_iterations: int = 3) -> dict[str, Any]:
    max_iterations = max(0, min(int(max_iterations), 10))
    report = {"schema": "repair_report/v1", "iterations": [], "removed_objects": [], "modified_objects": [], "final_status": "success" if validation.get("is_valid") else "partial_success"}
    current_objects = list(objects)
    current = placements
    current_validation = validation
    decorative = {"wall_art", "vase", "plant", "book", "mug", "cup", "water_bottle", "phone", "remote"}
    for idx in range(max_iterations):
        if current_validation.get("is_valid"):
            break
        hard = list(current_validation.get("hard_errors") or [])
        removed = []
        if hard and idx > 0:
            for obj in list(current_objects):
                if obj.get("role") == "accessory" or obj.get("subclass") in decorative:
                    current_objects.remove(obj)
                    removed.append(obj["id"])
                    break
        current = solve_placements(
            room_geometry,
            current_objects,
            relations,
            {},
            {"max_candidates_per_object": 16, "max_total_candidate_combinations": 512},
        )
        current_validation = validate_geometry(room_geometry, current_objects, relations, current)
        report["iterations"].append({"iteration": idx + 1, "hard_errors_before": hard, "removed": removed, "score_after": current_validation.get("score")})
        report["removed_objects"].extend(removed)
    if not current_validation.get("is_valid"):
        report["final_status"] = "failed" if any("main" in str(e) for e in current_validation.get("hard_errors", [])) else "partial_success"
    else:
        report["final_status"] = "success"
    report["placements"] = current
    report["objects"] = current_objects
    report["geometry_validation"] = current_validation
    report["max_iterations"] = max_iterations
    return report
