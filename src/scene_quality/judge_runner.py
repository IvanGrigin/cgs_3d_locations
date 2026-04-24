from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.prompt_compiler.llm_client import BaseLLMClient
from src.prompt_compiler.schemas import CompiledPolicy, GateResult, JudgeResult


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    with Image.open(path) as image:
        return {"exists": True, "width": image.width, "height": image.height, "mode": image.mode}


def _heuristic_judge(compiled_policy: CompiledPolicy, gate_result: GateResult, candidate_dir: Path) -> JudgeResult:
    real_object_count = int((gate_result.inventory_summary or {}).get("real_object_count", 0))
    if real_object_count == 0:
        score = 0.0
    elif gate_result.hard_failures:
        score = max(0.5, min(3.0, gate_result.rule_score))
    else:
        score = max(0.0, min(10.0, gate_result.rule_score - 0.1 * len(gate_result.soft_failures)))
    prompt_match = max(0.0, min(10.0, score - (1.0 if "empty_scene_generated" in gate_result.hard_failures else 0.0)))
    style_match = max(
        0.0,
        min(
            10.0,
            score
            - (1.5 if "style_whitelist_not_observed" in gate_result.soft_failures else 0.0)
            - (1.0 if real_object_count == 0 else 0.0),
        ),
    )
    functionality = max(
        0.0,
        min(
            10.0,
            score
            - (2.5 if "missing_required_bed" in gate_result.hard_failures or "bedroom_without_bed" in gate_result.hard_failures else 0.0),
        ),
    )
    composition = max(
        0.0,
        min(
            10.0,
            score
            - (1.0 if "repeated_factory_family" in gate_result.soft_failures else 0.0)
            - (1.5 if real_object_count == 0 else 0.0),
        ),
    )
    return JudgeResult(
        passed=not gate_result.hard_failures and score >= compiled_policy.acceptance_policy.min_judge_score,
        total_score=round(score, 3),
        functionality_score=round(functionality, 3),
        prompt_match_score=round(prompt_match, 3),
        style_match_score=round(style_match, 3),
        composition_score=round(composition, 3),
        strengths=["rule-based screening passed"] if gate_result.passed else [],
        weaknesses=gate_result.hard_failures + gate_result.soft_failures,
        notes="heuristic judge fallback",
        candidate_dir=str(candidate_dir),
        diagnostic_only=bool(gate_result.hard_failures),
    )


def run_judge(compiled_policy: CompiledPolicy, candidate_dir: str | Path, llm_client: BaseLLMClient | None) -> JudgeResult:
    candidate_path = Path(candidate_dir).expanduser().resolve()
    gate_result = GateResult.load(candidate_path / "rule_gate.json")
    prompt_template = (Path("config/judge/room_judge_prompt.md")).expanduser().resolve().read_text(encoding="utf-8")
    rubric = yaml.safe_load((Path("config/judge/room_judge_rubric.yaml")).expanduser().resolve().read_text(encoding="utf-8"))
    render_info = _image_summary(candidate_path / "render.png")
    user_prompt = json.dumps(
        {
            "original_prompt": compiled_policy.prompt_text,
            "room_type": compiled_policy.geometry.room_type.value,
            "area_sqm": compiled_policy.geometry.area_sqm,
            "area_bucket": compiled_policy.geometry.area_bucket.value,
            "style_label": compiled_policy.style_policy.style_label,
            "required_semantics": compiled_policy.program.required_semantics,
            "inventory_summary": gate_result.inventory_summary,
            "rule_gate": gate_result.model_dump(mode="json"),
            "render": render_info,
            "rubric": rubric,
        },
        ensure_ascii=False,
        indent=2,
    )
    schema = {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "total_score": {"type": "number"},
            "functionality_score": {"type": "number"},
            "prompt_match_score": {"type": "number"},
            "style_match_score": {"type": "number"},
            "composition_score": {"type": "number"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": [
            "passed",
            "total_score",
            "functionality_score",
            "prompt_match_score",
            "style_match_score",
            "composition_score",
            "strengths",
            "weaknesses",
            "notes",
        ],
        "additionalProperties": False,
    }
    if llm_client is None:
        result = _heuristic_judge(compiled_policy, gate_result, candidate_path)
    else:
        try:
            raw = llm_client.complete_json(prompt_template, user_prompt, schema)
            result = JudgeResult(
                passed=bool(raw.get("passed", False)),
                total_score=float(raw.get("total_score", 0.0)),
                functionality_score=float(raw.get("functionality_score", 0.0)),
                prompt_match_score=float(raw.get("prompt_match_score", 0.0)),
                style_match_score=float(raw.get("style_match_score", 0.0)),
                composition_score=float(raw.get("composition_score", 0.0)),
                strengths=list(raw.get("strengths") or []),
                weaknesses=list(raw.get("weaknesses") or []),
                notes=str(raw.get("notes") or ""),
                candidate_dir=str(candidate_path),
                diagnostic_only=bool(gate_result.hard_failures),
            )
        except Exception:
            result = _heuristic_judge(compiled_policy, gate_result, candidate_path)
    result.save(candidate_path / "judge.json")
    return result
