#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
appraiser.py

Главный orchestration-модуль Appraiser.

Что делает:
1. Загружает scene.v1.
2. Считает кодовые метрики через area_appraiser.py.
3. При необходимости запускает LLM-оценку через llm_appraiser.py.
4. Собирает итоговый score и сохраняет полный отчёт.

Рекомендуемый сценарий использования:
- code-only режим для быстрых массовых прогонов benchmark;
- hybrid режим для финальной оценки кандидатов;
- LLM-only режим для исследований и сравнения judge-моделей.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from .area_appraiser import AreaAppraisalResult, appraise_scene_area, load_json
    from .llm_appraiser import LLMAppraisalResult, appraise_scene_llm
except ImportError:
    from area_appraiser import AreaAppraisalResult, appraise_scene_area, load_json  # type: ignore
    from llm_appraiser import LLMAppraisalResult, appraise_scene_llm  # type: ignore


@dataclass
class FinalAppraisalResult:
    scene_path: Optional[str]
    prompt_text: Optional[str]
    mode: str
    score_10: float
    code_score_10: Optional[float]
    llm_score_10: Optional[float]
    code_result: Optional[dict[str, Any]]
    llm_result: Optional[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def round4(x: float) -> float:
    return round(float(x), 4)



def summarize_code_result(code_result: AreaAppraisalResult) -> dict[str, Any]:
    return {
        "score_10": code_result.score_10,
        "geometry_score_10": code_result.geometry_score_10,
        "prompt_match_score_10": code_result.prompt_match_score_10,
        "constraint_score_10": code_result.constraint_score_10,
        "largest_free_rectangle_ratio": code_result.largest_free_rectangle_ratio,
        "accessibility_ratio": code_result.accessibility_ratio,
        "overlap_ratio": code_result.overlap_ratio,
        "outside_room_ratio": code_result.outside_room_ratio,
        "accessible_objects": code_result.accessible_objects,
        "accessible_objects_total": code_result.accessible_objects_total,
        "room_area_m2": code_result.room_area_m2,
        "free_area_m2": code_result.free_area_m2,
        "largest_free_rectangle_area_m2": code_result.largest_free_rectangle_area_m2,
    }



def combine_scores(
    mode: str,
    code_result: Optional[AreaAppraisalResult],
    llm_result: Optional[LLMAppraisalResult],
    code_weight: float,
    llm_weight: float,
) -> float:
    if mode == "code":
        if code_result is None:
            raise RuntimeError("code mode requires code_result")
        return round4(code_result.score_10)

    if mode == "llm":
        if llm_result is None:
            raise RuntimeError("llm mode requires llm_result")
        return round4(llm_result.score_10)

    if code_result is None and llm_result is None:
        raise RuntimeError("No results to combine")
    if code_result is None:
        return round4(llm_result.score_10)
    if llm_result is None:
        return round4(code_result.score_10)

    total_w = code_weight + llm_weight
    if total_w <= 1e-9:
        raise RuntimeError("Weights sum to zero")

    combined = (code_weight * code_result.score_10 + llm_weight * llm_result.score_10) / total_w
    return round4(combined)



def appraise_scene(
    *,
    scene: dict[str, Any],
    prompt_text: Optional[str] = None,
    mode: str = "hybrid",
    grid_step_m: float = 0.10,
    clearance_m: float = 0.20,
    llm_provider: str = "ollama",
    llm_base_url: str = "http://127.0.0.1:11434",
    judge_models: Optional[list[str]] = None,
    chief_model: Optional[str] = None,
    llm_timeout_sec: float = 180.0,
    llm_temperature: float = 0.1,
    code_weight: float = 0.65,
    llm_weight: float = 0.35,
) -> FinalAppraisalResult:
    mode = mode.strip().lower()
    if mode not in {"code", "llm", "hybrid"}:
        raise RuntimeError(f"Unsupported mode: {mode}")

    code_result: Optional[AreaAppraisalResult] = None
    llm_result: Optional[LLMAppraisalResult] = None

    if mode in {"code", "hybrid"}:
        code_result = appraise_scene_area(
            scene=scene,
            prompt_text=prompt_text,
            grid_step_m=grid_step_m,
            clearance_m=clearance_m,
        )

    if mode in {"llm", "hybrid"}:
        code_summary = summarize_code_result(code_result) if code_result is not None else None
        llm_result = appraise_scene_llm(
            scene=scene,
            original_prompt=prompt_text,
            provider=llm_provider,
            base_url=llm_base_url,
            judge_models=judge_models,
            chief_model=chief_model,
            timeout_sec=llm_timeout_sec,
            temperature=llm_temperature,
            code_metrics_summary=code_summary,
        )

    final_score = combine_scores(mode, code_result, llm_result, code_weight=code_weight, llm_weight=llm_weight)

    summary: dict[str, Any] = {
        "mode": mode,
        "final_score_10": final_score,
    }

    if code_result is not None:
        summary["code"] = summarize_code_result(code_result)
    if llm_result is not None:
        summary["llm"] = {
            "score_10": llm_result.score_10,
            "aggregated_question_scores": llm_result.aggregated_question_scores,
            "chief_model": llm_result.chief_result.model,
            "judge_models": [x.model for x in llm_result.judge_results],
            "chief_confidence_10": llm_result.chief_result.confidence_10,
        }
    if mode == "hybrid":
        summary["weights"] = {
            "code_weight": code_weight,
            "llm_weight": llm_weight,
        }

    return FinalAppraisalResult(
        scene_path=None,
        prompt_text=prompt_text,
        mode=mode,
        score_10=final_score,
        code_score_10=code_result.score_10 if code_result is not None else None,
        llm_score_10=llm_result.score_10 if llm_result is not None else None,
        code_result=code_result.to_dict() if code_result is not None else None,
        llm_result=llm_result.to_dict() if llm_result is not None else None,
        summary=summary,
    )



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Main appraiser for scene.v1")
    p.add_argument("--scene", required=True, help="Path to scene JSON")
    p.add_argument("--prompt", default=None, help="Original prompt text")
    p.add_argument("--prompt-file", default=None, help="Path to prompt text file")
    p.add_argument("--mode", default="hybrid", choices=["code", "llm", "hybrid"], help="Appraiser mode")
    p.add_argument("--out", default=None, help="Output JSON file")

    p.add_argument("--grid-step", type=float, default=0.10, help="Grid step in meters for code metrics")
    p.add_argument("--clearance", type=float, default=0.20, help="Clearance around furniture for walking grid")

    p.add_argument("--llm-provider", default="ollama", help="LLM provider")
    p.add_argument("--llm-base-url", default="http://127.0.0.1:11434", help="LLM base URL")
    p.add_argument("--judge-model", action="append", default=[], help="Judge model, can be repeated")
    p.add_argument("--judge-models", default=None, help="Comma-separated judge models")
    p.add_argument("--chief-model", default=None, help="Chief appraiser model")
    p.add_argument("--llm-timeout", type=float, default=180.0, help="Timeout for LLM calls")
    p.add_argument("--llm-temperature", type=float, default=0.1, help="Temperature for LLM calls")

    p.add_argument("--code-weight", type=float, default=0.65, help="Weight of code metrics in hybrid mode")
    p.add_argument("--llm-weight", type=float, default=0.35, help="Weight of LLM metrics in hybrid mode")
    return p



def main() -> None:
    args = build_argparser().parse_args()
    scene = load_json(args.scene)

    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")

    judge_models = list(args.judge_model or [])
    if args.judge_models:
        judge_models.extend([x.strip() for x in args.judge_models.split(",") if x.strip()])

    result = appraise_scene(
        scene=scene,
        prompt_text=prompt_text,
        mode=args.mode,
        grid_step_m=args.grid_step,
        clearance_m=args.clearance,
        llm_provider=args.llm_provider,
        llm_base_url=args.llm_base_url,
        judge_models=judge_models,
        chief_model=args.chief_model,
        llm_timeout_sec=args.llm_timeout,
        llm_temperature=args.llm_temperature,
        code_weight=args.code_weight,
        llm_weight=args.llm_weight,
    )
    result.scene_path = str(Path(args.scene).expanduser().resolve())

    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
