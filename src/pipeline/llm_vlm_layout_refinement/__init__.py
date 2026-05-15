#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дополнительный контур (вне run_pipeline): LLM → запрос Infinigen (набор мебели + стиль)
→ несколько вариантов расстановки (разные screening seeds) → VLM по render.png
(при необходимости — быстрый EEVEE/low-res превью-рендер локальным Blender)
уточняет intent / max_counts и при необходимости перезапускает screening.

Подмодули:
- ``generation`` — LLM JSON, intent↔request, санитизация furniture/runtime, compile-хуки.
- ``evaluation`` — VLM, judge→repair, merge VLM, Blender/SSH вспомогательные функции.
- ``run`` — оркестратор ``run_llm_vlm_layout_refinement`` и CLI.
- ``evaluation_cli`` — только VLM по готовым артефактам: ``python -m ...evaluation_cli``.
- ``dynamic_prompt_image_eval`` — compile + VLM-инвентарь + ``evaluate_candidate`` (динамический ``rule_gate``).
- ``judge_cli`` — judge / динамический гейт: ``python -m ...judge_cli``.

Screening и политики — ``llm_vlm_screening``, ``llm_vlm_scene_policies.yaml``.
"""

from __future__ import annotations

from .generation import (
    apply_infinigen_runtime_block,
    apply_max_count_overrides,
    max_counts_from_request_and_intent,
    merge_runtime_dicts,
    parse_area_sqm_from_prompt,
    prompt_intent_from_infinigen_request,
    propose_infinigen_request_llm,
)
from .evaluation import merge_infinigen_request_with_vlm
from .run import build_cli, main, run_llm_vlm_layout_refinement

__all__ = [
    "apply_infinigen_runtime_block",
    "apply_max_count_overrides",
    "build_cli",
    "main",
    "max_counts_from_request_and_intent",
    "merge_infinigen_request_with_vlm",
    "merge_runtime_dicts",
    "parse_area_sqm_from_prompt",
    "prompt_intent_from_infinigen_request",
    "propose_infinigen_request_llm",
    "run_llm_vlm_layout_refinement",
]
