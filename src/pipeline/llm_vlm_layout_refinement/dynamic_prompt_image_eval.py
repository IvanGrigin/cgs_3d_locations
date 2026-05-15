#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Динамическая сборка артефактов «как в пайплайне» из одного текстового промпта + изображения:

1) compile (intent → ``CompiledPolicy``), как в ``run_llm_vlm_layout_refinement`` до screening;
2) vision-модель оценивает видимую мебель → ``inventory_summary.json`` (псевдо-инвентарь);
3) ``evaluate_candidate`` → настоящий ``rule_gate.json``;
4) опционально ``run_judge_llm_vlm`` → ``judge.json``.

Это **не** замена Infinigen: гейт и judge опираются на VLM-оценку счётчиков, а не на solver.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.prompt_compiler.compile_to_infinigen import compile_prompt_intent
from src.prompt_compiler.schemas import CompiledPolicy
from src.scene_quality.quality_gate import evaluate_candidate, write_gate_result

from . import evaluation as ev
from . import generation as gen
from src.pipeline.llm_vlm_screening import build_llm_client, load_policies_llm_vlm, run_judge_llm_vlm

_INVENTORY_VLM_SYSTEM = """You estimate 3D room inventory from ONE render image for downstream rule checks.
Return ONLY JSON matching the schema.
- core_semantic_counts: map canonical English furniture semantics (e.g. Bed, Chair, Table, Storage, Lighting, Rug) to non-negative integer counts of clearly visible instances.
- core_factory_counts: optional map of Infinigen-like factory names to counts; use {} if unsure.
- real_object_count: integer — approximate count of distinct major furniture/prop instances you see (>= sum of core counts is OK).
- evidence_notes: one short English sentence on confidence/limitations.
Be conservative: if unsure, lower counts. Empty room → all zeros.
No markdown."""

_INVENTORY_VLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "core_semantic_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
        "core_factory_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
        "real_object_count": {"type": "integer"},
        "evidence_notes": {"type": "string"},
    },
    "required": ["core_semantic_counts", "core_factory_counts", "real_object_count", "evidence_notes"],
    "additionalProperties": False,
}


def compile_like_first_pipeline_round(
    *,
    prompt_text: str,
    policies_path: str,
    llm_provider: str,
    ollama_model: str,
    ollama_url: str,
) -> tuple[CompiledPolicy, dict[str, Any]]:
    """Сборка compiled + infinigen_request без screening (параллель первому refine_round в ``run``)."""
    policies = load_policies_llm_vlm(policies_path)
    text_llm = build_llm_client(llm_provider, ollama_model, ollama_url)
    infinigen_req = gen.propose_infinigen_request_llm(prompt_text=prompt_text, llm=text_llm)
    infinigen_req.setdefault("infinigen_runtime", {})
    infinigen_req = gen._sanitize_request_furniture(infinigen_req, log_run_root=None)
    infinigen_req, _ = gen._ensure_min_factory_whitelist(infinigen_req)
    parsed_area = gen.parse_area_sqm_from_prompt(prompt_text)
    if parsed_area is not None:
        try:
            prev = float(infinigen_req.get("target_area_sqm")) if infinigen_req.get("target_area_sqm") is not None else None
        except Exception:
            prev = None
        if not prev or prev <= 0:
            infinigen_req["target_area_sqm"] = float(parsed_area)

    current_req = dict(infinigen_req)
    intent = gen.prompt_intent_from_infinigen_request(prompt_text=prompt_text, req=current_req)
    compiled = compile_prompt_intent(intent, policies)
    compiled.scene_id = "dynamic_prompt_image_eval"
    max_ov = gen.max_counts_from_request_and_intent(current_req, intent)
    if isinstance(current_req.get("max_count_overrides_vlm"), dict):
        max_ov.update({str(k): int(v) for k, v in current_req["max_count_overrides_vlm"].items()})
    gen.apply_max_count_overrides(compiled, max_ov)
    compiled = gen._force_min_factory_whitelist_in_compiled(compiled, log_run_root=None)
    compiled = gen._disable_child_restrictions_in_compiled(compiled, log_run_root=None)
    rt_block = current_req.get("infinigen_runtime")
    compiled = gen.apply_infinigen_runtime_block(
        compiled,
        rt_block if isinstance(rt_block, dict) else {},
        log_run_root=None,
    )
    compiled = gen._force_min_factory_whitelist_in_compiled(compiled, log_run_root=None)
    compiled = gen._disable_child_restrictions_in_compiled(compiled, log_run_root=None)
    return compiled, current_req


def _normalize_counts_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv > 0:
            out[key] = iv
    return out


def extract_inventory_from_image_vlm(
    *,
    image: Path,
    compiled: CompiledPolicy,
    prompt_text: str,
    ollama_url: str,
    vision_model: str,
    timeout_sec: int,
    temperature: float,
) -> dict[str, Any]:
    user_payload = json.dumps(
        {
            "original_prompt": prompt_text,
            "room_type": compiled.geometry.room_type.value,
            "area_sqm": compiled.geometry.area_sqm,
            "area_bucket": compiled.geometry.area_bucket.value,
            "style_label": compiled.style_policy.style_label,
            "required_semantics": compiled.program.required_semantics,
            "max_counts": compiled.program.max_counts,
            "acceptance_required": list(compiled.acceptance_policy.required_semantics or []),
        },
        ensure_ascii=False,
        indent=2,
    )
    return ev._ollama_vision_json(
        base_url=ollama_url,
        model=vision_model,
        system_prompt=_INVENTORY_VLM_SYSTEM,
        user_text=user_payload,
        image_paths=[image],
        timeout_sec=timeout_sec,
        temperature=temperature,
        response_json_schema=_INVENTORY_VLM_SCHEMA,
    )


def build_inventory_summary_from_vlm(raw: dict[str, Any]) -> dict[str, Any]:
    core_sem = _normalize_counts_map(raw.get("core_semantic_counts"))
    core_fact = _normalize_counts_map(raw.get("core_factory_counts"))
    try:
        roc = max(0, int(raw.get("real_object_count", 0)))
    except (TypeError, ValueError):
        roc = 0
    if roc <= 0 and core_sem:
        roc = max(roc, sum(int(x) for x in core_sem.values()))
    sem_all = dict(core_sem)
    fact_all = dict(core_fact)
    return {
        "raw_real_object_count": roc,
        "real_object_count": roc,
        "factory_counts": fact_all,
        "semantic_counts": sem_all,
        "core_factory_counts": core_fact,
        "core_semantic_counts": core_sem,
    }


def write_synthetic_candidate_dir(
    candidate_dir: Path,
    *,
    image_src: Path,
    inventory_summary: dict[str, Any],
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "inventory.json").write_text("[]", encoding="utf-8")
    (candidate_dir / "inventory_summary.json").write_text(
        json.dumps(inventory_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dst_render = candidate_dir / "render.png"
    shutil.copy2(image_src, dst_render)


def run_dynamic_prompt_image_eval(
    *,
    prompt_text: str,
    image: Path,
    out_dir: Path,
    policies_path: str,
    llm_provider: str,
    ollama_model: str,
    ollama_url: str,
    vision_model: str,
    timeout_sec: int,
    temperature: float,
    run_text_judge: bool,
) -> dict[str, Path]:
    """
    Пишет в ``out_dir``: synthetic_candidate/, compiled_policy.active.json,
    infinigen_request.dynamic.json, vlm_inventory_raw.json, rule_gate (внутри candidate), judge (опционально).
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = out_dir / "synthetic_candidate"
    image = Path(image).expanduser().resolve()

    compiled, req = compile_like_first_pipeline_round(
        prompt_text=prompt_text,
        policies_path=policies_path,
        llm_provider=llm_provider,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )
    (out_dir / "compiled_policy.active.json").write_text(compiled.model_dump_json_pretty(), encoding="utf-8")
    (out_dir / "infinigen_request.dynamic.json").write_text(
        json.dumps(req, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_inv = extract_inventory_from_image_vlm(
        image=image,
        compiled=compiled,
        prompt_text=prompt_text,
        ollama_url=ollama_url,
        vision_model=vision_model,
        timeout_sec=timeout_sec,
        temperature=temperature,
    )
    (out_dir / "vlm_inventory_raw.json").write_text(
        json.dumps(raw_inv, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = build_inventory_summary_from_vlm(raw_inv)
    write_synthetic_candidate_dir(cand, image_src=image, inventory_summary=summary)

    gate = evaluate_candidate(compiled, cand)
    write_gate_result(gate, cand / "rule_gate.json")

    paths: dict[str, Path] = {
        "candidate_dir": cand,
        "rule_gate": cand / "rule_gate.json",
        "compiled": out_dir / "compiled_policy.active.json",
    }
    if run_text_judge:
        judge_llm = build_llm_client(llm_provider, ollama_model, ollama_url)
        run_judge_llm_vlm(compiled, cand, judge_llm, log_run_root=out_dir)
        paths["judge"] = cand / "judge.json"
    return paths
