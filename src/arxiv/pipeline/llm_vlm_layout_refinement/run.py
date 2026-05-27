#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оркестрация refine-цикла и CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from src.prompt_compiler.compile_to_infinigen import compile_prompt_intent, write_compiled_artifacts
from src.prompt_compiler.schemas import CompiledPolicy
from src.scene_quality.repair_loop import apply_repair_plan, build_repair_plan

from . import evaluation as ev
from . import generation as gen
from src.pipeline.llm_vlm_screening import (
    build_llm_client,
    default_policies_path,
    emit_llm_vlm_log,
    format_timing_dur,
    is_valid_final_candidate,
    load_policies_llm_vlm,
    materialize_final,
    run_vlm_fast_preview_render,
    screen_and_score,
    select_best,
    truncate_for_log,
    write_run_status,
)

def run_llm_vlm_layout_refinement(
    *,
    prompt_text: str,
    run_root: Path,
    policies_path: str,
    llm_provider: str,
    ollama_url: str,
    ollama_model: str,
    vlm_model: str | None,
    layout_seeds: list[int],
    max_vlm_rounds: int,
    skip_judge: bool,
    remote_kwargs: dict[str, Any],
    timeout_sec: int = 300,
    max_inner_repairs: int = 2,
    ollama_judge_model: str | None = None,
    open_blend: bool = True,
    blender_bin: str | None = None,
    use_judge_feedback_llm: bool = True,
    judge_feedback_threshold: float = 6.0,
    vlm_fast_preview_render: bool = True,
    vlm_render_resolution_pct: int = 42,
    vlm_render_engine: str = "eevee",
    vlm_render_timeout_sec: int = 900,
) -> dict[str, Any]:
    """
    Полный цикл: LLM-запрос → compile+artifacts → infinigen_runtime → screening
    → rule-based repair (build_repair_plan) → VLM → уточнение запроса и повтор.

    При ``vlm_model`` и ``vlm_fast_preview_render``: если у лучшего кандидата нет
    ``render.png``, перед VLM вызывается локальный Blender (EEVEE или облегчённый Cycles,
    пониженное разрешение) — см. ``run_vlm_fast_preview_render``.

    Возвращает словарь со статусом и путями артефактов.
    """
    run_root = Path(run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "llm_vlm_run.log").write_text("", encoding="utf-8")
    # Чистим артефакты прошлого прогона, чтобы:
    #   1) Blender не открывал stale final/scene.blend от чужого промпта,
    #   2) run_status.json/trace не вводили в заблуждение, если новый прогон упадёт.
    stale_final = run_root / "final"
    if stale_final.is_dir():
        try:
            shutil.rmtree(stale_final)
        except Exception:
            pass
    for stale_name in ("run_status.json", "layout_refinement_trace.json"):
        stale_path = run_root / stale_name
        if stale_path.is_file():
            try:
                stale_path.unlink()
            except Exception:
                pass
    emit_llm_vlm_log(run_root, f"=== run start out_dir={run_root}")
    emit_llm_vlm_log(run_root, "cleaned previous-run artefacts: final/, run_status.json, layout_refinement_trace.json")
    t_run0 = perf_counter()
    emit_llm_vlm_log(run_root, f"user_prompt: {truncate_for_log(prompt_text, 800)}")
    effective_judge_model = (ollama_judge_model or ollama_model).strip()
    emit_llm_vlm_log(
        run_root,
        f"llm_provider={llm_provider!r} ollama_model={ollama_model!r} judge_model={effective_judge_model!r} "
        f"vlm_model={vlm_model!r} layout_seeds={layout_seeds} max_vlm_rounds={max_vlm_rounds} "
        f"max_inner_repairs={max_inner_repairs} skip_judge={skip_judge} "
        f"judge_feedback_llm={use_judge_feedback_llm} judge_feedback_threshold={judge_feedback_threshold} "
        f"vlm_fast_preview_render={vlm_fast_preview_render} vlm_render_resolution_pct={vlm_render_resolution_pct} "
        f"vlm_render_engine={vlm_render_engine!r}",
    )
    emit_llm_vlm_log(
        run_root,
        f"ollama_endpoint={ollama_url.rstrip('/') + '/api/chat'} request_model={ollama_model!r} "
        f"judge_model={effective_judge_model!r}",
    )
    emit_llm_vlm_log(run_root, f"policies_path={policies_path!r} remote_kwargs={truncate_for_log(remote_kwargs, 1200)}")
    policies = load_policies_llm_vlm(policies_path)
    text_llm = build_llm_client(llm_provider, ollama_model, ollama_url)
    if effective_judge_model == ollama_model.strip():
        judge_llm = text_llm
        emit_llm_vlm_log(run_root, "judge LLM client reuses request LLM (same model)")
    else:
        judge_llm = build_llm_client(llm_provider, effective_judge_model, ollama_url)
        emit_llm_vlm_log(
            run_root,
            f"judge LLM client built separately: model={effective_judge_model!r}",
        )

    trace: dict[str, Any] = {"rounds": []}

    remote_host = remote_kwargs.get("remote_host")
    remote_user = remote_kwargs.get("remote_user")
    if remote_kwargs.get("remote_cleanup_tmp") and remote_host and remote_user:
        cleanup_ok = ev._cleanup_remote_tmp(
            remote_host=remote_host,
            remote_user=remote_user,
            remote_port=remote_kwargs.get("remote_port"),
            remote_key=remote_kwargs.get("remote_key"),
            log_run_root=run_root,
        )
        if not cleanup_ok:
            emit_llm_vlm_log(run_root, "remote_cleanup_tmp finished with errors (см. выше) — продолжаем прогон.")
    # Параметр в remote_kwargs не предназначен для run_one_candidate_from_compiled.
    remote_kwargs = {k: v for k, v in remote_kwargs.items() if k != "remote_cleanup_tmp"}

    t_req0 = perf_counter()
    infinigen_req = gen.propose_infinigen_request_llm(prompt_text=prompt_text, llm=text_llm)
    emit_llm_vlm_log(
        run_root,
        f"[timing] gen.propose_infinigen_request_llm wall={format_timing_dur(perf_counter() - t_req0)}",
    )
    infinigen_req.setdefault("infinigen_runtime", {})
    infinigen_req = gen._sanitize_request_furniture(infinigen_req, log_run_root=run_root)
    infinigen_req, added_factories = gen._ensure_min_factory_whitelist(infinigen_req)
    if added_factories:
        emit_llm_vlm_log(
            run_root,
            f"factory_whitelist auto-extended for room_type={infinigen_req.get('room_type')!r}: "
            f"+{added_factories} (без них solver бракует слоты по coverage)",
        )
    parsed_area = gen.parse_area_sqm_from_prompt(prompt_text)
    if parsed_area is not None:
        prev = infinigen_req.get("target_area_sqm")
        try:
            prev_val = float(prev) if prev is not None else None
        except Exception:
            prev_val = None
        if not prev_val or prev_val <= 0:
            infinigen_req["target_area_sqm"] = float(parsed_area)
            emit_llm_vlm_log(
                run_root,
                f"area parsed from prompt: {parsed_area} m^2 (target_area_sqm не было задано LLM/intent)",
            )
        elif abs(prev_val - parsed_area) > 0.5:
            emit_llm_vlm_log(
                run_root,
                f"area mismatch: prompt={parsed_area} m^2, LLM target_area_sqm={prev_val}; оставляю LLM-значение",
            )
    else:
        emit_llm_vlm_log(run_root, "area parser: не нашёл «N м²/кв.м/sqm» в промпте")
    emit_llm_vlm_log(run_root, f"LLM infinigen_request (truncated):\n{truncate_for_log(infinigen_req, 4000)}")
    (run_root / "infinigen_request.initial.json").write_text(
        json.dumps(infinigen_req, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_root / "infinigen_request.active.json").write_text(
        json.dumps(infinigen_req, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    seeds = list(layout_seeds)
    best_row: dict[str, Any] | None = None
    current_req = dict(infinigen_req)

    for round_idx in range(max(1, max_vlm_rounds)):
        emit_llm_vlm_log(run_root, f"--- refine_round={round_idx} ---")
        t_round0 = perf_counter()
        emit_llm_vlm_log(
            run_root,
            f"current_req (truncated):\n{truncate_for_log(current_req, 3500)}",
        )
        intent = gen.prompt_intent_from_infinigen_request(prompt_text=prompt_text, req=current_req)
        emit_llm_vlm_log(
            run_root,
            f"intent room={intent.room_type.value} style={intent.style.style_label} "
            f"required={intent.objects.required!r} desired={intent.objects.desired!r}",
        )
        compiled = compile_prompt_intent(intent, policies)
        compiled.scene_id = run_root.name
        emit_llm_vlm_log(
            run_root,
            f"compiled solver_profile_key={compiled.infinigen_policy.solver_profile_key} "
            f"area_sqm={compiled.geometry.area_sqm} area_bucket={compiled.geometry.area_bucket.value} "
            f"required_semantics={compiled.program.required_semantics!r}",
        )
        max_ov = gen.max_counts_from_request_and_intent(current_req, intent)
        if isinstance(current_req.get("max_count_overrides_vlm"), dict):
            max_ov.update({str(k): int(v) for k, v in current_req["max_count_overrides_vlm"].items()})
        gen.apply_max_count_overrides(compiled, max_ov)

        compiled = gen._force_min_factory_whitelist_in_compiled(compiled, log_run_root=run_root)
        compiled = gen._disable_child_restrictions_in_compiled(compiled, log_run_root=run_root)
        if remote_kwargs is not None:
            remote_kwargs = ev._ensure_room_type_gin(
                remote_kwargs,
                str(compiled.geometry.room_type.value),
                log_run_root=run_root,
            )
        rt_block = current_req.get("infinigen_runtime")
        compiled = gen.apply_infinigen_runtime_block(
            compiled,
            rt_block if isinstance(rt_block, dict) else {},
            log_run_root=run_root,
        )
        compiled = gen._force_min_factory_whitelist_in_compiled(compiled, log_run_root=run_root)
        compiled = gen._disable_child_restrictions_in_compiled(compiled, log_run_root=run_root)
        emit_llm_vlm_log(
            run_root,
            f"after runtime_block solver_steps={compiled.infinigen_policy.solver_steps!r} "
            f"stage_flags={compiled.infinigen_policy.stage_flags!r} "
            f"max_counts={compiled.program.max_counts!r} "
            f"monkeypatch={truncate_for_log(compiled.infinigen_policy.monkeypatch_params, 600)}",
        )
        emit_llm_vlm_log(
            run_root,
            f"[timing] refine_round={round_idx} compile+intent+runtime wall={format_timing_dur(perf_counter() - t_round0)}",
        )

        round_dir = run_root / f"refine_round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        inner_trace: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for inner in range(max(0, max_inner_repairs) + 1):
            emit_llm_vlm_log(run_root, f"inner_repair iteration inner={inner}/{max_inner_repairs}")
            t_inner0 = perf_counter()
            write_compiled_artifacts(compiled, round_dir)
            (run_root / "compiled_policy.active.json").write_text(compiled.model_dump_json_pretty(), encoding="utf-8")
            screening_dir = round_dir / f"screening_i{inner}"
            screening_dir.mkdir(parents=True, exist_ok=True)
            results = screen_and_score(
                compiled,
                run_root,
                text_llm,
                screening_base_dir=screening_dir,
                seeds=seeds,
                skip_judge=skip_judge,
                remote_kwargs=remote_kwargs,
                judge_llm_client=judge_llm,
            )
            emit_llm_vlm_log(
                run_root,
                f"[timing] refine_round={round_idx} inner={inner} screen_and_score wall={format_timing_dur(perf_counter() - t_inner0)}",
            )
            remote_kwargs = ev._maybe_disable_fast_solve(results, remote_kwargs, log_run_root=run_root)
            best_row = select_best(results, compiled.acceptance_policy.min_judge_score)
            emit_llm_vlm_log(
                run_root,
                f"select_best min_judge={compiled.acceptance_policy.min_judge_score} "
                f"best_dir={best_row['candidate_dir'] if best_row else None!r} "
                f"best_combined={best_row['combined_score'] if best_row else None}",
            )
            inner_trace.append(
                {
                    "inner": inner,
                    "best_dir": str(best_row["candidate_dir"]) if best_row else None,
                    "best_combined": best_row["combined_score"] if best_row else None,
                    "gate_passed": best_row["gate"].passed if best_row else None,
                }
            )
            if is_valid_final_candidate(best_row, compiled.acceptance_policy.min_judge_score):
                break
            if inner >= max_inner_repairs:
                break
            if best_row is None:
                break
            judge_obj = best_row.get("judge") if best_row else None
            rp = build_repair_plan(compiled, best_row["gate"], judge_obj)
            ev._extend_repair_plan_with_judge(rp, judge_obj, compiled, log_run_root=run_root)

            # LLM-feedback цикл: судья дал низкий total → просим text-LLM
            # пересобрать infinigen_request с учётом его критики.
            refined_req: dict[str, Any] | None = None
            if use_judge_feedback_llm and judge_obj is not None:
                gate_for_feedback = {
                    "passed": bool(best_row["gate"].passed),
                    "rule_score": float(best_row["gate"].rule_score),
                    "hard_failures": list(best_row["gate"].hard_failures),
                    "soft_failures": list(best_row["gate"].soft_failures),
                }
                refined_req = ev._propose_request_refinement_via_judge_feedback(
                    current_req=current_req,
                    judge=judge_obj,
                    gate_summary=gate_for_feedback,
                    llm=text_llm,
                    threshold=judge_feedback_threshold,
                    log_run_root=run_root,
                )

            if refined_req is not None:
                refined_req.setdefault("infinigen_runtime", current_req.get("infinigen_runtime", {}) or {})
                refined_req = gen._sanitize_request_furniture(refined_req, log_run_root=run_root)
                refined_req, _ = gen._ensure_min_factory_whitelist(refined_req)
                current_req = refined_req
                (run_root / "infinigen_request.active.json").write_text(
                    json.dumps(current_req, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                emit_llm_vlm_log(
                    run_root,
                    "judge_feedback_llm: applying refined request → recompile from scratch (skip apply_repair_plan)",
                )
                intent = gen.prompt_intent_from_infinigen_request(prompt_text=prompt_text, req=current_req)
                compiled = compile_prompt_intent(intent, policies)
                compiled.scene_id = run_root.name
                max_ov = gen.max_counts_from_request_and_intent(current_req, intent)
                if isinstance(current_req.get("max_count_overrides_vlm"), dict):
                    max_ov.update({str(k): int(v) for k, v in current_req["max_count_overrides_vlm"].items()})
                gen.apply_max_count_overrides(compiled, max_ov)
                compiled = gen._force_min_factory_whitelist_in_compiled(compiled, log_run_root=run_root)
                compiled = gen._disable_child_restrictions_in_compiled(compiled, log_run_root=run_root)
                rt_block = current_req.get("infinigen_runtime")
                compiled = gen.apply_infinigen_runtime_block(
                    compiled,
                    rt_block if isinstance(rt_block, dict) else {},
                    log_run_root=run_root,
                )
                compiled = gen._force_min_factory_whitelist_in_compiled(compiled, log_run_root=run_root)
                compiled = gen._disable_child_restrictions_in_compiled(compiled, log_run_root=run_root)
                if remote_kwargs is not None:
                    remote_kwargs = ev._ensure_room_type_gin(
                        remote_kwargs,
                        str(compiled.geometry.room_type.value),
                        log_run_root=run_root,
                    )
                continue

            if not rp.reasons:
                emit_llm_vlm_log(run_root, "rule_repair: no reasons from build_repair_plan, stop inner loop")
                break
            emit_llm_vlm_log(
                run_root,
                f"rule_repair apply reasons={[r.value for r in rp.reasons]!r} actions={truncate_for_log(rp.actions, 800)}",
            )
            compiled = apply_repair_plan(compiled, rp)
            compiled = gen._force_min_factory_whitelist_in_compiled(compiled, log_run_root=run_root)
            compiled = gen._disable_child_restrictions_in_compiled(compiled, log_run_root=run_root)

        round_trace: dict[str, Any] = {
            "round": round_idx,
            "seeds": seeds,
            "inner_rule_repairs": inner_trace,
            "candidates": [
                {
                    "dir": str(r["candidate_dir"]),
                    "combined_score": r["combined_score"],
                    "gate_passed": r["gate"].passed,
                }
                for r in results
            ],
        }

        render_path: Path | None = None
        if best_row is not None:
            cand = Path(best_row["candidate_dir"])
            rp = cand / "render.png"
            if (
                vlm_model
                and vlm_fast_preview_render
                and (not rp.is_file() or rp.stat().st_size < 64)
            ):
                blender_exe = ev._resolve_blender_binary(blender_bin)
                if blender_exe:
                    run_vlm_fast_preview_render(
                        cand,
                        blender_executable=blender_exe,
                        resolution_pct=int(vlm_render_resolution_pct),
                        render_engine=str(vlm_render_engine or "eevee"),
                        log_run_root=run_root,
                        timeout_sec=int(vlm_render_timeout_sec),
                    )
                else:
                    emit_llm_vlm_log(
                        run_root,
                        "vlm_fast_render: Blender не найден (--blender-bin / BLENDER_PATH); "
                        "VLM возможен только при готовом render.png",
                    )
            if rp.is_file():
                render_path = rp

        vlm_payload: dict[str, Any] | None = None
        if vlm_model and render_path is not None:
            emit_llm_vlm_log(run_root, f"VLM render={render_path}")
            user_payload = json.dumps(
                {
                    "original_prompt": prompt_text,
                    "infinigen_request": current_req,
                    "compiled_summary": {
                        "room_type": compiled.geometry.room_type.value,
                        "style": compiled.style_policy.style_label,
                        "required_semantics": compiled.program.required_semantics,
                        "max_counts": compiled.program.max_counts,
                        "monkeypatch_params": compiled.infinigen_policy.monkeypatch_params,
                        "stage_flags": compiled.infinigen_policy.stage_flags,
                        "solver_steps": compiled.infinigen_policy.solver_steps,
                        "gin_overrides_head": (compiled.infinigen_policy.gin_overrides or [])[:24],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            emit_llm_vlm_log(run_root, f"VLM user_text (truncated):\n{truncate_for_log(user_payload, 4500)}")
            try:
                t_vlm0 = perf_counter()
                vlm_payload = ev._ollama_vision_json(
                    base_url=ollama_url,
                    model=vlm_model,
                    system_prompt=ev._vlm_system_prompt(),
                    user_text=user_payload,
                    image_paths=[render_path],
                    timeout_sec=timeout_sec,
                    temperature=0.0,
                )
                emit_llm_vlm_log(
                    run_root,
                    f"[timing] VLM ollama vision wall={format_timing_dur(perf_counter() - t_vlm0)} model={vlm_model!r}",
                )
                emit_llm_vlm_log(run_root, f"VLM response (truncated):\n{truncate_for_log(vlm_payload, 3000)}")
            except Exception as exc:
                round_trace["vlm_error"] = str(exc)
                emit_llm_vlm_log(
                    run_root,
                    f"[timing] VLM ollama vision wall={format_timing_dur(perf_counter() - t_vlm0)} (failed: {exc!r})",
                )
                vlm_payload = {
                    "satisfied": True,
                    "critique": f"VLM skipped: {exc}",
                    "add_required_objects": [],
                    "add_desired_objects": [],
                    "add_forbidden_objects": [],
                    "remove_objects": [],
                    "max_count_overrides": {},
                    "notes_append": "",
                    "next_screening_seeds": [],
                    "infinigen_runtime": {},
                }
        elif vlm_model:
            round_trace["vlm_error"] = "no_render_png_for_best_candidate"

        if vlm_payload:
            (round_dir / "vlm_refinement.json").write_text(
                json.dumps(vlm_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            round_trace["vlm"] = vlm_payload

        trace["rounds"].append(round_trace)

        if not vlm_model:
            break

        if vlm_payload is None:
            break

        if bool(vlm_payload.get("satisfied")):
            break

        if round_idx >= max(1, max_vlm_rounds) - 1:
            break

        nseeds_raw = vlm_payload.get("next_screening_seeds") or []
        nseeds: list[int] = []
        for x in nseeds_raw:
            try:
                nseeds.append(int(float(x)))
            except (TypeError, ValueError):
                continue
        if nseeds:
            seeds = nseeds
        current_req = ev.merge_infinigen_request_with_vlm(current_req, vlm_payload)
        emit_llm_vlm_log(
            run_root,
            f"merged infinigen_request.active (truncated):\n{truncate_for_log(current_req, 3500)}",
        )
        (run_root / "infinigen_request.active.json").write_text(
            json.dumps(current_req, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (run_root / "layout_refinement_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_llm_vlm_log(run_root, f"=== run end status written trace={run_root / 'layout_refinement_trace.json'}")
    emit_llm_vlm_log(run_root, f"[timing] run_llm_vlm_layout_refinement total wall={format_timing_dur(perf_counter() - t_run0)}")

    active_policy = run_root / "compiled_policy.active.json"
    if active_policy.is_file():
        compiled_last = CompiledPolicy.load(active_policy)
    else:
        compiled_last = compiled

    if is_valid_final_candidate(best_row, compiled_last.acceptance_policy.min_judge_score):
        assert best_row is not None
        materialize_final(run_root, best_row)
        write_run_status(run_root, status="ok", selected_candidate=str(best_row["candidate_dir"].name))
    else:
        write_run_status(
            run_root,
            status="no_valid_candidate",
            selected_candidate=str(best_row["candidate_dir"].name) if best_row else "",
            reason="llm_vlm_refinement_no_passing_candidate",
        )

    pipeline_ok = is_valid_final_candidate(best_row, compiled_last.acceptance_policy.min_judge_score)
    if open_blend:
        if not pipeline_ok:
            emit_llm_vlm_log(
                run_root,
                "open_blend skipped: pipeline finished with status=no_valid_candidate; "
                "не открываем stale .blend, чтобы не путать с реальным результатом.",
            )
        else:
            blend_to_open = ev._pick_blend_to_open(run_root, best_row)
            if blend_to_open is None:
                emit_llm_vlm_log(
                    run_root,
                    "open_blend skipped: no .blend artefact available "
                    "(ни final/scene.blend, ни best candidate/infinigen_clean_scene.blend).",
                )
            else:
                ev._open_blend_in_blender(blend_to_open, blender_bin=blender_bin, log_run_root=run_root)

    return {
        "run_root": str(run_root),
        "trace_path": str(run_root / "layout_refinement_trace.json"),
        "status": "ok" if is_valid_final_candidate(best_row, compiled_last.acceptance_policy.min_judge_score) else "no_valid_candidate",
    }


def _parse_seeds(s: str) -> list[int]:
    return [int(p.strip()) for p in str(s).split(",") if p.strip()]


def _parse_infinigen_configs(s: str | None) -> list[str] | None:
    if not s or not str(s).strip():
        return None
    return [p.strip() for p in str(s).split(",") if p.strip()]


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM→Infinigen запрос, варианты расстановки, VLM-уточнение (вне run_pipeline)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--policies",
        default=str(default_policies_path()),
        help="YAML политик (по умолчанию рядом с модулем: llm_vlm_scene_policies.yaml)",
    )
    p.add_argument("--llm-provider", default="ollama", choices=["ollama", "none"])
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument(
        "--ollama-model",
        default="qwen2.5:14b",
        help="LLM для генерации Infinigen-запроса и для judge (если --ollama-judge-model не задан).",
    )
    p.add_argument(
        "--ollama-judge-model",
        default=None,
        help="Отдельная модель для judge. Если не задано — используется --ollama-model.",
    )
    p.add_argument("--vlm-model", default=None, help="Например llava; если не задан, VLM-цикл отключён")
    p.add_argument(
        "--vlm-no-fast-render",
        action="store_true",
        help="Не собирать render.png через локальный Blender перед VLM (как раньше: только если PNG уже есть).",
    )
    p.add_argument(
        "--vlm-render-resolution-pct",
        type=int,
        default=42,
        help="Масштаб разрешения превью-рендера для VLM (5–100). Меньше — быстрее. По умолчанию: 42.",
    )
    p.add_argument(
        "--vlm-render-engine",
        default="eevee",
        choices=["eevee", "fast_cycles"],
        help="eevee: быстрый EEVEE; fast_cycles: Cycles ~64 spp + denoise (медленнее, чище). По умолчанию: eevee.",
    )
    p.add_argument(
        "--vlm-render-timeout-sec",
        type=int,
        default=900,
        help="Таймаут subprocess Blender для превью-рендера (сек). По умолчанию: 900.",
    )
    p.add_argument("--layout-seeds", default="0,1,2,3", help="Сиды для разных расстановок (screening)")
    p.add_argument("--max-vlm-rounds", type=int, default=2)
    p.add_argument(
        "--max-inner-repairs",
        type=int,
        default=2,
        help="После screening: число попыток rule-repair (build_repair_plan) до следующего VLM/раунда",
    )
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=22)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)
    p.add_argument("--remote-conda-env", default=None)
    p.add_argument("--remote-infinigen-src", default="/workspace/infinigen/src")
    p.add_argument("--infinigen-src", default=None)
    p.add_argument(
        "--infinigen-task",
        default="coarse",
        help="Аргумент --infinigen-task для run_infinigen_clean на удалённой машине",
    )
    p.add_argument(
        "--infinigen-configs",
        default=None,
        metavar="GINS",
        help="Через запятую .gin файлы (например singleroom.gin). По умолчанию: singleroom.gin,fast_solve.gin. "
        "Для более полного пула мебели попробуйте только singleroom.gin без fast_solve.gin",
    )
    p.add_argument(
        "--remote-cleanup-tmp",
        action="store_true",
        default=False,
        help="Перед запуском удалить старые /workspace/tmp/infinigen_clean_* на удалёнке (требует --remote-host/--remote-user). "
        "Полезно при REMOTE_DISK_FULL на vast.ai.",
    )
    p.add_argument(
        "--open-blend",
        dest="open_blend",
        action="store_true",
        default=True,
        help="По окончании прогона открыть лучший .blend в GUI Blender (без рендера). Default: on.",
    )
    p.add_argument(
        "--no-open-blend",
        dest="open_blend",
        action="store_false",
        help="Отключить авто-открытие .blend по окончании (например, на headless-сервере).",
    )
    p.add_argument(
        "--blender-bin",
        default=None,
        help="Путь к исполняемому Blender. Если не задан — ищем в PATH, BLENDER_PATH/BLENDER_BIN и стандартных местах.",
    )
    p.add_argument(
        "--judge-feedback-llm",
        dest="use_judge_feedback_llm",
        action="store_true",
        default=True,
        help=(
            "После judge передавать его notes/weaknesses в text-LLM и просить пересоставить "
            "infinigen_request с учётом критики (judge как доп-LLM-преобразователь). Default: on."
        ),
    )
    p.add_argument(
        "--no-judge-feedback-llm",
        dest="use_judge_feedback_llm",
        action="store_false",
        help="Отключить LLM-feedback цикл на основе judge.",
    )
    p.add_argument(
        "--judge-feedback-threshold",
        type=float,
        default=6.0,
        help=(
            "Порог judge.total_score, ниже которого вызывается LLM-feedback "
            "(пересборка infinigen_request). Default: 6.0."
        ),
    )
    return p


def main() -> None:
    args = build_cli().parse_args()
    remote_kwargs: dict[str, Any] = {
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "remote_user": args.remote_user,
        "remote_key": args.remote_key,
        "remote_conda_env": args.remote_conda_env,
        "remote_infinigen_src": args.remote_infinigen_src,
        "infinigen_src": args.infinigen_src,
        "infinigen_task": args.infinigen_task,
        "remote_cleanup_tmp": bool(args.remote_cleanup_tmp),
    }
    icf = _parse_infinigen_configs(args.infinigen_configs)
    if icf is not None:
        remote_kwargs["infinigen_configs"] = icf
    run_llm_vlm_layout_refinement(
        prompt_text=args.prompt,
        run_root=Path(args.out_dir),
        policies_path=args.policies,
        llm_provider=args.llm_provider,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_judge_model=args.ollama_judge_model,
        vlm_model=args.vlm_model,
        layout_seeds=_parse_seeds(args.layout_seeds),
        max_vlm_rounds=args.max_vlm_rounds,
        skip_judge=args.skip_judge,
        remote_kwargs=remote_kwargs,
        max_inner_repairs=args.max_inner_repairs,
        open_blend=args.open_blend,
        blender_bin=args.blender_bin,
        use_judge_feedback_llm=args.use_judge_feedback_llm,
        judge_feedback_threshold=args.judge_feedback_threshold,
        vlm_fast_preview_render=not bool(args.vlm_no_fast_render),
        vlm_render_resolution_pct=args.vlm_render_resolution_pct,
        vlm_render_engine=args.vlm_render_engine,
        vlm_render_timeout_sec=args.vlm_render_timeout_sec,
    )


if __name__ == "__main__":
    main()
