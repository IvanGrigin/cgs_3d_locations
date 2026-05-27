#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оценка и обратная связь: VLM по рендеру, judge→repair, merge VLM-патчей, вспомогательный Blender/SSH."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

from src.prompt_compiler.inventory_mapping import normalize_prompt_object
from src.prompt_compiler.llm_client import BaseLLMClient
from src.prompt_compiler.schemas import CompiledPolicy, JudgeResult, RepairPlan, RepairReason

from . import generation as gen
from src.pipeline.llm_vlm_screening import emit_llm_vlm_log, format_timing_dur, truncate_for_log

def _vlm_system_prompt() -> str:
    return (
        "You are a vision-language interior critic. You see a top-down or perspective render of ONE room candidate.\n"
        "Compare the image to the structured brief (room type, style, intended furniture semantics).\n"
        "Return ONLY JSON with: satisfied (bool), critique (short), optional list patches "
        "(add_required_objects, add_desired_objects, add_forbidden_objects, remove_objects as plain English or "
        "semantic names), max_count_overrides (semantic→int), notes_append, next_screening_seeds (integers).\n"
        "Optional infinigen_runtime object (same keys as LLM request): monkeypatch_params "
        "(e.g. furniture_fullness_pct, obj_interior_obj_pct), solver_steps, stage_flags, gin_overrides_append "
        "(strings like compose_indoors.solve_medium_enabled=True), add_factory_blacklist, add_factory_whitelist "
        "(*Factory), remove_optional_semantics, added_required_semantics, updated_max_counts (tighten counts).\n"
        "If the layout is acceptable, set satisfied=true and use empty patches.\n"
        "No markdown."
    )


def _ollama_vision_json(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_paths: list[Path],
    timeout_sec: int = 300,
    temperature: float = 0.0,
    response_json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    images_b64: list[str] = []
    for path in image_paths:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            continue
        images_b64.append(base64.standard_b64encode(p.read_bytes()).decode("ascii"))
    if not images_b64:
        raise FileNotFoundError("no readable images for VLM")
    url = base_url.rstrip("/") + "/api/chat"
    fmt: str | dict[str, Any] = "json"
    if response_json_schema is not None:
        fmt = response_json_schema
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "format": fmt,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text, "images": images_b64},
        ],
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    msg = ((raw.get("message") or {}).get("content") or "").strip()
    if not msg:
        raise RuntimeError("VLM empty content")
    return json.loads(msg)


def merge_infinigen_request_with_vlm(req: dict[str, Any], vlm: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(req)
    furniture = list(merged.get("furniture") or [])

    def _add_rows(objs: list[str], priority: str) -> None:
        for o in objs:
            sem = gen._coerce_semantic(str(o)) or normalize_prompt_object(str(o).lower())
            if not sem:
                continue
            furniture.append({"semantic": sem, "count": 1, "priority": priority})

    _add_rows([str(x) for x in (vlm.get("add_required_objects") or []) if str(x).strip()], "required")
    _add_rows([str(x) for x in (vlm.get("add_desired_objects") or []) if str(x).strip()], "desired")
    forbidden = list(merged.get("forbidden_objects") or [])
    for x in vlm.get("add_forbidden_objects") or []:
        t = str(x).strip()
        if t and t not in forbidden:
            forbidden.append(t)
    merged["forbidden_objects"] = forbidden

    remove = {str(x).strip().lower() for x in (vlm.get("remove_objects") or []) if str(x).strip()}
    if remove:
        furniture = [
            row
            for row in furniture
            if str(row.get("semantic", "")).strip().lower() not in remove
            and (gen._coerce_semantic(str(row.get("semantic", ""))) or "").lower() not in remove
        ]
    merged["furniture"] = furniture
    notes_append = str(vlm.get("notes_append") or "").strip()
    if notes_append:
        merged["notes"] = str(merged.get("notes") or "").strip()
        if merged["notes"]:
            merged["notes"] += "\n"
        merged["notes"] += "[VLM] " + notes_append
    mco = vlm.get("max_count_overrides") or {}
    if isinstance(mco, dict) and mco:
        merged["max_count_overrides_vlm"] = {str(k): int(v) for k, v in mco.items() if str(k)}
    v_rt = vlm.get("infinigen_runtime")
    if isinstance(v_rt, dict) and v_rt:
        merged_rt = dict(merged.get("infinigen_runtime") or {})
        merged["infinigen_runtime"] = gen.merge_runtime_dicts(merged_rt, v_rt)
    return merged


_FAST_SOLVE_GIN = "fast_solve.gin"

_ROOM_TYPE_GIN: dict[str, str] = {
    "Bedroom": "bedroom.gin",
    "Kitchen": "kitchen.gin",
    "Bathroom": "bathroom.gin",
    "LivingRoom": "livingroom.gin",
    "Office": "office.gin",
    "DiningRoom": "diningroom.gin",
}


def _ensure_room_type_gin(
    remote_kwargs: dict[str, Any],
    room_type: str,
    *,
    log_run_root: Path | None = None,
) -> dict[str, Any]:
    """Гарантирует наличие room-specific gin (bedroom.gin/kitchen.gin/...) в infinigen_configs.

    Без него Infinigen-indoor НЕ подключает constraint-граф для конкретной семантики комнаты:
    `home_furniture_constraints` стоит, `apply_greedy_restriction` помечает Variable(room) как
    Bedroom, но primary-asset usage для bedroom не активируется → pool под `restrict_child_primary`
    становится пустым и solve_large сразу выдаёт `No objects to be added`.
    """
    gin = _ROOM_TYPE_GIN.get(str(room_type).strip())
    if not gin:
        return remote_kwargs
    cfgs = list(remote_kwargs.get("infinigen_configs") or ["singleroom.gin"])
    if gin in cfgs:
        return remote_kwargs
    new_cfgs = list(cfgs)
    new_cfgs.append(gin)
    out = dict(remote_kwargs)
    out["infinigen_configs"] = new_cfgs
    if log_run_root is not None:
        emit_llm_vlm_log(
            log_run_root,
            f"infinigen_configs auto-extended for room_type={room_type!r}: +{gin!r} -> {new_cfgs}",
        )
    return out


def _early_failure_reason(result: dict[str, Any]) -> str | None:
    ef = result.get("early_failure")
    if isinstance(ef, dict):
        for key in ("reason", "type", "name", "code"):
            v = ef.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in ef.values():
            if isinstance(v, str) and "missing_required_bed" in v:
                return "missing_required_bed_early"
    return None


def _maybe_disable_fast_solve(
    results: list[dict[str, Any]],
    remote_kwargs: dict[str, Any],
    *,
    log_run_root: Path,
) -> dict[str, Any]:
    """Если все/большинство seed'ов упали с missing_required_bed_early и в configs ещё есть fast_solve.gin —
    выкидываем fast_solve.gin для следующих screening-итераций. Возвращает (возможно) обновлённый remote_kwargs.
    """
    if not results:
        return remote_kwargs
    bed_misses = sum(1 for r in results if (_early_failure_reason(r) or "").startswith("missing_required_bed"))
    if bed_misses == 0:
        return remote_kwargs
    cfgs = list(remote_kwargs.get("infinigen_configs") or ["singleroom.gin", _FAST_SOLVE_GIN])
    if _FAST_SOLVE_GIN not in cfgs:
        emit_llm_vlm_log(
            log_run_root,
            f"early-fail bed_misses={bed_misses}/{len(results)}; fast_solve уже отключён, пропускаем переключение",
        )
        return remote_kwargs
    new_cfgs = [c for c in cfgs if c != _FAST_SOLVE_GIN]
    if not new_cfgs:
        new_cfgs = ["singleroom.gin"]
    out = dict(remote_kwargs)
    out["infinigen_configs"] = new_cfgs
    emit_llm_vlm_log(
        log_run_root,
        f"early-fail bed_misses={bed_misses}/{len(results)}; отключаю '{_FAST_SOLVE_GIN}' "
        f"для следующих screening-итераций. infinigen_configs={new_cfgs}",
    )
    return out


_DEFAULT_BLENDER_CANDIDATES: tuple[str, ...] = (
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/Applications/Blender 4.2.app/Contents/MacOS/Blender",
    "/Applications/Blender 4.1.app/Contents/MacOS/Blender",
    "/Applications/Blender 4.0.app/Contents/MacOS/Blender",
    "/Applications/Blender 3.6.app/Contents/MacOS/Blender",
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    "/snap/bin/blender",
)


def _resolve_blender_binary(explicit: str | None) -> str | None:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
    env_path = os.environ.get("BLENDER_PATH") or os.environ.get("BLENDER_BIN")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
    which = shutil.which("blender")
    if which:
        return which
    for cand in _DEFAULT_BLENDER_CANDIDATES:
        p = Path(cand)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def _open_blend_in_blender(
    blend_path: Path,
    *,
    blender_bin: str | None,
    log_run_root: Path | None = None,
) -> None:
    """Открывает указанный .blend в GUI Blender (без блокировки/рендера).

    Никогда не падает: если Blender не найден или запуск не удался — просто пишет warning в лог.
    """
    blend_path = Path(blend_path).expanduser().resolve()
    if not blend_path.is_file():
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                f"open_blend skipped: .blend not found at {blend_path}",
            )
        return
    binary = _resolve_blender_binary(blender_bin)
    if not binary:
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                "open_blend skipped: Blender executable not found. "
                "Установите Blender или укажите путь через --blender-bin / BLENDER_PATH.",
            )
        return
    try:
        # Detach: не блокируем CLI; вывод blender уходит в /dev/null.
        creationflags = 0
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform.startswith("win"):
            creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            kwargs["creationflags"] = creationflags
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([binary, str(blend_path)], **kwargs)
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                f"open_blend: launched Blender GUI binary={binary!r} file={blend_path}",
            )
    except Exception as exc:
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                f"open_blend failed: {exc!r} (binary={binary!r}, file={blend_path})",
            )


def _cleanup_remote_tmp(
    *,
    remote_host: str,
    remote_user: str,
    remote_port: int | None,
    remote_key: str | None,
    log_run_root: Path | None,
) -> bool:
    """Удаляет старые /workspace/tmp/infinigen_clean_* на удалёнке и логирует df -h.

    Безопасный паттерн (только infinigen_clean_*) — другие файлы не трогаем.
    Возвращает True, если хотя бы df -h вернулся успешно; иначе False (и пишет warning).
    """
    base_ssh = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if remote_port:
        base_ssh += ["-p", str(int(remote_port))]
    if remote_key:
        base_ssh += ["-i", str(Path(remote_key).expanduser())]
    base_ssh.append(f"{remote_user}@{remote_host}")
    cleanup_cmd = base_ssh + [
        "bash -lc 'set -e; "
        "rm -rf /workspace/tmp/infinigen_clean_* 2>/dev/null || true; "
        "df -h /workspace 2>/dev/null || true'"
    ]
    try:
        completed = subprocess.run(
            cleanup_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                "remote_cleanup_tmp: timeout (120s); удалёнка не отвечает.",
            )
        return False
    except Exception as exc:
        if log_run_root is not None:
            emit_llm_vlm_log(log_run_root, f"remote_cleanup_tmp: error {exc!r}")
        return False
    out = (completed.stdout or "").strip()
    err = (completed.stderr or "").strip()
    if completed.returncode == 0:
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                f"remote_cleanup_tmp: removed /workspace/tmp/infinigen_clean_*; df:\n{out}",
            )
        return True
    if log_run_root is not None:
        emit_llm_vlm_log(
            log_run_root,
            f"remote_cleanup_tmp: returncode={completed.returncode} stderr={err[:300]!r} stdout={out[:300]!r}",
        )
    return False


def _pick_blend_to_open(
    run_root: Path,
    best_row: dict[str, Any] | None,
) -> Path | None:
    """Выбирает .blend для открытия. Приоритет:
    1) run_root/final/scene.blend (создан materialize_final при успешном кандидате),
    2) best_row['candidate_dir']/infinigen_clean_scene.blend (даже если final пуст),
    3) любой *.blend в screening-каталогах самого высокого combined_score.
    """
    final_blend = run_root / "final" / "scene.blend"
    if final_blend.is_file():
        return final_blend
    if best_row is not None:
        cand_dir: Path = best_row["candidate_dir"]
        cand_blend = cand_dir / "infinigen_clean_scene.blend"
        if cand_blend.is_file():
            return cand_blend
    return None


# --------------------------------------------------------------------------- #
# Judge-driven repair extension and LLM feedback loop
# --------------------------------------------------------------------------- #

# Ключевые слова в judge.notes/weaknesses → фабрики, которые судья просит «убрать».
_JUDGE_FACTORY_BLACKLIST_HINTS: dict[str, str] = {
    "floorlamp": "FloorLampFactory",
    "floor lamp": "FloorLampFactory",
    "напольн": "FloorLampFactory",
    "torchere": "FloorLampFactory",
    "tall lamp": "FloorLampFactory",
    "large shelf": "LargeShelfFactory",
    "shelf": "LargeShelfFactory",
    "bookcase": "SimpleBookcaseFactory",
    "bookshelf": "SimpleBookcaseFactory",
    "cell shelf": "CellShelfFactory",
    "kitchen cabinet": "KitchenCabinetFactory",
    "wardrobe": "WardrobeFactory",
    "tv": "TVFactory",
    "telly": "TVFactory",
}

_JUDGE_EMPTY_HINTS = ("empty", "sparse", "пуст", "недостаточно", "bare", "too few")
_JUDGE_CLUTTER_HINTS = ("clutter", "крайне много", "слишком много", "overcrowd", "too many", "busy")


def _judge_text_blob(judge: JudgeResult) -> str:
    parts = [judge.notes or ""]
    parts.extend(judge.weaknesses or [])
    return " | ".join(p for p in parts if p).lower()


def _extend_repair_plan_with_judge(
    rp: RepairPlan,
    judge: JudgeResult | None,
    compiled: CompiledPolicy,
    *,
    log_run_root: Path | None = None,
) -> None:
    """Расширяет уже построенный RepairPlan эвристиками на основе judge-оценок.

    Мутирует rp inplace. Не использует build_gin_overrides — только добавляет
    actions/reasons/updated_*/added_* поля, которые потом подхватит apply_repair_plan.
    """
    if judge is None:
        return

    text = _judge_text_blob(judge)
    cur_fullness = float(
        compiled.infinigen_policy.monkeypatch_params.get("furniture_fullness_pct", 0.5)
    )
    new_fullness = cur_fullness
    judge_changes: list[str] = []

    if judge.composition_score and judge.composition_score < 5.0:
        clutter_signal = any(k in text for k in _JUDGE_CLUTTER_HINTS)
        if clutter_signal:
            rp.reasons.append(RepairReason.TOO_CLUTTERED)
            new_fullness = max(0.32, new_fullness - 0.05)
            judge_changes.append(
                f"composition_score={judge.composition_score:.1f} + clutter signal → fullness -=0.05"
            )
        else:
            rp.reasons.append(RepairReason.TOO_EMPTY)
            new_fullness = min(0.78, new_fullness + 0.05)
            judge_changes.append(
                f"composition_score={judge.composition_score:.1f} (no clutter) → fullness +=0.05"
            )
        if "compose_indoors.solve_medium_enabled=True" not in rp.added_gin_overrides:
            rp.added_gin_overrides.append("compose_indoors.solve_medium_enabled=True")
            judge_changes.append("composition: enable solve_medium")

    if judge.functionality_score and judge.functionality_score < 5.0:
        rp.reasons.append(RepairReason.BAD_AREA_PROGRAM_FIT)
        soft_whitelist = [
            "BedFactory",
            "ChairFactory",
            "ArmChairFactory",
            "CoffeeTableFactory",
            "DiningTableFactory",
            "CeilingLightFactory",
            "LampFactory",
            "SingleCabinetFactory",
            "SideTableFactory",
        ]
        added = [f for f in soft_whitelist if f not in rp.added_factory_whitelist]
        rp.added_factory_whitelist.extend(added)
        if added:
            judge_changes.append(
                f"functionality_score={judge.functionality_score:.1f} → widen whitelist +{added}"
            )

    if judge.style_match_score and judge.style_match_score < 5.0:
        if RepairReason.STYLE_NOT_READABLE not in rp.reasons:
            rp.reasons.append(RepairReason.STYLE_NOT_READABLE)
        style_factories = list(compiled.style_policy.factory_whitelist)[:4]
        added = [f for f in style_factories if f not in rp.added_factory_whitelist]
        rp.added_factory_whitelist.extend(added)
        if added:
            judge_changes.append(
                f"style_match={judge.style_match_score:.1f} → +style whitelist {added}"
            )

    if any(k in text for k in _JUDGE_EMPTY_HINTS):
        if RepairReason.TOO_EMPTY not in rp.reasons:
            rp.reasons.append(RepairReason.TOO_EMPTY)
        new_fullness = min(0.78, max(new_fullness, cur_fullness) + 0.05)
        judge_changes.append("notes/weaknesses mention 'empty' → fullness +=0.05")

    if any(k in text for k in _JUDGE_CLUTTER_HINTS) and RepairReason.TOO_CLUTTERED not in rp.reasons:
        rp.reasons.append(RepairReason.TOO_CLUTTERED)
        new_fullness = max(0.32, min(new_fullness, cur_fullness) - 0.05)
        judge_changes.append("notes/weaknesses mention 'clutter' → fullness -=0.05")

    for kw, factory in _JUDGE_FACTORY_BLACKLIST_HINTS.items():
        if kw in text and factory not in rp.added_factory_blacklist:
            rp.added_factory_blacklist.append(factory)
            judge_changes.append(f"hint {kw!r} → blacklist {factory}")

    # Судья жалуется на «Storage» — по запросу пользователя расширяем blacklist
    # типичных storage-фабрик. Если рядом «lack»/«missing»/«not enough» —
    # вместо blacklisting добавляем Storage в required.
    if "storage" in text:
        lack_signal = any(w in text for w in ("lack", "missing", "not enough", "недост", "мало"))
        if lack_signal:
            if "Storage" not in rp.added_required_semantics:
                rp.added_required_semantics.append("Storage")
                judge_changes.append("storage critique + lack signal → require Storage")
        else:
            storage_factories = [
                "LargeShelfFactory",
                "CellShelfFactory",
                "SimpleBookcaseFactory",
            ]
            added = [f for f in storage_factories if f not in rp.added_factory_blacklist]
            rp.added_factory_blacklist.extend(added)
            if added:
                judge_changes.append(f"storage critique → blacklist {added}")

    if abs(new_fullness - cur_fullness) > 1e-6:
        rp.updated_monkeypatch_params["furniture_fullness_pct"] = round(new_fullness, 3)
        judge_changes.append(
            f"furniture_fullness_pct {cur_fullness:.3f} -> {new_fullness:.3f}"
        )

    if judge_changes:
        rp.actions.extend(f"judge: {c}" for c in judge_changes)
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                "judge-driven repair extension: " + "; ".join(judge_changes),
            )


_JUDGE_FEEDBACK_SYSTEM_PROMPT = """Ты — апскейлер 3D-сцены.
Тебе дают: (1) предыдущий infinigen_request, (2) оценки судьи (functionality,
prompt_match, style_match, composition + strengths/weaknesses/notes), (3)
свободу сделать «человеческое» исправление.

Верни тот же JSON-объект infinigen_request, но с правками:
- убери явные жалобы судьи (плохие фабрики в forbidden_objects / факт. blacklist),
- если scene «пустая» — добавь furniture-позиции (бытовая логика для room_type),
- если scene «переполнена» — убери лишние desired-позиции,
- если style_match низкий — уточни style_raw / favorite_colors / material_family,
- держи schema без изменений; semantics — только из supported-списка.

ВАЖНО: НЕ пиши новые solver_steps/stage_flags/gin_overrides (этим управляет
правило-репейр). Меняй только семантику и стиль."""


def _propose_request_refinement_via_judge_feedback(
    *,
    current_req: dict[str, Any],
    judge: JudgeResult | None,
    gate_summary: dict[str, Any],
    llm: BaseLLMClient | None,
    threshold: float = 6.0,
    log_run_root: Path | None = None,
) -> dict[str, Any] | None:
    """Спрашивает text-LLM пересобрать infinigen_request с учётом критики судьи.

    Возвращает новый request dict или None если LLM-feedback пропущен/упал.
    Срабатывает только если judge.total_score < threshold (есть что фиксить).
    """
    if llm is None or judge is None:
        return None
    if judge.total_score is None:
        return None
    if float(judge.total_score) >= float(threshold):
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                f"judge_feedback_llm skipped: total_score={judge.total_score:.2f} >= {threshold}",
            )
        return None

    user_payload = json.dumps(
        {
            "previous_request": current_req,
            "judge": {
                "total_score": judge.total_score,
                "functionality_score": judge.functionality_score,
                "prompt_match_score": judge.prompt_match_score,
                "style_match_score": judge.style_match_score,
                "composition_score": judge.composition_score,
                "strengths": list(judge.strengths or []),
                "weaknesses": list(judge.weaknesses or []),
                "notes": judge.notes or "",
            },
            "rule_gate": gate_summary,
            "supported_semantics": gen._ALLOWED_SEMANTICS,
        },
        ensure_ascii=False,
        indent=2,
    )
    if log_run_root is not None:
        emit_llm_vlm_log(
            log_run_root,
            f"judge_feedback_llm: total_score={judge.total_score:.2f} < {threshold}, calling LLM\n"
            f"feedback user payload (truncated):\n{truncate_for_log(user_payload, 2200)}",
        )

    t_fb0 = perf_counter()
    try:
        refined = llm.complete_json(
            _JUDGE_FEEDBACK_SYSTEM_PROMPT,
            user_payload,
            gen._infinigen_request_schema(),
        )
    except Exception as exc:
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                f"[timing] judge_feedback_llm wall={format_timing_dur(perf_counter() - t_fb0)} (error {type(exc).__name__})",
            )
            emit_llm_vlm_log(
                log_run_root,
                f"judge_feedback_llm failed: {type(exc).__name__}: {exc!r}",
            )
        return None
    if log_run_root is not None:
        emit_llm_vlm_log(
            log_run_root,
            f"[timing] judge_feedback_llm complete_json wall={format_timing_dur(perf_counter() - t_fb0)}",
        )
    if not isinstance(refined, dict) or not refined.get("furniture"):
        if log_run_root is not None:
            emit_llm_vlm_log(log_run_root, "judge_feedback_llm: degenerate response — skip")
        return None
    refined.setdefault("infinigen_runtime", current_req.get("infinigen_runtime", {}) or {})
    if log_run_root is not None:
        emit_llm_vlm_log(
            log_run_root,
            f"judge_feedback_llm produced refined request (truncated):\n"
            f"{truncate_for_log(refined, 2200)}",
        )
    return refined

