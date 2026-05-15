#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генерация и санитизация Infinigen-запроса: LLM JSON, intent↔request, runtime-блок для compile."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.prompt_compiler.compile_to_infinigen import build_gin_overrides
from src.prompt_compiler.inventory_mapping import (
    PRIMARY_SEMANTICS,
    SECONDARY_SEMANTICS,
    normalize_prompt_object,
    semantic_to_factory_family,
)
from src.prompt_compiler.llm_client import BaseLLMClient, StubLLMClient
from src.prompt_compiler.prompt_to_intent import extract_intent, normalize_intent
from src.prompt_compiler.schemas import (
    CompiledPolicy,
    DecorRichness,
    DensityLevel,
    ObjectsIntent,
    OpeningsIntent,
    PreferencesIntent,
    PromptIntent,
    RepairPlan,
    RoomType,
    StyleIntent,
    StyleLabel,
)
from src.scene_quality.repair_loop import apply_repair_plan

from src.pipeline.llm_vlm_screening import emit_llm_vlm_log

_ALLOWED_SEMANTICS = sorted((PRIMARY_SEMANTICS | SECONDARY_SEMANTICS) - {""})


# Семантики, известные intent-у, но без фабрик в SEMANTIC_TO_ALLOWED_FACTORIES.
# Без даунгрейда compile_prompt_intent падает с empty_candidate_pool_before_solve.
# Не трогая shared inventory_mapping.py, делаем «мягкий» маппинг на ближайший supported.
_SEMANTIC_DOWNGRADE: dict[str, str | None] = {
    "LoungeSeating": "Chair",
    "Seating": "Chair",
    "Sofa": "Chair",
    "Couch": "Chair",
    "KitchenAppliance": "Storage",
    "KitchenCounter": "Storage",
    "Furniture": None,  # совсем общий → удаляем строку из furniture
    "WallDecoration": None,  # secondary decor — solver добавит сам
}


def _downgrade_unsupported_semantic(sem: str) -> str | None:
    """Возвращает supported-семантик для intent / compile.

    Правила:
      - если есть прямой даунгрейд в _SEMANTIC_DOWNGRADE → берём его (None → удалить);
      - если для семантики есть фабрика в SEMANTIC_TO_ALLOWED_FACTORIES → возвращаем как есть;
      - иначе пробуем normalize_prompt_object (lower-case lookup);
      - иначе возвращаем None.
    """
    s = str(sem or "").strip()
    if not s:
        return None
    if s in _SEMANTIC_DOWNGRADE:
        return _SEMANTIC_DOWNGRADE[s]
    if semantic_to_factory_family(s):
        return s
    nm = normalize_prompt_object(s.lower())
    if nm and semantic_to_factory_family(nm):
        return nm
    return None


def _sanitize_request_furniture(req: dict[str, Any], *, log_run_root: Path | None = None) -> dict[str, Any]:
    """Прогоняет infinigen_request.furniture через _downgrade_unsupported_semantic.

    Удаляет дубли, выкидывает позиции, у которых нет ни даунгрейда, ни фабрик.
    Также чистит infinigen_runtime.added_required_semantics/remove_optional_semantics/add_factory_*
    от мусора, если LLM прислал что-то странное.
    """
    if not isinstance(req, dict):
        return req
    furniture = list(req.get("furniture") or [])
    changes: list[str] = []
    new_furniture: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in furniture:
        if not isinstance(row, dict):
            continue
        sem = str(row.get("semantic") or "").strip()
        if not sem:
            continue
        coerced = _downgrade_unsupported_semantic(sem)
        if coerced is None:
            changes.append(f"drop {sem!r}")
            continue
        if coerced != sem:
            changes.append(f"{sem!r}->{coerced!r}")
        prio = str(row.get("priority") or "desired").strip().lower() or "desired"
        key = (coerced, prio)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        new_row = dict(row)
        new_row["semantic"] = coerced
        new_row["priority"] = prio
        new_furniture.append(new_row)
    req["furniture"] = new_furniture
    rt = req.get("infinigen_runtime")
    if isinstance(rt, dict):
        for list_key in ("added_required_semantics", "remove_optional_semantics"):
            seq = rt.get(list_key)
            if not isinstance(seq, list):
                continue
            cleaned: list[str] = []
            for item in seq:
                coerced = _downgrade_unsupported_semantic(str(item))
                if coerced and coerced not in cleaned:
                    cleaned.append(coerced)
                elif coerced is None and str(item).strip():
                    changes.append(f"runtime.{list_key}: drop {item!r}")
            rt[list_key] = cleaned
        mc = rt.get("updated_max_counts")
        if isinstance(mc, dict):
            cleaned_mc: dict[str, int] = {}
            for k, v in mc.items():
                coerced = _downgrade_unsupported_semantic(str(k))
                if coerced is None:
                    changes.append(f"runtime.updated_max_counts: drop key {k!r}")
                    continue
                try:
                    cleaned_mc[coerced] = int(v)
                except Exception:
                    continue
            rt["updated_max_counts"] = cleaned_mc
        req["infinigen_runtime"] = rt
    if changes and log_run_root is not None:
        emit_llm_vlm_log(log_run_root, f"sanitize_request_furniture: {changes}")
    return req


def _infinigen_runtime_schema() -> dict[str, Any]:
    """LLM может задавать только семантические правки.
    Низкоуровневые поля (solver_steps / stage_flags / gin_overrides_append / monkeypatch_params)
    исключены: они зависят от конкретной версии Infinigen и легко ломают запуск.
    Их по-прежнему может присылать VLM (там они санитизируются и валидируются).
    """
    return {
        "type": "object",
        "properties": {
            "add_factory_blacklist": {"type": "array", "items": {"type": "string"}},
            "add_factory_whitelist": {"type": "array", "items": {"type": "string"}},
            "remove_optional_semantics": {"type": "array", "items": {"type": "string"}},
            "added_required_semantics": {"type": "array", "items": {"type": "string"}},
            "updated_max_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
        },
        "additionalProperties": False,
    }


def _infinigen_request_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "room_type": {"type": "string", "enum": [x.value for x in RoomType]},
            "style_label": {"type": "string", "enum": [x.value for x in StyleLabel]},
            "style_raw": {"type": "string"},
            "target_area_sqm": {"type": "number"},
            "width_m": {"type": "number"},
            "depth_m": {"type": "number"},
            "height_m": {"type": "number"},
            "density": {"type": "string", "enum": [x.value for x in DensityLevel]},
            "decor_richness": {"type": "string", "enum": [x.value for x in DecorRichness]},
            "furniture": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "semantic": {"type": "string"},
                        "count": {"type": "integer"},
                        "priority": {"type": "string", "enum": ["required", "desired"]},
                    },
                    "required": ["semantic", "count", "priority"],
                    "additionalProperties": False,
                },
            },
            "forbidden_objects": {"type": "array", "items": {"type": "string"}},
            "favorite_colors": {"type": "array", "items": {"type": "string"}},
            "avoid_colors": {"type": "array", "items": {"type": "string"}},
            "material_family": {"type": "array", "items": {"type": "string"}},
            "palette_hint": {"type": "array", "items": {"type": "string"}},
            "wants_door": {"type": "boolean"},
            "wants_window": {"type": "boolean"},
            "notes": {"type": "string"},
            "infinigen_runtime": _infinigen_runtime_schema(),
        },
        "required": ["room_type", "style_label", "furniture"],
        "additionalProperties": False,
    }


def _system_prompt_infinigen_request() -> str:
    allowed = ", ".join(_ALLOWED_SEMANTICS[:80]) + ("…" if len(_ALLOWED_SEMANTICS) > 80 else "")
    return (
        "You are an interior planner for a procedural room generator (Infinigen).\n"
        "From the user's natural-language prompt, produce ONE JSON object describing:\n"
        "- room_type and style_label (must match enums),\n"
        "- optional geometry and openness,\n"
        "- furniture: list of items with semantic (canonical English furniture/room semantic), "
        "integer count>=0, priority required|desired.\n"
        "Use only plausible semantics from this vocabulary when possible: "
        f"{allowed}\n"
        "If unsure, prefer coarse groups (Bed, Storage, Chair, Table, Lighting, Rug, …).\n"
        "Style fields (density, decor_richness) should reflect the user's prompt.\n"
        "Optional infinigen_runtime is for SEMANTIC adjustments only: "
        "add_factory_blacklist / add_factory_whitelist (*Factory names), "
        "remove_optional_semantics / added_required_semantics (canonical semantics), "
        "updated_max_counts (semantic -> integer). "
        "Do NOT set solver_steps, stage_flags, gin_overrides_append, monkeypatch_params here — "
        "those are managed by the pipeline.\n"
        "No markdown. JSON only."
    )


def _coerce_semantic(raw: str) -> str | None:
    s = str(raw or "").strip()
    if not s:
        return None
    if s in PRIMARY_SEMANTICS or s in SECONDARY_SEMANTICS:
        return s
    mapped = normalize_prompt_object(s.lower())
    if mapped and (mapped in PRIMARY_SEMANTICS or mapped in SECONDARY_SEMANTICS):
        return mapped
    compact = s.replace(" ", "").replace("_", "")
    for cand in _ALLOWED_SEMANTICS:
        if cand.replace("_", "").lower() == compact.lower():
            return cand
    return None


_BEDROOM_MIN_FACTORY_WHITELIST = (
    "BedFactory",
    "CeilingLightFactory",
    "LampFactory",
    "SideTableFactory",
    "SingleCabinetFactory",
)
_ROOM_MIN_FACTORY_WHITELIST: dict[str, tuple[str, ...]] = {
    "Bedroom": _BEDROOM_MIN_FACTORY_WHITELIST,
}


def _disable_child_restrictions_in_compiled(
    compiled: CompiledPolicy,
    *,
    log_run_root: Path | None = None,
) -> CompiledPolicy:
    """Отключает `restrict_solving.restrict_child_primary` для Infinigen-indoor.

    Внутри Infinigen `restrict_solving` делает `stages[k] = d.intersection(Domain(restrict_child_primary))`,
    но фабрики в `home_furniture_constraints` имеют многомерные tags (Bed+LargeFurniture+Wood+...),
    а наш список — лишь `{Bed, Lighting, Storage, SideTable, CeilingLight}`. Пересечение пустое →
    `No objects to be added` для всех слотов. Безопаснее опираться на factory_whitelist/blacklist,
    а ограничивать parent_rooms / solve_max_rooms (они не схлопывают pool).
    """
    was_enabled = bool(compiled.preflight.get("apply_child_restrictions"))
    if was_enabled:
        compiled.preflight["apply_child_restrictions"] = False
        if log_run_root is not None:
            emit_llm_vlm_log(
                log_run_root,
                "apply_child_restrictions disabled (override): "
                "restrict_solving.restrict_child_primary intersects домен в ноль "
                "при многомерных тегах фабрик; полагаемся на factory_whitelist.",
            )
    compiled.preflight["final_restrict_child_primary"] = []
    compiled.preflight["final_restrict_child_secondary"] = []
    overrides_in = list(compiled.infinigen_policy.gin_overrides or [])
    overrides_out = [
        ov for ov in overrides_in
        if "restrict_solving.restrict_child_primary" not in ov
        and "restrict_solving.restrict_child_secondary" not in ov
    ]
    if len(overrides_out) != len(overrides_in):
        compiled.infinigen_policy.gin_overrides = overrides_out
        # Логируем только при первом «снятии» (когда _force_min_factory_whitelist_in_compiled
        # отказывается возвращать restrict_solving обратно — этот блок выполняется один раз).
        if was_enabled and log_run_root is not None:
            removed = [ov for ov in overrides_in if ov not in overrides_out]
            emit_llm_vlm_log(
                log_run_root,
                f"removed gin_overrides: {removed}",
            )
    return compiled


def _force_min_factory_whitelist_in_compiled(
    compiled: CompiledPolicy,
    *,
    log_run_root: Path | None = None,
) -> CompiledPolicy:
    """Гарантирует, что во всех `factory_whitelist` (program/style/acceptance/infinigen)
    остаётся минимально-достаточный набор фабрик для room_type.

    `apply_repair_plan` в `scene_quality.repair_loop` собирает whitelist через **пересечение**
    нескольких политик, и через 2-3 итерации теряет важные фабрики (например, SideTableFactory).
    Мы здесь делаем UNION, не трогая старый repair_loop.
    """
    room_type = str(compiled.geometry.room_type.value)
    minimal = _ROOM_MIN_FACTORY_WHITELIST.get(room_type)
    if not minimal:
        return compiled
    added: list[str] = []
    for target in (
        compiled.program,
        compiled.style_policy,
        compiled.acceptance_policy,
        compiled.infinigen_policy,
    ):
        existing = list(getattr(target, "factory_whitelist", []) or [])
        for f in minimal:
            if f not in existing:
                existing.append(f)
                added.append(f)
        target.factory_whitelist = existing
    compiled.preflight["effective_factory_whitelist_count"] = len(compiled.program.factory_whitelist)
    try:
        rebuilt = build_gin_overrides(compiled)
        # Если ребятки выше отключили child-restrictions (apply_child_restrictions=False),
        # build_gin_overrides всё равно пересоберёт restrict_solving.restrict_child_primary/secondary
        # из исходного intent — выкидываем их, чтобы не оживляли мёртвое ограничение и не плодили
        # дубль логов в _disable_child_restrictions_in_compiled.
        if not compiled.preflight.get("apply_child_restrictions", True):
            rebuilt = [
                ov for ov in rebuilt
                if "restrict_solving.restrict_child_primary" not in ov
                and "restrict_solving.restrict_child_secondary" not in ov
            ]
        compiled.infinigen_policy.gin_overrides = sorted(
            set((compiled.infinigen_policy.gin_overrides or []) + rebuilt)
        )
    except Exception:
        pass
    if added and log_run_root is not None:
        emit_llm_vlm_log(
            log_run_root,
            f"min_factory_whitelist UNION для room_type={room_type!r}: "
            f"+{sorted(set(added))} (поверх pipeline-intersection)",
        )
    return compiled


def _ensure_min_factory_whitelist(req: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Гарантирует наличие минимально-достаточного набора фабрик для room_type.

    Иначе solver Infinigen бракует целые слоты по coverage (см. restrict_child_primary).
    """
    if not isinstance(req, dict):
        return req, []
    room_type = str(req.get("room_type") or "").strip()
    minimal = _ROOM_MIN_FACTORY_WHITELIST.get(room_type)
    if not minimal:
        return req, []
    out = dict(req)
    rt = dict(out.get("infinigen_runtime") or {})
    existing = [str(x).strip() for x in (rt.get("add_factory_whitelist") or []) if str(x).strip()]
    added: list[str] = []
    for f in minimal:
        if f not in existing:
            existing.append(f)
            added.append(f)
    rt["add_factory_whitelist"] = existing
    out["infinigen_runtime"] = rt
    return out, added


_AREA_REGEXES = (
    re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:м\s*²|м\s*\^?\s*2|кв\.?\s*м|m\s*²|m\s*\^?\s*2|sqm|sq\.?\s*m)", re.IGNORECASE),
    re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:квадрат(?:ных)?\s*метров?|квадратов)", re.IGNORECASE),
)


def parse_area_sqm_from_prompt(prompt_text: str) -> float | None:
    """Возвращает площадь в м² из русского/английского текста («15 м²», «15 кв. м», «15 sqm»…)."""
    text = str(prompt_text or "")
    for rgx in _AREA_REGEXES:
        m = rgx.search(text)
        if not m:
            continue
        raw = m.group(1).replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        if 1.0 <= value <= 1000.0:
            return value
    return None


def _ensure_area_in_request(prompt_text: str, req: dict[str, Any]) -> tuple[dict[str, Any], float | None]:
    """Если в запросе нет target_area_sqm, пытаемся вытащить из текста промпта.
    Возвращает (req, area_used_or_none)."""
    if not isinstance(req, dict):
        return req, None
    existing = req.get("target_area_sqm")
    try:
        existing_val = float(existing) if existing is not None else None
    except Exception:
        existing_val = None
    if existing_val and existing_val > 0:
        return req, existing_val
    area = parse_area_sqm_from_prompt(prompt_text)
    if area is None:
        return req, None
    out = dict(req)
    out["target_area_sqm"] = float(area)
    return out, float(area)


def propose_infinigen_request_llm(*, prompt_text: str, llm: BaseLLMClient | None) -> dict[str, Any]:
    """LLM: промпт → структурированный запрос (мебель + стиль). При отсутствии LLM — извлечение intent как fallback."""
    if llm is None:
        intent = extract_intent(prompt_text, StubLLMClient({}))
        return _request_dict_from_prompt_intent(intent)
    user = json.dumps(
        {
            "user_prompt": prompt_text,
            "allowed_semantics_sample": _ALLOWED_SEMANTICS,
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        out = llm.complete_json(_system_prompt_infinigen_request(), user, _infinigen_request_schema())
        if isinstance(out, dict):
            out.setdefault("infinigen_runtime", {})
        return out if isinstance(out, dict) else _request_dict_from_prompt_intent(extract_intent(prompt_text, StubLLMClient({})))
    except Exception:
        intent = extract_intent(prompt_text, StubLLMClient({}))
        return _request_dict_from_prompt_intent(intent)


def _request_dict_from_prompt_intent(intent: PromptIntent) -> dict[str, Any]:
    furniture: list[dict[str, Any]] = []
    for sem in intent.objects.required:
        coerced = _coerce_semantic(sem) or sem
        furniture.append({"semantic": coerced, "count": 1, "priority": "required"})
    for sem in intent.objects.desired:
        coerced = _coerce_semantic(sem) or sem
        furniture.append({"semantic": coerced, "count": 1, "priority": "desired"})
    return {
        "room_type": intent.room_type.value,
        "style_label": (intent.style.style_label or StyleLabel.JAPANDI).value,
        "style_raw": intent.style.style_raw or "",
        "target_area_sqm": intent.geometry.target_area_sqm,
        "width_m": intent.geometry.width_m,
        "depth_m": intent.geometry.depth_m,
        "height_m": intent.geometry.height_m,
        "density": (intent.style.density or DensityLevel.LOW).value,
        "decor_richness": (intent.style.decor_richness or DecorRichness.LOW).value,
        "furniture": furniture or [{"semantic": "Bed", "count": 1, "priority": "required"}],
        "forbidden_objects": list(intent.objects.forbidden),
        "favorite_colors": list(intent.preferences.favorite_colors),
        "avoid_colors": list(intent.preferences.avoid_colors),
        "material_family": list(intent.style.material_family),
        "palette_hint": list(intent.style.palette_hint),
        "wants_door": intent.openings.wants_door,
        "wants_window": intent.openings.wants_window,
        "notes": intent.preferences.notes or "heuristic_fallback_request",
        "infinigen_runtime": {},
    }


def _style_label_from_str(value: str | None) -> StyleLabel | None:
    if not value:
        return None
    v = str(value).strip().lower().replace("-", "_")
    for sl in StyleLabel:
        if sl.value == v:
            return sl
    return None


def _room_type_from_str(value: str | None) -> RoomType:
    if not value:
        return RoomType.BEDROOM
    v = str(value).strip()
    for rt in RoomType:
        if rt.value == v:
            return rt
    return RoomType.BEDROOM


def _density_from_str(value: str | None) -> DensityLevel | None:
    if not value:
        return None
    v = str(value).strip().lower()
    for d in DensityLevel:
        if d.value == v:
            return d
    return None


def _decor_from_str(value: str | None) -> DecorRichness | None:
    if not value:
        return None
    v = str(value).strip().lower()
    for d in DecorRichness:
        if d.value == v:
            return d
    return None


def prompt_intent_from_infinigen_request(*, prompt_text: str, req: dict[str, Any]) -> PromptIntent:
    required: list[str] = []
    desired: list[str] = []
    max_hint: dict[str, int] = {}
    for row in req.get("furniture") or []:
        if not isinstance(row, dict):
            continue
        sem = _coerce_semantic(str(row.get("semantic", "")))
        if not sem:
            continue
        try:
            count = max(0, int(row.get("count", 1)))
        except Exception:
            count = 1
        pri = str(row.get("priority", "desired")).lower()
        max_hint[sem] = max(max_hint.get(sem, 0), count)
        if count <= 0:
            continue
        if pri == "required":
            if sem not in required:
                required.append(sem)
        else:
            if sem not in desired and sem not in required:
                desired.append(sem)
    forbidden = [str(x).strip() for x in (req.get("forbidden_objects") or []) if str(x).strip()]
    notes = str(req.get("notes") or "").strip()
    if max_hint:
        notes = (notes + "\n" if notes else "") + "max_count_hints: " + json.dumps(max_hint, ensure_ascii=False)

    intent = PromptIntent(
        prompt_text=prompt_text,
        room_type=_room_type_from_str(req.get("room_type")),
        geometry={
            "target_area_sqm": req.get("target_area_sqm"),
            "width_m": req.get("width_m"),
            "depth_m": req.get("depth_m"),
            "height_m": req.get("height_m") or 2.7,
        },
        style=StyleIntent(
            style_label=_style_label_from_str(req.get("style_label")),
            style_raw=str(req.get("style_raw") or req.get("style_label") or ""),
            density=_density_from_str(req.get("density")),
            decor_richness=_decor_from_str(req.get("decor_richness")),
            palette_hint=[str(x).strip() for x in (req.get("palette_hint") or []) if str(x).strip()],
            material_family=[str(x).strip() for x in (req.get("material_family") or []) if str(x).strip()],
        ),
        openings=OpeningsIntent(
            wants_door=bool(req.get("wants_door", True)),
            wants_window=bool(req.get("wants_window", True)),
        ),
        objects=ObjectsIntent(
            required=required,
            desired=desired,
            forbidden=forbidden,
        ),
        preferences=PreferencesIntent(
            favorite_colors=[str(x).strip() for x in (req.get("favorite_colors") or []) if str(x).strip()],
            avoid_colors=[str(x).strip() for x in (req.get("avoid_colors") or []) if str(x).strip()],
            notes=notes,
        ),
    )
    return normalize_intent(intent)


def _parse_max_count_hints(notes: str) -> dict[str, int]:
    if "max_count_hints:" not in notes:
        return {}
    try:
        part = notes.split("max_count_hints:", 1)[1].strip().splitlines()[0]
        data = json.loads(part)
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items() if str(k) and int(v) >= 0}
    except Exception:
        return {}
    return {}


def max_counts_from_request_and_intent(req: dict[str, Any], intent: PromptIntent) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in req.get("furniture") or []:
        if not isinstance(row, dict):
            continue
        sem = _coerce_semantic(str(row.get("semantic", "")))
        if not sem:
            continue
        try:
            c = max(0, int(row.get("count", 1)))
        except Exception:
            c = 1
        if c > 0:
            out[sem] = max(out.get(sem, 0), c)
    out.update(_parse_max_count_hints(intent.preferences.notes))
    return out


def apply_max_count_overrides(compiled: CompiledPolicy, overrides: dict[str, int]) -> None:
    if not overrides:
        return
    merged = dict(compiled.program.max_counts)
    for key, value in overrides.items():
        if not key:
            continue
        merged[str(key)] = max(0, int(value))
    compiled.program.max_counts = merged
    compiled.infinigen_policy.max_counts = dict(merged)
    compiled.acceptance_policy.max_counts = dict(merged)
    pref = dict(compiled.preflight.get("max_counts") or {})
    pref.update(merged)
    compiled.preflight["max_counts"] = pref


def _repair_plan_from_runtime(rt: dict[str, Any]) -> RepairPlan | None:
    if not rt:
        return None
    mp = {str(k): float(v) for k, v in (rt.get("monkeypatch_params") or {}).items() if str(k).strip()}
    bl = [str(x).strip() for x in (rt.get("add_factory_blacklist") or []) if str(x).strip()]
    wl = [str(x).strip() for x in (rt.get("add_factory_whitelist") or []) if str(x).strip()]
    gin = [str(x).strip() for x in (rt.get("gin_overrides_append") or []) if str(x).strip()]
    rem = [str(x).strip() for x in (rt.get("remove_optional_semantics") or []) if str(x).strip()]
    add_req = [str(x).strip() for x in (rt.get("added_required_semantics") or []) if str(x).strip()]
    mc = {str(k): int(v) for k, v in (rt.get("updated_max_counts") or {}).items() if str(k).strip()}
    if not mp and not bl and not wl and not gin and not rem and not add_req and not mc:
        return None
    return RepairPlan(
        updated_monkeypatch_params=mp,
        added_factory_blacklist=sorted(set(bl)),
        added_factory_whitelist=sorted(set(wl)),
        added_gin_overrides=sorted(set(gin)),
        removed_optional_semantics=sorted(set(rem)),
        added_required_semantics=sorted(set(add_req)),
        updated_max_counts=mc,
    )


_NONZERO_SOLVER_STEPS = ("solve_steps_large", "solve_steps_medium")
_VALID_SOLVER_STEPS = {"solve_steps_large", "solve_steps_medium", "solve_steps_small"}
_VALID_STAGE_FLAGS = {
    "solve_large_enabled",
    "solve_medium_enabled",
    "solve_small_enabled",
    "populate_assets_enabled",
}


def _sanitize_runtime_block(compiled: CompiledPolicy, rt: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Чистит infinigen_runtime от неадекватных значений, возвращает санитизированный словарь и причины правок.
    Не модифицирует входной словарь.
    """
    if not isinstance(rt, dict):
        return {}, ["runtime_not_dict"]
    out = deepcopy(rt)
    notes: list[str] = []
    ss_in = out.get("solver_steps") or {}
    if isinstance(ss_in, dict):
        ss_clean: dict[str, int] = {}
        for k, v in ss_in.items():
            raw_key = str(k).strip()
            if not raw_key:
                continue
            key = raw_key.split(".")[-1] if "." in raw_key else raw_key
            if key != raw_key:
                notes.append(f"solver_steps key prefix stripped: {raw_key!r} -> {key!r}")
            if key not in _VALID_SOLVER_STEPS:
                notes.append(f"solver_steps.{raw_key} unknown(dropped)")
                continue
            try:
                iv = int(float(v))
            except Exception:
                notes.append(f"solver_steps.{key}=non-int(dropped)")
                continue
            if key in _NONZERO_SOLVER_STEPS and iv <= 0:
                cur = int(compiled.infinigen_policy.solver_steps.get(key, 0))
                notes.append(f"solver_steps.{key}={iv} dropped (<=0); kept compiled={cur}")
                continue
            ss_clean[key] = max(0, iv)
        out["solver_steps"] = ss_clean
    sf_in = out.get("stage_flags") or {}
    if isinstance(sf_in, dict):
        sf_clean: dict[str, bool] = {}
        for k, v in sf_in.items():
            raw_key = str(k).strip()
            if not raw_key:
                continue
            key = raw_key.split(".")[-1] if "." in raw_key else raw_key
            if key != raw_key:
                notes.append(f"stage_flags key prefix stripped: {raw_key!r} -> {key!r}")
            if key not in _VALID_STAGE_FLAGS:
                notes.append(f"stage_flags.{raw_key} unknown(dropped)")
                continue
            sf_clean[key] = bool(v)
        out["stage_flags"] = sf_clean
    mp_in = out.get("monkeypatch_params") or {}
    if isinstance(mp_in, dict):
        mp_clean: dict[str, Any] = {}
        for k, v in mp_in.items():
            key = str(k).strip()
            if not key:
                continue
            if key.endswith("_pct"):
                try:
                    fv = float(v)
                except Exception:
                    notes.append(f"monkeypatch_params.{key}=non-number(dropped)")
                    continue
                if fv < 0.0 or fv > 1.0:
                    clamped = max(0.0, min(1.0, fv))
                    notes.append(f"monkeypatch_params.{key}={fv} clamped to {clamped}")
                    fv = clamped
                mp_clean[key] = fv
            else:
                mp_clean[key] = v
        out["monkeypatch_params"] = mp_clean
    return out, notes


def apply_infinigen_runtime_block(
    compiled: CompiledPolicy,
    rt: dict[str, Any],
    *,
    log_run_root: Path | None = None,
) -> CompiledPolicy:
    """Правки уровня Infinigen из infinigen_runtime (после compile, до screening)."""
    if not rt:
        return compiled
    rt, notes = _sanitize_runtime_block(compiled, rt)
    for note in notes:
        emit_llm_vlm_log(log_run_root, f"runtime_block sanitize: {note}")
    base = CompiledPolicy.model_validate(compiled.model_dump(mode="json"))
    for k, v in (rt.get("solver_steps") or {}).items():
        key = str(k).strip()
        if not key:
            continue
        base.infinigen_policy.solver_steps[key] = int(float(v))
    for k, v in (rt.get("stage_flags") or {}).items():
        key = str(k).strip()
        if not key:
            continue
        base.infinigen_policy.stage_flags[key] = bool(v)
    plan = _repair_plan_from_runtime(rt)
    if plan is not None:
        base = apply_repair_plan(base, plan)
    elif rt.get("gin_overrides_append") or rt.get("solver_steps") or rt.get("stage_flags"):
        extra = [str(x).strip() for x in (rt.get("gin_overrides_append") or []) if str(x).strip()]
        base.infinigen_policy.gin_overrides = sorted(set(build_gin_overrides(base) + extra))
    base.preflight["stage_flags"] = dict(base.infinigen_policy.stage_flags)
    base.preflight["solver_steps"] = dict(base.infinigen_policy.solver_steps)
    return base


def merge_runtime_dicts(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in delta.items():
        if key in ("monkeypatch_params", "solver_steps", "stage_flags") and isinstance(val, dict):
            merged = dict(out.get(key) or {})
            for k in val:
                merged[str(k)] = val[k]
            out[key] = merged
        elif key in (
            "gin_overrides_append",
            "add_factory_blacklist",
            "add_factory_whitelist",
            "remove_optional_semantics",
            "added_required_semantics",
        ) and isinstance(val, list):
            prev = list(out.get(key) or [])
            prev.extend([str(x).strip() for x in val if str(x).strip()])
            out[key] = prev
        elif key == "updated_max_counts" and isinstance(val, dict):
            mc = dict(out.get("updated_max_counts") or {})
            mc.update({str(k): int(v) for k, v in val.items() if str(k).strip()})
            out["updated_max_counts"] = mc
    return out

