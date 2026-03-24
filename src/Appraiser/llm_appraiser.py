#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class JudgeQuestionResult:
    question_id: str
    question_text: str
    score_10: int
    reason: str
    raw_response_text: Optional[str] = None


@dataclass
class JudgeModelResult:
    model: str
    provider: str
    questions: list[JudgeQuestionResult] = field(default_factory=list)
    mean_score_10: float = 0.0


@dataclass
class ChiefAppraisalResult:
    model: str
    provider: str
    final_score_10: float
    confidence_10: float
    strengths: list[str]
    weaknesses: list[str]
    verdict: str
    raw_response_text: Optional[str] = None


@dataclass
class LLMAppraisalResult:
    judge_results: list[JudgeModelResult]
    chief_result: ChiefAppraisalResult
    aggregated_question_scores: dict[str, float]
    score_10: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_QUESTIONS: list[tuple[str, str]] = [
    (
        "q1_convenience",
        "How convenient is this layout for everyday use by a person living in this room? Give a score from 1 to 10.",
    ),
    (
        "q2_access",
        "Can a person realistically access and use all relevant objects in this room? Give a score from 1 to 10.",
    ),
    (
        "q3_circulation",
        "How well does this arrangement preserve circulation, free movement, and clear paths inside the room? Give a score from 1 to 10.",
    ),
    (
        "q4_logic",
        "How logically are the objects placed with respect to walls, symmetry, typical furniture usage, and practical bedroom composition? Give a score from 1 to 10.",
    ),
    (
        "q5_plausibility",
        "How plausible, realistic, and comfortable does this layout look as a real interior design solution? Give a score from 1 to 10.",
    ),
]


class OllamaClientError(RuntimeError):
    pass


class LLMParseError(RuntimeError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _compact_number(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    return x


def build_scene_summary(scene: dict[str, Any]) -> dict[str, Any]:
    room = scene.get("room") or {}
    placements = scene.get("placements") or []

    out_items = []
    for obj in placements:
        aabb = obj.get("aabb") or {}
        out_items.append(
            {
                "id": obj.get("id"),
                "name": obj.get("name"),
                "category": obj.get("category"),
                "mount_type": obj.get("mount_type") or (obj.get("constraints") or {}).get("mount_type") or "floor",
                "position_m": [_compact_number(x) for x in (obj.get("position_m") or [])],
                "size_m": [_compact_number(x) for x in (obj.get("size_m") or [])],
                "yaw_deg": _compact_number(obj.get("yaw_deg", obj.get("rotation_deg", 0))),
                "aabb": {k: _compact_number(v) for k, v in aabb.items()},
                "constraints": obj.get("constraints") or {},
            }
        )

    return {
        "schema": scene.get("schema", "scene.v1"),
        "room": {
            "id": room.get("id"),
            "name": room.get("name"),
            "room_type": room.get("room_type"),
            "width_m": _compact_number(room.get("width_m")),
            "depth_m": _compact_number(room.get("depth_m")),
            "area_m2": _compact_number(room.get("area_m2")),
            "ceiling_height_m": _compact_number(room.get("ceiling_height_m", room.get("ceiling_height"))),
            "floor_polygon": room.get("floor_polygon") or [],
            "doors": room.get("doors") or [],
            "windows": room.get("windows") or [],
            "openings": room.get("openings") or [],
        },
        "placements": out_items,
    }


def _project_root_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    out: list[Path] = []
    seen: set[str] = set()
    for p in [here.parents[2], here.parents[3]]:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def _load_project_chat_json() -> Optional[Any]:
    module_names = ["src.LLMModule.ollama_client", "LLMModule.ollama_client"]
    for module_name in module_names:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, "chat_json", None)
            if callable(fn):
                return fn
        except Exception:
            pass

    for path in _project_root_candidates():
        sp = str(path)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        for module_name in module_names:
            try:
                mod = importlib.import_module(module_name)
                fn = getattr(mod, "chat_json", None)
                if callable(fn):
                    return fn
            except Exception:
                pass
    return None


def _direct_ollama_chat_json(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Optional[dict[str, Any]],
    timeout_sec: float,
    temperature: float,
    extra_options: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    url = base_url.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            **(extra_options or {}),
        },
    }
    if json_schema is not None:
        payload["format"] = json_schema

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise OllamaClientError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise OllamaClientError(
            "Cannot connect to Ollama at "
            f"{base_url}. Check that the server is running and the port is correct. "
            f"Example check: curl {base_url.rstrip('/')}/api/tags. "
            f"Original error: {e}"
        ) from e

    try:
        obj = json.loads(body)
    except Exception as e:
        raise OllamaClientError(f"Ollama returned non-JSON body: {body[:500]}") from e

    text_candidates = _candidate_texts_from_response(obj, None)
    text = text_candidates[0] if text_candidates else body.strip()
    return obj, text


def chat_json_any(
    *,
    provider: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Optional[dict[str, Any]],
    timeout_sec: float,
    temperature: float,
    extra_options: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    provider = provider.strip().lower()
    if provider != "ollama":
        raise RuntimeError(f"Unsupported provider: {provider}")

    try:
        return _direct_ollama_chat_json(
            base_url=base_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            timeout_sec=timeout_sec,
            temperature=temperature,
            extra_options=extra_options,
        )
    except Exception:
        pass

    project_chat_json = _load_project_chat_json()
    if project_chat_json is not None:
        last_error: Optional[Exception] = None
        for extra_kwargs in [{"think": False}, {"think": "none"}, {"think": "low"}, {}]:
            try:
                response = project_chat_json(
                    base_url=base_url,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_schema=json_schema,
                    timeout_sec=timeout_sec,
                    temperature=temperature,
                    extra_options=extra_options or {},
                    **extra_kwargs,
                )
                text_candidates = _candidate_texts_from_response(response, None)
                text = text_candidates[0] if text_candidates else json.dumps(response, ensure_ascii=False)
                return response, text
            except TypeError as e:
                last_error = e
            except Exception as e:
                last_error = e
        if last_error is not None:
            raise last_error

    return _direct_ollama_chat_json(
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=json_schema,
        timeout_sec=timeout_sec,
        temperature=temperature,
        extra_options=extra_options,
    )


def build_judge_system_prompt() -> str:
    return (
        "You are an interior layout judge. "
        "Evaluate furniture layouts critically and practically. "
        "Use only the provided scene summary and prompt. "
        "Do not invent missing geometry. "
        "Return strict JSON only."
    )


def build_single_question_prompt(*, scene_summary: dict[str, Any], original_prompt: Optional[str], question_id: str, question_text: str) -> str:
    return (
        "Evaluate this room layout.\n\n"
        f"Original user prompt:\n{original_prompt or '<none>'}\n\n"
        f"Scene summary JSON:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Question id: {question_id}\n"
        f"Question: {question_text}\n\n"
        "Return strict JSON object only with these fields:\n"
        "- score_10: integer from 1 to 10\n"
        "- reason: short explanation, at most 2 sentences\n"
        "Do not wrap JSON in markdown."
    )


def build_repair_prompt(*, raw_text: str, kind: str) -> str:
    if kind == "judge":
        return (
            "Convert the following model output into strict JSON only.\n"
            "Required fields:\n"
            "- score_10: integer from 1 to 10\n"
            "- reason: short explanation\n\n"
            f"Raw model output:\n{raw_text}\n"
        )
    return (
        "Convert the following model output into strict JSON only.\n"
        "Required fields:\n"
        "- final_score_10: number from 1 to 10\n"
        "- confidence_10: number from 1 to 10\n"
        "- strengths: array of short strings\n"
        "- weaknesses: array of short strings\n"
        "- verdict: short paragraph\n\n"
        f"Raw model output:\n{raw_text}\n"
    )


def build_chief_system_prompt() -> str:
    return (
        "You are the chief appraiser for interior layouts. "
        "You receive multiple judge opinions and optional code metrics. "
        "Aggregate them into one balanced final decision. "
        "Be critical, practical, and concise. "
        "Return strict JSON only."
    )


def build_chief_prompt(*, scene_summary: dict[str, Any], original_prompt: Optional[str], judge_results: list[JudgeModelResult], code_metrics_summary: Optional[dict[str, Any]]) -> str:
    compact_judges = []
    for jr in judge_results:
        compact_judges.append(
            {
                "model": jr.model,
                "provider": jr.provider,
                "mean_score_10": jr.mean_score_10,
                "questions": [
                    {
                        "question_id": q.question_id,
                        "question_text": q.question_text,
                        "score_10": q.score_10,
                        "reason": q.reason,
                    }
                    for q in jr.questions
                ],
            }
        )
    return (
        "Aggregate these layout evaluations.\n\n"
        f"Original user prompt:\n{original_prompt or '<none>'}\n\n"
        f"Scene summary JSON:\n{json.dumps(scene_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Judge results JSON:\n{json.dumps(compact_judges, ensure_ascii=False, indent=2)}\n\n"
        f"Optional code metrics summary JSON:\n{json.dumps(code_metrics_summary or {}, ensure_ascii=False, indent=2)}\n\n"
        "Return strict JSON object only with fields:\n"
        "- final_score_10: number from 1 to 10\n"
        "- confidence_10: number from 1 to 10\n"
        "- strengths: array of short strings\n"
        "- weaknesses: array of short strings\n"
        "- verdict: short paragraph, 2-4 sentences\n"
        "Do not wrap JSON in markdown."
    )


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score_10": {"type": "integer", "minimum": 1, "maximum": 10},
        "reason": {"type": "string"},
    },
    "required": ["score_10", "reason"],
    "additionalProperties": False,
}

CHIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "final_score_10": {"type": "number", "minimum": 1, "maximum": 10},
        "confidence_10": {"type": "number", "minimum": 1, "maximum": 10},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string"},
    },
    "required": ["final_score_10", "confidence_10", "strengths", "weaknesses", "verdict"],
    "additionalProperties": False,
}


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _candidate_texts_from_response(raw_response: Any, text: Optional[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        s = value.strip()
        if not s:
            return
        if s not in seen:
            seen.add(s)
            out.append(s)

    add(text)
    if isinstance(raw_response, dict):
        msg = raw_response.get("message")
        if isinstance(msg, dict):
            add(msg.get("content"))
            add(msg.get("thinking"))
        for key in ["response", "text", "content", "thinking"]:
            add(raw_response.get(key))
        result_obj = raw_response.get("result")
        if isinstance(result_obj, dict):
            for key in ["content", "text", "thinking", "reason", "verdict", "summary"]:
                add(result_obj.get(key))
        try:
            add(json.dumps(raw_response, ensure_ascii=False))
        except Exception:
            pass
    return out


def _extract_json_text(text: str) -> str:
    text = text.strip()
    if not text:
        raise LLMParseError("Model returned empty text")
    if text.startswith("{") and text.endswith("}"):
        return text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(0).strip()
    raise LLMParseError(f"Model did not return JSON object: {text[:500]}")


def _parse_json_text(text: str) -> dict[str, Any]:
    raw = _extract_json_text(text)
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise LLMParseError(f"Model did not return valid JSON: {raw[:500]}") from e
    if not isinstance(obj, dict):
        raise LLMParseError(f"JSON response must be object, got: {type(obj).__name__}")
    return obj


def _try_parse_json_text(text: Any) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    try:
        return _parse_json_text(s)
    except Exception:
        return None


def _looks_like_judge_payload(obj: dict[str, Any]) -> bool:
    return bool(set(obj.keys()).intersection({"score_10", "score", "rating", "value", "grade", "final_score_10", "final_score"}))


def _looks_like_chief_payload(obj: dict[str, Any]) -> bool:
    return bool(set(obj.keys()).intersection({"final_score_10", "final_score", "score_10", "score", "rating", "confidence_10", "confidence", "certainty", "strengths", "weaknesses", "verdict"}))


def _extract_numeric_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid score")
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if not m:
            raise ValueError(f"Cannot parse numeric score from string: {value!r}")
        score = float(m.group(0))
    else:
        raise ValueError(f"Unsupported score type: {type(value).__name__}")
    if 0.0 <= score <= 1.0:
        score *= 10.0
    return score



def _coerce_score_1_10(value: Any) -> int:
    score = round(_extract_numeric_score(value))
    return max(1, min(10, int(score)))



def _coerce_score_float_1_10(value: Any) -> float:
    score = _extract_numeric_score(value)
    return max(1.0, min(10.0, float(score)))



def _heuristic_score_from_text(text: str) -> Optional[int]:
    patterns = [
        r'"(?:final_)?score_10"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'"(?:final_)?score"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'"rating"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'\b(?:final\s+score|overall\s+score|score|rating|grade|value)\b[^\d-]{0,20}(-?\d+(?:\.\d+)?)',
        r'(-?\d+(?:\.\d+)?)\s*/\s*10\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return _coerce_score_1_10(m.group(1))
            except Exception:
                pass
    return None



def _heuristic_score_float_from_text(text: str) -> Optional[float]:
    patterns = [
        r'"(?:final_)?score_10"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'"(?:final_)?score"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'"rating"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'\b(?:final\s+score|overall\s+score|score|rating|grade|value)\b[^\d-]{0,20}(-?\d+(?:\.\d+)?)',
        r'(-?\d+(?:\.\d+)?)\s*/\s*10\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return _coerce_score_float_1_10(m.group(1))
            except Exception:
                pass
    return None


def _heuristic_payload_from_text_candidates(text_candidates: list[str], *, kind: str) -> Optional[dict[str, Any]]:
    for txt in text_candidates:
        compact = re.sub(r"\s+", " ", txt).strip()[:400]
        if kind == "judge":
            score_i = _heuristic_score_from_text(txt)
            if score_i is None:
                continue
            return {"score_10": score_i, "reason": compact or "Parsed from non-JSON model output."}
        score_f = _heuristic_score_float_from_text(txt)
        if score_f is None:
            continue
        return {
            "final_score_10": score_f,
            "confidence_10": 6.0,
            "strengths": [],
            "weaknesses": [],
            "verdict": compact or "Parsed from non-JSON model output.",
        }
    return None


def _decode_json_object_from_llm_response(raw_response: Any, text: Optional[str], *, kind: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def add_candidate(x: Any) -> None:
        if isinstance(x, dict):
            candidates.append(x)

    for candidate_text in _candidate_texts_from_response(raw_response, text):
        parsed = _try_parse_json_text(candidate_text)
        if parsed is not None:
            add_candidate(parsed)
    add_candidate(raw_response)

    if isinstance(raw_response, dict):
        msg = raw_response.get("message")
        if isinstance(msg, dict):
            add_candidate(msg)
            for key in ["content", "thinking"]:
                parsed = _try_parse_json_text(msg.get(key))
                if parsed is not None:
                    add_candidate(parsed)
        for key in ["response", "text", "content", "thinking"]:
            parsed = _try_parse_json_text(raw_response.get(key))
            if parsed is not None:
                add_candidate(parsed)
        result_obj = raw_response.get("result")
        if isinstance(result_obj, dict):
            add_candidate(result_obj)
            for key in ["content", "text", "thinking"]:
                parsed = _try_parse_json_text(result_obj.get(key))
                if parsed is not None:
                    add_candidate(parsed)

    is_target = _looks_like_judge_payload if kind == "judge" else _looks_like_chief_payload
    for cand in candidates:
        if is_target(cand):
            return cand
        if isinstance(cand.get("result"), dict) and is_target(cand["result"]):
            return cand["result"]
        msg = cand.get("message")
        if isinstance(msg, dict):
            for key in ["content", "thinking"]:
                sub = _try_parse_json_text(msg.get(key))
                if isinstance(sub, dict) and is_target(sub):
                    return sub

    heuristic = _heuristic_payload_from_text_candidates(_candidate_texts_from_response(raw_response, text), kind=kind)
    if heuristic is not None:
        return heuristic

    debug_keys = sorted(raw_response.keys()) if isinstance(raw_response, dict) else []
    raise LLMParseError(
        f"Could not decode {kind} JSON payload from model response. Top-level keys: {debug_keys}. "
        f"Raw text prefix: {' | '.join(_candidate_texts_from_response(raw_response, text))[:500]}"
    )


def _pick_first(obj: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def _normalize_judge_response(obj: dict[str, Any]) -> tuple[int, str]:
    score_value = _pick_first(obj, ["score_10", "score", "rating", "value", "grade", "final_score_10", "final_score"])
    if score_value is None and isinstance(obj.get("result"), dict):
        score_value = _pick_first(obj["result"], ["score_10", "score", "rating", "value", "grade", "final_score_10", "final_score"])
    if score_value is None:
        raise LLMParseError(
            "Judge response does not contain score field. Expected one of: "
            "score_10, score, rating, value, grade. "
            f"Actual keys: {sorted(obj.keys())}"
        )
    reason_value = _pick_first(obj, ["reason", "rationale", "explanation", "comment", "verdict"]) 
    if reason_value is None and isinstance(obj.get("result"), dict):
        reason_value = _pick_first(obj["result"], ["reason", "rationale", "explanation", "comment", "verdict"])
    return _coerce_score_1_10(score_value), str(reason_value or "")


def _normalize_chief_response(obj: dict[str, Any]) -> dict[str, Any]:
    final_score_value = _pick_first(obj, ["final_score_10", "final_score", "score_10", "score", "rating"])
    confidence_value = _pick_first(obj, ["confidence_10", "confidence", "certainty"])
    if final_score_value is None:
        raise LLMParseError(
            "Chief response does not contain final score field. Expected one of: "
            "final_score_10, final_score, score_10, score, rating. "
            f"Actual keys: {sorted(obj.keys())}"
        )
    strengths = obj.get("strengths") or []
    weaknesses = obj.get("weaknesses") or []
    verdict = _pick_first(obj, ["verdict", "summary", "reason", "comment"]) or ""
    if not isinstance(strengths, list):
        strengths = [str(strengths)]
    if not isinstance(weaknesses, list):
        weaknesses = [str(weaknesses)]
    return {
        "final_score_10": round(_coerce_score_float_1_10(final_score_value), 4),
        "confidence_10": round(_coerce_score_float_1_10(confidence_value if confidence_value is not None else 7.0), 4),
        "strengths": [str(x) for x in strengths],
        "weaknesses": [str(x) for x in weaknesses],
        "verdict": str(verdict),
    }


def _repair_via_llm(*, provider: str, base_url: str, model: str, timeout_sec: float, temperature: float, raw_text: str, kind: str) -> Optional[dict[str, Any]]:
    schema = JUDGE_SCHEMA if kind == "judge" else CHIEF_SCHEMA
    raw_text = raw_text[:4000]
    try:
        raw_response, text = chat_json_any(
            provider=provider,
            base_url=base_url,
            model=model,
            system_prompt="You are a JSON repair tool. Return strict JSON only.",
            user_prompt=build_repair_prompt(raw_text=raw_text, kind=kind),
            json_schema=schema,
            timeout_sec=timeout_sec,
            temperature=min(temperature, 0.05),
            extra_options={"num_predict": 256, "num_ctx": 4096},
        )
        return _decode_json_object_from_llm_response(raw_response, text, kind=kind)
    except Exception:
        return None


def ask_single_question(*, provider: str, base_url: str, model: str, timeout_sec: float, temperature: float, scene_summary: dict[str, Any], original_prompt: Optional[str], question_id: str, question_text: str) -> JudgeQuestionResult:
    user_prompt = build_single_question_prompt(
        scene_summary=scene_summary,
        original_prompt=original_prompt,
        question_id=question_id,
        question_text=question_text,
    )
    raw_response, text = chat_json_any(
        provider=provider,
        base_url=base_url,
        model=model,
        system_prompt=build_judge_system_prompt(),
        user_prompt=user_prompt,
        json_schema=JUDGE_SCHEMA,
        timeout_sec=timeout_sec,
        temperature=temperature,
        extra_options={"num_predict": 256, "num_ctx": 8192},
    )
    try:
        obj = _decode_json_object_from_llm_response(raw_response, text, kind="judge")
        score, reason = _normalize_judge_response(obj)
        return JudgeQuestionResult(question_id=question_id, question_text=question_text, score_10=score, reason=reason, raw_response_text=text)
    except Exception:
        repaired = _repair_via_llm(
            provider=provider,
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
            temperature=temperature,
            raw_text=(text or json.dumps(raw_response, ensure_ascii=False))[:4000],
            kind="judge",
        )
        if repaired is not None:
            try:
                score, reason = _normalize_judge_response(repaired)
                return JudgeQuestionResult(question_id=question_id, question_text=question_text, score_10=score, reason=reason, raw_response_text=text)
            except Exception:
                pass

        heuristic = _heuristic_payload_from_text_candidates(_candidate_texts_from_response(raw_response, text), kind="judge")
        if heuristic is not None:
            score, reason = _normalize_judge_response(heuristic)
            return JudgeQuestionResult(question_id=question_id, question_text=question_text, score_10=score, reason=reason, raw_response_text=text)

        fallback_reason = re.sub(r"\s+", " ", (text or "Model returned non-JSON output."))[:240]
        return JudgeQuestionResult(
            question_id=question_id,
            question_text=question_text,
            score_10=5,
            reason=f"Fallback score used because the model did not return parseable JSON. Raw prefix: {fallback_reason}",
            raw_response_text=text,
        )


def evaluate_with_judge_model(*, provider: str, base_url: str, model: str, timeout_sec: float, temperature: float, scene_summary: dict[str, Any], original_prompt: Optional[str], questions: list[tuple[str, str]], sleep_sec_between_calls: float = 0.0) -> JudgeModelResult:
    results: list[JudgeQuestionResult] = []
    for qid, qtext in questions:
        try:
            results.append(
                ask_single_question(
                    provider=provider,
                    base_url=base_url,
                    model=model,
                    timeout_sec=timeout_sec,
                    temperature=temperature,
                    scene_summary=scene_summary,
                    original_prompt=original_prompt,
                    question_id=qid,
                    question_text=qtext,
                )
            )
        except Exception as e:
            results.append(JudgeQuestionResult(question_id=qid, question_text=qtext, score_10=5, reason=f"Question failed, fallback score used: {type(e).__name__}: {e}", raw_response_text=None))
        if sleep_sec_between_calls > 0:
            time.sleep(sleep_sec_between_calls)
    mean_score = sum(x.score_10 for x in results) / max(1, len(results))
    return JudgeModelResult(model=model, provider=provider, questions=results, mean_score_10=round(mean_score, 4))


def aggregate_question_scores(judge_results: list[JudgeModelResult]) -> dict[str, float]:
    acc: dict[str, list[int]] = {}
    for jr in judge_results:
        for q in jr.questions:
            acc.setdefault(q.question_id, []).append(q.score_10)
    return {qid: round(sum(vals) / max(1, len(vals)), 4) for qid, vals in acc.items()}


def _fallback_chief_from_judges(*, provider: str, model: str, judge_results: list[JudgeModelResult]) -> ChiefAppraisalResult:
    all_scores = [q.score_10 for jr in judge_results for q in jr.questions]
    final_score = round(sum(all_scores) / max(1, len(all_scores)), 4) if all_scores else 5.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    verdict = "Fallback chief verdict was used because the chief model did not return parseable JSON."
    if final_score >= 7.5:
        verdict += " Overall the layout is reasonably strong."
    elif final_score >= 5.5:
        verdict += " Overall the layout is mixed and likely usable with noticeable issues."
    else:
        verdict += " Overall the layout appears weak and requires revision."
    return ChiefAppraisalResult(
        model=model,
        provider=provider,
        final_score_10=float(final_score),
        confidence_10=5.0,
        strengths=strengths,
        weaknesses=weaknesses,
        verdict=verdict,
        raw_response_text=None,
    )


def ask_chief_appraiser(*, provider: str, base_url: str, model: str, timeout_sec: float, temperature: float, scene_summary: dict[str, Any], original_prompt: Optional[str], judge_results: list[JudgeModelResult], code_metrics_summary: Optional[dict[str, Any]]) -> ChiefAppraisalResult:
    user_prompt = build_chief_prompt(
        scene_summary=scene_summary,
        original_prompt=original_prompt,
        judge_results=judge_results,
        code_metrics_summary=code_metrics_summary,
    )
    raw_response, text = chat_json_any(
        provider=provider,
        base_url=base_url,
        model=model,
        system_prompt=build_chief_system_prompt(),
        user_prompt=user_prompt,
        json_schema=CHIEF_SCHEMA,
        timeout_sec=timeout_sec,
        temperature=temperature,
        extra_options={"num_predict": 512, "num_ctx": 8192},
    )
    try:
        obj = _decode_json_object_from_llm_response(raw_response, text, kind="chief")
        norm = _normalize_chief_response(obj)
        return ChiefAppraisalResult(
            model=model,
            provider=provider,
            final_score_10=float(norm["final_score_10"]),
            confidence_10=float(norm["confidence_10"]),
            strengths=[str(x) for x in norm["strengths"]],
            weaknesses=[str(x) for x in norm["weaknesses"]],
            verdict=str(norm["verdict"]),
            raw_response_text=text,
        )
    except Exception:
        repaired = _repair_via_llm(
            provider=provider,
            base_url=base_url,
            model=model,
            timeout_sec=timeout_sec,
            temperature=temperature,
            raw_text=(text or json.dumps(raw_response, ensure_ascii=False))[:4000],
            kind="chief",
        )
        if repaired is not None:
            try:
                norm = _normalize_chief_response(repaired)
                return ChiefAppraisalResult(
                    model=model,
                    provider=provider,
                    final_score_10=float(norm["final_score_10"]),
                    confidence_10=float(norm["confidence_10"]),
                    strengths=[str(x) for x in norm["strengths"]],
                    weaknesses=[str(x) for x in norm["weaknesses"]],
                    verdict=str(norm["verdict"]),
                    raw_response_text=text,
                )
            except Exception:
                pass
        heuristic = _heuristic_payload_from_text_candidates(_candidate_texts_from_response(raw_response, text), kind="chief")
        if heuristic is not None:
            norm = _normalize_chief_response(heuristic)
            return ChiefAppraisalResult(
                model=model,
                provider=provider,
                final_score_10=float(norm["final_score_10"]),
                confidence_10=float(norm["confidence_10"]),
                strengths=[str(x) for x in norm["strengths"]],
                weaknesses=[str(x) for x in norm["weaknesses"]],
                verdict=str(norm["verdict"]),
                raw_response_text=text,
            )
        return _fallback_chief_from_judges(provider=provider, model=model, judge_results=judge_results)


def appraise_scene_llm(
    scene: dict[str, Any],
    original_prompt: Optional[str] = None,
    *,
    provider: str = "ollama",
    base_url: str = "http://127.0.0.1:11434",
    judge_models: Optional[list[str]] = None,
    chief_model: Optional[str] = None,
    timeout_sec: float = 180.0,
    temperature: float = 0.1,
    questions: Optional[list[tuple[str, str]]] = None,
    code_metrics_summary: Optional[dict[str, Any]] = None,
    sleep_sec_between_calls: float = 0.0,
) -> LLMAppraisalResult:
    judge_models = [m for m in (judge_models or []) if isinstance(m, str) and m.strip()]
    if not judge_models:
        judge_models = ["qwen3:30b", "gpt-oss:20b"]
    if chief_model is None:
        chief_model = judge_models[0]
    questions = questions or DEFAULT_QUESTIONS
    scene_summary = build_scene_summary(scene)

    judge_results: list[JudgeModelResult] = []
    for model in judge_models:
        judge_results.append(
            evaluate_with_judge_model(
                provider=provider,
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
                temperature=temperature,
                scene_summary=scene_summary,
                original_prompt=original_prompt,
                questions=questions,
                sleep_sec_between_calls=sleep_sec_between_calls,
            )
        )

    agg_q = aggregate_question_scores(judge_results)
    chief = ask_chief_appraiser(
        provider=provider,
        base_url=base_url,
        model=chief_model,
        timeout_sec=timeout_sec,
        temperature=temperature,
        scene_summary=scene_summary,
        original_prompt=original_prompt,
        judge_results=judge_results,
        code_metrics_summary=code_metrics_summary,
    )

    return LLMAppraisalResult(
        judge_results=judge_results,
        chief_result=chief,
        aggregated_question_scores=agg_q,
        score_10=round(float(chief.final_score_10), 4),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM appraiser for scene.v1")
    p.add_argument("--scene", required=True, help="Path to scene JSON")
    p.add_argument("--prompt", default=None, help="Original prompt text")
    p.add_argument("--prompt-file", default=None, help="Prompt text file")
    p.add_argument("--provider", default="ollama", help="LLM provider, currently ollama")
    p.add_argument("--base-url", default="http://127.0.0.1:11434", help="LLM endpoint base URL")
    p.add_argument("--judge-model", action="append", default=[], help="Judge model; can be repeated")
    p.add_argument("--judge-models", default=None, help="Comma-separated judge models")
    p.add_argument("--chief-model", default=None, help="Chief appraiser model")
    p.add_argument("--timeout", type=float, default=180.0, help="Request timeout")
    p.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    p.add_argument("--out", default=None, help="Output JSON file")
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
    result = appraise_scene_llm(
        scene=scene,
        original_prompt=prompt_text,
        provider=args.provider,
        base_url=args.base_url,
        judge_models=judge_models,
        chief_model=args.chief_model,
        timeout_sec=args.timeout,
        temperature=args.temperature,
    )
    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
