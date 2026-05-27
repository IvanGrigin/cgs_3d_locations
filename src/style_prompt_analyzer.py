#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

try:
    from .LLMModule.ollama_client import chat_json
    from .LLMModule.retry_llm_json import ValidationResult, run_retry_loop
    from .style_profiles import (
        STYLE_PROFILES,
        compile_style_profile,
        default_style_label_for_room,
        infer_room_type_from_prompt,
        style_label_choices,
    )
except ImportError:
    from LLMModule.ollama_client import chat_json
    from LLMModule.retry_llm_json import ValidationResult, run_retry_loop
    from style_profiles import (
        STYLE_PROFILES,
        compile_style_profile,
        default_style_label_for_room,
        infer_room_type_from_prompt,
        style_label_choices,
    )


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_/\-+]+")


def _norm_text(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(value or "").lower().replace("ё", "е")))


def _token_set(value: str) -> set[str]:
    return set(_norm_text(value).split())


def _infer_colors(prompt_text: str) -> list[str]:
    tokens = _token_set(prompt_text)
    aliases = {
        "white": {"white", "белый", "белая", "cream", "ivory"},
        "black": {"black", "черный", "черная", "тёмный", "темный"},
        "gray": {"gray", "grey", "серый", "silver"},
        "beige": {"beige", "беж", "sand"},
        "brown": {"brown", "корич", "wood", "oak", "walnut"},
        "blue": {"blue", "синий", "голуб", "navy"},
        "green": {"green", "зелен", "olive", "sage"},
        "red": {"red", "красный", "burgundy", "бордо"},
        "yellow": {"yellow", "желтый", "gold", "golden"},
        "orange": {"orange", "оранж"},
        "purple": {"purple", "фиолет"},
        "pink": {"pink", "розов"},
    }
    out: list[str] = []
    for canonical, variants in aliases.items():
        if tokens & variants:
            out.append(canonical)
    return out


def _build_system_prompt() -> str:
    labels = ", ".join(style_label_choices())
    return (
        "You classify an interior design prompt into one dominant room style.\n"
        "Return only JSON with fields:\n"
        "{\n"
        '  "style_label": "<one label>",\n'
        '  "room_type": "<Bedroom|LivingRoom|Kitchen|Bathroom|DiningRoom>",\n'
        '  "confidence": <0..1>,\n'
        '  "preferred_colors": ["..."],\n'
        '  "avoid_colors": ["..."],\n'
        '  "material_family": ["..."],\n'
        '  "notes": "short summary"\n'
        "}\n"
        f"Allowed style labels: {labels}\n"
        "Choose the single best dominant style even if the prompt is mixed.\n"
        "Prefer concrete style words from the prompt over generic defaults.\n"
        "No markdown. JSON only."
    )


def _build_user_prompt(prompt_text: str, room_type_hint: str) -> str:
    payload = {
        "task": "Analyze an interior prompt and pick the best dominant style profile.",
        "prompt": prompt_text,
        "room_type_hint": room_type_hint,
        "allowed_style_labels": style_label_choices(),
        "allowed_room_types": ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "DiningRoom"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_analysis_obj(obj: Any, *, prompt_text: str, room_path: str | None = None) -> ValidationResult[dict[str, Any]]:
    if not isinstance(obj, dict):
        return ValidationResult(ok=False, feedback="Root JSON must be an object.")
    style_label = str(obj.get("style_label") or "").strip().lower().replace("-", "_")
    room_type = str(obj.get("room_type") or "").strip()
    if style_label not in STYLE_PROFILES:
        return ValidationResult(ok=False, feedback=f"style_label must be one of {style_label_choices()}.")
    allowed_room_types = {"Bedroom", "LivingRoom", "Kitchen", "Bathroom", "DiningRoom"}
    if room_type not in allowed_room_types:
        return ValidationResult(ok=False, feedback=f"room_type must be one of {sorted(allowed_room_types)}.")
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        return ValidationResult(ok=False, feedback="confidence must be a number.")
    normalized = {
        "style_label": style_label,
        "room_type": room_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "preferred_colors": [str(x).strip() for x in (obj.get("preferred_colors") or []) if str(x).strip()],
        "avoid_colors": [str(x).strip() for x in (obj.get("avoid_colors") or []) if str(x).strip()],
        "material_family": [str(x).strip() for x in (obj.get("material_family") or []) if str(x).strip()],
        "notes": str(obj.get("notes") or "").strip(),
    }
    return ValidationResult(ok=True, normalized=compile_style_profile(normalized, prompt_text=prompt_text, room_path=room_path))


def _validate_raw_json(raw_text: str, *, prompt_text: str, room_path: str | None = None) -> ValidationResult[dict[str, Any]]:
    try:
        obj = json.loads(raw_text)
    except Exception as exc:
        return ValidationResult(ok=False, feedback=f"Invalid JSON: {exc}")
    return _normalize_analysis_obj(obj, prompt_text=prompt_text, room_path=room_path)


def heuristic_style_profile(prompt_text: str, room_path: str | None = None) -> dict[str, Any]:
    room_type = infer_room_type_from_prompt(prompt_text, room_path)
    tokens = _token_set(prompt_text)
    best_label = default_style_label_for_room(room_type)
    best_score = -1
    for label, spec in STYLE_PROFILES.items():
        score = 0
        for keyword in spec.get("keywords") or []:
            kw_tokens = _token_set(keyword)
            if kw_tokens and kw_tokens <= tokens:
                score += 3
            elif kw_tokens & tokens:
                score += 1
        if label.replace("_", " ") in _norm_text(prompt_text):
            score += 4
        if score > best_score:
            best_label = label
            best_score = score
    analysis = {
        "style_label": best_label,
        "room_type": room_type,
        "confidence": 0.35 if best_score <= 0 else min(0.85, 0.45 + 0.08 * best_score),
        "preferred_colors": _infer_colors(prompt_text),
        "avoid_colors": [],
        "material_family": [],
        "notes": "heuristic fallback",
    }
    return compile_style_profile(analysis, prompt_text=prompt_text, room_path=room_path)


def analyze_prompt_to_style_profile(
    *,
    prompt_text: str,
    room_path: str | None = None,
    provider: str = "ollama",
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_models: Optional[list[str]] = None,
    timeout_sec: int = 180,
    temperature: float = 0.0,
    max_attempts: int = 4,
    think: str = "low",
    debug_dir: str | None = None,
) -> dict[str, Any]:
    room_type_hint = infer_room_type_from_prompt(prompt_text, room_path)
    if provider == "none":
        return heuristic_style_profile(prompt_text, room_path)

    models = [str(x).strip() for x in (ollama_models or []) if str(x).strip()]
    if not models:
        models = ["gpt-oss:20b", "qwen3:30b"]

    system_prompt = _build_system_prompt()
    initial_prompt = _build_user_prompt(prompt_text, room_type_hint)
    schema = {
        "type": "object",
        "properties": {
            "style_label": {"type": "string", "enum": style_label_choices()},
            "room_type": {"type": "string", "enum": ["Bedroom", "LivingRoom", "Kitchen", "Bathroom", "DiningRoom"]},
            "confidence": {"type": "number"},
            "preferred_colors": {"type": "array", "items": {"type": "string"}},
            "avoid_colors": {"type": "array", "items": {"type": "string"}},
            "material_family": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["style_label", "room_type", "confidence", "preferred_colors", "avoid_colors", "material_family", "notes"],
        "additionalProperties": False,
    }

    last_error: Optional[Exception] = None
    for model in models:
        def generate_fn(prompt: str) -> str:
            resp = chat_json(
                base_url=ollama_url,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                json_schema=schema,
                timeout_sec=timeout_sec,
                temperature=temperature,
                think=think,
            )
            msg = resp.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"].strip()
            if isinstance(resp.get("response"), str):
                return str(resp["response"]).strip()
            raise RuntimeError(f"Unexpected Ollama response keys: {list(resp.keys())}")

        def validate_fn(raw_text: str) -> ValidationResult[dict[str, Any]]:
            return _validate_raw_json(raw_text, prompt_text=prompt_text, room_path=room_path)

        try:
            result = run_retry_loop(
                generate_fn=generate_fn,
                validate_fn=validate_fn,
                initial_prompt=initial_prompt,
                max_attempts=max_attempts,
                debug_dir=str((Path(debug_dir) / model.replace(":", "_")).resolve()) if debug_dir else None,
            )
            profile = dict(result.normalized)
            profile.setdefault("llm", {})
            profile["llm"] = {
                "provider": provider,
                "model": model,
                "attempts_used": result.attempts_used,
            }
            return profile
        except Exception as exc:
            last_error = exc

    profile = heuristic_style_profile(prompt_text, room_path)
    profile.setdefault("llm", {})
    profile["llm"] = {
        "provider": "heuristic_fallback",
        "model": None,
        "error": str(last_error) if last_error else None,
    }
    return profile


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze prompt and compile a room style profile JSON")
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--room", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--llm-provider", choices=["none", "ollama"], default="ollama")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ollama-model", default="gpt-oss:20b")
    p.add_argument("--ollama-models", nargs="*", default=None)
    p.add_argument("--ollama-timeout", type=int, default=180)
    p.add_argument("--ollama-temperature", type=float, default=0.0)
    p.add_argument("--llm-max-attempts", type=int, default=4)
    p.add_argument("--llm-think", choices=["low", "medium", "high"], default="low")
    p.add_argument("--llm-debug-dir", default=None)
    return p


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return str(args.prompt)
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    raise RuntimeError("Need --prompt or --prompt-file")


def main() -> None:
    args = build_cli().parse_args()
    prompt_text = _read_prompt(args)
    models = [str(x).strip() for x in (args.ollama_models or []) if str(x).strip()]
    if not models and args.ollama_model:
        models = [str(args.ollama_model).strip()]
    profile = analyze_prompt_to_style_profile(
        prompt_text=prompt_text,
        room_path=args.room,
        provider=args.llm_provider,
        ollama_url=args.ollama_url,
        ollama_models=models,
        timeout_sec=int(args.ollama_timeout),
        temperature=float(args.ollama_temperature),
        max_attempts=int(args.llm_max_attempts),
        think=str(args.llm_think),
        debug_dir=args.llm_debug_dir,
    )
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()  # pragma: no cover
