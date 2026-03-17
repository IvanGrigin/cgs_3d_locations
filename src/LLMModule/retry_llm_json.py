#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/LLMModule/retry_llm_json.py

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, List, Optional, TypeVar


T = TypeVar("T")


@dataclass
class ValidationResult(Generic[T]):
    ok: bool
    normalized: Optional[T] = None
    feedback: str = ""


@dataclass
class RetryResult(Generic[T]):
    raw_text: str
    normalized: T
    attempts_used: int
    history: List[str]


def build_retry_prompt(
    base_prompt: str,
    previous_answer: str,
    feedback: str,
    attempt_no: int,
) -> str:
    return (
        base_prompt
        + "\n\n"
        + f"PREVIOUS ANSWER (attempt {attempt_no - 1}):\n"
        + previous_answer
        + "\n\n"
        + "THE ANSWER ABOVE IS INVALID.\n"
        + "FIX IT AND RETURN FULL JSON AGAIN.\n"
        + "ERRORS:\n"
        + feedback
        + "\n\n"
        + "RETURN ONLY JSON OBJECT. NO TEXT. NO MARKDOWN."
    )


def run_retry_loop(
    generate_fn: Callable[[str], str],
    validate_fn: Callable[[str], ValidationResult[T]],
    initial_prompt: str,
    max_attempts: int = 8,
    debug_dir: Optional[str] = None,
) -> RetryResult[T]:
    if max_attempts <= 0:
        raise ValueError("max_attempts должен быть > 0")

    prompt = initial_prompt
    history: List[str] = []
    last_feedback = ""
    dbg: Optional[Path] = None

    if debug_dir:
        dbg = Path(debug_dir).expanduser().resolve()
        dbg.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        raw_text = generate_fn(prompt)
        history.append(raw_text)

        if dbg is not None:
            (dbg / f"attempt_{attempt:02d}_raw.txt").write_text(raw_text, encoding="utf-8")
            (dbg / f"attempt_{attempt:02d}_prompt.txt").write_text(prompt, encoding="utf-8")

        result = validate_fn(raw_text)

        if dbg is not None:
            (dbg / f"attempt_{attempt:02d}_validation.txt").write_text(
                f"ok={result.ok}\nfeedback={result.feedback}\n",
                encoding="utf-8",
            )

        if result.ok:
            if result.normalized is None:
                raise RuntimeError("validate_fn вернул ok=True, но normalized=None")
            return RetryResult(
                raw_text=raw_text,
                normalized=result.normalized,
                attempts_used=attempt,
                history=history,
            )

        last_feedback = result.feedback.strip() or "Ответ не прошёл валидацию."
        prompt = build_retry_prompt(
            base_prompt=initial_prompt,
            previous_answer=raw_text,
            feedback=last_feedback,
            attempt_no=attempt + 1,
        )

    raise RuntimeError(
        "LLM не смогла выдать корректный ответ за "
        f"{max_attempts} попыток.\n"
        f"Последняя причина: {last_feedback}"
    )