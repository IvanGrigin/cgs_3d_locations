#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/LLMModule/ollama_client.py

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _is_gpt_oss_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return m.startswith("gpt-oss")


def _build_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    temperature: float,
    think: str,
    extra_options: Optional[Dict[str, Any]],
    schema_mode: bool,
    include_think: bool,
) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": 512,
        "num_ctx": 4096,
    }
    if extra_options:
        options.update(extra_options)

    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": options,
    }

    if schema_mode:
        payload["format"] = json_schema
    else:
        payload["format"] = "json"

    # think отправляем только для GPT-OSS.
    # Для qwen/mistral и других моделей лишние отличия в payload здесь не нужны.
    if include_think and _is_gpt_oss_model(model):
        payload["think"] = think

    return payload


def chat_json(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    timeout_sec: int = 300,
    temperature: float = 0.0,
    think: str = "low",
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    is_gpt_oss = _is_gpt_oss_model(model)

    attempts = []

    if is_gpt_oss:
        attempts.append((True, True))
        attempts.append((False, True))
        attempts.append((False, False))
    else:
        attempts.append((True, False))
        attempts.append((False, False))
        attempts.append((False, True))

    last_error: Optional[Exception] = None

    for schema_mode, include_think in attempts:
        payload = _build_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            temperature=temperature,
            think=think,
            extra_options=extra_options,
            schema_mode=schema_mode,
            include_think=include_think,
        )
        try:
            return _post_json(url, payload, timeout_sec=timeout_sec)
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code != 400:
                raise
            continue

    if last_error is not None:
        raise last_error

    raise RuntimeError("chat_json: unexpected failure without captured exception")
