#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/LLMModule/ollama_client.py

import json
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
        "think": think,   # для gpt-oss: low / medium / high
        "format": json_schema,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": options,
    }

    return _post_json(url, payload, timeout_sec=timeout_sec)