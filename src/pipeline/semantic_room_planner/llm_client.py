from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .schemas import write_json


def extract_json_object(text: str) -> dict[str, Any]:
    s = str(text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start < 0:
        raise ValueError("LLM response contains no JSON object")
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(s[start:idx + 1])
                if not isinstance(data, dict):
                    raise ValueError("Extracted JSON is not an object")
                return data
    raise ValueError("Could not extract a complete JSON object")


def _post_json(url: str, payload: dict[str, Any], timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_json_llm(
    messages: list[dict[str, str]],
    provider: str = "none",
    model: str | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
    openrouter_model: str | None = None,
    timeout: int = 180,
    temperature: float = 0.1,
    max_attempts: int = 3,
    debug_dir: str | Path | None = None,
    step_name: str = "llm",
) -> dict[str, Any]:
    if provider == "none":
        raise RuntimeError("provider=none cannot call LLM")
    dbg = Path(debug_dir).expanduser() if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    current = list(messages)
    for attempt in range(1, max_attempts + 1):
        if dbg:
            write_json(dbg / f"{step_name}.{attempt:02d}.prompt.json", {"messages": current})
        try:
            if provider == "ollama":
                data = _post_json(
                    f"{ollama_url.rstrip('/')}/api/chat",
                    {"model": model or "gpt-oss:20b", "messages": current, "stream": False, "format": "json", "options": {"temperature": temperature}},
                    timeout,
                )
                text = str((data.get("message") or {}).get("content") or data.get("response") or "")
            elif provider == "openrouter":
                key = os.environ.get("OPENROUTER_API_KEY")
                if not key:
                    raise RuntimeError("OPENROUTER_API_KEY is not set")
                data = _post_json(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"model": openrouter_model or model, "messages": current, "temperature": temperature, "response_format": {"type": "json_object"}},
                    timeout,
                    {"Authorization": f"Bearer {key}"},
                )
                text = str(data["choices"][0]["message"]["content"])
            else:
                raise ValueError(f"unsupported LLM provider: {provider}")
            if dbg:
                (dbg / f"{step_name}.{attempt:02d}.response.txt").write_text(text, encoding="utf-8")
            return extract_json_object(text)
        except (urllib.error.URLError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            last_error = exc
            current = current + [{"role": "user", "content": f"Previous response was invalid JSON or violated schema: {exc}. Return strict JSON object only."}]
            time.sleep(min(1.0 * attempt, 3.0))
    raise RuntimeError(f"LLM JSON call failed after {max_attempts} attempts: {last_error}")
