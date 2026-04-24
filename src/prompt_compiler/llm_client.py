from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from src.LLMModule.ollama_client import chat_json


class BaseLLMClient(ABC):
    @abstractmethod
    def complete_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class StubLLMClient(BaseLLMClient):
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {}

    def complete_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        del system_prompt, user_prompt, schema
        return dict(self.payload)


class OllamaJSONLLMClient(BaseLLMClient):
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "gpt-oss:20b",
        timeout_sec: int = 180,
        temperature: float = 0.0,
        think: str = "low",
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.think = think
        self.max_retries = max_retries

    def complete_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        prompt = user_prompt
        for attempt in range(self.max_retries + 1):
            try:
                payload = chat_json(
                    base_url=self.base_url,
                    model=self.model,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    json_schema=schema,
                    timeout_sec=self.timeout_sec,
                    temperature=self.temperature,
                    think=self.think,
                )
                message = (((payload.get("message") or {}).get("content")) or "").strip()
                if message:
                    return json.loads(message)
                if isinstance(payload.get("response"), dict):
                    return dict(payload["response"])
                if isinstance(payload.get("response"), str):
                    return json.loads(payload["response"])
                if isinstance(payload, dict):
                    return payload
                raise RuntimeError("LLM returned unsupported payload format")
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                prompt = (
                    user_prompt
                    + "\n\nPrevious JSON was invalid or incomplete. "
                    + "Return one valid JSON object that matches the schema exactly."
                )
        assert last_error is not None
        raise last_error
