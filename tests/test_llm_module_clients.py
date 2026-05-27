from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from src.LLMModule import ollama_client
from src.LLMModule import retry_llm_json


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _http_error(code: int = 400) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://ollama/api/chat", code, "bad request", hdrs=None, fp=None)


def test_ollama_client_payload_fallbacks_and_urlopen(monkeypatch):
    posted = []

    def fake_urlopen(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        posted.append((req.full_url, timeout, body))
        return FakeHTTPResponse({"message": {"content": "{\"ok\": true}"}})

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
    result = ollama_client.chat_json(
        base_url="http://localhost:11434/",
        model="gpt-oss:20b",
        system_prompt="system",
        user_prompt="user",
        json_schema={"type": "object"},
        timeout_sec=12,
        temperature=0.2,
        think="low",
        extra_options={"num_predict": 10},
    )
    assert result["message"]["content"]
    assert posted[0][0] == "http://localhost:11434/api/chat"
    assert posted[0][1] == 12
    assert posted[0][2]["format"] == {"type": "object"}
    assert posted[0][2]["think"] == "low"
    assert posted[0][2]["options"]["num_predict"] == 10

    attempts = []

    def fallback_urlopen(req, timeout):
        body = json.loads(req.data.decode("utf-8"))
        attempts.append(body)
        if len(attempts) < 3:
            raise _http_error(400)
        return FakeHTTPResponse({"ok": True})

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fallback_urlopen)
    assert ollama_client.chat_json("http://ollama", "qwen3:30b", "s", "u", {"type": "object"}) == {"ok": True}
    assert attempts[0]["format"] == {"type": "object"}
    assert attempts[1]["format"] == "json"
    assert attempts[2]["format"] == "json"
    assert "think" not in attempts[2]

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(500)))
    with pytest.raises(urllib.error.HTTPError):
        ollama_client.chat_json("http://ollama", "qwen3:30b", "s", "u", {"type": "object"})

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(_http_error(400)))
    with pytest.raises(urllib.error.HTTPError) as exc:
        ollama_client.chat_json("http://ollama", "gpt-oss:20b", "s", "u", {"type": "object"})
    assert exc.value.code == 400


def test_retry_llm_json_success_retry_debug_and_failure(tmp_path: Path):
    assert "PREVIOUS ANSWER" in retry_llm_json.build_retry_prompt("base", "{}", "bad", 2)
    answers = iter(["bad", '{"ok": true}'])

    def generate(prompt: str) -> str:
        return next(answers)

    def validate(raw: str) -> retry_llm_json.ValidationResult[dict]:
        if raw.startswith("{"):
            return retry_llm_json.ValidationResult(ok=True, normalized={"ok": True})
        return retry_llm_json.ValidationResult(ok=False, feedback="not json")

    result = retry_llm_json.run_retry_loop(generate, validate, "base", max_attempts=2, debug_dir=str(tmp_path))
    assert result.normalized == {"ok": True}
    assert result.attempts_used == 2
    assert (tmp_path / "attempt_01_raw.txt").read_text(encoding="utf-8") == "bad"
    assert "not json" in (tmp_path / "attempt_01_validation.txt").read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="normalized=None"):
        retry_llm_json.run_retry_loop(lambda prompt: "{}", lambda raw: retry_llm_json.ValidationResult(ok=True), "base", max_attempts=1)

    with pytest.raises(RuntimeError, match="Последняя причина"):
        retry_llm_json.run_retry_loop(lambda prompt: "bad", lambda raw: retry_llm_json.ValidationResult(ok=False, feedback="still bad"), "base", max_attempts=1)

    with pytest.raises(ValueError, match="max_attempts"):
        retry_llm_json.run_retry_loop(lambda prompt: "{}", validate, "base", max_attempts=0)
