import json
import sys
import types

import pytest

from src.style_prompt_analyzer import (
    _build_user_prompt,
    _read_prompt,
    _infer_colors,
    _normalize_analysis_obj,
    _validate_raw_json,
    analyze_prompt_to_style_profile,
    build_cli,
    heuristic_style_profile,
    main,
)


def test_infer_colors():
    colors = _infer_colors("Белый и beige, немного green")
    assert colors == ["white", "beige", "green"]


def test_normalize_analysis_validation():
    result = _normalize_analysis_obj(
        {
            "style_label": "minimalism",
            "room_type": "Bedroom",
            "confidence": 0.9,
            "preferred_colors": ["white", "gray"],
            "avoid_colors": [],
            "material_family": [],
            "notes": "ok",
        },
        prompt_text="квартира",
        room_path=None,
    )
    assert result.ok
    assert result.normalized["style_label"] == "minimalism"

    bad = _normalize_analysis_obj(
        {"style_label": "unknown", "room_type": "Bedroom", "confidence": 0.1},
        prompt_text="x",
        room_path=None,
    )
    assert not bad.ok

    assert not _normalize_analysis_obj([], prompt_text="x", room_path=None).ok
    assert not _normalize_analysis_obj({"style_label": "minimalism", "room_type": "Garage"}, prompt_text="x", room_path=None).ok
    assert not _normalize_analysis_obj(
        {"style_label": "minimalism", "room_type": "Bedroom", "confidence": "bad"},
        prompt_text="x",
        room_path=None,
    ).ok


def test_validate_raw_json_invalid():
    bad = _validate_raw_json("{not json", prompt_text="x", room_path=None)
    assert not bad.ok
    assert "Invalid JSON" in bad.feedback

    good = _validate_raw_json(
        json.dumps(
            {
                "style_label": "minimalism",
                "room_type": "Bedroom",
                "confidence": 2.0,
                "preferred_colors": ["white", ""],
                "avoid_colors": ["red"],
                "material_family": ["wood"],
                "notes": "ok",
            }
        ),
        prompt_text="white bedroom",
        room_path=None,
    )
    assert good.ok
    assert good.normalized["confidence"] == 1.0


def test_analyze_prompt_provider_none(monkeypatch):
    profile = analyze_prompt_to_style_profile(prompt_text="Небольшая спальня для отдыха", provider="none")
    assert profile["schema"] == "room_style_profile/v1"
    assert profile["room_type"] == "Bedroom"
    assert profile["style_label"] in {"minimalism", "contemporary", "scandinavian", "japandi"}
    assert heuristic_style_profile("industrial loft bedroom with black metal")["confidence"] > 0.35


def test_analyze_prompt_with_ollama_uses_fallback_when_needed(monkeypatch):
    # If LLM returns non-JSON, function should fall back to heuristic and keep error in llm metadata.
    monkeypatch.setattr("src.style_prompt_analyzer.chat_json", lambda *_, **__: {"message": {"content": "not json"}})
    profile = analyze_prompt_to_style_profile(
        prompt_text="светлая гостиная",
        provider="ollama",
        ollama_models=["mock:model"],
        max_attempts=1,
    )
    assert profile["schema"] == "room_style_profile/v1"
    assert profile["llm"]["provider"] == "heuristic_fallback"


def test_analyze_prompt_unexpected_ollama_response_falls_back(monkeypatch):
    monkeypatch.setattr("src.style_prompt_analyzer.chat_json", lambda **_kwargs: {"unexpected": True})

    def fake_retry_loop(*, generate_fn, initial_prompt, **_kwargs):
        generate_fn(initial_prompt)

    monkeypatch.setattr("src.style_prompt_analyzer.run_retry_loop", fake_retry_loop)
    profile = analyze_prompt_to_style_profile(prompt_text="modern bedroom", provider="ollama", ollama_models=["mock:model"])
    assert profile["llm"]["provider"] == "heuristic_fallback"
    assert "Unexpected Ollama response keys" in profile["llm"]["error"]


def test_analyze_prompt_success_response_and_cli_main(tmp_path, monkeypatch):
    seen = {}

    payload = {
        "style_label": "minimalism",
        "room_type": "Bedroom",
        "confidence": 0.8,
        "preferred_colors": ["white"],
        "avoid_colors": [],
        "material_family": ["wood"],
        "notes": "ok",
    }

    def fake_chat_json(**kwargs):
        seen["chat"] = kwargs
        return {"response": json.dumps(payload)}

    def fake_retry_loop(*, generate_fn, validate_fn, initial_prompt, max_attempts, debug_dir):
        raw = generate_fn(initial_prompt)
        validation = validate_fn(raw)
        assert validation.ok
        seen["retry"] = {"max_attempts": max_attempts, "debug_dir": debug_dir, "prompt": initial_prompt}
        return types.SimpleNamespace(normalized=validation.normalized, attempts_used=1)

    monkeypatch.setattr("src.style_prompt_analyzer.chat_json", fake_chat_json)
    monkeypatch.setattr("src.style_prompt_analyzer.run_retry_loop", fake_retry_loop)

    profile = analyze_prompt_to_style_profile(
        prompt_text="minimalist white bedroom",
        provider="ollama",
        ollama_models=None,
        debug_dir=str(tmp_path / "debug"),
    )

    assert profile["llm"]["provider"] == "ollama"
    assert profile["llm"]["model"] == "gpt-oss:20b"
    assert seen["chat"]["model"] == "gpt-oss:20b"
    assert seen["retry"]["debug_dir"].endswith("gpt-oss_20b")
    assert "allowed_style_labels" in _build_user_prompt("x", "Bedroom")

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("prompt from file", encoding="utf-8")
    assert _read_prompt(types.SimpleNamespace(prompt="inline", prompt_file=str(prompt_file))) == "inline"
    assert _read_prompt(types.SimpleNamespace(prompt=None, prompt_file=str(prompt_file))) == "prompt from file"
    with pytest.raises(RuntimeError, match="Need --prompt"):
        _read_prompt(types.SimpleNamespace(prompt=None, prompt_file=None))

    args = build_cli().parse_args(["--prompt", "x", "--out", str(tmp_path / "out.json"), "--llm-provider", "none"])
    assert args.prompt == "x"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "style_prompt_analyzer",
            "--prompt-file",
            str(prompt_file),
            "--out",
            str(tmp_path / "profile.json"),
            "--llm-provider",
            "none",
            "--ollama-models",
            " ",
            "mock:model",
        ],
    )
    main()
    written = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert written["schema"] == "room_style_profile/v1"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "style_prompt_analyzer",
            "--prompt",
            "inline prompt",
            "--out",
            str(tmp_path / "profile_default_model.json"),
            "--llm-provider",
            "none",
        ],
    )
    main()
    assert json.loads((tmp_path / "profile_default_model.json").read_text(encoding="utf-8"))["schema"] == "room_style_profile/v1"
