from __future__ import annotations

import json
import sys
import types

from src.suppliers.kitchen import kitchen_llm_decisions as kllm


def test_json_parsing_settings_and_chat_json_import_fallback(monkeypatch) -> None:
    assert not kllm._settings_enabled(None)
    assert not kllm._settings_enabled({"provider": "none"})
    assert kllm._settings_enabled({"provider": "ollama"})
    assert kllm._extract_text({"message": {"content": "  hi  "}}) == "hi"
    assert kllm._extract_text({"response": " ok "}) == "ok"
    assert kllm._parse_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert kllm._parse_json_object("prefix {\"b\": 2} suffix") == {"b": 2}
    assert kllm._parse_json_object("no json") == {}

    fake_module = types.ModuleType("src.LLMModule.ollama_client")
    calls = []

    def fake_chat_json(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps({"ok": True})}}

    fake_module.chat_json = fake_chat_json
    monkeypatch.setitem(sys.modules, "src.LLMModule.ollama_client", fake_module)

    parsed = kllm._chat_json(
        {"provider": "ollama", "ollama_url": "http://ollama", "ollama_model": "model", "ollama_timeout": 5},
        "system",
        {"payload": True},
        {"type": "object"},
    )
    assert parsed == {"ok": True}
    assert calls[0]["base_url"] == "http://ollama"
    assert calls[0]["model"] == "model"


def test_preferences_prompt_application_and_dining_plan(monkeypatch) -> None:
    assert kllm.infer_prompt_preferences_with_llm(
        user_prompt="",
        room={},
        kitchen_zone={},
        required_appliances={},
        llm_settings={"provider": "none"},
    ) == {"status": "skipped", "reason": "provider_none"}

    monkeypatch.setattr(
        kllm,
        "_chat_json",
        lambda *args, **kwargs: {
            "palette": {"facades": ["white"], "countertop": ["stone"], "backsplash": ["green"], "accent": ["brass"]},
            "style_keywords": ["minimal"],
            "appliance_hints": {"sink": ["black"], "cooktop": ["induction"]},
            "reason": "prompt",
        },
    )
    prefs = kllm.infer_prompt_preferences_with_llm(
        user_prompt="white kitchen with black sink",
        room={"room_type": "kitchen"},
        kitchen_zone={"wall": "north"},
        required_appliances={"sink": True},
        llm_settings={"provider": "ollama"},
    )
    assert prefs["status"] == "ok"

    spec = kllm.apply_llm_preferences_to_design_spec({"palette": {}}, prefs)
    assert spec["palette"]["facades"] == ["white"]
    assert spec["materials_intent"]["llm_style_keywords"] == ["minimal"]
    assert "Kitchen appliance preferences" in kllm.append_appliance_hints_to_prompt("base", prefs)
    assert kllm.append_appliance_hints_to_prompt("base", {"status": "failed"}) == "base"

    monkeypatch.setattr(kllm, "_chat_json", lambda *args, **kwargs: {"add_dining": True, "chair_count": 4, "table": {"shape": "round"}})
    dining = kllm.plan_dining_with_llm(room={"width_m": 5}, prompt_text="add dining", llm_settings={"provider": "ollama"})
    assert dining["status"] == "ok"
    assert dining["chair_count"] == 4

    monkeypatch.setattr(kllm, "_chat_json", lambda *args, **kwargs: {})
    failed = kllm.plan_dining_with_llm(room={}, prompt_text="", llm_settings={"provider": "ollama"})
    assert failed == {"status": "failed", "reason": "empty_llm_response"}


def test_material_and_appliance_rerank_with_llm_choices(monkeypatch) -> None:
    selected_materials = {
        "materials": {
            "facades": {
                "top_candidates": [
                    {
                        "material": {
                            "sku": "m1",
                            "name": "White facade",
                            "kitchen_role": "facades",
                            "visual": {"base_colors": ["white"], "tone": "light"},
                        },
                        "final_score": 0.7,
                    },
                    {"material": {"sku": "m2", "name": "Dark facade"}, "final_score": 0.9},
                ]
            }
        }
    }
    assert kllm._compact_material_candidate(selected_materials["materials"]["facades"]["top_candidates"][0])["sku"] == "m1"
    monkeypatch.setattr(kllm, "_chat_json", lambda *args, **kwargs: {"choices": {"facades": {"sku": "m2", "reason": "better"}}})
    reranked = kllm.rerank_material_bindings_with_llm(
        selected_materials=selected_materials,
        design_spec={"palette": {"facades": ["dark"]}},
        user_prompt="dark kitchen",
        mode="balanced",
        llm_settings={"provider": "ollama"},
    )
    assert reranked["materials"]["facades"]["chosen_material"]["sku"] == "m2"
    assert reranked["materials"]["facades"]["llm_reason"] == "better"
    assert kllm.rerank_material_bindings_with_llm(
        selected_materials=selected_materials,
        design_spec={},
        user_prompt="",
        mode="",
        llm_settings={"provider": "none"},
    ) is selected_materials

    appliance_assets = {
        "appliances": {
            "sink": {
                "top_candidates": [
                    {"unique_key": "a1", "title": "Steel sink", "asset_local_path": "/tmp/sink.fbx"},
                    {"unique_key": "a2", "title": "Black sink", "color": "black"},
                ]
            }
        }
    }
    compact = kllm._compact_asset_candidate(appliance_assets["appliances"]["sink"]["top_candidates"][0])
    assert compact["has_local_asset"] is True
    monkeypatch.setattr(kllm, "_chat_json", lambda *args, **kwargs: {"choices": {"sink": {"unique_key": "a2", "reason": "black"}}})
    appliance_reranked = kllm.rerank_appliance_assets_with_llm(
        appliance_assets=appliance_assets,
        user_prompt="black sink",
        design_spec={"palette": {"accent": ["black"]}},
        llm_settings={"provider": "ollama"},
    )
    assert appliance_reranked["appliances"]["sink"]["chosen_asset"]["unique_key"] == "a2"
    assert appliance_reranked["appliances"]["sink"]["llm_selected"] is True

    monkeypatch.setattr(kllm, "_chat_json", lambda *args, **kwargs: {"choices": {}})
    unchanged = kllm.rerank_appliance_assets_with_llm(
        appliance_assets=appliance_assets,
        user_prompt="",
        design_spec={},
        llm_settings={"provider": "ollama"},
    )
    assert unchanged is appliance_assets

