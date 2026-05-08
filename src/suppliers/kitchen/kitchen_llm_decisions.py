from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any


def _settings_enabled(settings: dict[str, Any] | None) -> bool:
    if not settings:
        return False
    return str(settings.get("provider") or "none").strip().lower() == "ollama"


def _extract_text(response: dict[str, Any]) -> str:
    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    if isinstance(response.get("response"), str):
        return str(response["response"]).strip()
    return json.dumps(response, ensure_ascii=False)


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _chat_json(settings: dict[str, Any], system_prompt: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    for module_name in ("src.LLMModule.ollama_client", "LLMModule.ollama_client"):
        try:
            module = __import__(module_name, fromlist=["chat_json"])
            chat_json = getattr(module, "chat_json")
            response = chat_json(
                base_url=str(settings.get("ollama_url") or "http://127.0.0.1:11434"),
                model=str(settings.get("ollama_model") or "gpt-oss:20b"),
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                json_schema=schema,
                timeout_sec=int(settings.get("ollama_timeout") or 180),
                temperature=float(settings.get("ollama_temperature") or 0.1),
                think=str(settings.get("ollama_think") or "low"),
                extra_options={
                    "num_ctx": int(settings.get("ollama_num_ctx") or 8192),
                    "num_predict": int(settings.get("ollama_num_predict") or 1024),
                },
            )
            parsed = _parse_json_object(_extract_text(response))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue
    return {}


def _compact_material_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    material = candidate.get("material") if isinstance(candidate.get("material"), dict) else candidate.get("chosen_material")
    material = material if isinstance(material, dict) else {}
    visual = material.get("visual") if isinstance(material.get("visual"), dict) else {}
    return {
        "sku": material.get("sku"),
        "name": material.get("name"),
        "role": material.get("kitchen_role"),
        "price": material.get("price"),
        "dimensions": material.get("dimensions"),
        "visual": {
            "base_colors": visual.get("base_colors"),
            "tone": visual.get("tone"),
            "pattern": visual.get("pattern"),
            "finish": visual.get("finish"),
            "style_tags": visual.get("style_tags"),
        },
        "score": candidate.get("final_score"),
        "score_breakdown": candidate.get("score_breakdown"),
    }


def _compact_asset_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "unique_key": candidate.get("unique_key"),
        "title": candidate.get("title"),
        "category_norm": candidate.get("category_norm"),
        "color": candidate.get("color"),
        "price": candidate.get("price"),
        "dimensions_cm": candidate.get("dimensions_cm"),
        "has_local_asset": bool(candidate.get("asset_local_path")),
        "asset_format": candidate.get("asset_format"),
        "score": candidate.get("score"),
        "score_breakdown": candidate.get("score_breakdown"),
    }


def infer_prompt_preferences_with_llm(
    *,
    user_prompt: str,
    room: dict[str, Any],
    kitchen_zone: dict[str, Any],
    required_appliances: dict[str, Any],
    llm_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _settings_enabled(llm_settings):
        return {"status": "skipped", "reason": "provider_none"}
    settings = dict(llm_settings or {})
    schema = {
        "type": "object",
        "properties": {
            "palette": {
                "type": "object",
                "properties": {
                    "facades": {"type": "array", "items": {"type": "string"}},
                    "countertop": {"type": "array", "items": {"type": "string"}},
                    "backsplash": {"type": "array", "items": {"type": "string"}},
                    "accent": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            "style_keywords": {"type": "array", "items": {"type": "string"}},
            "appliance_hints": {
                "type": "object",
                "properties": {
                    "sink": {"type": "array", "items": {"type": "string"}},
                    "faucet": {"type": "array", "items": {"type": "string"}},
                    "cooktop": {"type": "array", "items": {"type": "string"}},
                    "hood": {"type": "array", "items": {"type": "string"}},
                    "fridge": {"type": "array", "items": {"type": "string"}},
                    "microwave": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            "reason": {"type": "string"},
        },
        "required": ["palette", "appliance_hints"],
        "additionalProperties": False,
    }
    payload = {
        "user_prompt": user_prompt,
        "room": room,
        "kitchen_zone": kitchen_zone,
        "required_appliances": required_appliances,
        "rules": [
            "Extract only practical color/material/appliance preferences from the prompt.",
            "Use broad color families plus useful synonyms: white, gray, black, blue, green, red, burgundy, wood, light_wood, dark_wood, stone.",
            "For cooktop hints include induction/electric/gas only if prompt implies it.",
            "Do not decide module coordinates.",
        ],
    }
    parsed = _chat_json(
        settings,
        "You extract kitchen design preferences for a procedural generator. Return strict JSON only.",
        payload,
        schema,
    )
    if not parsed:
        return {"status": "failed", "reason": "empty_llm_response"}
    parsed["status"] = "ok"
    return parsed


def apply_llm_preferences_to_design_spec(design_spec: dict[str, Any], preferences: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(design_spec)
    if preferences.get("status") != "ok":
        return out
    palette = preferences.get("palette") if isinstance(preferences.get("palette"), dict) else {}
    out_palette = out.setdefault("palette", {})
    for key in ("facades", "countertop", "backsplash", "accent"):
        values = palette.get(key)
        if isinstance(values, list) and values:
            out_palette[key] = [str(x) for x in values if str(x).strip()]
    style_keywords = preferences.get("style_keywords")
    if isinstance(style_keywords, list) and style_keywords:
        intent = out.setdefault("materials_intent", {})
        intent.setdefault("llm_style_keywords", [str(x) for x in style_keywords])
    out.setdefault("llm_preferences", preferences)
    return out


def append_appliance_hints_to_prompt(user_prompt: str, preferences: dict[str, Any]) -> str:
    if preferences.get("status") != "ok":
        return user_prompt
    hints = preferences.get("appliance_hints") if isinstance(preferences.get("appliance_hints"), dict) else {}
    parts: list[str] = []
    for role, values in hints.items():
        if isinstance(values, list) and values:
            parts.append(f"{role}: {', '.join(str(x) for x in values[:5])}")
    if not parts:
        return user_prompt
    return user_prompt + "\nKitchen appliance preferences: " + "; ".join(parts)


def rerank_material_bindings_with_llm(
    *,
    selected_materials: dict[str, Any],
    design_spec: dict[str, Any],
    user_prompt: str,
    mode: str,
    llm_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _settings_enabled(llm_settings):
        return selected_materials
    settings = dict(llm_settings or {})
    materials = selected_materials.get("materials") if isinstance(selected_materials.get("materials"), dict) else {}
    candidate_payload: dict[str, Any] = {}
    for role, entry in materials.items():
        top = entry.get("top_candidates") if isinstance(entry, dict) else []
        if isinstance(top, list):
            candidate_payload[str(role)] = [_compact_material_candidate(candidate) for candidate in top[:8]]
    if not candidate_payload:
        return selected_materials
    schema = {
        "type": "object",
        "properties": {
            "choices": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
            "palette_consistency_reason": {"type": "string"},
        },
        "required": ["choices"],
        "additionalProperties": False,
    }
    parsed = _chat_json(
        settings,
        "You choose final kitchen surface materials from top-k candidates. Prefer prompt match first, then compatibility, then price. Return strict JSON.",
        {
            "user_prompt": user_prompt,
            "mode": mode,
            "design_spec": {
                "style": design_spec.get("style"),
                "palette": design_spec.get("palette"),
                "materials_intent": design_spec.get("materials_intent"),
            },
            "candidates_by_role": candidate_payload,
            "rules": [
                "Choose only sku values present in candidates_by_role.",
                "Facade, countertop and backsplash must look coherent together.",
                "For cheapest mode avoid expensive premium materials unless needed for role/color.",
            ],
        },
        schema,
    )
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
    if not choices:
        return selected_materials
    out = deepcopy(selected_materials)
    out.setdefault("llm_rerank", {"status": "ok", "choices": choices, "reason": parsed.get("palette_consistency_reason")})
    for role, choice in choices.items():
        if role not in out.get("materials", {}) or not isinstance(choice, dict):
            continue
        wanted_sku = str(choice.get("sku") or "").strip()
        entry = out["materials"][role]
        top = entry.get("top_candidates") if isinstance(entry.get("top_candidates"), list) else []
        matched = None
        for candidate in top:
            material = candidate.get("material") if isinstance(candidate.get("material"), dict) else {}
            if str(material.get("sku") or "") == wanted_sku:
                matched = candidate
                break
        if not matched:
            continue
        entry["chosen_material"] = deepcopy(matched["material"])
        entry["final_score"] = matched.get("final_score")
        entry["score_breakdown"] = matched.get("score_breakdown")
        entry["llm_selected"] = True
        entry["llm_reason"] = choice.get("reason")
    return out


def rerank_appliance_assets_with_llm(
    *,
    appliance_assets: dict[str, Any],
    user_prompt: str,
    design_spec: dict[str, Any],
    llm_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _settings_enabled(llm_settings):
        return appliance_assets
    appliances = appliance_assets.get("appliances") if isinstance(appliance_assets.get("appliances"), dict) else {}
    candidates_by_role: dict[str, Any] = {}
    for role, entry in appliances.items():
        top = entry.get("top_candidates") if isinstance(entry, dict) else []
        if isinstance(top, list):
            candidates_by_role[str(role)] = [_compact_asset_candidate(candidate) for candidate in top[:8]]
    if not candidates_by_role:
        return appliance_assets
    schema = {
        "type": "object",
        "properties": {
            "choices": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"unique_key": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["unique_key"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["choices"],
        "additionalProperties": False,
    }
    parsed = _chat_json(
        dict(llm_settings or {}),
        "You choose final FBX/asset kitchen appliances from top-k candidates. Match the prompt first, then dimensions and color. Return strict JSON.",
        {
            "user_prompt": user_prompt,
            "palette": (design_spec.get("palette") or {}),
            "candidates_by_role": candidates_by_role,
            "rules": [
                "Choose only unique_key values present in candidates_by_role.",
                "Prefer real local FBX/3D assets.",
                "Sink, faucet and cooktop type/color should follow the prompt when possible.",
            ],
        },
        schema,
    )
    choices = parsed.get("choices") if isinstance(parsed.get("choices"), dict) else {}
    if not choices:
        return appliance_assets
    out = deepcopy(appliance_assets)
    out.setdefault("llm_rerank", {"status": "ok", "choices": choices})
    for role, choice in choices.items():
        entry = (out.get("appliances") or {}).get(role)
        if not isinstance(entry, dict) or not isinstance(choice, dict):
            continue
        wanted = str(choice.get("unique_key") or "").strip()
        top = entry.get("top_candidates") if isinstance(entry.get("top_candidates"), list) else []
        matched = next((candidate for candidate in top if str(candidate.get("unique_key") or "") == wanted), None)
        if not matched:
            continue
        entry["chosen_asset"] = deepcopy(matched)
        entry["llm_selected"] = True
        entry["llm_reason"] = choice.get("reason")
    return out


def plan_dining_with_llm(
    *,
    room: dict[str, Any],
    prompt_text: str,
    llm_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    if not _settings_enabled(llm_settings):
        return {"status": "skipped", "reason": "provider_none"}
    schema = {
        "type": "object",
        "properties": {
            "add_dining": {"type": "boolean"},
            "table": {
                "type": "object",
                "properties": {
                    "shape": {"type": "string"},
                    "width_m": {"type": "number"},
                    "depth_m": {"type": "number"},
                    "x_m": {"type": "number"},
                    "y_m": {"type": "number"},
                    "yaw_deg": {"type": "number"},
                },
                "additionalProperties": False,
            },
            "chair_count": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["add_dining", "chair_count"],
        "additionalProperties": False,
    }
    parsed = _chat_json(
        dict(llm_settings or {}),
        "You propose a dining table placement for a kitchen room. Return approximate dimensions and position only; code will clamp it.",
        {
            "room": room,
            "prompt": prompt_text,
            "rules": [
                "Do not block the 0.65 m kitchen work zone along the back wall.",
                "For small rooms choose no table or a compact 2-chair table.",
                "For larger rooms choose 2-4 chairs.",
                "Return center position in meters from the room origin.",
            ],
        },
        schema,
    )
    if not parsed:
        return {"status": "failed", "reason": "empty_llm_response"}
    parsed["status"] = "ok"
    return parsed
