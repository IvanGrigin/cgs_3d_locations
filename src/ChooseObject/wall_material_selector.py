from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.ChooseObject.wall_material_normalizer import (
    WallMaterial,
    analyze_wallpaper_colors,
    load_normalized_wall_materials,
)


@dataclass
class WallMaterialRequest:
    version: str = "wall_material.request.v1"
    prompt: str = ""
    style: str = "contemporary"
    room_type: str = "unknown_room"
    room_description: str = ""
    preferred_colors: list[str] = field(default_factory=list)
    preferred_tones: list[str] = field(default_factory=list)
    preferred_patterns: list[str] = field(default_factory=list)
    preferred_material_types: list[str] = field(default_factory=list)
    avoid_colors: list[str] = field(default_factory=list)
    avoid_patterns: list[str] = field(default_factory=list)
    nice_to_have_terms: list[str] = field(default_factory=list)
    visual_requirements: dict[str, Any] = field(default_factory=dict)


@dataclass
class WallMaterialCandidate:
    material: WallMaterial
    final_score: float
    prompt_score: float
    style_score: float
    room_score: float
    visual_score: float
    matched_terms: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "sku": self.material.sku,
            "name": self.material.name,
            "product_url": self.material.product_url,
            "final_score": round(self.final_score, 4),
            "prompt_score": round(self.prompt_score, 4),
            "style_score": round(self.style_score, 4),
            "room_score": round(self.room_score, 4),
            "visual_score": round(self.visual_score, 4),
            "color": self.material.color,
            "tone": self.material.tone,
            "pattern": self.material.pattern,
            "average_rgb": self.material.average_rgb,
            "average_hex": self.material.average_hex,
            "dominant_colors_hex": self.material.dominant_colors_hex,
            "matched_terms": self.matched_terms,
            "penalties": self.penalties,
        }


@dataclass
class WallMaterialSelection:
    version: str
    room_id: str
    request: WallMaterialRequest
    selected_material: WallMaterial | None
    selection_reason: dict[str, Any]
    top_candidates: list[WallMaterialCandidate]
    filtered_count: int = 0
    llm_rerank: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "room_id": self.room_id,
            "request": asdict(self.request),
            "selected_material": self.selected_material.to_dict() if self.selected_material else None,
            "llm_rerank": self.llm_rerank,
            "selection_reason": self.selection_reason,
            "top_candidates": [c.summary_dict() for c in self.top_candidates],
        }


class WallPromptAnalyzer:
    COLORS = [
        ("white", ["бел", "white", "молоч", "ivory"]),
        ("gray", ["сер", "gray", "grey", "графит"]),
        ("beige", ["беж", "beige", "крем", "песоч", "ivory"]),
        ("brown", ["корич", "brown"]),
        ("green", ["зелен", "green", "олив", "sage", "mint", "emerald", "шалф", "мят"]),
        ("blue", ["син", "blue", "голуб"]),
        ("pink", ["роз", "pink"]),
        ("red", ["красн", "red"]),
        ("black", ["черн", "black"]),
        ("gold", ["золот", "gold"]),
    ]
    TONES = [
        ("light", ["светл", "light", "воздуш"]),
        ("dark", ["темн", "dark"]),
        ("neutral", ["нейтрал", "neutral"]),
        ("warm_light", ["тепл", "warm", "уют"]),
    ]
    PATTERNS = [
        ("plain", ["однотон", "фон", "plain", "без рисунка"]),
        ("concrete", ["бетон", "concrete"]),
        ("brick", ["кирпич", "brick"]),
        ("stone", ["камень", "stone"]),
        ("marble", ["мрамор", "marble"]),
        ("wood", ["дерев", "wood"]),
        ("stripe", ["полоск", "stripe"]),
        ("geometric", ["геометр", "geometry"]),
        ("botanical", ["листь", "растен", "botanical"]),
        ("floral", ["цвет", "floral"]),
        ("ornament", ["орнамент", "венз", "ornament"]),
        ("damask", ["дамаск", "damask"]),
        ("kids", ["детск", "динозав", "kids"]),
        ("textile", ["текстил", "лен", "linen"]),
        ("plaster", ["штукатур", "plaster"]),
    ]
    STYLE_ALIASES = {
        "modern": "contemporary",
        "industrial": "loft",
        "classicism": "classic",
        "neoclassical": "classic",
        "wabi_sabi": "japandi",
        "mid_century_modern": "contemporary",
        "art_deco": "classic",
        "baroque_inspired": "baroque",
    }

    def _text(self, *parts: str | None) -> str:
        return " ".join(x or "" for x in parts).lower().replace("ё", "е")

    def _extract(self, patterns: list[tuple[str, list[str]]], text: str) -> list[str]:
        return [value for value, needles in patterns if any(n in text for n in needles)]

    def build_request(self, prompt: str, style: str | None, room_type: str | None, room_description: str | None) -> WallMaterialRequest:
        text = self._text(prompt, room_description)
        style_norm = str(style or "contemporary").strip().lower().replace("-", "_")
        style_norm = self.STYLE_ALIASES.get(style_norm, style_norm)
        colors = self._extract(self.COLORS, text)
        tones = self._extract(self.TONES, text)
        patterns = self._extract(self.PATTERNS, text)
        if "black" in colors and "dark" not in tones:
            tones.append("dark")
        avoid_colors: list[str] = []
        avoid_patterns: list[str] = []
        if style_norm in {"scandinavian", "japandi", "minimalism"}:
            avoid_colors.extend(["black", "red"])
            avoid_patterns.extend(["damask", "ornament"])
        if "без рисун" in text:
            patterns.append("plain")
            avoid_patterns.extend(["floral", "ornament", "damask", "kids"])
        preferred_material_types: list[str] = []
        if any(x in text for x in ("wallpaper", "обои", "тактильн", "textile wall", "painted-look", "painted look")):
            preferred_material_types.append("wallpaper")
        if any(x in text for x in ("painted-look", "painted look", "крашен", "штукатур", "plaster")):
            preferred_material_types.append("wallpaper")
        nice_terms = [w for w in re.findall(r"[а-яa-zA-Z0-9]+", text) if len(w) >= 4][:24]
        return WallMaterialRequest(
            prompt=prompt,
            style=style_norm,
            room_type=str(room_type or "unknown_room").strip().lower().replace(" ", "_"),
            room_description=room_description or "",
            preferred_colors=list(dict.fromkeys(colors)),
            preferred_tones=list(dict.fromkeys(tones)),
            preferred_patterns=list(dict.fromkeys(patterns)),
            preferred_material_types=list(dict.fromkeys(preferred_material_types)),
            avoid_colors=[c for c in dict.fromkeys(avoid_colors) if c not in colors],
            avoid_patterns=list(dict.fromkeys(avoid_patterns)),
            nice_to_have_terms=nice_terms,
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _intersects(left: list[Any], right: list[Any]) -> set[Any]:
    return set(x for x in left if x is not None).intersection(x for x in right if x is not None)


def _json_loads_or(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return default
        try:
            return json.loads(match.group(0))
        except Exception:
            return default


def _extract_ollama_text(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(response.get("response"), str):
            return response["response"]
        if isinstance(response.get("content"), str):
            return response["content"]
    return str(response or "")


class WallMaterialSelector:
    STYLE_RULES = {
        "scandinavian": {
            "preferred_colors": ["white", "beige", "gray"],
            "preferred_tones": ["light", "neutral", "warm_light"],
            "preferred_patterns": ["plain", "textile", "plaster"],
            "positive_tags": ["scandinavian", "minimalism", "plain", "light"],
            "negative_tags": ["baroque", "damask", "black"],
        },
        "japandi": {
            "preferred_colors": ["beige", "white", "gray", "brown"],
            "preferred_tones": ["warm_light", "light", "neutral"],
            "preferred_patterns": ["plain", "textile", "plaster", "wood"],
            "positive_tags": ["japandi", "minimalism", "beige", "plain"],
            "negative_tags": ["baroque", "damask", "industrial"],
        },
        "minimalism": {
            "preferred_colors": ["white", "gray", "beige"],
            "preferred_tones": ["light", "neutral"],
            "preferred_patterns": ["plain", "concrete", "plaster", "geometric"],
            "positive_tags": ["minimalism", "plain", "gray", "light"],
            "negative_tags": ["baroque", "ornament", "floral"],
        },
        "loft": {
            "preferred_colors": ["gray", "brown", "black"],
            "preferred_tones": ["dark", "neutral", "warm_dark"],
            "preferred_patterns": ["concrete", "brick", "plaster", "wood"],
            "positive_tags": ["loft", "industrial", "concrete", "brick"],
            "negative_tags": ["floral", "pink", "baroque"],
        },
        "classic": {
            "preferred_colors": ["beige", "brown", "white", "gold"],
            "preferred_tones": ["warm_light", "warm_dark", "neutral"],
            "preferred_patterns": ["damask", "ornament", "floral", "textile", "plain"],
            "positive_tags": ["classic", "ornament", "damask", "floral"],
            "negative_tags": ["industrial", "kids"],
        },
        "baroque": {
            "preferred_colors": ["gold", "beige", "brown", "black"],
            "preferred_tones": ["warm_dark", "dark", "warm_light"],
            "preferred_patterns": ["damask", "ornament", "floral"],
            "positive_tags": ["baroque", "classic", "damask", "ornament"],
            "negative_tags": ["minimalism", "industrial", "kids"],
        },
        "contemporary": {
            "preferred_colors": ["white", "gray", "beige", "green", "blue"],
            "preferred_tones": ["light", "neutral", "warm_light"],
            "preferred_patterns": ["plain", "plaster", "geometric", "botanical", "textile"],
            "positive_tags": ["contemporary", "plain", "gray", "beige"],
            "negative_tags": ["baroque"],
        },
    }

    def __init__(self, materials_path: Path):
        self.materials_path = Path(materials_path)
        self.materials_base_dir = self.materials_path if self.materials_path.is_dir() else self.materials_path.parent
        self.materials = load_normalized_wall_materials(materials_path)
        self.analyzer = WallPromptAnalyzer()

    def select(
        self,
        prompt: str,
        style: str | None = None,
        room_type: str | None = None,
        room_description: str | None = None,
        top_k: int = 10,
        room_id: str = "room_001",
        llm_settings: dict[str, Any] | None = None,
    ) -> WallMaterialSelection:
        request = self.analyzer.build_request(prompt, style, room_type, room_description)
        source = self.filter_materials(self.materials, request)
        candidates = [self.score_material(m, request) for m in source]
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        shortlist_n = max(max(1, top_k) * 4, int((llm_settings or {}).get("top_n") or 5), 32)
        shortlist = candidates[:shortlist_n]
        if shortlist:
            rescored = [self.score_material(self._ensure_color_analysis(c.material), request) for c in shortlist]
            rescored.sort(key=lambda c: c.final_score, reverse=True)
            candidates = rescored + candidates[shortlist_n:]
        top = candidates[:max(1, top_k)]
        top, llm_rerank = self._llm_rerank_candidates(top, request, llm_settings)
        selected = top[0] if top else None
        reason = {}
        if selected:
            reason = {
                "final_score": round(selected.final_score, 4),
                "prompt_score": round(selected.prompt_score, 4),
                "style_score": round(selected.style_score, 4),
                "room_score": round(selected.room_score, 4),
                "visual_score": round(selected.visual_score, 4),
                "matched_terms": selected.matched_terms,
                "penalties": selected.penalties,
                "average_rgb": selected.material.average_rgb,
                "average_hex": selected.material.average_hex,
                "dominant_colors_rgb": selected.material.dominant_colors_rgb,
                "dominant_colors_hex": selected.material.dominant_colors_hex,
            }
        return WallMaterialSelection(
            "wall_material.selection.v1",
            room_id,
            request,
            selected.material if selected else None,
            reason,
            top,
            len(source),
            llm_rerank,
        )

    def filter_materials(self, materials: list[WallMaterial], request: WallMaterialRequest) -> list[WallMaterial]:
        base = [m for m in materials if m.parse_status in {"", "ok"}]
        known = [m for m in base if m.material_type != "unknown_wall_material"]
        base = known if len(known) >= 3 else base
        if request.preferred_colors:
            color_matched = [m for m in base if m.color in request.preferred_colors]
            if color_matched:
                base = color_matched
        if request.preferred_material_types:
            type_matched = [m for m in base if m.material_type in request.preferred_material_types]
            if len(type_matched) >= 3:
                base = type_matched
        return base

    def _ensure_color_analysis(self, material: WallMaterial) -> WallMaterial:
        if material.average_rgb or not material.local_image_paths:
            return material
        info = analyze_wallpaper_colors(self.materials_base_dir, material.local_image_paths)
        material.average_rgb = info.get("average_rgb")
        material.average_hex = info.get("average_hex")
        material.dominant_colors_rgb = info.get("dominant_colors_rgb") or []
        material.dominant_colors_hex = info.get("dominant_colors_hex") or []
        return material

    def score_material(self, m: WallMaterial, r: WallMaterialRequest) -> WallMaterialCandidate:
        prompt_score, matched, prompt_penalties = self._prompt_score(m, r)
        style_score, style_penalties = self._style_score(m, r)
        room_score, room_penalties = self._room_score(m, r)
        visual_score, visual_penalties = self._visual_score(m)
        final = 0.46 * prompt_score + 0.24 * style_score + 0.12 * room_score + 0.18 * visual_score
        return WallMaterialCandidate(
            m,
            _clamp(final),
            prompt_score,
            style_score,
            room_score,
            visual_score,
            sorted(set(matched)),
            prompt_penalties + style_penalties + room_penalties + visual_penalties,
        )

    def _prompt_score(self, m: WallMaterial, r: WallMaterialRequest) -> tuple[float, list[str], list[str]]:
        score = 0.35
        matched: list[str] = []
        penalties: list[str] = []
        text = m.search_text or m.name.lower()
        for term in r.nice_to_have_terms:
            if term and term in text:
                score += 0.025
                matched.append(term)
        if m.color in r.preferred_colors:
            score += 0.22
            matched.append(m.color or "")
        elif r.preferred_colors:
            score -= 0.16
            penalties.append(f"explicit_color_mismatch:{m.color}")
        if m.tone in r.preferred_tones:
            score += 0.16
            matched.append(m.tone or "")
        if m.pattern in r.preferred_patterns:
            score += 0.20
            matched.append(m.pattern or "")
        if m.material_type in r.preferred_material_types:
            score += 0.22
            matched.append(m.material_type)
        elif r.preferred_material_types:
            score -= 0.18
            penalties.append(f"material_type_not_preferred:{m.material_type}")
        if m.color in r.avoid_colors:
            score -= 0.22
            penalties.append(f"avoid_color:{m.color}")
        if m.pattern in r.avoid_patterns:
            score -= 0.24
            penalties.append(f"avoid_pattern:{m.pattern}")
        return _clamp(score), matched, penalties

    def _style_score(self, m: WallMaterial, r: WallMaterialRequest) -> tuple[float, list[str]]:
        rules = self.STYLE_RULES.get(r.style, self.STYLE_RULES["contemporary"])
        score = 0.35
        penalties: list[str] = []
        if m.color in rules.get("preferred_colors", []):
            score += 0.14
        if m.tone in rules.get("preferred_tones", []):
            score += 0.12
        if m.pattern in rules.get("preferred_patterns", []):
            score += 0.16
        positives = _intersects(m.style_tags, rules.get("positive_tags", []))
        negatives = _intersects(m.style_tags, rules.get("negative_tags", []))
        score += min(0.18, 0.05 * len(positives))
        if negatives:
            score -= min(0.24, 0.08 * len(negatives))
            penalties.extend(f"negative_style_tag:{x}" for x in sorted(negatives))
        return _clamp(score), penalties

    def _room_score(self, m: WallMaterial, r: WallMaterialRequest) -> tuple[float, list[str]]:
        score = 0.52
        penalties: list[str] = []
        if r.room_type in m.room_suitability:
            score += 0.16
        if r.room_type == "children" and m.pattern == "kids":
            score += 0.20
        if r.room_type == "bedroom" and m.pattern in {"kids", "brick"}:
            score -= 0.12
            penalties.append(f"bedroom_pattern:{m.pattern}")
        return _clamp(score), penalties

    def _visual_score(self, m: WallMaterial) -> tuple[float, list[str]]:
        score = 0.25
        penalties: list[str] = []
        if m.local_image_paths:
            score += 0.36
        elif m.image_urls:
            score += 0.18
        else:
            penalties.append("no_images")
        if m.average_rgb:
            score += 0.17
        if len(m.dominant_colors_rgb) >= 3:
            score += 0.12
        elif m.dominant_colors_rgb:
            score += 0.06
        return _clamp(score), penalties

    def _llm_candidate_payload(self, candidate: WallMaterialCandidate) -> dict[str, Any]:
        m = candidate.material
        return {
            "sku": m.sku,
            "name": m.name,
            "brand": m.brand,
            "final_score": round(candidate.final_score, 4),
            "material_type": m.material_type,
            "base_material": m.base_material,
            "color": m.color,
            "tone": m.tone,
            "pattern": m.pattern,
            "average_rgb": m.average_rgb,
            "average_hex": m.average_hex,
            "dominant_colors_rgb": m.dominant_colors_rgb[:6],
            "dominant_colors_hex": m.dominant_colors_hex[:6],
            "style_tags": m.style_tags[:12],
            "matched_terms": candidate.matched_terms,
            "penalties": candidate.penalties,
        }

    def _llm_rerank_candidates(
        self,
        candidates: list[WallMaterialCandidate],
        request: WallMaterialRequest,
        llm_settings: dict[str, Any] | None,
    ) -> tuple[list[WallMaterialCandidate], dict[str, Any] | None]:
        settings = dict(llm_settings or {})
        provider = str(settings.get("provider") or "none").strip().lower()
        if provider == "none" or len(candidates) <= 1:
            return candidates, None
        if provider != "ollama":
            return candidates, {"status": "skipped", "reason": f"unsupported_provider:{provider}"}
        top_n = max(1, min(int(settings.get("top_n") or len(candidates)), len(candidates)))
        candidate_slice = candidates[:top_n]
        by_sku = {str(c.material.sku): c for c in candidate_slice if c.material.sku}
        if len(by_sku) <= 1:
            return candidates, None
        try:
            from src.LLMModule.ollama_client import chat_json
        except Exception as exc:
            return candidates, {"status": "failed", "reason": f"ollama_import_failed:{type(exc).__name__}:{exc}"}
        schema = {
            "type": "object",
            "properties": {
                "chosen_sku": {"type": "string"},
                "ordered_skus": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["chosen_sku"],
            "additionalProperties": False,
        }
        payload = {
            "request": asdict(request),
            "ranking_policy": [
                "Respect explicit user color and pattern over generic style defaults.",
                "Use average_rgb and dominant_colors_rgb to reason about real image color, not only product text.",
                "Prefer coherent wall coverings for the room style and avoid visually loud patterns unless requested.",
                "Choose exactly one sku from candidates.",
            ],
            "candidates": [self._llm_candidate_payload(c) for c in candidate_slice],
        }
        try:
            response = chat_json(
                base_url=str(settings.get("ollama_url") or "http://127.0.0.1:11434"),
                model=str(settings.get("ollama_model") or "gpt-oss:20b"),
                system_prompt="You are an interior wall-covering reranker. Return strict JSON only.",
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                json_schema=schema,
                timeout_sec=int(settings.get("ollama_timeout") or 180),
                temperature=float(settings.get("ollama_temperature") or 0.0),
                think=str(settings.get("ollama_think") or "low"),
                extra_options={"num_ctx": int(settings.get("ollama_num_ctx") or 8192), "num_predict": 256},
            )
            parsed = _json_loads_or(_extract_ollama_text(response), None)
            if not isinstance(parsed, dict):
                raise RuntimeError("LLM did not return JSON object")
        except Exception as exc:
            return candidates, {"status": "failed", "reason": f"ollama_rerank_failed:{type(exc).__name__}:{exc}"}
        chosen_sku = str(parsed.get("chosen_sku") or "").strip()
        ordered = [str(x).strip() for x in parsed.get("ordered_skus") or [] if str(x).strip() in by_sku]
        if chosen_sku not in by_sku:
            return candidates, {"status": "failed", "reason": "ollama_returned_unknown_sku", "raw_response": parsed}
        if chosen_sku not in ordered:
            ordered = [chosen_sku] + [sku for sku in ordered if sku != chosen_sku]
        ordered += [sku for sku in by_sku if sku not in ordered]
        return [by_sku[sku] for sku in ordered] + candidates[top_n:], {
            "status": "applied",
            "provider": provider,
            "model": str(settings.get("ollama_model") or ""),
            "top_n": top_n,
            "chosen_sku": chosen_sku,
            "ordered_skus": ordered,
            "reason": str(parsed.get("reason") or "").strip() or None,
        }

    def save_selection(self, selection: WallMaterialSelection, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(selection.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
