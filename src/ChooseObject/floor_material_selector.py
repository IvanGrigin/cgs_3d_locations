from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.ChooseObject.floor_material_normalizer import FloorMaterial, load_normalized_materials


@dataclass
class FlooringRequest:
    version: str = "flooring.request.v1"
    prompt: str = ""
    style: str = "contemporary"
    room_type: str = "unknown_room"
    room_description: str = ""
    preferred_material_types: list[str] = field(default_factory=list)
    preferred_decors: list[str] = field(default_factory=list)
    preferred_designs: list[str] = field(default_factory=list)
    preferred_tones: list[str] = field(default_factory=list)
    avoid_tones: list[str] = field(default_factory=list)
    preferred_gloss: list[str | None] = field(default_factory=list)
    technical_requirements: dict[str, Any] = field(default_factory=dict)
    visual_requirements: dict[str, Any] = field(default_factory=dict)
    must_have_terms: list[str] = field(default_factory=list)
    nice_to_have_terms: list[str] = field(default_factory=list)
    avoid_terms: list[str] = field(default_factory=list)


@dataclass
class FloorMaterialCandidate:
    material: FloorMaterial
    final_score: float
    prompt_score: float
    style_score: float
    room_score: float
    technical_score: float
    visual_score: float
    matched_terms: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "sku": self.material.sku,
            "name": self.material.name,
            "product_url": self.material.product_url,
            "final_score": round(self.final_score, 4),
            "prompt_score": round(self.prompt_score, 4),
            "style_score": round(self.style_score, 4),
            "room_score": round(self.room_score, 4),
            "technical_score": round(self.technical_score, 4),
            "visual_score": round(self.visual_score, 4),
            "matched_terms": self.matched_terms,
            "penalties": self.penalties,
        }


@dataclass
class FlooringSelection:
    version: str
    room_id: str
    request: FlooringRequest
    selected_material: FloorMaterial | None
    selection_reason: dict[str, Any]
    top_candidates: list[FloorMaterialCandidate]
    filtered_count: int = 0
    texture_candidate: dict[str, Any] | None = None
    texture_usable_in_blender: bool = False
    llm_rerank: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        selected = self.selected_material.to_dict() if self.selected_material else None
        return {
            "version": self.version,
            "room_id": self.room_id,
            "request": asdict(self.request),
            "selected_material": selected,
            "texture_candidate": self.texture_candidate,
            "texture_usable_in_blender": self.texture_usable_in_blender,
            "llm_rerank": self.llm_rerank,
            "selection_reason": self.selection_reason,
            "top_candidates": [c.summary_dict() for c in self.top_candidates],
        }


class FloorPromptAnalyzer:
    ROOM_PATTERNS = [
        ("bedroom", ["спальня", "bedroom"]),
        ("living_room", ["гостиная", "living"]),
        ("kitchen", ["кухня", "kitchen"]),
        ("bathroom", ["ванная", "санузел", "туалет", "wc", "restroom", "bathroom"]),
        ("hallway", ["прихожая", "коридор", "hallway"]),
        ("office", ["кабинет", "office"]),
        ("children", ["детская", "kids", "child"]),
    ]
    STYLE_PATTERNS = [
        ("scandinavian", ["скандинав", "scandinavian"]),
        ("japandi", ["japandi", "джапанди"]),
        ("minimalism", ["минимал", "minimalism"]),
        ("loft", ["loft", "лофт"]),
        ("classic", ["classic", "классик"]),
        ("baroque", ["baroque", "барокко"]),
        ("contemporary", ["contemporary", "современный"]),
    ]
    TONES = [
        ("light", ["светл", "light"]),
        ("dark", ["темн", "dark"]),
        ("gray", ["сер", "gray", "grey"]),
        ("beige", ["беж", "beige"]),
        ("natural", ["натурал", "natural"]),
        ("brown", ["коричнев", "brown"]),
        ("white", ["бел", "white"]),
        ("black", ["черн", "black"]),
    ]
    DECORS = [("oak", ["дуб"]), ("ash", ["ясень"]), ("walnut", ["орех"]), ("pine", ["сосна"])]
    DESIGNS = [
        ("wood", ["дерево", "деревян", "дуб", "ясень", "орех"]),
        ("concrete", ["бетон"]),
        ("stone", ["камень"]),
        ("marble", ["мрамор"]),
        ("tile", ["плитка"]),
        ("plain", ["однотон"]),
    ]
    STYLE_ALIASES = {
        "modern": "contemporary",
        "industrial": "loft",
        "classicism": "classic",
        "neoclassical": "classic",
        "wabi_sabi": "japandi",
        "mid_century_modern": "contemporary",
        "art_deco": "classic",
        "rustic": "classic",
        "coastal": "scandinavian",
    }
    ROOM_ALIASES = {
        "toilet": "bathroom",
        "wc": "bathroom",
        "restroom": "bathroom",
        "bath": "bathroom",
        "санузел": "bathroom",
        "туалет": "bathroom",
    }

    def _text(self, *parts: str | None) -> str:
        return " ".join(x or "" for x in parts).lower().replace("ё", "е")

    def extract_room_type(self, prompt: str, room_description: str | None = None) -> str:
        text = self._text(prompt, room_description)
        return next((room for room, needles in self.ROOM_PATTERNS if any(n in text for n in needles)), "unknown_room")

    def normalize_room_type(self, room_type: str | None) -> str | None:
        if not room_type:
            return None
        room = str(room_type).strip().lower().replace("-", "_").replace(" ", "_")
        return self.ROOM_ALIASES.get(room, room)

    def extract_style(self, prompt: str, fallback_style: str | None = None) -> str:
        text = self._text(prompt, fallback_style)
        style = next((style for style, needles in self.STYLE_PATTERNS if any(n in text for n in needles)), fallback_style or "contemporary")
        style = str(style or "contemporary").strip().lower().replace("-", "_")
        return self.STYLE_ALIASES.get(style, style)

    def extract_preferred_tones(self, prompt: str) -> list[str]:
        text = self._text(prompt)
        return [tone for tone, needles in self.TONES if any(n in text for n in needles)]

    def extract_preferred_decors(self, prompt: str) -> list[str]:
        text = self._text(prompt)
        return [decor for decor, needles in self.DECORS if any(n in text for n in needles)]

    def extract_preferred_designs(self, prompt: str) -> list[str]:
        text = self._text(prompt)
        return [design for design, needles in self.DESIGNS if any(n in text for n in needles)]

    def extract_technical_requirements(self, prompt: str, room_type: str) -> dict[str, Any]:
        text = self._text(prompt)
        min_class = 32
        water = False
        if room_type == "bathroom":
            water = True
        if room_type == "hallway":
            min_class = 33
        if room_type == "bedroom":
            min_class = 31 if "31" in text else 32
        if room_type in {"kitchen", "children"}:
            min_class = 32
        warm = True if "теплый пол" in text or "теплый пол" in text or "warm floor" in text else None
        return {"min_class": min_class, "water_resistant": water, "warm_floor_compatible": warm}

    def build_flooring_request(
        self,
        prompt: str,
        style: str | None = None,
        room_type: str | None = None,
        room_description: str | None = None,
    ) -> FlooringRequest:
        resolved_room = self.normalize_room_type(room_type) or self.extract_room_type(prompt, room_description)
        resolved_style = self.extract_style(prompt, style)
        tones = self.extract_preferred_tones(prompt)
        if "black" in tones and "dark" not in tones:
            tones.append("dark")
        if "black" in tones or "dark" in tones:
            tones = [tone for tone in tones if tone != "natural"]
        decors = self.extract_preferred_decors(prompt)
        designs = self.extract_preferred_designs(prompt)
        technical = self.extract_technical_requirements(prompt, resolved_room)
        nice_terms = [w for w in re.findall(r"[а-яa-zA-Z0-9]+", prompt.lower()) if len(w) >= 4][:20]
        avoid_terms: list[str] = []
        preferred_material_types: list[str] = []
        text = self._text(prompt)
        if re.search(r"(не\s+люблю|без|не\s+нужен|не\s+хочу|не)\s+\w*\s*ламинат", text):
            avoid_terms.append("ламинат")
        if "паркет" in text:
            preferred_material_types.extend(["parquet_board", "engineered_wood"])
        if "кварц" in text or "spc" in text or "пвх" in text or "винил" in text:
            preferred_material_types.append("vinyl_or_spc")
        visual_requirements: dict[str, Any] = {}
        if (
            "без затемн" in text
            or "без потемн" in text
            or "без утемн" in text
            or "без темных пят" in text
            or "без темных пятен" in text
            or "без сучк" in text
            or "не нравится" in text and ("затемн" in text or "потемн" in text or "утемн" in text or "сучк" in text)
            or "ровный цвет" in text
            or "равномерный" in text
            or "однотон" in text
        ):
            visual_requirements["avoid_strong_color_variation"] = True
            visual_requirements["max_color_variation_score"] = 0.42
        return FlooringRequest(
            prompt=prompt,
            style=resolved_style,
            room_type=resolved_room,
            room_description=room_description or "",
            preferred_decors=decors,
            preferred_designs=designs,
            preferred_tones=tones,
            preferred_material_types=preferred_material_types,
            avoid_tones=["black", "very_dark"] if resolved_style in {"scandinavian", "japandi"} else [],
            technical_requirements=technical,
            visual_requirements=visual_requirements,
            nice_to_have_terms=nice_terms,
            avoid_terms=avoid_terms,
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _intersects(left: list[Any], right: list[Any]) -> set[Any]:
    return set(x for x in left if x is not None).intersection(x for x in right if x is not None)


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


class FloorMaterialSelector:
    def __init__(self, materials_path: Path, style_rules_path: Path):
        self.materials_path = Path(materials_path)
        self.materials_base_dir = self.materials_path if self.materials_path.is_dir() else self.materials_path.parent
        self.materials = load_normalized_materials(materials_path)
        self.style_rules = json.loads(Path(style_rules_path).read_text(encoding="utf-8"))
        self.analyzer = FloorPromptAnalyzer()
        self.last_filtered_count = 0
        self._texture_candidate_cache: dict[str, dict[str, Any]] = {}

    def select(
        self,
        prompt: str,
        style: str | None = None,
        room_type: str | None = None,
        room_description: str | None = None,
        top_k: int = 10,
        room_id: str = "room_001",
        llm_settings: dict[str, Any] | None = None,
    ) -> FlooringSelection:
        request = self.analyzer.build_flooring_request(prompt, style, room_type, room_description)
        candidates_source = self.filter_materials(self.materials, request)
        if not candidates_source:
            candidates_source = self.materials
        self.last_filtered_count = len(candidates_source)
        candidates = [self.score_material(m, request) for m in candidates_source]
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        top = candidates[:max(1, top_k)]
        top, llm_rerank = self._llm_rerank_candidates(top, request, llm_settings)
        selected = top[0] if top else None
        reason = {}
        if selected:
            texture_candidate = self.select_texture_candidate(selected.material)
            reason = {
                "final_score": round(selected.final_score, 4),
                "prompt_score": round(selected.prompt_score, 4),
                "style_score": round(selected.style_score, 4),
                "room_score": round(selected.room_score, 4),
                "technical_score": round(selected.technical_score, 4),
                "visual_score": round(selected.visual_score, 4),
                "matched_terms": selected.matched_terms,
                "penalties": selected.penalties,
                "notes": selected.notes,
                "texture_analysis": texture_candidate,
            }
        else:
            texture_candidate = None
        return FlooringSelection(
            "flooring.selection.v1",
            room_id,
            request,
            selected.material if selected else None,
            reason,
            top,
            len(candidates_source),
            texture_candidate,
            bool(texture_candidate and texture_candidate.get("usable_in_blender")),
            llm_rerank,
        )

    def filter_materials(self, materials: list[FloorMaterial], request: FlooringRequest) -> list[FloorMaterial]:
        base = [m for m in materials if m.parse_status in {"", "ok"}]
        known = [m for m in base if m.material_type != "unknown_floor_material"]
        if len(known) >= 3:
            base = known
        if "ламинат" in request.avoid_terms:
            non_laminate = [m for m in base if m.material_type != "laminate" and "ламинат" not in (m.search_text or "").lower()]
            if len(non_laminate) >= 3:
                base = non_laminate
        if request.preferred_material_types:
            preferred = [m for m in base if m.material_type in request.preferred_material_types]
            if len(preferred) >= 3:
                base = preferred
        if request.room_type == "bathroom":
            strict = [
                m for m in base
                if m.material_type in {"vinyl_or_spc", "ceramic_tile", "porcelain_tile", "linoleum"}
                or (m.material_type == "laminate" and m.water_resistant)
            ]
            if strict:
                return strict
        return base

    def score_material(self, material: FloorMaterial, request: FlooringRequest) -> FloorMaterialCandidate:
        prompt_score, matched, prompt_penalties = self._prompt_score(material, request)
        style_score, style_penalties = self._style_score(material, request)
        room_score, room_penalties, notes = self._room_score(material, request)
        technical_score, technical_penalties = self._technical_score(material, request)
        visual_score, visual_penalties = self._visual_score(material, request)
        final = (
            0.30 * prompt_score + 0.25 * style_score + 0.20 * room_score
            + 0.15 * technical_score + 0.10 * visual_score
        )
        return FloorMaterialCandidate(
            material=material,
            final_score=_clamp(final),
            prompt_score=prompt_score,
            style_score=style_score,
            room_score=room_score,
            technical_score=technical_score,
            visual_score=visual_score,
            matched_terms=sorted(set(matched)),
            penalties=prompt_penalties + style_penalties + room_penalties + technical_penalties + visual_penalties,
            notes=notes,
        )

    def _prompt_score(self, m: FloorMaterial, r: FlooringRequest) -> tuple[float, list[str], list[str]]:
        score = 0.35
        matched: list[str] = []
        penalties: list[str] = []
        text = m.search_text or " ".join([m.name, m.description]).lower()
        for term in r.must_have_terms + r.nice_to_have_terms:
            if term and term.lower() in text:
                score += 0.04
                matched.append(term)
        if m.tone in r.preferred_tones:
            score += 0.18
            matched.append(m.tone or "")
        if m.decor in r.preferred_decors:
            score += 0.18
            matched.append(m.decor or "")
        if m.design in r.preferred_designs:
            score += 0.18
            matched.append(m.design or "")
        if m.material_type in r.preferred_material_types:
            score += 0.2
            matched.append(m.material_type)
        if m.tone in r.avoid_tones:
            score -= 0.25
            penalties.append(f"avoid_tone:{m.tone}")
        for term in r.avoid_terms:
            if term.lower() in text:
                score -= 0.35
                penalties.append(f"avoid_term:{term}")
        return _clamp(score), matched, penalties

    def _style_score(self, m: FloorMaterial, r: FlooringRequest) -> tuple[float, list[str]]:
        rules = self.style_rules.get(r.style, {})
        score = 0.35
        penalties: list[str] = []
        checks = [
            ("preferred_material_types", m.material_type, 0.16),
            ("preferred_decors", m.decor, 0.12),
            ("preferred_designs", m.design, 0.12),
            ("preferred_tones", m.tone, 0.12),
            ("preferred_gloss", m.gloss, 0.06),
        ]
        for key, value, delta in checks:
            if value in rules.get(key, []):
                score += delta
        if m.tone in r.preferred_tones:
            score += 0.08
        positives = _intersects(m.style_tags, rules.get("positive_tags", []))
        negatives = _intersects(m.style_tags, rules.get("negative_tags", []))
        score += min(0.18, 0.045 * len(positives))
        if negatives:
            score -= min(0.25, 0.08 * len(negatives))
            penalties.extend(f"negative_style_tag:{x}" for x in sorted(negatives))
        if m.tone in rules.get("avoid_tones", []):
            score -= 0.18
            penalties.append(f"style_avoid_tone:{m.tone}")
        if m.design in rules.get("avoid_designs", []):
            score -= 0.18
            penalties.append(f"style_avoid_design:{m.design}")
        return _clamp(score), penalties

    def _room_score(self, m: FloorMaterial, r: FlooringRequest) -> tuple[float, list[str], list[str]]:
        score = 0.45
        penalties: list[str] = []
        notes: list[str] = []
        room = r.room_type
        if room in m.room_suitability:
            score += 0.25
            notes.append(f"Материал подходит для room_type={room}.")
        if room in m.bad_for:
            score -= 0.35
            penalties.append(f"bad_for:{room}")
        if room == "bathroom":
            if m.material_type in {"vinyl_or_spc", "ceramic_tile", "porcelain_tile"}:
                score += 0.25
            elif m.material_type in {"laminate", "parquet_board", "engineered_wood"} and not m.water_resistant:
                score -= 0.5
                penalties.append("bathroom_non_waterproof_wood")
        if room == "kitchen":
            score += 0.12 if m.water_resistant else -0.05
            if m.class_value and m.class_value >= 32:
                score += 0.12
        if room == "hallway" and m.class_value and m.class_value >= 33:
            score += 0.22
        if room == "bedroom":
            if m.design == "wood" or m.material_type in {"laminate", "parquet_board", "engineered_wood"}:
                score += 0.2
            if m.material_type in {"ceramic_tile", "porcelain_tile"} and "tile" not in r.preferred_designs:
                score -= 0.25
                penalties.append("bedroom_cold_tile")
        if room == "children":
            if m.class_value and m.class_value >= 32:
                score += 0.18
            if m.tone in {"dark", "black"}:
                score -= 0.18
        return _clamp(score), penalties, notes

    def _technical_score(self, m: FloorMaterial, r: FlooringRequest) -> tuple[float, list[str]]:
        req = r.technical_requirements
        score = 0.55
        penalties: list[str] = []
        min_class = req.get("min_class")
        if min_class and m.class_value:
            if m.class_value >= min_class:
                score += 0.15
            else:
                score -= 0.18
                penalties.append(f"class_below_{min_class}")
        if req.get("water_resistant") is True:
            if m.water_resistant or m.material_type in {"vinyl_or_spc", "ceramic_tile", "porcelain_tile", "linoleum"}:
                score += 0.18
            else:
                score -= 0.35
                penalties.append("water_resistance_required")
        if req.get("warm_floor_compatible") is True:
            if m.warm_floor_compatible is True:
                score += 0.12
            elif m.warm_floor_compatible is None:
                score += 0.02
            else:
                score -= 0.18
                penalties.append("warm_floor_incompatible")
        if m.material_type == "laminate" and m.thickness_mm is not None:
            score += 0.12 if 8 <= m.thickness_mm <= 12 else -0.05
        if m.availability == "in_stock":
            score += 0.08
        elif m.availability == "out_of_stock":
            score -= 0.15
            penalties.append("out_of_stock")
        return _clamp(score), penalties

    def _visual_score(self, m: FloorMaterial, r: FlooringRequest) -> tuple[float, list[str]]:
        count = len(m.local_image_paths) + len(m.image_urls)
        if count == 0:
            return 0.15, ["no_images"]
        if count == 1:
            score = 0.72
        else:
            score = 0.9
        penalties: list[str] = []
        if (r.visual_requirements or {}).get("avoid_strong_color_variation"):
            texture = self.select_texture_candidate(m)
            analysis = texture.get("analysis") or {}
            variation = analysis.get("color_variation") or {}
            variation_score = float(variation.get("variation_score") or 0.0)
            max_score = float((r.visual_requirements or {}).get("max_color_variation_score") or 0.42)
            if bool(variation.get("natural_darkening_risk")) or variation_score > max_score:
                penalty = min(0.55, max(0.18, variation_score - max_score + 0.18))
                score -= penalty
                penalties.append(f"color_variation_too_high:{variation_score:.3f}")
        return _clamp(score), penalties

    def _llm_candidate_payload(self, candidate: FloorMaterialCandidate) -> dict[str, Any]:
        m = candidate.material
        return {
            "sku": m.sku,
            "name": m.name,
            "brand": m.brand,
            "final_score": round(candidate.final_score, 4),
            "prompt_score": round(candidate.prompt_score, 4),
            "style_score": round(candidate.style_score, 4),
            "room_score": round(candidate.room_score, 4),
            "technical_score": round(candidate.technical_score, 4),
            "material_type": m.material_type,
            "design": m.design,
            "decor": m.decor,
            "decor_name": m.decor_name,
            "tone": m.tone,
            "average_rgb": m.average_rgb,
            "average_hex": m.average_hex,
            "dominant_colors_hex": m.dominant_colors_hex[:6],
            "class": m.class_value,
            "thickness_mm": m.thickness_mm,
            "water_resistant": m.water_resistant,
            "availability": m.availability,
            "style_tags": m.style_tags[:12],
            "room_suitability": m.room_suitability,
            "bad_for": m.bad_for,
            "matched_terms": candidate.matched_terms,
            "penalties": candidate.penalties,
        }

    def _llm_rerank_candidates(
        self,
        candidates: list[FloorMaterialCandidate],
        request: FlooringRequest,
        llm_settings: dict[str, Any] | None,
    ) -> tuple[list[FloorMaterialCandidate], dict[str, Any] | None]:
        settings = dict(llm_settings or {})
        provider = str(settings.get("provider") or "none").strip().lower()
        if provider == "none" or len(candidates) <= 1:
            return candidates, None
        if provider != "ollama":
            return candidates, {"status": "skipped", "reason": f"unsupported_provider:{provider}"}

        top_n = max(1, min(int(settings.get("top_n") or len(candidates)), len(candidates)))
        candidate_slice = candidates[:top_n]
        candidate_by_sku = {str(c.material.sku): c for c in candidate_slice if c.material.sku}
        if len(candidate_by_sku) <= 1:
            return candidates, None

        chat_json = None
        import_error: Exception | None = None
        for module_name in ("src.LLMModule.ollama_client", "LLMModule.ollama_client"):
            try:
                module = __import__(module_name, fromlist=["chat_json"])
                chat_json = getattr(module, "chat_json", None)
                if callable(chat_json):
                    break
            except Exception as exc:
                import_error = exc
                chat_json = None
        if not callable(chat_json):
            return candidates, {
                "status": "failed",
                "reason": f"ollama_import_failed:{type(import_error).__name__ if import_error else 'RuntimeError'}:{import_error or 'chat_json_not_found'}",
            }

        system_prompt = (
            "You are an interior floor-covering reranker. "
            "Choose only from provided candidates. "
            "The shortlist was produced by deterministic retrieval/scoring; use LLM judgment only to break ties "
            "and better interpret the user's prompt, style, room, color, decor, and practical constraints. "
            "Do not invent products. Return strict JSON only."
        )
        payload = {
            "request": asdict(request),
            "ranking_policy": [
                "Respect explicit prompt color/tone/decor over generic style defaults.",
                "For bedrooms prefer wood-like laminate/parquet/engineered wood unless prompt asks tile.",
                "Avoid candidates marked bad_for the room.",
                "Prefer usable texture images for Blender when candidates are otherwise close.",
                "Choose exactly one sku from candidates.",
            ],
            "candidates": [self._llm_candidate_payload(c) for c in candidate_slice],
            "required_output": {
                "chosen_sku": "string sku from candidates",
                "ordered_skus": ["sku strings from best to worst"],
                "reason": "short Russian explanation",
            },
        }
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

        try:
            response = chat_json(
                base_url=str(settings.get("ollama_url") or "http://127.0.0.1:11434"),
                model=str(settings.get("ollama_model") or "gpt-oss:20b"),
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
                json_schema=schema,
                timeout_sec=int(settings.get("ollama_timeout") or 180),
                temperature=float(settings.get("ollama_temperature") or 0.0),
                think=str(settings.get("ollama_think") or "low"),
                extra_options={"num_ctx": int(settings.get("ollama_num_ctx") or 8192), "num_predict": 256},
            )
            raw_text = _extract_ollama_text(response)
            parsed = _json_loads_or(raw_text, None)
            if not isinstance(parsed, dict):
                raise RuntimeError(f"LLM did not return JSON object: {raw_text[:300]}")
        except Exception as exc:
            return candidates, {
                "status": "failed",
                "provider": provider,
                "model": str(settings.get("ollama_model") or ""),
                "reason": f"ollama_rerank_failed:{type(exc).__name__}:{exc}",
            }

        chosen_sku = str(parsed.get("chosen_sku") or "").strip()
        ordered_skus = [str(x).strip() for x in parsed.get("ordered_skus") or [] if str(x).strip() in candidate_by_sku]
        if chosen_sku not in candidate_by_sku:
            return candidates, {
                "status": "failed",
                "provider": provider,
                "model": str(settings.get("ollama_model") or ""),
                "reason": "ollama_returned_unknown_sku",
                "raw_response": parsed,
            }

        if chosen_sku not in ordered_skus:
            ordered_skus = [chosen_sku] + [sku for sku in ordered_skus if sku != chosen_sku]
        ordered_skus += [sku for sku in candidate_by_sku if sku not in ordered_skus]
        reranked = [candidate_by_sku[sku] for sku in ordered_skus] + candidates[top_n:]

        return reranked, {
            "status": "applied",
            "provider": provider,
            "model": str(settings.get("ollama_model") or ""),
            "top_n": top_n,
            "chosen_sku": chosen_sku,
            "ordered_skus": ordered_skus,
            "reason": str(parsed.get("reason") or "").strip() or None,
        }

    def _resolve_texture_path(self, path_value: str) -> Path | None:
        if not path_value:
            return None
        path = Path(path_value)
        if path.is_absolute() and path.exists():
            return path
        candidate = self.materials_base_dir / path
        if candidate.exists():
            return candidate
        return None

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    def _analyze_color_variation(
        self,
        pixels: list[tuple[int, int, int]],
        width: int,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
    ) -> dict[str, Any]:
        luminance: list[float] = []
        chroma: list[float] = []
        for y in range(min_y, max_y + 1):
            row_offset = y * width
            for x in range(min_x, max_x + 1):
                r, g, b = pixels[row_offset + x]
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                luminance.append(lum)
                chroma.append(max(r, g, b) - min(r, g, b))
        if not luminance:
            return {
                "variation_score": 1.0,
                "natural_darkening_risk": True,
                "reason": "empty_texture_crop",
            }
        mean = sum(luminance) / len(luminance)
        variance = sum((v - mean) ** 2 for v in luminance) / len(luminance)
        std = variance ** 0.5
        p05 = self._percentile(luminance, 0.05)
        p10 = self._percentile(luminance, 0.10)
        p50 = self._percentile(luminance, 0.50)
        p90 = self._percentile(luminance, 0.90)
        p95 = self._percentile(luminance, 0.95)
        dark_threshold = min(p50 * 0.72, mean - 34.0)
        dark_patch_ratio = sum(1 for v in luminance if v <= dark_threshold) / len(luminance)
        high_contrast_ratio = sum(1 for v in luminance if abs(v - mean) >= 48.0) / len(luminance)
        chroma_mean = sum(chroma) / len(chroma)
        range_90 = p95 - p05
        variation_score = _clamp(
            0.44 * min(std / 58.0, 1.0)
            + 0.30 * min(range_90 / 135.0, 1.0)
            + 0.18 * min(dark_patch_ratio / 0.18, 1.0)
            + 0.08 * min(chroma_mean / 70.0, 1.0)
        )
        natural_darkening_risk = bool(
            variation_score >= 0.52
            or dark_patch_ratio >= 0.12
            or (range_90 >= 96.0 and dark_patch_ratio >= 0.06)
            or high_contrast_ratio >= 0.18
        )
        return {
            "variation_score": round(variation_score, 4),
            "natural_darkening_risk": natural_darkening_risk,
            "luminance_mean": round(mean, 3),
            "luminance_std": round(std, 3),
            "luminance_p05": round(p05, 3),
            "luminance_p10": round(p10, 3),
            "luminance_p50": round(p50, 3),
            "luminance_p90": round(p90, 3),
            "luminance_p95": round(p95, 3),
            "luminance_range_p05_p95": round(range_90, 3),
            "dark_patch_ratio": round(dark_patch_ratio, 4),
            "high_contrast_ratio": round(high_contrast_ratio, 4),
            "mean_chroma": round(chroma_mean, 3),
            "reason": "strong_natural_darkening" if natural_darkening_risk else "acceptable_color_variation",
        }

    def analyze_texture_image(self, image_path: Path) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError:
            return {
                "path": str(image_path),
                "usable_in_blender": False,
                "reason": "Pillow is not installed",
            }

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((512, 512))
            width, height = image.size
            pixels = list(image.getdata())
        except Exception as exc:
            return {
                "path": str(image_path),
                "usable_in_blender": False,
                "reason": f"image_read_failed:{exc}",
            }

        white_threshold = 245
        non_white_points: list[tuple[int, int]] = []
        white_count = 0
        for idx, (r, g, b) in enumerate(pixels):
            is_white = r >= white_threshold and g >= white_threshold and b >= white_threshold
            if is_white:
                white_count += 1
            else:
                non_white_points.append((idx % width, idx // width))

        total = max(1, width * height)
        white_ratio = white_count / total
        if not non_white_points:
            return {
                "path": str(image_path),
                "usable_in_blender": False,
                "reason": "all_white_after_mask",
                "white_ratio": round(white_ratio, 4),
            }

        xs = [p[0] for p in non_white_points]
        ys = [p[1] for p in non_white_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        bbox_area = max(1, bbox_w * bbox_h)
        non_white_ratio = len(non_white_points) / total
        bbox_area_ratio = bbox_area / total
        fill_ratio = len(non_white_points) / bbox_area
        aspect = bbox_w / max(1, bbox_h)

        mask = [[False for _ in range(width)] for _ in range(height)]
        for x, y in non_white_points:
            mask[y][x] = True

        def rect_density(x0: int, y0: int, x1: int, y1: int) -> float:
            x0 = max(min_x, min(x0, max_x))
            x1 = max(min_x, min(x1, max_x))
            y0 = max(min_y, min(y0, max_y))
            y1 = max(min_y, min(y1, max_y))
            if x1 < x0 or y1 < y0:
                return 0.0
            count = 0
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            for yy in range(y0, y1 + 1):
                row = mask[yy]
                for xx in range(x0, x1 + 1):
                    if row[xx]:
                        count += 1
            return count / max(1, area)

        edge = max(2, min(16, min(bbox_w, bbox_h) // 16))
        corner = max(3, min(24, min(bbox_w, bbox_h) // 10))
        top_density = rect_density(min_x, min_y, max_x, min_y + edge - 1)
        bottom_density = rect_density(min_x, max_y - edge + 1, max_x, max_y)
        left_density = rect_density(min_x, min_y, min_x + edge - 1, max_y)
        right_density = rect_density(max_x - edge + 1, min_y, max_x, max_y)
        corner_densities = [
            rect_density(min_x, min_y, min_x + corner - 1, min_y + corner - 1),
            rect_density(max_x - corner + 1, min_y, max_x, min_y + corner - 1),
            rect_density(min_x, max_y - corner + 1, min_x + corner - 1, max_y),
            rect_density(max_x - corner + 1, max_y - corner + 1, max_x, max_y),
        ]
        edge_density_min = min(top_density, bottom_density, left_density, right_density)
        corner_density_min = min(corner_densities)

        # A useful Blender texture may still sit on a large white background.
        # What matters is that the non-white region is an axis-aligned rectangle
        # that can be cropped. Perspective product shots fail on side/corner fill.
        is_axis_aligned_rectangle = (
            fill_ratio >= 0.86
            and edge_density_min >= 0.72
            and corner_density_min >= 0.60
        )
        has_texture_aspect = 0.20 <= aspect <= 8.0
        has_enough_pixels = bbox_w >= 64 and bbox_h >= 64
        has_crop_margin = white_ratio >= 0.02 and bbox_area_ratio <= 0.98
        usable = bool(is_axis_aligned_rectangle and has_texture_aspect and has_enough_pixels)

        reason = "axis_aligned_crop_texture" if usable else "not_axis_aligned_rectangular_texture"
        score = 0.0
        score += min(0.46, bbox_area_ratio * 0.46)
        score += min(0.24, fill_ratio * 0.24)
        score += min(0.12, edge_density_min * 0.12)
        score += min(0.08, corner_density_min * 0.08)
        if has_texture_aspect:
            score += 0.04
        if has_crop_margin:
            score += 0.04
        if usable:
            score += 0.02
        color_variation = self._analyze_color_variation(
            pixels=pixels,
            width=width,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
        )
        luminance_range = float(color_variation.get("luminance_range_p05_p95") or 0.0)
        luminance_std = float(color_variation.get("luminance_std") or 0.0)
        # Product photos with angled boards often include side faces and shadows.
        # Prefer large, dense rectangular swatches with reasonably even color,
        # while still allowing natural wood grain.
        uniformity_score = _clamp(1.0 - (0.65 * min(luminance_range / 120.0, 1.0) + 0.35 * min(luminance_std / 55.0, 1.0)))
        rectangularity_score = _clamp(
            0.42 * fill_ratio
            + 0.24 * edge_density_min
            + 0.24 * corner_density_min
            + 0.10 * (1.0 if has_texture_aspect else 0.0)
        )
        size_score = _clamp(bbox_area_ratio / 0.72)
        crop_margin_score = 1.0 if has_crop_margin else 0.65
        texture_selection_score = _clamp(
            0.46 * rectangularity_score
            + 0.28 * size_score
            + 0.16 * uniformity_score
            + 0.06 * crop_margin_score
            + 0.04 * (1.0 if usable else 0.0)
        )

        return {
            "path": str(image_path),
            "usable_in_blender": usable,
            "reason": reason,
            "thumbnail_width": width,
            "thumbnail_height": height,
            "white_ratio": round(white_ratio, 4),
            "non_white_ratio": round(non_white_ratio, 4),
            "bbox": [min_x, min_y, max_x, max_y],
            "bbox_area_ratio": round(bbox_area_ratio, 4),
            "fill_ratio": round(fill_ratio, 4),
            "aspect_ratio": round(aspect, 4),
            "edge_density_min": round(edge_density_min, 4),
            "corner_density_min": round(corner_density_min, 4),
            "has_crop_margin": has_crop_margin,
            "edge_densities": {
                "top": round(top_density, 4),
                "bottom": round(bottom_density, 4),
                "left": round(left_density, 4),
                "right": round(right_density, 4),
            },
            "corner_densities": [round(x, 4) for x in corner_densities],
            "crop_bbox": [min_x, min_y, max_x + 1, max_y + 1],
            "color_variation": color_variation,
            "uniformity_score": round(uniformity_score, 4),
            "rectangularity_score": round(rectangularity_score, 4),
            "texture_selection_score": round(texture_selection_score, 4),
            "score": round(score, 4),
        }

    def _save_color_variation_map(self, image_path: Path, analysis: dict[str, Any], material: FloorMaterial, image_index: int) -> str | None:
        crop_bbox = analysis.get("crop_bbox")
        if not isinstance(crop_bbox, list) or len(crop_bbox) != 4:
            return None
        try:
            from PIL import Image, ImageFilter
            image = Image.open(image_path).convert("RGB")
            original_w, original_h = image.size
            thumb_w = max(1, int(analysis.get("thumbnail_width") or 0))
            thumb_h = max(1, int(analysis.get("thumbnail_height") or 0))
            if not thumb_w or not thumb_h:
                return None
            sx = original_w / thumb_w
            sy = original_h / thumb_h
            left = max(0, int(round(float(crop_bbox[0]) * sx)))
            top = max(0, int(round(float(crop_bbox[1]) * sy)))
            right = min(original_w, int(round(float(crop_bbox[2]) * sx)))
            bottom = min(original_h, int(round(float(crop_bbox[3]) * sy)))
            if right - left < 32 or bottom - top < 32:
                return None
            crop = image.crop((left, top, right, bottom)).resize((256, 256))
            gray = crop.convert("L")
            smooth = gray.filter(ImageFilter.GaussianBlur(radius=3))
            pixels = list(smooth.getdata())
            mean = sum(pixels) / max(1, len(pixels))
            heat_pixels: list[tuple[int, int, int]] = []
            for value in pixels:
                delta = int(max(-96, min(96, value - mean)))
                if delta < 0:
                    strength = min(255, int(abs(delta) * 2.65))
                    heat_pixels.append((strength, 24, 32))
                else:
                    strength = min(255, int(delta * 2.2))
                    heat_pixels.append((32, 80 + strength // 2, strength))
            heat = Image.new("RGB", (256, 256))
            heat.putdata(heat_pixels)
            out_dir = self.materials_base_dir / "texture_color_maps" / str(material.sku or "unknown")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_index:02d}_variation.jpg"
            heat.save(out_path, quality=95)
            return str(out_path.relative_to(self.materials_base_dir))
        except Exception:
            return None

    def _save_texture_crop(self, image_path: Path, analysis: dict[str, Any], material: FloorMaterial, image_index: int) -> str | None:
        crop_bbox = analysis.get("crop_bbox")
        if not analysis.get("usable_in_blender") or not isinstance(crop_bbox, list) or len(crop_bbox) != 4:
            return None
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            original_w, original_h = image.size
            thumb_w = max(1, int(analysis.get("thumbnail_width") or 0))
            thumb_h = max(1, int(analysis.get("thumbnail_height") or 0))
            if not thumb_w or not thumb_h:
                return None
            sx = original_w / thumb_w
            sy = original_h / thumb_h
            left = max(0, int(round(float(crop_bbox[0]) * sx)))
            top = max(0, int(round(float(crop_bbox[1]) * sy)))
            right = min(original_w, int(round(float(crop_bbox[2]) * sx)))
            bottom = min(original_h, int(round(float(crop_bbox[3]) * sy)))
            if right - left < 32 or bottom - top < 32:
                return None
            out_dir = self.materials_base_dir / "texture_crops" / str(material.sku or "unknown")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_index:02d}.jpg"
            image.crop((left, top, right, bottom)).save(out_path, quality=95)
            return str(out_path.relative_to(self.materials_base_dir))
        except Exception:
            return None

    def select_texture_candidate(self, material: FloorMaterial) -> dict[str, Any]:
        cache_key = str(material.sku or material.product_url or material.name)
        if cache_key in self._texture_candidate_cache:
            return self._texture_candidate_cache[cache_key]
        analyses: list[dict[str, Any]] = []
        resolved_paths: list[tuple[str, Path]] = []
        for raw_path in material.local_image_paths:
            resolved = self._resolve_texture_path(raw_path)
            if not resolved:
                continue
            resolved_paths.append((raw_path, resolved))
            analysis = self.analyze_texture_image(resolved)
            analysis["local_image_path"] = raw_path
            analyses.append(analysis)

        if analyses:
            usable = [a for a in analyses if a.get("usable_in_blender")]
            chosen = max(
                usable,
                key=lambda a: (
                    float(a.get("texture_selection_score") or 0.0),
                    float(a.get("rectangularity_score") or 0.0),
                    float(a.get("bbox_area_ratio") or 0.0),
                ),
            ) if usable else analyses[0]
            for analysis in analyses:
                analysis["selected"] = analysis is chosen
            texture_path = chosen.get("local_image_path") or chosen.get("path")
            texture_abs_path = chosen.get("path")
            if usable:
                chosen_index = analyses.index(chosen)
                crop_path = self._save_texture_crop(resolved_paths[chosen_index][1], chosen, material, chosen_index + 1)
                if crop_path:
                    texture_path = crop_path
                    texture_abs_path = str((self.materials_base_dir / crop_path).resolve())
                    chosen["cropped_texture_path"] = crop_path
                    chosen["cropped_texture_abs_path"] = texture_abs_path
                variation_map_path = self._save_color_variation_map(resolved_paths[chosen_index][1], chosen, material, chosen_index + 1)
                if variation_map_path:
                    chosen["color_variation"]["variation_map_path"] = variation_map_path
                    chosen["color_variation"]["variation_map_abs_path"] = str((self.materials_base_dir / variation_map_path).resolve())
            result = {
                "texture_path": texture_path,
                "texture_abs_path": texture_abs_path,
                "usable_in_blender": bool(chosen.get("usable_in_blender")),
                "reason": chosen.get("reason"),
                "analysis": chosen,
                "all_local_image_analyses": analyses,
            }
            self._texture_candidate_cache[cache_key] = result
            return result

        fallback_url = material.image_urls[0] if material.image_urls else None
        fallback_local = material.local_image_paths[0] if material.local_image_paths else None
        result = {
            "texture_path": fallback_local or fallback_url,
            "texture_abs_path": str((self.materials_base_dir / fallback_local).resolve()) if fallback_local else fallback_url,
            "usable_in_blender": False,
            "reason": "no_local_images_to_analyze",
            "analysis": None,
            "all_local_image_analyses": [],
        }
        self._texture_candidate_cache[cache_key] = result
        return result

    def save_selection(self, selection: FlooringSelection, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(selection.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
