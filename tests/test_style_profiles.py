from src.style_profiles import (
    build_chooser_style_prompt,
    build_style_hint,
    compile_style_profile,
    default_style_label_for_room,
    infer_room_type_from_prompt,
    build_style_profile_from_compiled_policy,
    attach_style_hint_to_room_json,
    STYLE_PROFILES,
)


def test_room_type_inference():
    assert infer_room_type_from_prompt("Уютная ванная с душем") == "Bathroom"
    assert infer_room_type_from_prompt("Кухня в скандинавском стиле") == "Kitchen"
    assert infer_room_type_from_prompt("Спальня для семьи", room_path="bedroom_01.json") == "Bedroom"


def test_default_style_fallback_and_defaults():
    assert default_style_label_for_room("LivingRoom") == "contemporary"
    assert default_style_label_for_room("Unknown") == "minimalism"


def test_compile_style_profile_fills_defaults():
    analysis = {
        "room_type": "Bathroom",
        "style_label": "unknown_style",
        "confidence": 0.2,
    }
    profile = compile_style_profile(analysis, prompt_text="спокойная ванная", room_path="data/bathroom.json")
    assert profile["style_label"] == "minimalism"
    assert profile["room_type"] == "Bathroom"
    assert profile["style_hint"]
    assert profile["chooser_prompt"]


def test_style_hint_and_chooser_prompt():
    profile = STYLE_PROFILES["scandinavian"].copy()
    profile["style_label"] = "scandinavian"
    profile["room_type"] = "Bedroom"
    profile["preferred_colors"] = ["beige", "gray"]
    profile["wall_palette"] = ["beige"]
    profile["floor_palette"] = ["beige"]
    profile["furniture_palette"] = ["beige"]
    profile["material_family"] = ["wood"]
    profile["palette_base"] = ["beige", "gray", "white"]
    profile["surface_design_brief"] = "soft surfaces"

    hint = build_style_hint(profile)
    chooser = build_chooser_style_prompt("Нужна комната", profile)
    assert "scandinavian" in hint
    assert "beige" in chooser

    room_json = {"room": {}}
    attached = attach_style_hint_to_room_json(room_json, profile)
    assert attached["room"]["style_label"] == "scandinavian"
    assert attached["room"]["style_profile"]["style_label"] == "scandinavian"
