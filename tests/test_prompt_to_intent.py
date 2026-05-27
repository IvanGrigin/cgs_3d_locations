import pytest

pytest.skip("legacy test for archived prompt compiler modules", allow_module_level=True)

from src.prompt_compiler.llm_client import StubLLMClient
from src.prompt_compiler.prompt_to_intent import extract_intent, normalize_intent
from src.prompt_compiler.schemas import RoomType, StyleLabel


def test_style_normalization_japanese_minimalist() -> None:
    intent = extract_intent(
        "small minimalist japanese bedroom, 5 sqm, one bed",
        StubLLMClient(
            {
                "room_type": "Bedroom",
                "style_label": "japandi",
                "target_area_sqm": 5.0,
                "required_objects": ["bed"],
            }
        ),
    )
    assert intent.style.style_label == StyleLabel.JAPANDI
    assert intent.room_type == RoomType.BEDROOM


def test_object_normalization() -> None:
    intent = normalize_intent(
        extract_intent(
            "compact bedroom with nightstand and wardrobe",
            StubLLMClient(
                {
                    "room_type": "Bedroom",
                    "style_label": "scandinavian",
                    "required_objects": ["nightstand", "wardrobe"],
                }
            ),
        )
    )
    assert "SideTable" in intent.objects.required
    assert "Storage" in intent.objects.required


def test_area_extraction_from_dimensions() -> None:
    intent = extract_intent("bedroom 2 x 3 m in scandinavian style", StubLLMClient())
    assert intent.geometry.target_area_sqm == 6.0
