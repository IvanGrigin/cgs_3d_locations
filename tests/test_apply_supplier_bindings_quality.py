from pathlib import Path

import pytest

from src import apply_supplier_bindings as asb
from tests.helpers.scene_builders import aabb_item, box


def test_gaming_monitor_is_computer_not_tv():
    item = {"id": "monitor_1", "semantic_group": "tv_projector_screen", "category": "WallMountedTVFactory", "name": "LG UltraGear gaming monitor"}

    assert asb._semantic_group_for_item(item, None) == "computer"
    assert not asb._scene_has_tv([item], {})


def test_real_tv_still_counts_as_tv():
    item = {"id": "tv_1", "semantic_group": "tv_projector_screen", "category": "WallMountedTVFactory", "name": "Samsung OLED TV"}

    assert asb._semantic_group_for_item(item, None) == "tv_projector_screen"
    assert asb._scene_has_tv([item], {})


def test_computer_kind_classification_keeps_imac_separate_from_macbook():
    assert asb._computer_text_kind("Apple iMac 2017 all-in-one") == "all_in_one"
    assert asb._computer_text_kind("MacBook Pro 2015 laptop") == "laptop"
    assert asb._computer_text_kind("Alienware gaming monitor") == "monitor"
    assert asb._computer_text_kind("mouse and keyboard bluetooth combo") == "keyboard_mouse"


@pytest.fixture(scope="module")
def canonical_supplier_catalog():
    catalog = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
    if not catalog.is_file():
        pytest.skip("supplier catalog is not available")
    return asb.read_json(catalog)


def test_catalog_quality_for_computer_and_tv_examples(monkeypatch, canonical_supplier_catalog):
    monkeypatch.setattr(asb, "read_json", lambda _path: canonical_supplier_catalog)

    monitor_candidate = asb._candidate_from_supplier_catalog_json(
        {"laptop_computer_keyboard_mouse", "computer", "computer_monitor", "tv_projector_screen"},
        [0.6, 0.25, 0.55],
        computer_kind="monitor",
    )
    assert monitor_candidate is not None
    monitor_text = " ".join(str(monitor_candidate.get(k) or "") for k in ("title", "category_norm", "category_raw"))
    assert asb._computer_text_kind(monitor_text) in {"monitor", "all_in_one"}
    assert asb._computer_text_kind(monitor_text) != "laptop"

    laptop_candidate = asb._candidate_from_supplier_catalog_json(
        {"laptop_computer_keyboard_mouse", "computer", "computer_monitor", "tv_projector_screen"},
        [0.35, 0.25, 0.03],
        computer_kind="laptop",
    )
    assert laptop_candidate is not None
    laptop_text = " ".join(str(laptop_candidate.get(k) or "") for k in ("title", "category_norm", "category_raw"))
    assert asb._computer_text_kind(laptop_text) == "laptop"

    tv_candidate = asb._candidate_from_supplier_catalog_json({"tv_projector_screen"}, [1.1, 0.06, 0.65])
    assert tv_candidate is not None
    tv_text = " ".join(str(tv_candidate.get(k) or "") for k in ("title", "category_norm", "category_raw"))
    assert "monitor" not in tv_text.lower()
    assert tv_candidate.get("category_norm") == "tv_projector_screen"


def test_imac_replacement_sits_on_table_and_suppresses_keyboard_overlap(monkeypatch):
    monkeypatch.setattr(
        asb,
        "_candidate_from_supplier_catalog_json",
        lambda *_args, **_kwargs: {
            "unique_key": "test::imac",
            "title": "Apple iMac all-in-one",
            "category_norm": "computer",
            "width_cm": 60,
            "depth_cm": 20,
            "height_cm": 45,
        },
    )
    scene = {
        "room": {"floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
        "placements": [
            aabb_item("desk", "SimpleDeskFactory", box(0.7, 2.1, 0.8, 1.6, 0.0, 0.76)),
            aabb_item("imac_target", "MonitorFactory", box(1.1, 1.7, 1.0, 1.35, 1.0, 1.6), name="iMac computer"),
            aabb_item("keyboard", "KeyboardFactory", box(1.15, 1.65, 1.05, 1.35, 0.76, 0.84), name="keyboard mouse"),
        ],
    }

    out = asb.apply_supplier_bindings_to_data(scene, {"bindings": []})
    by_id = {item["id"]: item for item in out["placements"]}

    assert "imac_target" in by_id
    assert "keyboard" not in by_id
    assert by_id["imac_target"]["aabb"]["z_min"] == pytest.approx(0.764)
    assert asb._computer_item_kind(by_id["imac_target"]) == "all_in_one"
