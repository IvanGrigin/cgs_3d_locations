import unittest
from pathlib import Path

from src.apply_supplier_bindings import (
    _candidate_from_supplier_catalog_json,
    _computer_item_kind,
    _computer_text_kind,
    _scene_has_tv,
    _semantic_group_for_item,
    apply_supplier_bindings_to_data,
)


class SupplierBindingQualityTests(unittest.TestCase):
    def test_gaming_monitor_is_computer_not_tv(self) -> None:
        item = {
            "id": "monitor_1",
            "semantic_group": "tv_projector_screen",
            "category": "WallMountedTVFactory",
            "name": "LG UltraGear gaming monitor",
        }

        self.assertEqual(_semantic_group_for_item(item, None), "computer")
        self.assertFalse(_scene_has_tv([item], {}))

    def test_real_tv_still_counts_as_tv(self) -> None:
        item = {
            "id": "tv_1",
            "semantic_group": "tv_projector_screen",
            "category": "WallMountedTVFactory",
            "name": "Samsung OLED TV",
        }

        self.assertEqual(_semantic_group_for_item(item, None), "tv_projector_screen")
        self.assertTrue(_scene_has_tv([item], {}))

    def test_computer_kind_classification_keeps_imac_separate_from_macbook(self) -> None:
        self.assertEqual(_computer_text_kind("Apple iMac 2017 all-in-one"), "all_in_one")
        self.assertEqual(_computer_text_kind("MacBook Pro 2015 laptop"), "laptop")
        self.assertEqual(_computer_text_kind("Alienware gaming monitor"), "monitor")
        self.assertEqual(_computer_text_kind("mouse and keyboard bluetooth combo"), "keyboard_mouse")

    def test_catalog_quality_for_computer_and_tv_examples(self) -> None:
        catalog = Path("data/sourse/suppliers/supplier_catalog_canonical.json")
        if not catalog.is_file():
            self.skipTest("supplier catalog is not available")

        monitor_candidate = _candidate_from_supplier_catalog_json(
            {"laptop_computer_keyboard_mouse", "computer", "computer_monitor", "tv_projector_screen"},
            [0.6, 0.25, 0.55],
            computer_kind="monitor",
        )
        self.assertIsNotNone(monitor_candidate)
        assert monitor_candidate is not None
        monitor_text = " ".join(str(monitor_candidate.get(k) or "") for k in ("title", "category_norm", "category_raw"))
        self.assertIn(_computer_text_kind(monitor_text), {"monitor", "all_in_one"})
        self.assertNotEqual(_computer_text_kind(monitor_text), "laptop")

        laptop_candidate = _candidate_from_supplier_catalog_json(
            {"laptop_computer_keyboard_mouse", "computer", "computer_monitor", "tv_projector_screen"},
            [0.35, 0.25, 0.03],
            computer_kind="laptop",
        )
        self.assertIsNotNone(laptop_candidate)
        assert laptop_candidate is not None
        laptop_text = " ".join(str(laptop_candidate.get(k) or "") for k in ("title", "category_norm", "category_raw"))
        self.assertEqual(_computer_text_kind(laptop_text), "laptop")

        tv_candidate = _candidate_from_supplier_catalog_json({"tv_projector_screen"}, [1.1, 0.06, 0.65])
        self.assertIsNotNone(tv_candidate)
        assert tv_candidate is not None
        tv_text = " ".join(str(tv_candidate.get(k) or "") for k in ("title", "category_norm", "category_raw"))
        self.assertNotIn("monitor", tv_text.lower())
        self.assertEqual(tv_candidate.get("category_norm"), "tv_projector_screen")

    def test_imac_replacement_sits_on_table_and_suppresses_keyboard_overlap(self) -> None:
        scene = {
            "room": {"floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
            "placements": [
                {
                    "id": "desk",
                    "category": "SimpleDeskFactory",
                    "name": "desk",
                    "position_m": [1.4, 1.2, 0.38],
                    "aabb": {"x_min": 0.7, "x_max": 2.1, "y_min": 0.8, "y_max": 1.6, "z_min": 0.0, "z_max": 0.76},
                },
                {
                    "id": "imac_target",
                    "category": "MonitorFactory",
                    "name": "iMac computer",
                    "position_m": [1.4, 1.2, 1.2],
                    "aabb": {"x_min": 1.1, "x_max": 1.7, "y_min": 1.0, "y_max": 1.35, "z_min": 1.0, "z_max": 1.6},
                },
                {
                    "id": "keyboard",
                    "category": "KeyboardFactory",
                    "name": "keyboard mouse",
                    "position_m": [1.4, 1.2, 0.80],
                    "aabb": {"x_min": 1.15, "x_max": 1.65, "y_min": 1.05, "y_max": 1.35, "z_min": 0.76, "z_max": 0.84},
                },
            ],
        }

        out = apply_supplier_bindings_to_data(scene, {"bindings": []})
        by_id = {item["id"]: item for item in out["placements"]}
        self.assertIn("imac_target", by_id)
        self.assertNotIn("keyboard", by_id)
        self.assertAlmostEqual(by_id["imac_target"]["aabb"]["z_min"], 0.764, places=3)
        self.assertEqual(_computer_item_kind(by_id["imac_target"]), "all_in_one")


if __name__ == "__main__":
    unittest.main()
