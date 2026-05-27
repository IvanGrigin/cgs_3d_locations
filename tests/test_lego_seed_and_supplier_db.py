from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.ml import lego_seed_scene as lego
from src.suppliers import db, db_core
from src.suppliers.models import ProductRecord
from src.suppliers.site_models import SupplierAssetRecord


def _room() -> dict:
    return {
        "room": {
            "id": "r1",
            "ceiling_height": 3.0,
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}, {"x": 0, "y": 3}],
        }
    }


def _product(unique_key: str = "unit::1", title: str = "Chair") -> ProductRecord:
    return ProductRecord(
        unique_key=unique_key,
        source_site="unit",
        source_url="https://example.test/item",
        parsed_at="2026-01-01T00:00:00Z",
        external_id="1",
        title=title,
        product_url="https://example.test/item",
        category_norm="chair",
        price_value=10.0,
        tags_json='["tag"]',
        images_json='["image.jpg"]',
        extra_json='{"ok": true}',
    )


def test_lego_seed_scene_geometry_sorting_io_and_generation(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "room.json"
    lego.save_json(path, _room())
    assert lego.load_json(path)["room"]["id"] == "r1"

    poly = lego.room_polygon_xy(_room())
    assert lego.polygon_area(poly) == 12.0
    assert lego.room_area_m2(_room()) == 12.0
    assert lego.point_in_polygon_xy(2, 2, poly)
    assert not lego.point_in_polygon_xy(5, 5, poly)
    with pytest.raises(RuntimeError):
        lego.room_polygon_xy({"room": {"floor_polygon": [[0, 0], [1, 0]]}})

    monkeypatch.setattr(lego.random, "uniform", lambda lo, hi: lo + (hi - lo) * 0.25)
    assert lego.sample_point_in_polygon(poly, max_tries=1) == (1.0, 0.75)
    monkeypatch.setattr(lego.random, "uniform", lambda lo, hi: hi + 10.0)
    assert lego.sample_point_in_polygon(poly, max_tries=2) == (2.0, 1.5)

    aabb = lego.build_aabb_from_center_size([2, 2, 1], [2, 4, 2])
    assert aabb == {"x_min": 1.0, "x_max": 3.0, "y_min": 0.0, "y_max": 4.0, "z_min": 0.0, "z_max": 2.0}
    assert lego.object_footprint_m2({"size_m": [2, 3, 1]}) == 6.0
    assert lego.object_footprint_m2({"size_m": "bad"}) == 0.0

    objects = {
        "objects": [
            {"id": "lamp", "name": "ceiling lamp", "category": "lamp", "size_m": [1, 1, 0.2], "constraints": {"mount_type": "ceiling"}},
            {"id": "chair", "name": "chair", "category": "chair", "size_m": [0.5, 0.5, 1.0]},
            {"id": "bed", "name": "bed", "category": "bed", "size_m": [2, 2, 1.0]},
            "bad",
        ]
    }
    assert lego.total_objects_footprint_m2(objects) == pytest.approx(4.25)
    sorted_objects = lego.sort_objects_for_generation(objects)
    assert [obj["id"] for obj in sorted_objects["objects"]] == ["bed", "chair", "lamp"]
    assert [obj["id"] for obj in lego.crop_last_object(sorted_objects)["objects"]] == ["bed", "chair"]

    monkeypatch.setattr(lego.random, "uniform", lambda lo, hi: (lo + hi) / 2.0)
    scene, placement = lego.build_seed_scene_and_placement(_room(), sorted_objects, seed=7)
    assert scene["schema"] == "scene.v1"
    assert placement["schema"] == "placement.v1"
    by_id = {item["id"]: item for item in placement["placements"]}
    assert by_id["lamp"]["position_m"][2] == pytest.approx(2.9)
    assert by_id["bed"]["aabb"]["z_min"] == pytest.approx(0.0)
    assert by_id["chair"]["rotation_deg"] in lego.ALLOWED_ROTATIONS


def test_supplier_db_and_db_core_create_upsert_and_log_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "suppliers.sqlite"
    db_core.init_db(db_path)
    product = _product()
    db.upsert_product(db_path, product)
    updated = _product(title="Updated chair")
    db.upsert_products(db_path, [updated])
    db.insert_fetch_log(db_path, "unit", "https://example.test/item", "2026-01-01T00:00:00Z", True, None)
    db_core.insert_download(
        db_path,
        unique_key=product.unique_key,
        downloaded_at="2026-01-01T00:00:01Z",
        final_url="https://cdn.example.test/model.glb",
        local_path="/tmp/model.glb",
        filename="model.glb",
        content_type="model/gltf-binary",
        ok=True,
        size_bytes=123,
        error=None,
    )
    asset = SupplierAssetRecord(
        unique_key=product.unique_key,
        updated_at="2026-01-01T00:00:02Z",
        source_site="unit",
        product_url=product.product_url,
        title=product.title,
        asset_status="ready",
        asset_format=".glb",
        asset_source_url="https://cdn.example.test/model.glb",
        asset_local_path="/tmp/model.glb",
        preview_local_path="/tmp/preview.png",
        blender_job_path="/tmp/job",
        notes_json='["ok"]',
        extra_json='{"source": "unit"}',
    )
    db_core.upsert_asset(db_path, asset)
    db_core.upsert_asset(db_path, SupplierAssetRecord(**{**asset.__dict__, "title": "Updated asset"}))

    with sqlite3.connect(db_path) as con:
        product_row = con.execute("SELECT title, tags_json, images_json, extra_json FROM supplier_product WHERE unique_key=?", (product.unique_key,)).fetchone()
        assert product_row == ("Updated chair", '["tag"]', '["image.jpg"]', '{"ok": true}')
        assert con.execute("SELECT ok, error FROM supplier_fetch_log").fetchone() == (1, None)
        assert con.execute("SELECT ok, size_bytes FROM supplier_download").fetchone() == (1, 123)
        assert con.execute("SELECT title, asset_status FROM supplier_asset WHERE unique_key=?", (product.unique_key,)).fetchone() == ("Updated asset", "ready")
