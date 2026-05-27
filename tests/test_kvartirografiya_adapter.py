from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.skip("legacy test for archived module src.tools.kvartirografiya_adapter", allow_module_level=True)

from src.tools.kvartirografiya_adapter import FloorInput, convert_floor


def _polygon(coords, props):
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [coords]},
    }


def test_convert_floor_exports_filtered_room_specs(tmp_path: Path) -> None:
    project_dir = tmp_path / "1001"
    project_dir.mkdir()
    geojson = project_dir / "project_1001_floor_2.geojson"
    data = {
        "type": "FeatureCollection",
        "features": [
            _polygon(
                [[37.0, 55.0], [37.0002, 55.0], [37.0002, 55.0002], [37.0, 55.0002], [37.0, 55.0]],
                {"type": "apartment", "totalArea": 40.0, "livingArea": 20.0, "apartmentType": 1},
            ),
            _polygon(
                [[37.00001, 55.00001], [37.00009, 55.00001], [37.00009, 55.00009], [37.00001, 55.00009], [37.00001, 55.00001]],
                {"type": "room", "roomType": "KITCHEN", "area": 9.0},
            ),
            _polygon(
                [[37.00011, 55.00001], [37.00019, 55.00001], [37.00019, 55.00009], [37.00011, 55.00009], [37.00011, 55.00001]],
                {"type": "room", "roomType": "BEDROOM", "area": 12.0},
            ),
            _polygon(
                [[37.00001, 55.00011], [37.00009, 55.00011], [37.00009, 55.00019], [37.00001, 55.00019], [37.00001, 55.00011]],
                {"type": "room", "roomType": "BALCONY", "area": 3.0},
            ),
        ],
    }
    geojson.write_text(json.dumps(data), encoding="utf-8")

    manifest = convert_floor(
        FloorInput(
            project_id="1001",
            floor=2,
            project_dir=project_dir,
            geojson_path=geojson,
            dxf_path=None,
            floor_xlsx_path=None,
            project_xlsx_path=None,
            floor_jpg_path=None,
            overview_jpg_path=None,
        ),
        tmp_path / "out",
        {"KITCHEN", "BEDROOM"},
    )

    assert manifest["apartments_count"] == 1
    assert manifest["apartment_specs_count"] == 1
    assert manifest["rooms_count"] == 2
    apartment = json.loads(Path(manifest["apartments"][0]["room_json"]).read_text(encoding="utf-8"))["room"]
    bundle_manifest = json.loads(Path(manifest["apartments"][0]["bundle_manifest"]).read_text(encoding="utf-8"))
    assert bundle_manifest["rooms_count"] == 2
    assert all(Path(row["room_json"]).is_file() for row in bundle_manifest["rooms"])
    assert apartment["room_type"] == "apartment"
    assert apartment["type"] == "apartment"
    assert apartment["meta"]["apartment_id"] == "apt_0001"
    assert len(apartment["meta"]["child_rooms"]) == 2
    graph = apartment["meta"]["synthetic_door_graph"]
    assert graph["is_connected"] is True
    assert graph["internal_doors_count"] == 1
    assert graph["minimum_internal_doors_for_connectivity"] == 1
    assert len(graph["doors"]) == 2
    assert graph["doors"][0]["kind"] == "entrance"
    assert {door["kind"] for door in graph["doors"]} == {"entrance", "internal"}
    assert apartment["meta"]["door_graph"]["source"] == "synthetic"
    assert len(apartment["doors"]) == 1
    assert apartment["meta"]["window_graph"]["source"] == "none"
    assert apartment["meta"]["window_graph"]["windows_count"] == 0
    assert all(room["synthetic_door_ids"] for room in apartment["meta"]["child_rooms"])
    assert all(room["door_ids"] for room in apartment["meta"]["child_rooms"])
    assert {row["source_room_type"] for row in manifest["rooms"]} == {"KITCHEN", "BEDROOM"}
    for row in manifest["rooms"]:
        payload = json.loads(Path(row["room_json"]).read_text(encoding="utf-8"))
        assert payload["units"] == "m"
        room = payload["room"]
        assert room["type"] == room["room_type"]
        assert room["meta"]["apartment_id"] == "apt_0001"
        assert len(room["floor_polygon"]) == 4
        assert room["width_m"] > 0.0
        assert room["depth_m"] > 0.0
        assert room["doors"]
        assert isinstance(room["windows"], list)


def test_convert_floor_prefers_real_generated_doors(tmp_path: Path) -> None:
    project_dir = tmp_path / "1002"
    project_dir.mkdir()
    geojson = project_dir / "project_1002_floor_2.geojson"
    data = {
        "type": "FeatureCollection",
        "features": [
            _polygon(
                [[37.0, 55.0], [37.0002, 55.0], [37.0002, 55.0002], [37.0, 55.0002], [37.0, 55.0]],
                {
                    "type": "apartment",
                    "totalArea": 40.0,
                    "livingArea": 20.0,
                    "apartmentType": 1,
                    "entranceDoor": [
                        {
                            "point": {"type": "Point", "coordinates": [37.00002, 55.00001]},
                            "line": {"type": "LineString", "coordinates": [[37.00001, 55.00001], [37.00003, 55.00001]]},
                            "width": 0.9,
                        }
                    ],
                    "interiorDoors": [
                        {
                            "point": {"type": "Point", "coordinates": [37.0001, 55.00005]},
                            "line": {"type": "LineString", "coordinates": [[37.0001, 55.00004], [37.0001, 55.00006]]},
                            "width": 0.8,
                        }
                    ],
                },
            ),
            _polygon(
                [[37.00001, 55.00001], [37.00009, 55.00001], [37.00009, 55.00009], [37.00001, 55.00009], [37.00001, 55.00001]],
                {"type": "room", "roomType": "HALL", "area": 9.0},
            ),
            _polygon(
                [[37.00011, 55.00001], [37.00019, 55.00001], [37.00019, 55.00009], [37.00011, 55.00009], [37.00011, 55.00001]],
                {"type": "room", "roomType": "BEDROOM", "area": 12.0},
            ),
        ],
    }
    geojson.write_text(json.dumps(data), encoding="utf-8")

    manifest = convert_floor(
        FloorInput(
            project_id="1002",
            floor=2,
            project_dir=project_dir,
            geojson_path=geojson,
            dxf_path=None,
            floor_xlsx_path=None,
            project_xlsx_path=None,
            floor_jpg_path=None,
            overview_jpg_path=None,
        ),
        tmp_path / "out",
        {"HALL", "BEDROOM"},
    )

    apartment = json.loads(Path(manifest["apartments"][0]["room_json"]).read_text(encoding="utf-8"))["room"]
    graph = apartment["meta"]["door_graph"]
    assert graph["source"] == "real"
    assert len(apartment["doors"]) == 1
    assert graph["real_doors_count"] == 2
    assert {door["source"] for door in graph["doors"]} == {"real_generated_linear_object"}
    assert apartment["meta"]["real_door_graph"]["is_connected"] is True
    assert all(room["real_door_ids"] for room in apartment["meta"]["child_rooms"])
