import json
from pathlib import Path

from src.pipeline import curtain_stage as cs


def test_load_curtain_catalog_filters_by_source_and_image_paths(tmp_path: Path) -> None:
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()

    jsonl = catalog_dir / "shtorystore_curtains.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source": "shtorystore",
                        "title": "main",
                        "local_image_paths": ["preview.jpg"],
                    }
                ),
                json.dumps({"source": "other", "local_image_paths": ["preview.jpg"]}),
                json.dumps({"source": "shtorystore", "local_image_paths": []}),
            ]
        ),
        encoding="utf-8",
    )

    rows, base_dir = cs.load_curtain_catalog(catalog_dir / "shtorystore_curtains.jsonl")
    assert len(rows) == 1
    assert str(rows[0]["source"]) == "shtorystore"
    assert base_dir == catalog_dir.resolve()

    (catalog_dir / "products.jsonl").write_text(
        "\n".join(
            [
                "",
                "{not json",
                json.dumps({"source": "shtorystore", "title": "secondary", "local_image_paths": ["a.jpg"]}),
            ]
        ),
        encoding="utf-8",
    )
    (catalog_dir / "shtorystore_curtains.jsonl").unlink()
    rows, base_dir = cs.load_curtain_catalog(catalog_dir)
    assert [row["title"] for row in rows] == ["secondary"]
    assert base_dir == catalog_dir.resolve()

    assert cs.load_curtain_catalog(catalog_dir / "missing.jsonl")[0] == []
    out_json = tmp_path / "nested" / "out.json"
    cs.write_json(out_json, {"ok": True})
    assert json.loads(out_json.read_text(encoding="utf-8")) == {"ok": True}


def test_discover_curtain_models_sorted_paths(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "a.fbx").write_text("x", encoding="utf-8")
    (models / "b.glb").write_text("x", encoding="utf-8")
    (models / "c.obj").write_text("x", encoding="utf-8")
    (models / "d.fbx").write_text("x", encoding="utf-8")

    discovered = cs.discover_curtain_models(models)
    assert len(discovered) == 4
    assert discovered[0].endswith("a.fbx")
    assert discovered[1].endswith("d.fbx")


def test_discover_curtain_models_empty_returns_empty() -> None:
    assert cs.discover_curtain_models(None) == []
    assert cs.discover_curtain_models("/definitely/missing") == []


def test_is_primary_plain_curtain_model() -> None:
    assert cs._is_primary_plain_curtain_model({"asset_local_path": "/tmp/shtora.fbx", "title": "Стора"})
    assert not cs._is_primary_plain_curtain_model({"asset_local_path": "/tmp/curtain_2.fbx", "title": "Гофрированный штор"})
    assert not cs._is_primary_plain_curtain_model("/tmp/random.glb")
    assert cs._curtain_model_rank_key({"asset_local_path": "/tmp/shtora.fbx", "title": "штора"})[0] < 0
    assert cs._curtain_model_rank_key({"asset_local_path": "/tmp/french_lace.obj", "title": "French lace"})[0] > 0


def test_discover_supplier_curtain_models_prefers_manual_assets(tmp_path: Path) -> None:
    manual = tmp_path / "manual_assets"
    (manual / "nested").mkdir(parents=True)
    (manual / "nested" / "shtora.fbx").write_text("x", encoding="utf-8")
    (manual / "nested" / "ignore.txt").write_text("x", encoding="utf-8")

    rows = cs.discover_supplier_curtain_models(
        supplier_catalog_path=None,
        manual_assets_root=manual,
    )
    assert len(rows) >= 1
    assert rows[0]["source"] == "supplier_manual_assets"
    assert rows[0]["title"] == "nested"


def test_discover_supplier_curtain_models_catalog_and_fallback_sorting(tmp_path: Path) -> None:
    catalog_asset = tmp_path / "catalog_shtora.fbx"
    catalog_asset.write_text("fbx", encoding="utf-8")
    catalog_ignored = tmp_path / "catalog_ignore.obj"
    catalog_ignored.write_text("obj", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "category_norm": "curtain_blinds",
                        "asset_local_path": str(catalog_asset),
                        "title": "Шторы простые",
                        "unique_key": "curtain-key",
                        "asset_status": "ready",
                    },
                    {"category_norm": "chair", "asset_local_path": str(catalog_ignored)},
                    "bad",
                ]
            }
        ),
        encoding="utf-8",
    )

    manual = tmp_path / "manual"
    (manual / "lace").mkdir(parents=True)
    lace = manual / "lace" / "french_curtain.obj"
    lace.write_text("obj", encoding="utf-8")
    rows = cs.discover_supplier_curtain_models(catalog, manual)

    assert {row["source"] for row in rows} == {"supplier_catalog", "supplier_manual_assets"}
    assert any(row["unique_key"] == "curtain-key" for row in rows)
    assert all(row["asset_format"] in {"fbx", "obj"} for row in rows)

    bad_catalog = tmp_path / "bad_catalog.json"
    bad_catalog.write_text("{not json", encoding="utf-8")
    fallback_rows = cs.discover_supplier_curtain_models(bad_catalog, manual)
    assert len(fallback_rows) == 1
    assert fallback_rows[0]["source"] == "supplier_manual_assets"
