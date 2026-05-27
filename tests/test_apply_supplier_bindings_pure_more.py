from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src import apply_supplier_bindings as asb
from tests.helpers.scene_builders import scene_with_room


def box(x1, x2, y1, y2, z1, z2):
    return {"x_min": x1, "x_max": x2, "y_min": y1, "y_max": y2, "z_min": z1, "z_max": z2}


def item(item_id, category, aabb, **extra):
    payload = {
        "id": item_id,
        "name": category,
        "category": category,
        "position_m": asb._aabb_center(aabb),
        "size_m": [aabb["x_max"] - aabb["x_min"], aabb["y_max"] - aabb["y_min"], aabb["z_max"] - aabb["z_min"]],
        "aabb": dict(aabb),
    }
    payload.update(extra)
    return payload


def candidate(path: Path, *, unique_key: str = "cand", fmt: str | None = None, group: str = "chair"):
    return {
        "unique_key": unique_key,
        "title": f"{group} candidate",
        "semantic_group": group,
        "category_norm": group,
        "asset_local_path": str(path),
        "asset_format": fmt or path.suffix.lstrip("."),
        "asset_status": "local_supplier_asset",
        "width_cm": 50,
        "depth_cm": 60,
        "height_cm": 90,
        "source_site": "test",
        "product_url": "https://example.test/p",
    }


def selected_binding(candidate_payload, *, target_id="target", group="chair"):
    return {
        "target_id": target_id,
        "semantic_group": group,
        "selection_status": "heuristic_top1_selected",
        "provenance": {"final_asset_source": "supplier_catalog"},
        "chosen_candidate": candidate_payload,
        "top_candidates": [candidate_payload, dict(candidate_payload, unique_key="dup")],
    }


def test_candidate_asset_reference_and_geometry_helpers(tmp_path: Path):
    scene = tmp_path / "scene.json"
    blend = tmp_path / "infinigen_clean_scene.blend"
    scene.write_text(json.dumps({"meta": {"placement_meta": {"scene_blend": "other.blend"}}}), encoding="utf-8")
    blend.write_bytes(b"blend")
    assert asb._infer_reference_scene_blend_path(scene, json.loads(scene.read_text(encoding="utf-8"))) == str(blend.resolve())

    mesh_dir = tmp_path / "asset"
    mesh_dir.mkdir()
    fbx = mesh_dir / "model.fbx"
    obj = mesh_dir / "model.obj"
    glb = mesh_dir / "model.glb"
    proxy = mesh_dir / "proxy.glb"
    for path in (fbx, obj, glb, proxy):
        path.write_bytes(b"x")

    cand = candidate(mesh_dir, unique_key="asset-dir")
    paths = asb._candidate_asset_paths(cand)
    assert str(fbx.resolve()) in paths
    assert str(obj.resolve()) in paths
    assert str(glb.resolve()) in paths

    fbx_asset, placeholder = asb._candidate_asset(candidate(fbx), require_local_asset=True)
    assert fbx_asset["asset_format"] == "fbx"
    assert placeholder is False

    glb_asset, _ = asb._candidate_asset(dict(candidate(glb), asset_status=asb.TRELLIS_ASSET_STATUS), require_local_asset=False)
    assert glb_asset["asset_source"] == "trellis_generated_local_asset"
    assert glb_asset["mesh_fit_mode"] == "stretch"

    proxy_asset, placeholder = asb._candidate_asset({"asset_status": "needs_blender_rebuild"}, require_local_asset=False, fallback_mode=asb.ASSET_FALLBACK_MODE_FBX_OBJ_PROXY)
    assert proxy_asset["kind"] == "procedural_proxy"
    assert placeholder is False

    empty_asset, placeholder = asb._candidate_asset({"asset_status": "needs_blender_rebuild"}, require_local_asset=False)
    assert empty_asset == {}
    assert placeholder is True

    binding = selected_binding(candidate(fbx), group="desk")
    pool = asb._compact_candidate_pool(binding, limit=2)
    assert pool[0]["unique_key"] == "cand"
    assert asb._replacement_mesh_fit_mode({"semantic_group": "lamp_ceiling"}, {"constraints": {"mount_type": "ceiling"}}) == "stretch"

    rotated = item("desk", "SimpleDeskFactory", box(0, 1, 0, 1, 0, 1), rotation_deg=90)
    asb._apply_geometry_from_candidate(rotated, [2, 1, 0.5])
    assert rotated["size_m"] == [2.0, 1.0, 0.5]
    assert pytest.approx(rotated["aabb"]["x_max"] - rotated["aabb"]["x_min"]) == 1.0
    assert pytest.approx(rotated["aabb"]["y_max"] - rotated["aabb"]["y_min"]) == 2.0
    assert asb._should_apply_candidate_geometry(rotated, {"semantic_group": "computer"}, [1, 1, 1], {"mesh_path": "x"})
    assert not asb._should_apply_candidate_geometry(rotated, {"semantic_group": "chair"}, None, {})


def test_catalog_candidate_semantic_and_asset_edge_branches(monkeypatch, tmp_path: Path):
    proxy = tmp_path / "built" / "proxy.glb"
    proxy.parent.mkdir()
    proxy.write_bytes(b"proxy")
    real_glb = tmp_path / "real.glb"
    real_glb.write_bytes(b"glb")
    real_obj = tmp_path / "real.obj"
    real_obj.write_bytes(b"obj")

    assert asb._candidate_asset(None, require_local_asset=False) == ({}, False)
    assert asb._candidate_asset({"asset_local_path": str(proxy)}, require_local_asset=True) == ({}, False)
    generated_proxy = {
        "asset_status": asb.TRELLIS_ASSET_STATUS,
        "asset_local_path": str(real_glb),
        "extra": {"trellis_generated_asset": {"asset_generation_method": "proxy_glb_fallback"}},
    }
    asset_block, placeholder = asb._candidate_asset(generated_proxy, require_local_asset=False)
    assert placeholder is False
    assert asset_block["asset_source"] == "supplier_catalog_procedural_proxy"
    obj_block, _placeholder = asb._candidate_asset({"asset_local_path": str(real_obj)}, require_local_asset=False)
    assert obj_block["asset_format"] == "obj"
    assert asb._candidate_asset({}, require_local_asset=False) == ({}, False)

    assert (
        asb._semantic_group_for_item(
            {
                "category": "WallMountedTVFactory",
                "name": "display",
                "meta": {"supplier_candidate": {"semantic_group": "tv_projector_screen", "title": "gaming monitor"}},
            }
        )
        == "computer"
    )
    assert asb._semantic_group_for_item({"name": "настольная лампа"}) == "lamp_table"
    assert asb._semantic_group_for_item({"name": "coffee table"}) == "coffee_table"
    assert asb._semantic_group_for_item({"name": "кресло мягкое"}) == "armchair"
    assert asb._semantic_group_for_item({"name": "unknown"}) == ""
    assert asb._computer_text_kind("keyboard and mouse") == "keyboard_mouse"
    assert asb._computer_text_kind("MacBook laptop") == "laptop"
    assert asb._computer_text_kind("gaming monitor") == "monitor"
    assert asb._computer_text_kind("desktop pc") == "desktop"

    catalog_rows = [
        "bad",
        {"title": "Tiny TV", "category_norm": "tv_projector_screen", "width_cm": 50, "depth_cm": 10, "height_cm": 30, "asset_local_path": str(real_glb)},
        {"title": "OLED smart TV", "category_norm": "tv_projector_screen", "width_cm": 120, "depth_cm": 7, "height_cm": 70, "asset_local_path": str(real_glb), "source_site": "3ddd"},
        {"title": "Office washing machine", "category_norm": "computer", "width_cm": "bad", "asset_local_path": str(real_glb)},
        {"title": "Gaming monitor display", "category_norm": "tv_projector_screen", "width_cm": 60, "depth_cm": 18, "height_cm": 45, "asset_local_path": str(real_glb)},
        {"title": "MacBook laptop", "category_norm": "laptop_computer_keyboard_mouse", "width_cm": 32, "depth_cm": 22, "height_cm": 2, "asset_local_path": str(real_glb), "asset_status": "local_dir_preferred"},
        {"title": "iMac all-in-one", "category_norm": "computer", "width_cm": 60, "depth_cm": 20, "height_cm": 45, "asset_local_path": str(real_glb)},
    ]
    monkeypatch.setattr(asb.Path, "is_file", lambda self: True)
    monkeypatch.setattr(asb, "read_json", lambda _path: {"items": catalog_rows})
    monkeypatch.setattr(asb, "_candidate_has_supported_local_asset", lambda candidate: bool(candidate.get("asset_local_path")))

    tv = asb._candidate_from_supplier_catalog_json({"tv_projector_screen"}, [1.2, 0.08, 0.7])
    assert tv["title"] == "OLED smart TV"
    assert tv["semantic_group"] == "tv_projector_screen"
    monitor = asb._candidate_from_supplier_catalog_json({"computer", "tv_projector_screen"}, [0.6, 0.2, 0.45], computer_kind="monitor")
    assert monitor["semantic_group"] == "computer"
    laptop = asb._candidate_from_supplier_catalog_json({"laptop_computer_keyboard_mouse"}, [0.32, 0.22, 0.03], computer_kind="laptop")
    assert laptop["title"] == "MacBook laptop"
    assert asb._candidate_from_supplier_catalog_json({"missing"}, [1, 1, 1]) is None

    monkeypatch.setattr(asb, "read_json", lambda _path: [])
    assert asb._candidate_from_supplier_catalog_json({"computer"}, [1, 1, 1]) is None
    monkeypatch.setattr(asb, "read_json", lambda _path: (_ for _ in ()).throw(RuntimeError("bad json")))
    assert asb._candidate_from_supplier_catalog_json({"computer"}, [1, 1, 1]) is None

    fallback_items = [item("a", "chair", box(2, 4, 3, 5, 0, 1))]
    assert asb._room_center_xy({"room": {}}, fallback_items) == (3.0, 4.0)
    assert asb._room_center_xy({"room": {}}, []) == (0.0, 0.0)
    assert asb._room_xy_bounds({"room": {}}, fallback_items) == {"x_min": 2, "x_max": 4, "y_min": 3, "y_max": 5}
    assert asb._room_xy_bounds({"room": {}}, []) is None


def test_candidate_path_geometry_semantic_and_catalog_remaining_edges(monkeypatch, tmp_path: Path):
    asset_dir = tmp_path / "assets"
    nested_dir = asset_dir / "nested"
    nested_dir.mkdir(parents=True)
    fbx = asset_dir / "model.fbx"
    gltf = nested_dir / "model.gltf"
    txt = asset_dir / "readme.txt"
    for path in (fbx, gltf, txt):
        path.write_bytes(b"x")

    payload = {
        "asset": {"mesh_local_path": str(asset_dir)},
        "source": {"downloaded_path": str(txt)},
        "extra": {"trellis_generated_asset": {"gltf_path": str(gltf)}},
    }
    paths = asb._candidate_asset_paths(payload)
    assert str(fbx.resolve()) in paths
    assert str(gltf.resolve()) in paths
    assert all(not p.endswith(".txt") for p in paths)
    normalized = asb._normalize_supplier_catalog_candidate(
        {"asset_local_path": str(fbx), "dimensions_cm": {"width": 12, "depth": 34, "height": 56}}
    )
    assert normalized["width_cm"] == 12
    assert normalized["asset_format"] == "fbx"

    ceiling = item("ceiling", "CeilingLightFactory", box(0, 1, 0, 1, 2, 3), constraints={"mount_type": "ceiling"})
    asb._apply_geometry_from_candidate(ceiling, [0.5, 0.4, 0.2])
    assert ceiling["aabb"]["z_max"] == pytest.approx(3.0)
    wall = {"id": "wall", "position_m": [1, 2, 1.5], "size_m": [1, 1, 1], "constraints": {"mount_type": "wall"}}
    asb._apply_geometry_from_candidate(wall, [0.6, 0.2, 0.8])
    assert wall["position_m"][2] == pytest.approx(1.5)
    floor = {"id": "floor", "position_m": [1, 2, 1.5], "size_m": [1, 1, 1]}
    asb._apply_geometry_from_candidate(floor, [0.6, 0.2, 0.8])
    assert floor["aabb"]["z_min"] == pytest.approx(1.1)
    assert asb._item_has_scene_geometry({"size_m": [1, 2, 3]})
    assert not asb._item_has_scene_geometry({"size_m": [1, 2]})
    assert asb._should_apply_candidate_geometry({"constraints": {"mount_type": "ceiling"}}, {"semantic_group": "chair"}, [1, 1, 1], {"mesh_path": "x"})
    assert asb._replacement_mesh_fit_mode({"semantic_group": "chair"}, {"asset": {"asset_source": "trellis_generated_local_asset"}}) == "trellis_stretch"
    assert asb._replacement_mesh_fit_mode({"semantic_group": "dresser"}, {"category": ""}) == "uniform"
    assert asb._replacement_mesh_fit_mode({"semantic_group": "chair"}, {"category": "SingleCabinetFactory"}) == "uniform"

    for text, expected in [
        ("tv stand cabinet", "tv_stand"),
        ("large television panel", "tv_projector_screen"),
        ("king кровать", "bed"),
        ("диван угловой", "sofa"),
        ("люстра crystal", "lamp_ceiling"),
        ("dining table", "dining_table"),
        ("office desk", "desk"),
        ("wooden chair", "chair"),
    ]:
        assert asb._semantic_group_for_item({"name": text}) == expected
    assert asb._looks_like_tv_text("OLED smart tv")
    assert not asb._looks_like_tv_text("gaming monitor")
    assert asb._computer_item_kind({"name": "keyboard mouse only"}) == "keyboard_mouse"

    real_glb = tmp_path / "catalog.glb"
    real_glb.write_bytes(b"glb")
    catalog_rows = [
        {"title": "Samsung smart TV", "category_norm": "tv_projector_screen", "width_cm": 20, "depth_cm": 30, "height_cm": 20, "asset_local_path": str(real_glb)},
        {"title": "Thick TV box", "category_norm": "tv_projector_screen", "width_cm": 160, "depth_cm": 35, "height_cm": 90, "asset_local_path": str(real_glb)},
        {"title": "Thin OLED TV", "category_norm": "tv_projector_screen", "width_cm": 150, "depth_cm": 6, "height_cm": 85, "asset_local_path": str(real_glb), "source_site": "3ddd"},
        {"title": "Desktop PC workstation", "category_norm": "computer", "width_cm": 60, "depth_cm": 50, "height_cm": 50, "asset_local_path": str(real_glb)},
        {"title": "Router device", "category_norm": "computer", "width_cm": 20, "depth_cm": 20, "height_cm": 10, "asset_local_path": str(real_glb)},
        {"title": "Gaming monitor", "category_norm": "tv_projector_screen", "width_cm": 60, "depth_cm": 12, "height_cm": 40, "asset_local_path": str(real_glb)},
    ]
    monkeypatch.setattr(asb.Path, "is_file", lambda self: True)
    monkeypatch.setattr(asb, "read_json", lambda _path: {"items": catalog_rows})
    monkeypatch.setattr(asb, "_candidate_has_supported_local_asset", lambda candidate: bool(candidate.get("asset_local_path")))
    assert asb._candidate_from_supplier_catalog_json({"tv_projector_screen"}, [1.5, 0.08, 0.85])["title"] == "Thin OLED TV"
    assert asb._candidate_from_supplier_catalog_json({"computer"}, [0.6, 0.5, 0.5], computer_kind="desktop")["title"] == "Desktop PC workstation"
    assert asb._candidate_from_supplier_catalog_json({"computer", "tv_projector_screen"}, [0.6, 0.12, 0.4], computer_kind="all_in_one")["semantic_group"] == "computer"


def test_related_generated_items_and_light_normalization(tmp_path: Path):
    mesh = tmp_path / "bed.fbx"
    mesh.write_bytes(b"x")
    bed = item("bed", "BedFactory", box(0, 2, 0, 2, 0, 0.7))
    pillow = item("pillow", "PillowFactory", box(0.5, 1.0, 0.5, 1.0, 0.68, 0.9))
    lamp = item("lamp", "DeskLampFactory", box(0.6, 0.9, 0.6, 0.9, 0.7, 1.1))
    bindings = {"bed": selected_binding(candidate(mesh, group="bed"), target_id="bed", group="bed")}
    actions = asb._related_generated_item_actions([bed, pillow, lamp], bindings, preserve_generated_bedding=True)
    assert actions["pillow"]["action"] == "reanchor"
    suppressed = asb._related_generated_item_actions([bed, pillow], bindings, preserve_generated_bedding=False)
    assert suppressed["pillow"]["action"] == "suppress"

    desk = item("desk", "SimpleDeskFactory", box(0, 2, 0, 1, 0, 0.75))
    table_lamp = item("table_lamp", "DeskLampFactory", box(1.8, 2.1, 0.8, 1.1, 0.2, 0.7))
    by_target_id = {"desk": {"semantic_group": "desk"}, "table_lamp": {"semantic_group": "lamp_table"}}
    moved_items, info = asb._normalize_supported_light_placements({}, [desk, table_lamp], by_target_id)
    assert moved_items[1]["meta"]["supplier_light_position_normalized"] is True
    assert info["moved_count"] == 1

    room = scene_with_room(floor_polygon=[{"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 4}, {"x": 0, "y": 4}]).build()
    light1 = item("l1", "CeilingLightFactory", box(0, 0.5, 0, 0.5, 2.7, 3.0), meta={"supplier_candidate": {"unique_key": "same"}})
    light2 = item("l2", "CeilingLightFactory", box(4, 4.5, 3, 3.5, 2.7, 3.0), meta={"supplier_candidate": {"unique_key": "same"}})
    collapsed, collapse_info = asb._collapse_ceiling_lights(room, [light1, light2], {})
    assert collapse_info["count_preserved"] is True
    assert len(collapsed) == 2
    assert all(obj["meta"]["ceiling_supplier_coverage_normalized"] for obj in collapsed)


def test_support_actions_tv_geometry_and_room_keepout_edges(tmp_path: Path):
    mesh = tmp_path / "asset.glb"
    mesh.write_bytes(b"x")

    fallback_item = {"position_m": [1.0, 2.0, 0.5], "size_m": [2.0, 1.0, 1.0]}
    assert asb._item_aabb(fallback_item) == box(0.0, 2.0, 1.5, 2.5, 0.0, 1.0)
    assert asb._item_position({"aabb": box(0, 2, 0, 4, 0, 2)}) == [1.0, 2.0, 1.0]
    assert asb._xy_inside_expanded(box(0, 1, 0, 1, 0, 1), [1.05, 0.5], 0.1)
    assert asb._aabb_overlaps_expanded(box(0, 1, 0, 1, 0, 1), box(1.05, 2, 0.2, 0.8, 1.05, 1.2), x_margin=0.1, y_margin=0, z_margin_below=0, z_margin_above=0.1)

    dresser = item("dresser", "DresserFactory", box(0, 2, 0, 1, 0, 0.8))
    strict_lamp = item("lamp", "DeskLampFactory", box(0.2, 0.5, 0.2, 0.5, 0.82, 1.2))
    book_inside = item("books", "BookStackFactory", box(0.3, 0.7, 0.3, 0.7, 0.2, 0.5))
    desk = item("desk", "SimpleDeskFactory", box(3, 5, 0, 1, 0, 0.75))
    trinket = item("trinket", "NatureShelfTrinketsFactory", box(3.5, 3.8, 0.3, 0.6, 0.76, 0.9))
    by_target = {
        "dresser": selected_binding(candidate(mesh, group="dresser"), target_id="dresser", group="dresser"),
        "desk": selected_binding(candidate(mesh, group="desk"), target_id="desk", group="desk"),
    }
    actions = asb._related_generated_item_actions([dresser, strict_lamp, book_inside, desk, trinket], by_target)
    assert actions["lamp"]["support_mode"] == "top"
    assert actions["books"]["support_mode"] == "volume"
    assert actions["trinket"]["support_mode"] == "top"

    tall_stand = box(0, 0.5, 0, 2, 0, 0.6)
    tv_on_tall, yaw_tall = asb._tv_aabb_on_stand(tall_stand, (1.2, 0.08, 0.7))
    assert yaw_tall == 90.0
    assert tv_on_tall["y_max"] - tv_on_tall["y_min"] == pytest.approx(1.2)
    wide_stand = box(0, 2, 0, 0.5, 0, 0.6)
    _tv_on_wide, yaw_wide = asb._tv_aabb_on_stand(wide_stand, (1.2, 0.08, 0.7))
    assert yaw_wide == 0.0

    data = scene_with_room(
        floor_polygon=[[0, 0], [5, 0], [5, 4], [0, 4]],
        doors=[
            {"segment": {"x1": 0.5, "y1": 0, "x2": 1.5, "y2": 0}},
            {"segment": {"x1": 0, "y1": 1, "x2": 0, "y2": 2}},
            {"segment": {"bad": True}},
        ],
    ).build()
    keepouts = asb._door_keepout_aabbs(data)
    assert len(keepouts) == 2
    assert asb._xy_aabb_overlap(keepouts[0], box(0.7, 0.8, 0.2, 0.3, 0, 1))

    sofa_right = item("sofa_r", "SofaFactory", box(4.0, 4.5, 1.5, 2.2, 0, 0.8), rotation_deg=90)
    pose_r = asb._wall_tv_pose_for_anchor(data, [sofa_right], sofa_right, (1.2, 0.08, 0.7), anchor_group="sofa")
    assert pose_r is not None and pose_r[1] == 90.0
    sofa_left = item("sofa_l", "SofaFactory", box(0.5, 1.0, 1.5, 2.2, 0, 0.8), rotation_deg=270)
    pose_l = asb._wall_tv_pose_for_anchor(data, [sofa_left], sofa_left, (1.2, 0.08, 0.7), anchor_group="sofa")
    assert pose_l is not None and pose_l[1] == 270.0
    sofa_up = item("sofa_u", "SofaFactory", box(2.0, 2.7, 3.0, 3.5, 0, 0.8), rotation_deg=180)
    pose_u = asb._wall_tv_pose_for_anchor(data, [sofa_up], sofa_up, (1.2, 0.08, 0.7), anchor_group="bed")
    assert pose_u is not None and pose_u[1] == 180.0
    sofa_down = item("sofa_d", "SofaFactory", box(2.0, 2.7, 0.5, 1.0, 0, 0.8), rotation_deg=0)
    pose_d = asb._wall_tv_pose_for_anchor(data, [sofa_down], sofa_down, (1.2, 0.08, 0.7), anchor_group="sofa")
    assert pose_d is not None and pose_d[1] == 0.0

    table = item("table", "SimpleDeskFactory", box(1, 3, 1, 2, 0, 0.75))
    chair_e = item("chair_e", "ChairFactory", box(3.05, 3.45, 1.2, 1.8, 0, 0.9))
    chair_n = item("chair_n", "ChairFactory", box(1.5, 2.2, 2.05, 2.45, 0, 0.9))
    assert asb._chair_side_for_table(table["aabb"], chair_e["aabb"]) == "east"
    assert asb._chair_side_for_table(table["aabb"], chair_n["aabb"]) == "north"
    assert asb._chair_is_on_table_long_edge(table["aabb"], chair_n["aabb"])
    assert asb._has_nearby_chair(table, [table, chair_e], {})
    assert asb._has_usable_nearby_chair(table, [table, chair_n], {})


def test_room_table_chair_tv_and_computer_affordances(monkeypatch, tmp_path: Path):
    room = scene_with_room(
        type="living_room",
        description="home cinema with tv",
        floor_polygon=[[0, 0], [5, 0], [5, 4], [0, 4]],
        doors=[{"segment": {"x1": 1, "y1": 0, "x2": 2, "y2": 0}}],
    ).build()
    room["meta"] = {"prompt": "add smart tv"}
    table = item("desk", "SimpleDeskFactory", box(2, 3.2, 1.5, 2.1, 0, 0.74))
    old_chair = item("chair", "ChairFactory", box(0.2, 0.7, 0.2, 0.8, 0, 0.9))
    assert asb._table_requires_chair(table, "desk")
    assert not asb._has_usable_nearby_chair(table, [table, old_chair], {})

    moved_items, chair_info = asb._ensure_table_chair_affordances(room, [table, old_chair], {})
    assert chair_info["moved_count"] == 1
    moved_chair = next(obj for obj in moved_items if obj["id"] == "chair")
    assert moved_chair["meta"]["affordance"] == "table_chair"
    assert asb._has_usable_nearby_chair(table, moved_items, {})

    chair_mesh = tmp_path / "chair.fbx"
    chair_mesh.write_bytes(b"x")
    monkeypatch.setattr(asb, "_candidate_from_supplier_db", lambda group, target_size: candidate(chair_mesh, group=group))
    added_items, added_info = asb._ensure_table_chair_affordances(room, [table], {})
    assert added_info["added_count"] == 1
    assert added_items[-1]["source"]["asset_source"] == "supplier_catalog_local_asset"

    tv_mesh = tmp_path / "tv.glb"
    tv_mesh.write_bytes(b"x")
    tv_candidate = dict(candidate(tv_mesh, group="tv_projector_screen"), width_cm=120, depth_cm=7, height_cm=70, title="OLED TV")
    monkeypatch.setattr(asb, "_candidate_from_supplier_catalog_json", lambda *_args, **_kwargs: tv_candidate)
    stand = item("stand", "TVStandFactory", box(1, 3, 0.3, 0.8, 0, 0.55))
    with_tv, tv_info = asb._ensure_tv_affordance(room, [stand], {})
    assert tv_info["status"] == "added"
    assert with_tv[-1]["meta"]["affordance"] == "tv_on_stand"
    assert asb._scene_has_tv(with_tv, {})

    sofa = item("sofa", "SofaFactory", box(1.5, 3.0, 2.5, 3.3, 0, 0.9), rotation_deg=180)
    wall_tv_items, wall_tv_info = asb._ensure_tv_affordance(room, [sofa], {})
    assert wall_tv_info["status"] == "added"
    assert wall_tv_items[-1]["constraints"]["mount_type"] == "wall"

    computer_mesh = tmp_path / "imac.glb"
    computer_mesh.write_bytes(b"x")
    computer_candidate = dict(candidate(computer_mesh, group="computer"), title="Apple iMac all-in-one", width_cm=60, depth_cm=20, height_cm=45)
    monkeypatch.setattr(asb, "_candidate_from_supplier_catalog_json", lambda *_args, **_kwargs: computer_candidate)
    monitor = item("monitor", "MonitorFactory", box(2.2, 2.8, 1.6, 2.0, 0.74, 1.2), name="iMac computer")
    keyboard = item("keyboard", "KeyboardFactory", box(2.25, 2.75, 1.65, 1.95, 0.75, 0.82), name="keyboard mouse")
    replaced, computer_info = asb._ensure_computer_replacements([table, monitor, keyboard], {})
    assert computer_info["replaced_count"] == 1
    assert "keyboard" in computer_info["suppressed_keyboard_ids"]
    assert next(obj for obj in replaced if obj["id"] == "monitor")["meta"]["computer_candidate_kind"] == "all_in_one"


def test_json_entrypoint_and_invalid_modes(monkeypatch, tmp_path: Path):
    mesh = tmp_path / "chair.obj"
    mesh.write_bytes(b"x")
    scene = {
        "room": {"floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
        "placements": [item("target", "ChairFactory", box(1, 1.5, 1, 1.5, 0, 0.9))],
    }
    bindings = {
        "bindings": [
            selected_binding(
                dict(candidate(mesh), width_cm=55, depth_cm=60, height_cm=92),
                target_id="target",
                group="chair",
            )
        ]
    }
    input_path = tmp_path / "scene.json"
    bindings_path = tmp_path / "bindings.json"
    out_path = tmp_path / "out.json"
    input_path.write_text(json.dumps(scene), encoding="utf-8")
    bindings_path.write_text(json.dumps(bindings), encoding="utf-8")

    result_path = asb.apply_supplier_bindings_to_json(
        input_json_path=input_path,
        bindings_json_path=bindings_path,
        output_json_path=out_path,
        require_local_asset=True,
        fallback_mode=asb.ASSET_FALLBACK_MODE_NONE,
    )
    assert result_path == out_path
    out = json.loads(out_path.read_text(encoding="utf-8"))
    replaced = out["placements"][0]
    assert replaced["asset"]["mesh_path"] == str(mesh.resolve())
    assert replaced["meta"]["supplier_binding_applied"] is True

    permissive = asb.apply_supplier_bindings_to_data(scene, bindings, fallback_mode="bad")
    assert permissive["meta"]["supplier_binding_summary"]["supplier_asset_fallback_mode"] == "bad"


def test_cli_main_parallel_items_sync_and_invalid_inputs(monkeypatch, tmp_path: Path, capsys):
    mesh = tmp_path / "bed.obj"
    mesh.write_bytes(b"x")
    bed_aabb = box(0, 2, 0, 2, 0, 0.7)
    pillow_aabb = box(0.4, 0.8, 0.5, 0.9, 0.68, 0.9)
    scene = {
        "room": {"floor_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]]},
        "placements": [
            item("bed", "BedFactory", bed_aabb),
            item("pillow", "PillowFactory", pillow_aabb),
            item("keep", "DecorFactory", box(3, 3.2, 1, 1.2, 0, 0.3)),
        ],
        "items": [
            item("bed", "BedFactory", bed_aabb),
            item("pillow", "PillowFactory", pillow_aabb),
            item("parallel_only", "PlantFactory", box(4, 4.3, 1, 1.3, 0, 1.0)),
        ],
    }
    bindings = {
        "bindings": [
            selected_binding(
                dict(candidate(mesh, group="bed"), width_cm=210, depth_cm=220, height_cm=80),
                target_id="bed",
                group="bed",
            )
        ]
    }
    scene_path = tmp_path / "scene.json"
    bindings_path = tmp_path / "bindings.json"
    out_path = tmp_path / "out.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    bindings_path.write_text(json.dumps(bindings), encoding="utf-8")

    parser = asb.build_cli()
    parsed = parser.parse_args(
        [
            "--input-json",
            str(scene_path),
            "--bindings-json",
            str(bindings_path),
            "--out",
            str(out_path),
            "--require-local-asset",
            "--suppress-generated-bedding",
            "--supplier-asset-fallback-mode",
            asb.ASSET_FALLBACK_MODE_FBX_OBJ_PROXY,
        ]
    )
    assert parsed.require_local_asset is True
    assert parsed.suppress_generated_bedding is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_supplier_bindings",
            "--input-json",
            str(scene_path),
            "--bindings-json",
            str(bindings_path),
            "--out",
            str(out_path),
            "--require-local-asset",
            "--suppress-generated-bedding",
        ],
    )
    asb.main()
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert "replaced = 1" in capsys.readouterr().out
    assert "pillow" not in {obj["id"] for obj in out["placements"]}
    assert "pillow" not in {obj["id"] for obj in out["items"]}
    assert "parallel_only" in {obj["id"] for obj in out["items"]}
    assert out["meta"]["supplier_bed_postprocess"]["policy"] == "suppress_generated_bedding_when_replacing_bed"

    with pytest.raises(RuntimeError, match="нет placements/items"):
        asb.apply_supplier_bindings_to_data({"room": {}}, bindings)
    with pytest.raises(RuntimeError, match="нет bindings"):
        asb.apply_supplier_bindings_to_data({"items": []}, {"bindings": {"bad": True}})


def test_apply_supplier_bindings_to_items_collection_edge_branches(tmp_path: Path):
    mesh = tmp_path / "replacement.fbx"
    mesh.write_bytes(b"fbx")
    scene = {
        "room": {"floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
        "items": [
            "raw",
            item("unbound", "ChairFactory", box(0, 0.5, 0, 0.5, 0, 0.8)),
            item("bad_status", "ChairFactory", box(0.6, 1.0, 0, 0.5, 0, 0.8)),
            item("bad_source", "ChairFactory", box(1.1, 1.5, 0, 0.5, 0, 0.8)),
            item("needs_local", "ChairFactory", box(1.6, 2.0, 0, 0.5, 0, 0.8)),
            {"id": "placeholder", "category": "DecorFactory"},
            item("local", "ChairFactory", box(2.2, 2.7, 0, 0.5, 0, 0.8)),
        ],
        "placements": [
            item("local", "ChairFactory", box(2.2, 2.7, 0, 0.5, 0, 0.8)),
            item("parallel_keep", "PlantFactory", box(3, 3.3, 1, 1.3, 0, 1.0)),
        ],
    }
    good_candidate = candidate(mesh, unique_key="local", group="chair")
    bindings = {
        "bindings": [
            {"target_id": "bad_status", "selection_status": "unmatched", "provenance": {"final_asset_source": "supplier_catalog"}, "chosen_candidate": good_candidate},
            {"target_id": "bad_source", "selection_status": "heuristic_top1_selected", "provenance": {"final_asset_source": "manual"}, "chosen_candidate": good_candidate},
            {"target_id": "needs_local", "selection_status": "heuristic_top1_selected", "provenance": {"final_asset_source": "supplier_catalog"}, "chosen_candidate": {"unique_key": "missing", "title": "Missing"}},
            {"target_id": "placeholder", "selection_status": "heuristic_top1_selected", "provenance": {"final_asset_source": "supplier_catalog"}, "category": "decor", "chosen_candidate": {"unique_key": "placeholder", "title": "Placeholder"}},
            selected_binding(good_candidate, target_id="local", group="chair"),
        ]
    }
    out = asb.apply_supplier_bindings_to_data(
        scene,
        bindings,
        require_local_asset=True,
        fallback_mode=asb.ASSET_FALLBACK_MODE_NONE,
    )

    items_by_id = {obj.get("id"): obj for obj in out["items"] if isinstance(obj, dict)}
    assert items_by_id["bad_status"]["name"] == "ChairFactory"
    assert items_by_id["bad_source"]["name"] == "ChairFactory"
    assert items_by_id["needs_local"]["name"] == "ChairFactory"
    assert items_by_id["placeholder"]["category"] == "DecorFactory"
    assert items_by_id["local"]["asset"]["mesh_path"] == str(mesh.resolve())
    assert any(obj == "raw" for obj in out["items"])
    assert any(obj.get("id") == "parallel_keep" for obj in out["placements"] if isinstance(obj, dict))
    assert out["meta"]["supplier_binding_summary"]["local_asset_replaced_count"] == 1


def test_apply_supplier_bindings_reanchors_preserved_generated_support_items(tmp_path: Path):
    bed_mesh = tmp_path / "bed.fbx"
    dresser_mesh = tmp_path / "dresser.fbx"
    bed_mesh.write_bytes(b"fbx")
    dresser_mesh.write_bytes(b"fbx")
    scene = {
        "room": {
            "type": "bedroom",
            "floor_polygon": [[0, 0], [5, 0], [5, 4], [0, 4]],
        },
        "placements": [
            item("bed", "BedFactory", box(0, 2, 0, 2, 0, 0.7)),
            item("pillow", "PillowFactory", box(0.45, 0.85, 0.45, 0.85, 0.68, 0.90)),
            item("dresser", "DresserFactory", box(3, 4, 0.4, 1.3, 0, 0.9)),
            item("books", "BookStackFactory", box(3.25, 3.65, 0.65, 1.0, 0.20, 0.42)),
        ],
    }
    bindings = {
        "bindings": [
            selected_binding(candidate(bed_mesh, group="bed"), target_id="bed", group="bed"),
            selected_binding(candidate(dresser_mesh, group="dresser"), target_id="dresser", group="dresser"),
        ]
    }

    out = asb.apply_supplier_bindings_to_data(scene, bindings, require_local_asset=True, preserve_generated_bedding=True)
    by_id = {obj["id"]: obj for obj in out["placements"]}

    assert by_id["bed"]["meta"]["supplier_binding_applied"] is True
    assert by_id["dresser"]["meta"]["supplier_binding_applied"] is True
    assert by_id["pillow"]["meta"]["supplier_support_reanchored"] is True
    assert by_id["pillow"]["meta"]["supplier_support_mode"] == "top"
    assert by_id["pillow"]["meta"]["supplier_bedding_preserved"] is True
    assert by_id["books"]["meta"]["supplier_support_reanchored"] is True
    assert by_id["books"]["meta"]["supplier_support_mode"] == "volume"
    assert out["meta"]["supplier_binding_summary"]["reanchored_generated_related_count"] == 2
    assert out["meta"]["supplier_bed_postprocess"]["preserved_bedding_ids"] == ["pillow"]


def test_apply_supplier_bindings_remaining_low_level_edges(tmp_path: Path):
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8")
    assert (
        asb._infer_reference_scene_blend_path(
            scene_path,
            {"meta": {"placement_meta": {"scene_blend": "scene_infinigen_clean.blend"}}},
        )
        is None
    )

    assert asb._candidate_size_m({"width_cm": None, "depth_cm": 1, "height_cm": 1}) is None
    assert asb._candidate_size_m({"width_cm": "bad", "depth_cm": 1, "height_cm": 1}) is None
    assert asb._item_aabb({"id": "missing_geometry"}) is None
    assert asb._item_position({"id": "missing_geometry"}) is None

    assert not asb._selected_supplier_binding({"chosen_candidate": None})
    assert not asb._selected_supplier_binding(
        {
            "chosen_candidate": {},
            "selection_status": "rejected",
            "provenance": {"final_asset_source": "supplier_catalog"},
        }
    )
    assert not asb._candidate_has_supported_local_asset(None)

    proxy = tmp_path / "built" / "proxy.glb"
    proxy.parent.mkdir()
    proxy.write_bytes(b"proxy")
    unsupported = tmp_path / "asset.txt"
    unsupported.write_text("txt", encoding="utf-8")
    real = tmp_path / "real.obj"
    real.write_text("obj", encoding="utf-8")

    assert asb._candidate_asset_paths(None) == []
    assert asb._candidate_asset_paths({"asset_local_path": [None, str(tmp_path / "missing.obj"), str(unsupported), str(real)]}) == [
        str(real.resolve())
    ]
    assert not asb._candidate_has_supported_local_asset({"asset_local_path": str(proxy)})
    assert not asb._candidate_has_supported_local_asset(
        {"asset_local_path": str(real), "asset_status": "needs_blender_rebuild"}
    )


def test_apply_supplier_bindings_remaining_geometry_and_affordance_helpers(tmp_path: Path):
    missing_candidate_asset, missing_placeholder = asb._candidate_asset(
        {"unique_key": "missing"},
        require_local_asset=True,
    )
    assert missing_candidate_asset == {}
    assert missing_placeholder is False

    proxy_asset, proxy_placeholder = asb._candidate_asset(
        {"unique_key": "proxy"},
        require_local_asset=False,
        fallback_mode=asb.ASSET_FALLBACK_MODE_FBX_OBJ_PROXY,
    )
    assert proxy_asset["kind"] == "procedural_proxy"
    assert proxy_placeholder is False

    low_quality_asset, low_quality_placeholder = asb._candidate_asset(
        {"unique_key": "low", "asset_status": "needs_blender_rebuild"},
        require_local_asset=False,
    )
    assert low_quality_asset == {}
    assert low_quality_placeholder is True

    assert asb._semantic_group_for_item(
        {
            "category": "WallMountedTVFactory",
            "name": "gaming monitor",
            "meta": {"supplier_candidate": {"semantic_group": "tv_projector_screen", "title": "gaming monitor"}},
        }
    ) == "computer"
    assert asb._semantic_group_for_item({"category": "custom", "name": "tv stand low тумба"}) == "tv_stand"
    assert asb._semantic_group_for_item({"category": "custom", "name": "телевизор wall"}) == "tv_projector_screen"
    assert asb._semantic_group_for_item({"category": "custom", "name": "диван"}) == "sofa"
    assert asb._semantic_group_for_item({"category": "custom", "name": "люстра"}) == "lamp_ceiling"
    assert asb._semantic_group_for_item({"category": "custom", "name": "настольная лампа"}) == "lamp_table"

    data = {"room": {"floor_polygon": [{"x": 0, "y": 0}, [4, 0], [4, 3], {"x": 0, "y": 3}]}}
    assert asb._room_center_xy(data, []) == (2.0, 1.5)
    assert asb._room_xy_bounds(data, []) == {"x_min": 0.0, "x_max": 4.0, "y_min": 0.0, "y_max": 3.0}
    assert asb._point_in_room_xy({"room": {}}, 100, 100) is True
    assert asb._point_in_room_xy(data, 2, 1) is True
    assert asb._point_in_room_xy(data, 5, 1) is False
    assert asb._room_polygon_points(data) == [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
    assert asb._dist_to_room_edges([(0, 0)], 1, 1) == 999.0
    assert asb._ceiling_coverage_points(data, 0) == ([], 0.0)
    assert asb._ceiling_coverage_points({"room": {"floor_polygon": [[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]]}}, 2)[0] == [
        (0.1, 0.1),
        (0.1, 0.1),
    ]

    movable = item("move", "ChairFactory", box(0, 1, 0, 1, 0, 1))
    asb._set_item_center_xy(movable, 2, 3)
    assert movable["position_m"] == [2.0, 3.0, 0.5]
    asb._set_item_center_xyz(movable, 4, 5, 6)
    assert movable["position_m"] == [4.0, 5.0, 6.0]
    no_geom = {"id": "no_geom"}
    asb._set_item_center_xy(no_geom, 1, 1)
    asb._set_item_center_xyz(no_geom, 1, 1, 1)
    assert no_geom == {"id": "no_geom"}

    assert asb._table_requires_chair({"category": "CoffeeTableFactory", "name": "coffee table"}, "coffee_table") is False
    assert asb._table_requires_chair({"category": "DiningTableFactory", "name": "table"}, "unknown") is True

    table = item("desk", "DeskFactory", box(0, 1, 0, 1, 0, 0.8))
    chair = item("chair", "ChairFactory", box(0.2, 0.6, -0.7, -0.2, 0, 0.8))
    far_chair = item("far", "ChairFactory", box(10, 11, 10, 11, 0, 0.8))
    assert asb._has_nearby_chair({"id": "no_pos"}, [chair], {}) is False
    assert asb._has_nearby_chair(table, [table, far_chair], {}) is False
    assert asb._has_nearby_chair(table, [table, chair], {}) is True
    assert asb._has_usable_nearby_chair({"id": "no_pos"}, [chair], {}) is False
    assert asb._has_usable_nearby_chair(table, [table, chair], {}) is True

    assert asb._door_keepout_aabbs(
        {
            "room": {
                "doors": [
                    "bad",
                    {"segment": {"x1": 0, "y1": 0, "x2": 1, "y2": 0}},
                    {"segment": {"x1": 2, "y1": 2, "x2": 3, "y2": 2}},
                    {"segment": {"x1": 0, "y1": 0, "x2": 0, "y2": 1}},
                    {"segment": {"x1": 2, "y1": 1, "x2": 2, "y2": 2}},
                ]
            }
        },
        depth_m=0.5,
        side_margin_m=0.1,
    ) == [
        {"x_min": -0.1, "x_max": 1.1, "y_min": 0.0, "y_max": 0.5, "z_min": 0.0, "z_max": 2.2},
        {"x_min": 1.9, "x_max": 3.1, "y_min": 1.5, "y_max": 2.0, "z_min": 0.0, "z_max": 2.2},
        {"x_min": 0.0, "x_max": 0.5, "y_min": -0.1, "y_max": 1.1, "z_min": 0.0, "z_max": 2.2},
        {"x_min": 1.5, "x_max": 2.0, "y_min": 0.9, "y_max": 2.1, "z_min": 0.0, "z_max": 2.2},
    ]


def test_apply_supplier_bindings_db_light_tv_and_placeholder_edge_branches(monkeypatch, tmp_path: Path):
    import sqlite3

    monkeypatch.setattr(asb, "_candidate_asset_paths", lambda _candidate: [str(tmp_path / "unsupported.txt")])
    assert asb._candidate_has_supported_local_asset({"asset_status": "local_supplier_asset"}) is False
    monkeypatch.undo()

    assert asb._compact_candidate_pool({"chosen_candidate": {"unique_key": "missing"}}, limit=1) == []

    real = tmp_path / "chair.fbx"
    real.write_bytes(b"fbx")
    monkeypatch.chdir(tmp_path)
    db_dir = tmp_path / "data" / "sourse" / "suppliers"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "site_assets_imodern_clean.db"
    db_path.write_text("not a sqlite database", encoding="utf-8")
    assert asb._candidate_from_supplier_db("chair", [0.5, 0.55, 0.9]) is None
    db_path.unlink()

    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE supplier_asset (unique_key TEXT, source_site TEXT, title TEXT, product_url TEXT, "
            "asset_status TEXT, asset_format TEXT, asset_local_path TEXT, extra_json TEXT)"
        )
        con.execute(
            "CREATE TABLE supplier_product (unique_key TEXT, brand TEXT, collection TEXT, category_raw TEXT, "
            "category_norm TEXT, width_cm REAL, depth_cm REAL, height_cm REAL, price_value REAL, "
            "price_currency TEXT, style TEXT, color TEXT, materials TEXT, description TEXT)"
        )
        con.execute(
            "INSERT INTO supplier_asset VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("badfmt", "imodern", "Bad Chair", "u", "local_supplier_asset", "txt", str(real), "{}"),
        )
        con.execute(
            "INSERT INTO supplier_product VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("badfmt", "b", "c", "chair", "chair", 50, 55, 90, 1, "RUB", "modern", "gray", "wood", "bad"),
        )
        con.execute(
            "INSERT INTO supplier_asset VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "good",
                "imodern",
                "Compact Chair",
                "https://example.test/chair",
                "",
                "",
                str(real),
                '{"model_page_url":"mp","model_download_url":"md"}',
            ),
        )
        con.execute(
            "INSERT INTO supplier_product VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("good", "brand", "coll", "chair", "chair", 51, 56, 91, 100, "RUB", "modern", "gray", "wood", "ok"),
        )

    db_candidate = asb._candidate_from_supplier_db("chair", [0.5, 0.55, 0.9])
    assert db_candidate["unique_key"] == "good"
    assert db_candidate["asset_format"] == "fbx"
    assert db_candidate["model_page_url"] == "mp"
    assert asb._catalog_candidate_asset({"mesh_local_path": str(real)})["mesh_path"] == str(real.resolve())
    assert asb._catalog_candidate_asset({"asset_local_path": str(tmp_path / "missing.fbx")}) == {}

    candidate_aabb = box(0, 1, 0, 1, 0, 1)
    collide_items = [
        "raw",
        {"id": "nogeom", "category": "PlantFactory"},
        item("skip", "ChairFactory", box(0, 1, 0, 1, 0, 1)),
        item("chair", "ChairFactory", box(0.1, 0.5, 0.1, 0.5, 0.1, 0.5)),
        item("plant", "PlantFactory", box(0.6, 0.9, 0.6, 0.9, 0.1, 0.5)),
    ]
    assert asb._light_collides_xy(candidate_aabb, collide_items, skip_id="skip", by_target_id={}) == 5

    small_support = item("small", "NightstandFactory", box(1, 1.2, 1, 1.2, 0, 0.5))
    lamp = item("lamp", "DeskLampFactory", box(0.8, 1.4, 0.8, 1.4, 0.48, 1.08), meta=[])
    floor_lamp = item("floor", "FloorLampFactory", box(3, 3.2, 3, 3.2, 0, 1.6))
    no_support_lamp = item("orphan", "DeskLampFactory", box(8, 8.2, 8, 8.2, 0, 0.5))
    moved_lights, light_info = asb._normalize_supported_light_placements(
        {},
        ["raw", {"id": "bad"}, lamp, small_support, floor_lamp, no_support_lamp],
        {"small": {"semantic_group": "nightstand"}},
    )
    moved_lamp = next(obj for obj in moved_lights if isinstance(obj, dict) and obj.get("id") == "lamp")
    assert light_info["moved_count"] == 1
    assert moved_lamp.get("meta") == []

    assert asb._candidate_tv_size(None) == (1.1, 0.06, 0.65)
    assert asb._candidate_tv_size({"width_cm": "bad"}) == (1.1, 0.06, 0.65)
    assert asb._candidate_size_m_or_fallback({"width_cm": "bad"}, [1, 2, 3]) == (1.0, 2.0, 3.0)
    assert asb._scene_has_tv(["raw", {"meta": {"supplier_candidate": {"category_norm": "tv_projector_screen", "title": "OLED TV"}}}], {}) is True
    assert not asb._has_clear_tv_volume(box(0, 1, 0, 1, 0, 1), ["raw", item("ignore", "TVStandFactory", box(0, 1, 0, 1, 0, 1))])
    ids = {"auto_tv_for_stand", "auto_tv_for_stand_2"}
    assert asb._next_generated_id("auto_tv_for_stand", ids) == "auto_tv_for_stand_3"

    bedroom = {"room": {"type": "bedroom", "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}}
    assert asb._ensure_tv_affordance(bedroom, [], {})[1]["skipped_reason"] == "bedroom_tv_not_requested"
    living = {"room": {"type": "living_room", "description": "home cinema tv", "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}}
    monkeypatch.setattr(asb, "_candidate_from_supplier_catalog_json", lambda *_args, **_kwargs: None)
    assert asb._ensure_tv_affordance(living, [], {})[1]["skipped_reason"] == "missing_supplier_tv_asset"

    tv_mesh = tmp_path / "tv.glb"
    tv_mesh.write_bytes(b"glb")
    monkeypatch.setattr(
        asb,
        "_candidate_from_supplier_catalog_json",
        lambda *_args, **_kwargs: dict(candidate(tv_mesh, group="tv_projector_screen"), title="OLED TV", width_cm=120, depth_cm=6, height_cm=70),
    )
    blocker = item("block", "PlantFactory", box(1.3, 2.7, 0.45, 0.75, 0.7, 1.8))
    wall_blocker = item("wall_block", "PlantFactory", box(1.3, 2.7, 0.0, 0.2, 0.7, 1.8))
    stand = item("stand", "TVStandFactory", box(1, 3, 0.3, 0.8, 0, 0.55))
    sofa = item("sofa", "SofaFactory", box(1.4, 2.6, 1.4, 2.2, 0, 0.9), rotation_deg=0)
    no_clear_items, no_clear = asb._ensure_tv_affordance(living, ["raw", {"id": "bad_stand", "category": "TVStandFactory"}, stand, sofa, blocker, wall_blocker], {})
    assert no_clear_items[-1]["id"] == "wall_block"
    assert no_clear["status"] == "skipped"
    assert no_clear["skipped_reason"] == "no_clear_tv_location"
    assert any(attempt["clear"] is False for attempt in no_clear["attempts"])

    assert asb._wall_tv_pose_for_anchor({"room": {}}, [], {"id": "anchor"}, (1, 0.1, 0.5), anchor_group="sofa") is None
    centered_anchor = item("center", "SofaFactory", box(1.5, 2.5, 1.0, 2.0, 0, 0.8), rotation_deg=0)
    pose = asb._wall_tv_pose_for_anchor(
        {"room": {"floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}},
        [centered_anchor],
        centered_anchor,
        (1, 0.1, 0.5),
        anchor_group="sofa",
    )
    assert pose is not None and pose[1] == 0.0


def test_apply_supplier_bindings_placeholders_proxy_reference_and_sync_edges(monkeypatch, tmp_path: Path):
    placeholder_scene = {
        "room": {"type": "living_room", "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
        "placements": [
            "raw-placement",
            {"id": "placeholder", "category": "DecorFactory", "name": "placeholder"},
            {"id": "proxy", "category": "DecorFactory", "name": "proxy"},
        ],
        "items": [
            {"id": "placeholder", "category": "DecorFactory", "name": "old"},
            {"id": "", "category": "RawIdFactory"},
            "raw-item",
        ],
    }
    placeholder_binding = {
        "target_id": "placeholder",
        "selection_status": "heuristic_top1_selected",
        "provenance": {"final_asset_source": "supplier_catalog_pending"},
        "chosen_candidate": {"unique_key": "placeholder", "title": "Placeholder Supplier"},
        "top_candidates": [{"unique_key": "", "asset_local_path": ""}],
    }
    proxy_binding = {
        "target_id": "proxy",
        "selection_status": "heuristic_top1_selected",
        "provenance": {"final_asset_source": "supplier_catalog"},
        "chosen_candidate": {"unique_key": "proxy", "title": "Proxy Supplier"},
    }
    monkeypatch.setattr(asb, "_ensure_tv_affordance", lambda _data, items, _bindings: (items, {"added_count": 0}))
    monkeypatch.setattr(asb, "_related_generated_item_actions", lambda *_args, **_kwargs: {})
    out_placeholder = asb.apply_supplier_bindings_to_data(placeholder_scene, {"bindings": [placeholder_binding]})
    summary = out_placeholder["meta"]["supplier_binding_summary"]
    assert summary["placeholder_replaced_count"] == 1
    assert out_placeholder["placements"][1]["source"]["asset_source"] == "supplier_catalog_placeholder"
    assert "raw-placement" in out_placeholder["items"]

    out_proxy = asb.apply_supplier_bindings_to_data(
        placeholder_scene,
        {"bindings": [proxy_binding]},
        fallback_mode=asb.ASSET_FALLBACK_MODE_FBX_OBJ_PROXY,
    )
    assert out_proxy["meta"]["supplier_binding_summary"]["proxy_asset_replaced_count"] == 1
    assert out_proxy["placements"][2]["asset"]["kind"] == "procedural_proxy"

    ref_scene = {
        "room": {"floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]},
        "placements": [item("chair", "ChairFactory", box(0, 0.5, 0, 0.5, 0, 0.8))],
    }
    scene_path = tmp_path / "scene.json"
    bindings_path = tmp_path / "bindings.json"
    out_path = tmp_path / "out.json"
    (tmp_path / "scene_infinigen_clean.blend").write_bytes(b"blend")
    mesh = tmp_path / "chair.obj"
    mesh.write_bytes(b"obj")
    scene_path.write_text(json.dumps(ref_scene), encoding="utf-8")
    bindings_path.write_text(
        json.dumps({"bindings": [selected_binding(candidate(mesh), target_id="chair", group="chair")]}),
        encoding="utf-8",
    )
    asb.apply_supplier_bindings_to_json(scene_path, bindings_path, out_path, require_local_asset=True)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["meta"]["placement_meta"]["scene_blend"] == str((tmp_path / "scene_infinigen_clean.blend").resolve())
