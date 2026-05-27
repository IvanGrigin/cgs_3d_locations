import json
import sys
import types
from pathlib import Path

import pytest


from tests.helpers.fake_blender import (
    FakeCollection,
    FakeCollectionObjects,
    FakeLinks,
    FakeMaterialStore,
    FakeMatrix,
    FakeMesh,
    FakeMeshStore,
    FakeNode,
    FakeNodes,
    FakeNodeSocket,
    FakeObject,
    FakeObjectStore,
    FakeObjects,
    FakeVector,
    builder,
)


def aabb(x1=0.0, x2=1.0, y1=0.0, y2=1.0, z1=0.0, z2=1.0):
    return {"x_min": x1, "x_max": x2, "y_min": y1, "y_max": y2, "z_min": z1, "z_max": z2}


def test_cli_aabb_and_support_solver_helpers(builder):
    args = builder._parse_argv(["blender", "--", "--json", "scene.json", "--bbox-fallback", "--render-layer", "windows"])
    assert args.json == "scene.json"
    assert args.no_bbox_fallback is False
    assert args.render_layer == "windows"

    assert builder._median([5, 1, 2]) == 2.0
    assert builder._median([1, 9]) == 5.0
    assert builder._parse_id_set("a, b,,c") == {"a", "b", "c"}

    box = aabb()
    moved = builder._translate_aabb_xy(box, 2.0, -1.0)
    assert moved["x_min"] == 2.0
    assert moved["y_max"] == 0.0
    assert builder._aabb_xy_overlap_area(box, aabb(0.5, 1.5, 0.25, 0.75, 0, 1)) == pytest.approx(0.25)
    assert builder._aabb_inside_xy(box, aabb(-0.1, 1.1, -0.1, 1.1, 0, 1))

    center, size = builder._aabb_to_center_size(aabb(0, 2, 1, 3, 4, 8))
    assert center == (1.0, 2.0, 6.0)
    assert size == (2.0, 2.0, 4.0)

    clamped = builder.clamp_item_aabb_to_room_bounds(
        aabb(-2, -1, -2, -1, -2, -1),
        FakeVector((0, 0, 0)),
        FakeVector((5, 5, 3)),
        margin=0.1,
    )
    assert clamped["x_min"] == pytest.approx(0.1)
    assert clamped["z_min"] == pytest.approx(0.1)

    anchor = {"category": "shelf", "meta": {"supplier_candidate": {"semantic_group": "shelf"}}}
    planes = builder._infer_support_planes_from_anchor_item(anchor, aabb(0, 2, 0, 1, 0, 1.2))
    assert len(planes) >= 4
    assert all("clearance_height" in plane for plane in planes)
    chosen = builder._choose_support_plane(aabb(0.2, 0.6, 0.2, 0.6, 1.19, 1.3), planes, mode="near")
    assert chosen is not None

    solver = builder.MLSupportSolver(room_floor_z=0.0)
    solved = solver.solve(
        item_aabb=aabb(0.0, 0.2, 0.0, 0.2, 0.0, 0.1),
        anchor_aabb=aabb(0, 1, 0, 1, 0, 1),
        planes=[{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z": 0.5, "area": 1.0, "clearance_height": 1.0}],
        occupied_aabbs=[aabb(0.8, 1.0, 0.8, 1.0, 0.5, 0.7)],
        mode="top",
    )
    assert solved is not None
    assert solved["z_min"] == pytest.approx(0.504)
    assert builder._aabb_inside_xy(solved, aabb(0, 1, 0, 1, 0, 1), margin=0.02)


def test_item_semantics_sources_and_render_layers(builder):
    item = {
        "id": "desk1",
        "category": "SimpleDeskFactory",
        "source": {"blend_object_name": "DeskFactory(1).spawn_asset(2).001"},
        "meta": {
            "source": {"original_blend_object_name": "DeskOriginal"},
            "original_generated_item": {"blend_object_name": "GeneratedDesk"},
        },
    }
    assert builder._blend_source_names_from_item(item) == [
        "DeskFactory(1).spawn_asset(2).001",
        "DeskOriginal",
        "GeneratedDesk",
    ]
    assert builder._blend_source_name_from_item(item).startswith("DeskFactory")
    assert builder._strip_blender_numeric_suffix("Cube.001") == "Cube"
    assert "1:2" in builder._source_family_tokens_from_item(item)

    assert builder._item_semantic_group(
        {
            "category": "WallMountedTVFactory",
            "name": "monitor gaming",
            "meta": {"supplier_candidate": {"semantic_group": "tv_projector_screen"}},
        }
    ) == "computer"
    assert builder._item_semantic_group({"name": "торшер modern"}) == "lamp_floor"
    assert builder._item_mount_mode({"meta": {"supplier_support_reanchored": True}}) == "support"
    assert builder._item_mount_mode({"constraints": {"mount_type": "wall"}}) == "wall"
    assert builder._is_supplier_light_replacement_item(
        {"category": "DeskLampFactory", "meta": {"supplier_binding_applied": True}}
    )
    assert builder._should_lock_supplier_rotation({"orientation_rule": {}, "meta": {"supplier_binding_applied": True}}, "desk")
    assert builder._rotation_candidates_for_semantic_group(45, "desk") == [45.0, 135.0, 225.0, 315.0]
    assert builder._rotation_candidates_for_semantic_group(45, "lamp_table") == [45.0]

    kitchen = {"category": "x", "asset": {"kind": "procedural_kitchen"}}
    assert builder._is_procedural_kitchen_item(kitchen)
    assembly = builder._kitchen_assembly_from_scene_item(
        {"id": "kit", "position": [1, 2, 3], "rotation": [0, 0, 90], "meta": {"foo": "bar"}},
        aabb(0, 2, 0, 1, 0, 2),
    )
    assert assembly["assembly_type"] == "procedural_kitchen"
    assert assembly["_scene_root_position"] == [1.0, 2.0, 3.0]
    assert assembly["rotation"] == [0.0, 0.0, 0.0]

    assert builder._matches_room_shell_name("bedroom_0/0.wall")
    assert builder._matches_room_shell_name("ceiling_light_diffuser")
    assert builder._looks_like_room_wrapper_name("room_shell.001")
    assert builder._looks_like_kitchen_wallpaper_overlay_name("Kitchen_Room_Wallpaper_SupplierOverlay")
    assert builder._looks_like_overlay_helper_name("chair_AABB")
    assert builder._looks_like_bbox_helper_name("invalid_item_aabb")
    assert builder._looks_like_architectural_door_name("entry_door_frame")
    assert not builder._looks_like_architectural_door_name("base_cabinet_door_line")
    assert builder._generic_object_basename("/tmp/Group__Cube.003") == "cube"

    coll = types.SimpleNamespace(name="Kitchen")
    obj = FakeObject("Dining chair", collections=[coll])
    obj["cgs_procedural_assembly"] = "kitchen"
    assert builder._object_matches_render_layer(obj, "kitchen")
    assert builder._object_matches_render_layer(obj, "tables_chairs")
    assert not builder._object_matches_render_layer(obj, "windows")


def test_asset_path_supplier_and_duplicate_helpers(builder, tmp_path):
    mesh = tmp_path / "model.obj"
    mesh.write_text("# obj", encoding="utf-8")
    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()

    item = {
        "id": "wardrobe1",
        "category": "wardrobe",
        "asset": {
            "mesh_path": "model.obj",
            "mesh_texture_dirs": ["textures"],
            "asset_yaw_offset_deg": "15",
            "supplier_unique_key": "asset-key",
        },
        "meta": {
            "supplier_candidate": {"unique_key": "primary", "semantic_group": "wardrobe"},
            "supplier_candidate_pool": [{"unique_key": "primary"}, {"unique_key": "second"}],
        },
        "aabb": aabb(0, 2, 0, 0.6, 0, 2),
    }
    assert builder._resolve_path_maybe(tmp_path, "model.obj") == str(mesh.resolve())
    assert builder._item_mesh_path_raw(item) == "model.obj"
    assert builder._item_mesh_texture_dirs_raw(item) == ["textures"]
    assert builder._item_mesh_fit_mode({"category": "toilet_cabinet", "mesh_fit_mode": "fit"}) == "stretch"
    assert builder._item_mesh_yaw_offset_deg(item) == 15.0
    assert [c.get("unique_key") for c in builder._item_supplier_candidate_pool(item)] == ["primary", "second"]
    assert builder._candidate_mesh_path_raw({"asset_local_path": "/tmp/x.glb"}, item) == "/tmp/x.glb"
    assert builder._item_has_existing_mesh_file(item, tmp_path)

    assert builder._supplier_record_key({"unique_key": "record-key"}, item) == "record-key"
    assert builder._supplier_record_key({}, item) == "primary"
    assert builder._supplier_candidate_dimensions_m({"dimensions_cm": {"width": 100, "depth": 50, "height": 200}}) == (1.0, 0.5, 2.0)
    assert builder._sizes_materially_different((1, 1, 1), (2, 1, 1))
    assert builder._rigid_supplier_group("wardrobe")

    items = [
        item,
        {
            "id": "desk1",
            "category": "desk",
            "aabb": aabb(2, 3, 0, 1, 0, 1),
            "meta": {"supplier_candidate": {"unique_key": "primary", "semantic_group": "desk"}},
        },
    ]
    reuse_index = builder._build_supplier_reuse_size_index(items)
    assert builder._supplier_candidate_reuse_reject_reason(items[1], {"unique_key": "primary"}, reuse_index)

    assert builder._aabb_size_tuple(aabb(0, 2, 0, 3, 0, 4)) == (2.0, 3.0, 4.0)
    assert builder._aabb_nearly_identical(aabb(), aabb(0.005, 1.005, 0, 1, 0, 1))
    assert builder._aabb_xy_intersection_ratio(aabb(), aabb(0.5, 1.5, 0, 1, 0, 1)) == pytest.approx(0.5)
    assert builder._aabb_intersection_ratio(aabb(), aabb(0.5, 1.5, 0, 1, 0, 1)) == pytest.approx(0.5)
    assert builder._duplicate_semantics_compatible({"category": "DeskLampFactory"}, {"category": "FloorLampFactory"})

    replacement = {
        "id": "new_bed",
        "category": "bed",
        "aabb": aabb(0, 2, 0, 2, 0, 1),
        "source": {"asset_source": "supplier_catalog_local_asset", "blend_object_name": "BedFactory(1).spawn_asset(2)"},
        "meta": {"supplier_candidate": {"unique_key": "bed-key", "semantic_group": "bed"}},
    }
    companion = {
        "id": "old_pillow",
        "category": "PillowFactory",
        "aabb": aabb(0.2, 1.8, 0.2, 1.8, 0.5, 0.9),
        "source": {"blend_object_name": "PillowFactory(1).spawn_asset(2)"},
    }
    duplicate = {
        "id": "old_bed",
        "category": "bed",
        "aabb": aabb(0, 2, 0, 2, 0, 1),
    }
    assert builder._is_replacement_render_item(replacement)
    assert builder._is_replacement_bed_item(replacement)
    assert builder._is_bed_companion_item(companion)
    assert builder._find_duplicate_render_item_ids([replacement, companion, duplicate]) == ["old_bed", "old_pillow"]

    assert builder._is_procedural_proxy_item({"category": "chair", "asset": {"kind": "procedural_placeholder"}})
    assert not builder._is_procedural_proxy_item({"mesh_path": "model.obj", "asset": {"kind": "procedural_proxy"}})


def test_texture_file_parsing_and_map_guessing(builder, tmp_path):
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    obj = asset_dir / "chair.obj"
    mtl = asset_dir / "chair.mtl"
    diffuse = asset_dir / "chair_diffuse.jpg"
    normal = asset_dir / "chair_normal.png"
    roughness = asset_dir / "chair_roughness.jpg"
    preview = asset_dir / "chair_preview.jpg"
    for path in (diffuse, normal, roughness, preview):
        path.write_bytes(b"img")
    obj.write_text('mtllib "chair.mtl"\nmtllib extra.mtl chair.mtl\n', encoding="utf-8")
    mtl.write_text(
        """
newmtl Fabric
map_Kd -o 1 2 3 "chair_diffuse.jpg"
bump chair_normal.png
map_Ks chair_roughness.jpg
map_d alpha.png
""",
        encoding="utf-8",
    )

    assert builder.parse_obj_mtl_files(str(obj)) == ["chair.mtl", "extra.mtl"]
    assert builder._strip_mtl_opts('-bm 0.5 "chair_normal.png"') == "chair_normal.png"
    refs = builder.parse_mtl_refs(str(mtl))
    assert refs["basecolor_ref"] == "chair_diffuse.jpg"
    materials = builder.parse_mtl_materials(str(mtl))
    assert materials["Fabric"]["bump_ref"] == "chair_normal.png"

    idx = builder.build_image_index([str(asset_dir)])
    assert "chair_preview.jpg" not in idx
    assert builder.resolve_texture_ref("chair_diffuse.jpg", asset_dir, idx) == str(diffuse.resolve())
    assert list(builder._walk_images_limited(str(asset_dir), max_depth=0, max_files=2))

    assert builder.detect_asset_root(str(obj)) == asset_dir.resolve()
    search_dirs = builder.build_search_dirs(str(obj), ["textures", str(asset_dir)])
    assert str(asset_dir.resolve()) in search_dirs
    maps = builder._guess_maps_from_scan(str(obj), [str(asset_dir)], explicit_base=None)
    assert maps.basecolor == str(diffuse)
    assert maps.normal == str(normal)
    assert maps.roughness == str(roughness)

    assert builder._mesh_ext_priority(".glb") < builder._mesh_ext_priority(".fbx")
    group_cache = builder._group_glb_cache_path(obj)
    group_cache.write_bytes(b"glb")
    candidates = builder._discover_mesh_import_candidates(str(obj))
    assert str(group_cache.resolve()) == candidates[0]
    assert str(obj.resolve()) in candidates

    rgb = builder._hsv_to_rgb(0.0, 1.0, 1.0)
    assert rgb == (1.0, 0.0, 0.0)
    assert builder._named_color_rgb("серый") == (0.55, 0.55, 0.55)
    assert builder._supplier_candidate_tint_rgb({"meta": {"supplier_candidate": {"color": "#112233"}}}, "fallback") == pytest.approx(
        (17 / 255, 34 / 255, 51 / 255)
    )
    assert builder._should_apply_tint_rgb((0.2, 0.2, 0.2))
    assert not builder._should_apply_tint_rgb((0.7, 0.7, 0.7))
    assert builder._blend_rgba((0, 0, 0, 1), (1, 0, 0), 0.25) == (0.25, 0.0, 0.0, 1.0)
    assert builder._pick_best_match("chair", idx, ["diffuse"]) == str(diffuse)


def test_room_spec_geometry_and_merge_helpers(builder, tmp_path, monkeypatch):
    room = {
        "type": "bathroom",
        "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}, {"x": 0, "y": 3}],
        "doors": [{"id": "door1", "wall_id": "w0", "s": 1.0, "width": 0.8, "height": 2.0}],
        "openings": {
            "windows": [{"id": "win1", "wall_id": "w1", "s": 0.5, "width": 1.0, "sill_height": 0.9, "height": 1.1}]
        },
    }
    walls = builder._synthesize_walls_from_floor_polygon(room)
    assert [wall["id"] for wall in walls] == ["w0", "w1", "w2", "w3"]
    poly_xy = [(0, 0), (4, 0), (4, 3), (0, 3)]
    assert builder._wall_points_from_spec(walls[1], poly_xy) == ((4, 0), (4, 3))
    assert builder._wall_points_from_spec({"from": {"x": 1, "y": 2}, "to": [3, 4]}, poly_xy) == ((1.0, 2.0), (3.0, 4.0))
    assert builder._room_opening_list(room, "openings")
    by_wall = builder._room_openings_by_wall(room)
    assert set(by_wall) == {"w0", "w1"}
    assert builder._opening_kind({"id": "window_main", "sill_height": 0.8}) == "window"
    assert builder._opening_z0({"sill_height": "0.7"}, 0.2) == 0.7
    assert builder._opening_height({"height": "-1"}, 2.0) == 0.0

    spec = builder._room_spec_from_bounds({}, (0, 5, 0, 4, 0, 3))
    assert spec["room"]["ceiling_height"] == 3
    assert builder._poly_signed_area_xy(poly_xy) == 12.0
    assert builder._point_in_bounds_xy(2, 2, (0, 4, 0, 3))
    assert builder._polygon_area_xy_from_vectors([FakeVector((0, 0, 0)), FakeVector((2, 0, 0)), FakeVector((2, 2, 0))]) == 2.0
    assert builder._normalize2(0.0, 0.0) == (1.0, 0.0)
    assert builder._normalize2(3.0, 4.0) == (0.6, 0.8)

    segments = builder._subtract_wall_opening_from_segments(
        [(0.0, 4.0, 0.0, 3.0)],
        opening_s0=1.0,
        opening_s1=2.0,
        opening_z0=0.0,
        opening_z1=2.0,
    )
    assert (1.0, 2.0, 2.0, 3.0) in segments
    assert (0.0, 1.0, 0.0, 3.0) in segments

    assert builder._has_room_spec({"room": room})
    merged = builder._merge_nested_dict({"a": {"x": 1}, "b": 2}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 2}
    render_items = builder._merge_render_items_with_placements(
        [{"id": "a", "asset": {"x": 1}, "meta": {"old": True}}],
        [{"id": "a", "asset": {"y": 2}, "meta": {"new": True}}, {"id": "b", "category": "chair"}],
    )
    assert render_items[0]["asset"] == {"x": 1, "y": 2}
    assert render_items[0]["meta"] == {"old": True, "new": True}
    assert render_items[1]["id"] == "b"

    floor_tex = tmp_path / "floor.jpg"
    wall_tex = tmp_path / "wall.jpg"
    floor_tex.write_bytes(b"img")
    wall_tex.write_bytes(b"img")
    textured_room = {
        "floor_material": {"texture_path": str(floor_tex), "texture_tiling": {"tile_size_m": 2.0, "mode": "mirror"}},
        "wall_material": {"texture_path": str(wall_tex), "wall_tiling": {"tile_size_m": 0.5}},
    }
    assert builder._floor_material_texture_info(textured_room) == (str(floor_tex), 0.5, True)
    assert builder._wall_material_texture_info(textured_room) == (str(wall_tex), 2.0)
    assert builder._is_sanitary_room_spec(room)

    assert builder._procedural_requirement_role({"name": "компактный унитаз"}) == "toilet"
    assert builder._is_procedural_flat_ceiling_light_item({"asset": {"kind": "procedural_flat_ceiling_light"}})
    assert builder._is_procedural_requirement_item({"category": "toilet", "asset": {"kind": "procedural_placeholder"}})


def test_import_mesh_filtering_and_cluster_helpers(builder, monkeypatch):
    objs = [
        FakeObject("body_a"),
        FakeObject("body_b"),
        FakeObject("body_c"),
        FakeObject("mat_preview"),
        FakeObject("helper_plane"),
        FakeObject("Armature", obj_type="ARMATURE"),
    ]
    bounds = {
        "body_a": (FakeVector((0, 0, 0)), FakeVector((1, 1, 1))),
        "body_b": (FakeVector((1.1, 0, 0)), FakeVector((2.1, 1, 1))),
        "body_c": (FakeVector((0, 1.1, 0)), FakeVector((1, 2.1, 1))),
        "mat_preview": (FakeVector((0, 0, 0)), FakeVector((0.2, 0.2, 0.2))),
        "helper_plane": (FakeVector((0, 0, 0)), FakeVector((100, 100, 0.01))),
    }

    monkeypatch.setattr(builder, "_world_bounds_single_mesh_object", lambda obj: bounds[obj.name])

    kept, dropped_preview = builder._drop_material_preview_meshes(objs)
    assert "mat_preview" in dropped_preview
    assert all(obj.name != "mat_preview" for obj in kept if obj.type == "MESH")

    kept, dropped_outliers = builder._filter_imported_mesh_outliers(objs)
    assert any(row["name"] == "helper_plane" for row in dropped_outliers)
    assert all(obj.name != "helper_plane" for obj in kept if obj.type == "MESH")

    mirror_a = FakeObject("mirror_600x900")
    mirror_b = FakeObject("mirror_1200x300")
    bounds.update(
        {
            "mirror_600x900": (FakeVector((0, 0, 0)), FakeVector((0.6, 0.03, 0.9))),
            "mirror_1200x300": (FakeVector((0, 0, 0)), FakeVector((1.2, 0.03, 0.3))),
        }
    )
    selected, dropped = builder._select_single_import_variant_mesh(
        [mirror_a, mirror_b],
        semantic_group="mirror",
        target_aabb=aabb(0, 0.6, 0, 0.05, 0, 0.9),
    )
    assert selected == [mirror_a]
    assert dropped == ["mirror_1200x300"]

    near_a = FakeObject("near_a")
    near_b = FakeObject("near_b")
    far = FakeObject("far")
    bounds.update(
        {
            "near_a": (FakeVector((0, 0, 0)), FakeVector((1, 1, 1))),
            "near_b": (FakeVector((1.2, 0, 0)), FakeVector((2.2, 1, 1))),
            "far": (FakeVector((100, 100, 0)), FakeVector((101, 101, 1))),
        }
    )
    clustered, dropped_cluster = builder._keep_primary_import_cluster([near_a, near_b, far])
    assert clustered == [near_a, near_b]
    assert dropped_cluster[0]["name"] == "far"

    non_mesh = FakeObject("CameraHelper", obj_type="CAMERA")
    kept, removed = builder._remove_or_hide_non_mesh_import_objects([near_a, non_mesh, objs[-1]])
    assert near_a in kept
    assert objs[-1] in kept and objs[-1].hide_render
    assert removed[0]["type"] == "CAMERA"


def test_object_family_lookup_visibility_and_bounds(builder, monkeypatch):
    parent = FakeObject("DeskFactory(1).spawn_asset(2)")
    child = FakeObject("DeskLamp", parent=parent)
    suffix = FakeObject("Chair.001")
    builder.bpy.data.objects[:] = [parent, child, suffix]

    monkeypatch.setattr(
        builder,
        "_world_bounds_mesh_objects",
        lambda _family: (FakeVector((1, 2, 0)), FakeVector((3, 5, 2))),
    )
    assert builder._aabb_from_blend_object_name(parent.name) == {
        "x_min": 1.0,
        "x_max": 3.0,
        "y_min": 2.0,
        "y_max": 5.0,
        "z_min": 0.0,
        "z_max": 2.0,
    }
    assert builder._aabb_from_object_family_root(parent)["y_max"] == 5.0
    assert builder._get_scene_source_object(parent.name) is parent
    assert builder._get_scene_source_object("Chair") is suffix
    assert builder._get_scene_source_object("DeskFactory(1).spawn_asset(2).001") is parent

    assert builder._hide_object_family(parent) == 2
    assert parent.hide_render and child.hide_render
    assert builder._show_object_family(parent) == 2
    assert not parent.hide_render and not child.hide_render

    assert builder._looks_like_reference_light_fixture(child)
    assert builder._collect_reference_light_fixture_roots() == [child]
    parent.hide_render = True
    assert builder._object_or_parent_hidden(child)
    parent.hide_render = False
    suffix.hide_set(True)
    assert builder._object_or_parent_hidden(suffix)
    suffix.hide_set(False)
    assert builder._restore_hidden_reference_light_fixtures([child]) == 0
    assert builder._duplicate_light_objects_from_family(parent) == 0

    parent.location = FakeVector((0, 0, 0))
    child.location = FakeVector((0, 0, 0))
    moved = builder._move_object_family_to_target_aabb(parent, aabb(4, 6, 4, 8, 0, 3))
    assert moved is True
    assert parent.location.x == pytest.approx(3.0)
    assert child.location.y == pytest.approx(2.5)

    deltas = []
    monkeypatch.setattr(builder, "_translate_object_family", lambda root, delta: deltas.append(delta))
    monkeypatch.setattr(builder, "_aabb_from_object_family_root", lambda root: aabb(10, 12, 10, 12, 0, 2))
    exact = builder._move_object_family_to_exact_aabb(parent, aabb(0, 2, 0, 2, 1, 3), aabb(4, 8, 6, 8, 0, 2))
    assert exact["x_min"] == 10
    assert deltas[-1] == FakeVector((5.0, 6.0, -1.0))


def test_support_plane_extraction_snap_and_scene_bounds(builder, monkeypatch):
    root = FakeObject("ShelfRoot", obj_type="EMPTY")
    shelf_a = FakeObject("shelf_a", parent=root)
    shelf_b = FakeObject("shelf_b", parent=root)
    real_mesh_face_candidates = builder._mesh_face_support_plane_candidates

    monkeypatch.setattr(builder, "_mesh_face_support_plane_candidates", lambda _obj: [])
    bounds = {
        "shelf_a": (FakeVector((0, 0, 0)), FakeVector((1, 1, 0.08))),
        "shelf_b": (FakeVector((0, 0, 0.5)), FakeVector((1, 1, 0.58))),
    }
    monkeypatch.setattr(builder, "_world_bounds_single_mesh_object", lambda obj: bounds[obj.name])
    planes = builder._extract_support_planes_from_object_family(root)
    assert [round(plane["z"], 2) for plane in planes] == [0.58, 0.08]
    assert planes[1]["clearance_height"] == pytest.approx(0.5)

    tiny = FakeObject("tiny", parent=root)
    flat = FakeObject("flat", parent=root)
    block = FakeObject("block", parent=root)
    dup = FakeObject("dup", parent=root)
    bounds.update(
        {
            "tiny": (FakeVector((0, 0, 0)), FakeVector((0.01, 1, 0.05))),
            "flat": (FakeVector((0, 0, 0)), FakeVector((0.08, 0.08, 0.05))),
            "block": (FakeVector((0, 0, 0)), FakeVector((1, 1, 1))),
            "dup": (FakeVector((0.01, 0.01, 0.0)), FakeVector((1.01, 1.01, 0.081))),
        }
    )
    planes_with_rejects = builder._extract_support_planes_from_object_family(root)
    assert planes_with_rejects
    assert sum(1 for plane in planes_with_rejects if round(plane["z"], 2) == 0.08) == 1

    empty_mesh_obj = FakeObject("empty_mesh")
    empty_mesh_obj.to_mesh = lambda: None
    assert real_mesh_face_candidates(empty_mesh_obj) == []

    class Poly:
        def __init__(self, area, normal, vertices):
            self.area = area
            self.normal = FakeVector(normal)
            self.vertices = vertices

    candidate_obj = FakeObject("candidate_faces")
    candidate_obj.data.vertices = [
        types.SimpleNamespace(co=FakeVector((0, 0, 0.2))),
        types.SimpleNamespace(co=FakeVector((1, 0, 0.2))),
        types.SimpleNamespace(co=FakeVector((1, 1, 0.2))),
        types.SimpleNamespace(co=FakeVector((0, 1, 0.2))),
        types.SimpleNamespace(co=FakeVector((1.02, 0, 0.205))),
        types.SimpleNamespace(co=FakeVector((2.0, 0, 0.205))),
        types.SimpleNamespace(co=FakeVector((2.0, 1, 0.205))),
        types.SimpleNamespace(co=FakeVector((1.02, 1, 0.205))),
        types.SimpleNamespace(co=FakeVector((0, 0, 0.8))),
        types.SimpleNamespace(co=FakeVector((0.2, 0, 0.8))),
        types.SimpleNamespace(co=FakeVector((0.2, 0.02, 0.8))),
    ]
    candidate_obj.data.polygons = [
        Poly(1e-5, (0, 0, 1), (0, 1, 2)),
        Poly(1.0, (0, 0, 0), (0, 1, 2)),
        Poly(1.0, (0, 1, 0), (0, 1, 2)),
        Poly(1.0, (0, 0, 1), (0, 1)),
        Poly(1.0, (0, 0, 1), (8, 9, 10)),
        Poly(1.0, (0, 0, 1), (0, 1, 2, 3)),
        Poly(1.0, (0, 0, 1), (4, 5, 6, 7)),
    ]
    merged_planes = real_mesh_face_candidates(candidate_obj)
    assert len(merged_planes) == 1
    assert merged_planes[0]["x_max"] == pytest.approx(2.0)

    translations = []
    monkeypatch.setattr(builder, "_extract_support_planes_from_object_family", lambda _root: [{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z": 0.5, "area": 1.0, "clearance_height": 2.0}])
    monkeypatch.setattr(builder, "_translate_object_family", lambda root, delta: translations.append(delta))
    monkeypatch.setattr(builder, "_aabb_from_object_family_root", lambda _root: aabb(0, 0.2, 0, 0.2, 0.504, 0.704))
    snapped = builder._snap_object_family_to_support_plane(root, aabb(0, 0.2, 0, 0.2, 0.1, 0.3), shelf_a, mode="top")
    assert snapped["z_min"] == pytest.approx(0.504)
    assert translations[-1].z == pytest.approx(0.404)

    default_min = FakeVector((-1, -1, 0))
    default_max = FakeVector((1, 1, 2))
    builder.bpy.data.objects[:] = []
    assert builder._visible_mesh_bounds(default_min, default_max) == (default_min, default_max)
    visible = FakeObject("visible")
    hidden = FakeObject("hidden")
    hidden.hide_render = True
    builder.bpy.data.objects[:] = [visible, hidden]
    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", lambda objs: (FakeVector((0, 0, 0)), FakeVector((2, 3, 4))))
    assert builder._visible_mesh_bounds(default_min, default_max)[1] == FakeVector((2, 3, 4))

    builder._store_scene_room_bounds(FakeVector((0, 0, 0)), FakeVector((4, 3, 2.5)))
    assert builder._scene_room_bounds()[1] == FakeVector((4, 3, 2.5))
    wall_dir, room_dir, dist = builder._nearest_room_wall_context(aabb(0.1, 0.3, 1, 1.2, 0, 1))
    assert wall_dir == FakeVector((-1, 0, 0))
    assert dist == pytest.approx(0.2)
    assert room_dir.length > 0


def test_render_layer_cleanup_and_light_helpers(builder, monkeypatch):
    wall = FakeObject("Room_Wall_w0")
    light = FakeObject("Room_Wall_Light", obj_type="LIGHT")
    wrapper = FakeObject("room_shell")
    overlay = FakeObject("chair_AABB")
    door = FakeObject("entry_door_panel")
    kitchen = FakeObject("KitchenCabinet")
    kitchen["cgs_procedural_assembly"] = "kitchen"
    chair = FakeObject("Dining chair")
    builder.bpy.data.objects[:] = [wall, light, wrapper, overlay, door, kitchen, chair]

    assert builder._hide_room_shell_objects() == 1
    assert wall.hide_render and not light.hide_render
    assert builder._hide_room_wrapper_objects() == 1
    assert wrapper.hide_render
    assert builder._apply_render_layer_visibility("kitchen") == 6
    assert kitchen.hide_render is False
    assert chair.hide_render is True

    changed = builder._set_overlay_helpers_render_visibility(show=False)
    assert changed >= 2
    assert overlay.hide_render is True

    monkeypatch.setattr(builder, "_replace_missing_texture_materials", lambda objects=None: 2)
    monkeypatch.setattr(builder, "_remove_bbox_helper_objects", lambda: 1)
    monkeypatch.setattr(builder, "_hide_room_wrapper_objects", lambda: 3)
    monkeypatch.setattr(builder, "_hide_architectural_door_objects", lambda: 4)
    cleanup = builder._cleanup_final_visual_helpers()
    assert cleanup["removed_bbox_helpers"] == 1
    assert cleanup["replaced_missing_texture_materials"] == 2
    assert cleanup["hidden_architectural_doors"] == 4

    assert builder._functional_light_location(aabb(0, 2, 0, 2, 0, 3), "lamp_ceiling")[2] == pytest.approx(0.22)
    assert builder._functional_light_location(aabb(0, 2, 0, 2, 0, 3), "lamp_floor")[2] == pytest.approx(2.34)

    created_lights = []

    class LightStore:
        def new(self, name, light_type):
            data = types.SimpleNamespace(name=name, type=light_type, energy=0.0, shadow_soft_size=0.0)
            created_lights.append(data)
            return data

    class ObjectStore(FakeObjects):
        def new(self, name, data):
            obj = FakeObject(name, obj_type="LIGHT")
            obj.data = data
            self.append(obj)
            return obj

    builder.bpy.data.lights = LightStore()
    builder.bpy.data.objects = ObjectStore(builder.bpy.data.objects)
    builder.bpy.context.scene.collection.objects.link = lambda obj: None
    assert builder._add_functional_light_for_item(
        {"id": "lamp1", "category": "DeskLampFactory", "meta": {"supplier_binding_applied": True}},
        aabb(0, 1, 0, 1, 0, 1),
    ) == 1
    assert created_lights[0].energy == 130.0
    assert builder.bpy.data.objects.get("CGS_FunctionalLight_lamp1") is not None


def test_texture_image_socket_and_material_helpers(builder, tmp_path, monkeypatch):
    real = tmp_path / "real_color.jpg"
    real.write_bytes(b"img")
    placeholder = types.SimpleNamespace(name="Map #13", filepath="", filepath_raw="", source="FILE", packed_file=None, size=(0, 0), has_data=False)
    generated = types.SimpleNamespace(name="generated", filepath="", filepath_raw="", source="GENERATED", packed_file=None, size=(4, 4), has_data=True)
    real_img = types.SimpleNamespace(name="real_color", filepath=str(real), filepath_raw="", source="FILE", packed_file=None, size=(4, 4), has_data=True)
    normal_img = types.SimpleNamespace(name="chair_normal.png", filepath=str(real), filepath_raw="", source="FILE", packed_file=None, size=(4, 4), has_data=True)

    assert builder._is_placeholder_image(placeholder)
    assert builder._image_has_real_pixels(generated)
    assert builder._image_has_real_pixels(real_img)
    assert builder._image_is_missing_file(placeholder)
    assert builder._image_looks_non_color_texture(normal_img)
    assert not builder._image_looks_non_color_texture(real_img)

    class Node:
        def __init__(self, node_type, image=None, inputs=None):
            self.type = node_type
            self.image = image
            self.inputs = inputs or []

        def as_pointer(self):
            return id(self)

    tex_node = Node("TEX_IMAGE", image=real_img)
    normal_node = Node("TEX_IMAGE", image=normal_img)
    linked_socket = types.SimpleNamespace(is_linked=True, links=[types.SimpleNamespace(from_node=tex_node)])
    normal_socket = types.SimpleNamespace(is_linked=True, links=[types.SimpleNamespace(from_node=normal_node)])
    assert builder._socket_chain_has_real_image(linked_socket)
    assert builder._socket_chain_has_real_color_image(linked_socket)
    assert not builder._socket_chain_has_real_color_image(normal_socket)

    mesh = FakeObject("mesh")
    good_mat = types.SimpleNamespace(use_nodes=True, node_tree=types.SimpleNamespace(nodes=[]))
    mesh.data.materials = [good_mat]
    monkeypatch.setattr(builder, "_material_has_effective_basecolor_texture", lambda mat: mat is good_mat)
    assert builder._object_has_any_material_slots(mesh)
    assert builder._has_loaded_textures(mesh)

    bad_mat = types.SimpleNamespace(diffuse_color=(0.8, 0.1, 0.9, 1.0), use_nodes=False, node_tree=None)
    assert builder._material_looks_magenta_missing(bad_mat)

    neutral = types.SimpleNamespace(name="neutral")
    monkeypatch.setattr(builder, "_make_neutral_missing_texture_material", lambda: neutral)
    monkeypatch.setattr(builder, "_material_has_effective_basecolor_texture", lambda mat: False)
    monkeypatch.setattr(builder, "_material_has_missing_basecolor_texture", lambda mat: mat is bad_mat)
    mesh.data.materials = [bad_mat]
    assert builder._replace_missing_texture_materials([mesh]) == 1
    assert mesh.data.materials[0] is neutral

    assert builder._as_float_or_none("3.5") == 3.5
    assert builder._as_float_or_none("") is None
    floor = tmp_path / "floor.jpg"
    wall = tmp_path / "wall.jpg"
    floor.write_bytes(b"img")
    wall.write_bytes(b"img")
    assert builder._floor_material_texture_info({"floor_material": {"texture_path": str(floor), "plank_length_mm": 2400}})[1] == pytest.approx(1 / 2.4)
    assert builder._wall_material_texture_info({"wall_material": {"texture_path": "http://example.com/wall.jpg"}}) == (None, 1.0)
    assert builder._wall_material_texture_info({"wall_material": {"texture_path": str(wall), "wall_tiling": {"tile_size_m": 0.5}}}) == (str(wall), 2.0)


def test_env_texture_opening_and_placeholder_edge_cases(builder, tmp_path):
    for name in ("floor_oak.jpg", "wall_white.png", "window_clear.webp", "door_wood.jpeg"):
        (tmp_path / name).write_bytes(b"img")
    textures = builder._list_env_textures(str(tmp_path))
    assert len(textures["floor"]) == 1
    assert builder._choose_texture("floor", textures["floor"], "style", __import__("random").Random(1)) in textures["floor"]
    assert builder._choose_texture("floor", [], None, __import__("random").Random(1)) is None

    assert builder._point_xy_from_wall_endpoint({"x": "1", "y": "2"}) == (1.0, 2.0)
    assert builder._point_xy_from_wall_endpoint([3, 4, 5]) == (3.0, 4.0)
    assert builder._wall_points_from_spec({"from_vertex": 9, "to_vertex": 1}, [(0, 0), (1, 0)]) is None

    room = {
        "doors": [{"id": "same", "wall_id": "w0", "s": 1, "width": 1, "height": 2}],
        "openings": {
            "doors": [{"id": "same", "wall_id": "w0", "s": 1, "width": 1, "height": 2}],
            "windows": [{"id": "win", "wall_id": "w1", "s": 2, "width": 1, "sill_height": 0.8, "height": 1}],
        },
    }
    assert len(builder._room_opening_list(room, "openings")) == 2
    by_wall = builder._room_openings_by_wall(room)
    assert sorted(by_wall) == ["w0", "w1"]
    assert builder._opening_kind({"id": "door_main"}) == "door"
    assert builder._opening_kind({"id": "passage"}) == "opening"
    assert builder._opening_z0({"id": "bad", "z0": "bad"}, 0.3) == 0.3
    assert builder._opening_height({"height": "bad"}, 2.1) == 2.1

    assert builder._should_skip_placeholder_bbox({"meta": {"placeholder_bbox": True, "procedural_requirement": True}})
    assert builder._should_skip_placeholder_bbox({"name": "scatter:fruit", "meta": {"placeholder_bbox": True}})
    assert builder._should_skip_placeholder_bbox({"name": "Cube.001", "meta": {"placeholder_bbox": True}})
    assert not builder._should_skip_placeholder_bbox({"name": "real chair", "meta": {"placeholder_bbox": True}})

    assert builder._item_semantic_group({"category": "WallMountedTVFactory", "name": "television"}) == "tv_projector_screen"
    assert builder._item_semantic_group({"name": "macbook on desk"}) == "computer"
    assert builder._item_mount_mode({"constraints": {"under_ceiling": True}}) == "ceiling"
    assert builder._item_mount_mode({"category": "ChairFactory"}) == "floor"
    assert builder._should_lock_supplier_rotation({"meta": {"affordance": "table_chair"}}, "chair")
    assert builder._should_lock_supplier_rotation({"meta": {"bed_headboard_repaired": True}}, "bed")


def test_world_preview_material_chain_and_reference_floor_edges(builder, tmp_path, capsys):
    materials = FakeMaterialStore()
    builder.bpy.data.materials = materials

    neutral = builder._make_neutral_missing_texture_material()
    assert neutral.name == "MAT_CGS_MISSING_TEXTURE_NEUTRAL_GRAY"
    assert neutral.use_nodes is True
    assert any(node.type == "BSDF_PRINCIPLED" for node in neutral.node_tree.nodes)

    missing_image = types.SimpleNamespace(name="Map #1", filepath="", filepath_raw="", source="FILE", packed_file=None, size=(0, 0), has_data=False)
    missing_node = types.SimpleNamespace(
        type="TEX_IMAGE",
        image=missing_image,
        inputs=[],
        as_pointer=lambda: id(missing_image),
    )
    root_socket = FakeNodeSocket("Base Color")
    root_socket.is_linked = True
    root_socket.links = [types.SimpleNamespace(from_node=missing_node)]
    assert builder._socket_chain_has_missing_image(root_socket)

    mat = materials.new("MAT_missing")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].is_linked = True
    bsdf.inputs["Base Color"].links = [types.SimpleNamespace(from_node=missing_node)]
    assert builder._material_has_missing_basecolor_texture(mat)

    mesh_obj = FakeObject("mesh_with_missing_mat")
    mesh_obj.data.materials = [mat]
    assert builder._replace_missing_texture_materials([mesh_obj]) == 1
    assert mesh_obj.data.materials[0] is neutral

    world = types.SimpleNamespace(use_nodes=False, node_tree=types.SimpleNamespace(nodes=FakeNodes(), links=FakeLinks()))
    builder.bpy.data.worlds = types.SimpleNamespace(new=lambda name: world)
    builder.bpy.context.scene.world = None
    builder._ensure_world()
    assert builder.bpy.context.scene.world is world
    assert any(node.type == "ShaderNodeBackground" for node in world.node_tree.nodes)

    shading = types.SimpleNamespace(type="", use_scene_lights=False, use_scene_world=True)
    view = types.SimpleNamespace(type="VIEW_3D", spaces=types.SimpleNamespace(active=types.SimpleNamespace(shading=shading)))
    ignored = types.SimpleNamespace(type="OUTLINER", spaces=types.SimpleNamespace(active=types.SimpleNamespace()))
    builder.bpy.context.window_manager.windows = [
        types.SimpleNamespace(screen=None),
        types.SimpleNamespace(screen=types.SimpleNamespace(areas=[ignored, view])),
    ]
    builder._force_material_preview_if_ui()
    assert shading.type == "MATERIAL"
    assert shading.use_scene_lights is True
    assert shading.use_scene_world is False

    class RaisingObject(FakeObject):
        def evaluated_get(self, _depsgraph):
            raise RuntimeError("bad depsgraph")

    valid = FakeObject("reference_floor")
    valid.data.vertices = [
        types.SimpleNamespace(co=FakeVector((0.0, 0.0, 0.12))),
        types.SimpleNamespace(co=FakeVector((1.0, 0.0, 0.12))),
        types.SimpleNamespace(co=FakeVector((1.0, 1.0, 0.12))),
        types.SimpleNamespace(co=FakeVector((0.0, 1.0, 0.12))),
    ]
    valid.data.polygons = [types.SimpleNamespace(normal=FakeVector((0.0, 0.0, 1.0)), vertices=(0, 1, 2, 3))]
    hidden = FakeObject("hidden_floor")
    hidden.hide_render = True
    overlay = FakeObject("chair_supplieroverlay_aabb")
    bad = RaisingObject("bad_floor")
    builder.bpy.data.objects = FakeObjects([FakeObject("curve", obj_type="CURVE"), hidden, overlay, bad, valid])

    inferred = builder._infer_reference_floor_z([(0, 0), (2, 0), (2, 2), (0, 2)], fallback_z=0.0, verbose=True)
    assert inferred == pytest.approx(0.12)
    assert "inferred reference floor z" in capsys.readouterr().out
    assert builder._infer_reference_floor_z([(0, 0), (1, 0)], fallback_z=0.4) == 0.4


def test_tint_incoming_links_and_basic_lights(builder):
    materials = FakeMaterialStore()
    builder.bpy.data.materials = materials

    def make_linked_mat(image):
        mat = materials.new(f"mat_{getattr(image, 'name', 'img')}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        source_node = types.SimpleNamespace(
            type="TEX_IMAGE",
            image=image,
            inputs=[],
            as_pointer=lambda: id(image),
        )
        source_socket = FakeNodeSocket("Color", source_node)
        link = types.SimpleNamespace(from_node=source_node, from_socket=source_socket, to_node=bsdf, to_socket=bsdf.inputs["Base Color"])
        mat.node_tree.links.links.append(link)
        bsdf.inputs["Base Color"].is_linked = True
        bsdf.inputs["Base Color"].links = [link]
        return mat, bsdf

    real_image = types.SimpleNamespace(name="real.jpg", filepath="/tmp/real.jpg", filepath_raw="", source="FILE", packed_file=None, size=(8, 8), has_data=True)
    real_mat, real_bsdf = make_linked_mat(real_image)
    assert builder._apply_tint_to_material_nodes(real_mat, (0.2, 0.3, 0.4), strength=0.25)
    assert any(node.label == "SUPPLIER_TINT_MIX" for node in real_mat.node_tree.nodes)

    missing_image = types.SimpleNamespace(name="missing.jpg", filepath="", filepath_raw="", source="FILE", packed_file=None, size=(0, 0), has_data=False)
    missing_mat, missing_bsdf = make_linked_mat(missing_image)
    assert builder._apply_tint_to_material_nodes(missing_mat, (0.1, 0.2, 0.3), strength=0.5)
    assert missing_bsdf.inputs["Base Color"].default_value == (0.1, 0.2, 0.3, 1.0)

    parent = FakeObject("parent", obj_type="EMPTY")
    child = FakeObject("child", parent=parent)
    child.data.materials = [real_mat, missing_mat]
    assert builder._apply_tint_to_existing_materials(parent, (0.3, 0.4, 0.5), strength=0.2) == 2
    assert builder._apply_tint_to_existing_materials(parent, (0.7, 0.7, 0.7), strength=0.2) == 0

    created_lights = []
    scene_collection = FakeCollection("Scene")
    builder.bpy.context.scene.collection.objects = scene_collection.objects
    builder.bpy.data.lights = types.SimpleNamespace(
        new=lambda name, light_type: created_lights.append(types.SimpleNamespace(name=name, type=light_type, energy=0.0, size=0.0)) or created_lights[-1]
    )
    object_store = FakeObjectStore()
    builder.bpy.data.objects = object_store
    builder._add_basic_lights(FakeVector((0, 0, 0)), FakeVector((2, 2, 2)))
    assert [light.type for light in created_lights] == ["SUN", "AREA"]
    assert {"Sun", "Key"} <= {obj.name for obj in object_store}


def _install_geometry_backend(builder):
    objects = FakeObjectStore()
    meshes = FakeMeshStore()
    materials = FakeMaterialStore()
    scene_collection = FakeCollection("Scene")
    builder.bpy.data.objects = objects
    builder.bpy.data.meshes = meshes
    builder.bpy.data.materials = materials
    builder.bpy.data.images = types.SimpleNamespace(
        load=lambda path, check_existing=True: types.SimpleNamespace(
            name=Path(path).name,
            filepath=str(path),
            filepath_raw=str(path),
            source="FILE",
            packed_file=None,
            size=(4, 4),
            has_data=True,
            colorspace_settings=types.SimpleNamespace(name=""),
        )
    )
    builder.bpy.context.scene.collection.objects = scene_collection.objects
    builder.bpy.context.view_layer.update = lambda: None
    return objects, scene_collection


def _install_fake_bmesh_backend(builder):
    class FakeBMVertStore(list):
        def new(self, co):
            vertex = types.SimpleNamespace(co=co)
            self.append(vertex)
            return vertex

    class FakeBMFaceStore(list):
        def new(self, verts):
            self.append(list(verts))
            return self[-1]

    class FakeBM:
        def __init__(self):
            self.verts = FakeBMVertStore()
            self.faces = FakeBMFaceStore()

        def to_mesh(self, mesh):
            index_by_vertex = {id(vertex): index for index, vertex in enumerate(self.verts)}
            verts = [vertex.co for vertex in self.verts]
            faces = [tuple(index_by_vertex[id(vertex)] for vertex in face) for face in self.faces]
            mesh.from_pydata(verts, [], faces)

        def free(self):
            return None

    builder.bmesh.new = FakeBM
    builder.bmesh.ops = types.SimpleNamespace(triangulate=lambda *_args, **_kwargs: None)


def test_supplier_proxy_curtain_and_import_edge_branches(builder, monkeypatch, tmp_path):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")

    wardrobe = builder._make_exact_supplier_proxy_in_aabb(
        item={"id": "wardrobe1", "meta": {"supplier_candidate": {"color_hex": "#886644"}}},
        aabb=aabb(0, 2.2, 0, 0.55, 0, 2.4),
        collection=coll,
        name="Wide wardrobe",
        group="wardrobe",
    )
    assert wardrobe is not None
    assert any(obj.name.endswith("wardrobe_handle_l") for obj in coll.objects)

    wall_art = builder._make_exact_supplier_proxy_in_aabb(
        item={"id": "art1"},
        aabb=aabb(2.8, 2.86, 0.2, 1.4, 0.7, 1.6),
        collection=coll,
        name="Tall wall art",
        group="wall_art",
    )
    assert wall_art is not None
    assert any(obj.name.endswith("wall_art_canvas") for obj in coll.objects)

    rug = builder._make_exact_supplier_proxy_in_aabb(
        item={"id": "rug1"},
        aabb=aabb(0, 1.2, 0, 1.0, 0, 0.004),
        collection=coll,
        name="Soft rug",
        group="rug",
    )
    assert rug is not None
    assert rug["cgs_supplier_proxy_fallback"] == "exact_aabb"

    def primitive_cylinder_add(**kwargs):
        mesh = builder.bpy.data.meshes.new("Cylinder_mesh")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [], [(0, 1, 2)])
        mesh.update()
        rod = builder.bpy.data.objects.new("Cylinder", mesh)
        rod.location = kwargs.get("location", (0, 0, 0))
        rod.rotation_euler = kwargs.get("rotation", (0, 0, 0))
        builder.bpy.context.object = rod

    builder.bpy.ops.mesh = types.SimpleNamespace(primitive_cylinder_add=primitive_cylinder_add)
    curtain = builder._make_curtain_proxy_mesh(
        item={"id": "curtain1", "size_m": [1.5, 0.08, 1.8]},
        aabb=aabb(0, 1.5, 0, 0.08, 0.2, 2.0),
        rotation_deg_engine=12.0,
        texture_path=None,
        collection=coll,
        name="CurtainProxy",
    )
    assert curtain is not None
    assert any(obj.name == "CurtainProxy_rod" for obj in coll.objects)

    assert builder._build_obj_import_override() is None
    region = types.SimpleNamespace(type="WINDOW")
    area = types.SimpleNamespace(type="VIEW_3D", regions=[region], spaces=types.SimpleNamespace(active="space"))
    screen = types.SimpleNamespace(areas=[area])
    builder.bpy.context.window_manager = types.SimpleNamespace(windows=[types.SimpleNamespace(screen=screen)])
    override = builder._build_obj_import_override()
    assert override["region"] is region
    assert override["space_data"] == "space"

    class TempOverride:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    builder.bpy.context.temp_override = lambda **_kwargs: TempOverride()
    builder.bpy.context.mode = "EDIT"
    mode_changes = []

    def add_imported(name):
        obj = builder.bpy.data.objects.new(name, FakeMesh(f"{name}_mesh"))
        return obj

    builder.bpy.ops.object = types.SimpleNamespace(mode_set=lambda mode: mode_changes.append(mode))
    builder.bpy.ops.wm = types.SimpleNamespace(obj_import=lambda filepath: (_ for _ in ()).throw(RuntimeError("wm fail")))
    builder.bpy.ops.import_scene = types.SimpleNamespace(
        obj=lambda filepath: add_imported("ImportedOBJ"),
        fbx=lambda filepath: add_imported("ImportedFBX"),
        gltf=lambda filepath: add_imported("ImportedGLB"),
    )
    assert [obj.name for obj in builder.import_supported_mesh(str(tmp_path / "model.obj"))] == ["ImportedOBJ"]
    assert mode_changes == ["OBJECT"]
    assert [obj.name for obj in builder.import_supported_mesh(str(tmp_path / "model.fbx"))] == ["ImportedFBX"]
    assert [obj.name for obj in builder.import_supported_mesh(str(tmp_path / "model.glb"))] == ["ImportedGLB"]
    with pytest.raises(RuntimeError):
        builder.import_supported_mesh(str(tmp_path / "model.max"))


def test_procedural_catalog_proxy_factory_branches(builder, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")

    pillow_proxy = builder._build_procedural_catalog_proxy_in_aabb(
        item={"id": "pillow1", "category": "pillow", "semantic_group": "pillow"},
        aabb=aabb(0, 0.6, 0, 0.4, 0.5, 0.53),
        rotation_deg_engine=0,
        collection=coll,
        name="PillowProxy",
        fit_mode="fit",
    )
    assert pillow_proxy is not None
    assert pillow_proxy["cgs_supplier_proxy_fallback"] == "exact_aabb"

    seen = {}

    def build_from_catalog_item(candidate, loc, fallback_subclass, collection_name):
        seen.update(
            {
                "candidate": dict(candidate),
                "loc": loc,
                "fallback_subclass": fallback_subclass,
                "collection_name": collection_name,
            }
        )
        built = builder.bpy.data.objects.new("FactoryBuilt", FakeMesh("FactoryBuiltMesh"))
        return [built]

    fake_factory = types.SimpleNamespace(build_from_catalog_item=build_from_catalog_item)
    monkeypatch.setitem(sys.modules, "src.Plasement.procedural_object_factory_blender", fake_factory)
    placed_root = FakeObject("PlacedFactoryRoot", obj_type="EMPTY")
    monkeypatch.setattr(builder, "place_in_aabb", lambda **kwargs: placed_root)

    result = builder._build_procedural_catalog_proxy_in_aabb(
        item={
            "id": "cab1",
            "size_m": [0.7, 0.3, 1.1],
            "asset": {"base_type": "wall_shelf", "taxonomy_subclass": "open", "material": "wood", "color": "#112233"},
            "meta": {"supplier_candidate": {"semantic_group": "shelf", "_source": "catalog"}},
        },
        aabb=aabb(0, 0.7, 0, 0.3, 0, 1.1),
        rotation_deg_engine=45,
        collection=coll,
        name="CatalogProxy",
        fit_mode="fit",
    )
    assert result is placed_root
    assert seen["candidate"]["width_cm"] == pytest.approx(70.0)
    assert seen["candidate"]["taxonomy_subclass"] == "open"
    assert "_source" not in seen["candidate"]

    fake_factory.build_from_catalog_item = lambda *args, **kwargs: []
    assert (
        builder._build_procedural_catalog_proxy_in_aabb(
            item={"id": "empty", "category": "chair"},
            aabb=aabb(),
            rotation_deg_engine=0,
            collection=coll,
            name="EmptyFactory",
            fit_mode="fit",
        )
        is None
    )

    fake_factory.build_from_catalog_item = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("factory fail"))
    assert (
        builder._build_procedural_catalog_proxy_in_aabb(
            item={"id": "boom", "category": "chair"},
            aabb=aabb(),
            rotation_deg_engine=0,
            collection=coll,
            name="FailFactory",
            fit_mode="fit",
        )
        is None
    )

    cleanup_obj = builder.bpy.data.objects.new("CleanupBuilt", FakeMesh("CleanupMesh"))
    fake_factory.build_from_catalog_item = lambda *args, **kwargs: [cleanup_obj]
    monkeypatch.setattr(builder, "place_in_aabb", lambda **kwargs: None)
    removed = []
    monkeypatch.setattr(builder, "_remove_object_family", lambda obj: (_ for _ in ()).throw(RuntimeError("remove fail")))
    original_remove = builder.bpy.data.objects.remove

    def tracking_remove(obj, do_unlink=True):
        removed.append(obj.name)
        original_remove(obj, do_unlink=do_unlink)

    builder.bpy.data.objects.remove = tracking_remove
    assert (
        builder._build_procedural_catalog_proxy_in_aabb(
            item={"id": "cleanup", "category": "chair"},
            aabb=aabb(),
            rotation_deg_engine=0,
            collection=coll,
            name="CleanupFactory",
            fit_mode="fit",
        )
        is None
    )
    assert removed == ["CleanupBuilt"]


def test_place_in_aabb_real_scaling_and_rejection_paths(builder, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")
    mesh_obj = FakeObject("chair_body")
    objects.append(mesh_obj)
    for name in ("preview_mesh", "outlier_mesh", "cluster_mesh", "variant_mesh"):
        objects.append(FakeObject(name))

    monkeypatch.setattr(builder, "_drop_material_preview_meshes", lambda objs: (objs, ["preview_mesh"]))
    monkeypatch.setattr(
        builder,
        "_filter_imported_mesh_outliers",
        lambda objs: (objs, [{"name": "outlier_mesh", "diag": 10.0, "longest": 9.0, "footprint": 0.01, "thin_ratio": 0.001}]),
    )
    monkeypatch.setattr(
        builder,
        "_keep_primary_import_cluster",
        lambda objs: (objs, [{"name": "cluster_mesh", "diag": 1.0, "longest": 1.0, "distance_to_origin": 99.0}]),
    )
    monkeypatch.setattr(builder, "_select_single_import_variant_mesh", lambda objs, *_args: (objs, ["variant_mesh"]))

    def fake_bounds(mesh_objs):
        parent = mesh_objs[0].parent
        loc = parent.location
        scale = parent.scale
        return loc.copy(), FakeVector((loc.x + scale.x, loc.y + scale.y, loc.z + scale.z))

    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", fake_bounds)

    placed = builder.place_in_aabb(
        [mesh_obj],
        aabb(0, 2, 0, 2, 0, 1),
        rotation_deg_engine=10,
        fit_mode="stretch",
        parent_name="PlacedChair",
        collection=coll,
        snap_to_floor=True,
        floor_offset=0.02,
        semantic_group="decor",
        lock_rotation=True,
    )
    assert placed is not None
    assert placed.name == "PlacedChair"
    assert placed["cgs_placement_confidence"] in {"high", "medium", "low"}
    assert objects.get("preview_mesh") is None
    assert objects.get("outlier_mesh") is None

    rigid_mesh = FakeObject("wardrobe_body")
    objects.append(rigid_mesh)
    rejected = builder.place_in_aabb(
        [rigid_mesh],
        aabb(0, 4, 0, 0.4, 0, 1),
        rotation_deg_engine=0,
        fit_mode="stretch",
        parent_name="RejectedWardrobe",
        collection=coll,
        snap_to_floor=True,
        floor_offset=0.0,
        semantic_group="wardrobe",
        lock_rotation=True,
    )
    assert rejected is None
    assert objects.get("RejectedWardrobe") is None


def test_place_in_aabb_fit_modes_ceiling_and_wall_penalty(builder, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")
    builder._store_scene_room_bounds(FakeVector((0, 0, 0)), FakeVector((4, 4, 3)))
    builder.mathutils.Matrix = types.SimpleNamespace(Rotation=lambda *_args, **_kwargs: FakeMatrix())

    bounds_size = {"value": FakeVector((1.0, 1.0, 1.0))}

    def fake_bounds(mesh_objs):
        parent = mesh_objs[0].parent
        loc = parent.location
        scale = parent.scale
        size = bounds_size["value"]
        return loc.copy(), FakeVector((loc.x + abs(scale.x) * size.x, loc.y + abs(scale.y) * size.y, loc.z + abs(scale.z) * size.z))

    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", fake_bounds)
    monkeypatch.setattr(builder, "_drop_material_preview_meshes", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "_filter_imported_mesh_outliers", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "_keep_primary_import_cluster", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "_select_single_import_variant_mesh", lambda objs, *_args: (objs, []))

    cases = [
        ("UniformFit", "uniform", aabb(0, 2, 0, 3, 0, 4), "bed", False, None),
        ("WallHeightFit", "wall_height", aabb(0, 2, 0, 1, 0, 4), "bathroom_sink", False, None),
        ("CurtainWide", "curtain_soft_width", aabb(0, 3, 0, 0.4, 0, 2), "curtain", False, None),
        ("CurtainTallAxis", "curtain_window_soft_width", aabb(0, 0.4, 0, 3, 0, 2), "curtain", False, None),
        ("CeilingSnap", "stretch", aabb(1, 2, 1, 2, 2, 3), "lamp_ceiling", True, None),
        ("TargetSizeFit", "uniform", aabb(0, 1, 0, 1, 0, 1), "chair", False, (0.8, 0.6, 1.2)),
    ]
    for name, fit_mode, box, group, snap_ceiling, target_size in cases:
        mesh = FakeObject(f"{name}_mesh")
        objects.append(mesh)
        placed = builder.place_in_aabb(
            [mesh],
            box,
            rotation_deg_engine=90,
            fit_mode=fit_mode,
            parent_name=name,
            collection=coll,
            snap_to_floor=not snap_ceiling,
            floor_offset=0.0,
            semantic_group=group,
            snap_to_ceiling=snap_ceiling,
            ceiling_offset=0.02 if snap_ceiling else 0.0,
            fit_target_size_m=target_size,
            enforce_scale_guard=False,
        )
        assert placed is not None
        assert placed["cgs_placement_rotation_deg"] in {0.0, 90.0, 180.0, 270.0}


def test_room_build_and_supplier_wall_floor_overlay_paths(builder, tmp_path, monkeypatch):
    _objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Room")
    floor_tex = tmp_path / "floor_oak.jpg"
    wall_tex = tmp_path / "wall_paint.jpg"
    floor_tex.write_bytes(b"img")
    wall_tex.write_bytes(b"img")
    room = {
        "room": {
            "type": "bathroom",
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 3}, {"x": 0, "y": 3}],
            "ceiling_height": 2.8,
            "walls": [
                {"id": "w0", "from_vertex": 0, "to_vertex": 1},
                {"id": "bad", "from_vertex": 99, "to_vertex": 1},
                {"id": "w1", "from_vertex": 1, "to_vertex": 2},
            ],
            "doors": [{"id": "door", "wall_id": "w0", "s": 0.5, "width": 0.8, "height": 2.0}],
            "windows": [{"id": "win", "wall_id": "w1", "s": 0.4, "width": 1.0, "sill_height": 0.8, "height": 1.0}],
            "floor_material": {"texture_path": str(floor_tex), "texture_tiling": {"tile_size_m": 1.0, "mode": "mirror"}},
            "wall_material": {"texture_path": str(wall_tex), "wall_tiling": {"tile_size_m": 0.5}},
        }
    }

    calls = {"floor": 0, "wall": 0, "decal": 0, "assigned": 0}

    def fake_floor(name, poly_xy, z, coll):
        calls["floor"] += 1
        obj = FakeObject(name)
        obj.data = FakeMesh(name + "_mesh")
        coll.objects.link(obj)
        return obj

    def fake_wall(name, p0, p1, z0, z1, coll):
        calls["wall"] += 1
        obj = FakeObject(name)
        obj.data = FakeMesh(name + "_mesh")
        coll.objects.link(obj)
        return obj

    def fake_decal(name, **kwargs):
        calls["decal"] += 1
        return [fake_wall(name + "_Inner", kwargs["wall_p0"], kwargs["wall_p1"], kwargs["z0"], kwargs["z0"] + kwargs["height"], kwargs["coll"])]

    monkeypatch.setattr(builder, "_make_floor_from_polygon", fake_floor)
    monkeypatch.setattr(builder, "_make_wall_quad", fake_wall)
    monkeypatch.setattr(builder, "_make_double_sided_decal_on_wall", fake_decal)
    monkeypatch.setattr(builder, "_set_uvs_floor_xy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_set_uvs_wall_sz", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_assign_material_to_object", lambda *_args, **_kwargs: calls.__setitem__("assigned", calls["assigned"] + 1))
    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", lambda _objs: (FakeVector((0, 0, 0)), FakeVector((4, 3, 2.8))))

    objs, bmin, bmax = builder.build_room_from_spec(room, coll, str(tmp_path), "warm style", 1, verbose=True)
    assert len(objs) >= 5
    assert calls["floor"] == 1
    assert calls["wall"] >= 2
    assert calls["decal"] == 2
    assert bmax == FakeVector((4, 3, 2.8))

    floor_overlay = builder.add_supplier_floor_overlay_from_spec(room, coll, floor_z_override=0.1, verbose=True)
    assert floor_overlay is not None
    wall_overlay = builder.add_supplier_wall_overlay_from_spec(room, coll, floor_z_override=0.1, verbose=True)
    assert wall_overlay

    ref_wall = FakeObject("Room_Wall_Reference")
    ref_window = FakeObject("Room_Window_Reference")
    builder.bpy.data.objects.extend([ref_wall, ref_window])
    assert builder.apply_supplier_wall_material_to_reference_scene(room, verbose=True) == 1


def test_supplier_proxy_and_procedural_requirement_meshes(builder, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")

    wardrobe = builder._make_exact_supplier_proxy_in_aabb(
        item={"id": "wardrobe", "asset": {"color": "#556677"}},
        aabb=aabb(0, 1.2, 0, 2.0, 0, 2.4),
        collection=coll,
        name="WardrobeProxy",
        group="wardrobe",
    )
    assert wardrobe is not None
    assert wardrobe["cgs_supplier_proxy_fallback"] == "exact_aabb"
    assert len(coll.objects) >= 2

    wide_wall_art = builder._make_exact_supplier_proxy_in_aabb(
        item={"id": "art"},
        aabb=aabb(0, 2.0, 0, 0.08, 1.0, 2.0),
        collection=coll,
        name="ArtProxy",
        group="wall_art",
    )
    assert wide_wall_art is not None
    assert any("wall_art_canvas" in obj.name for obj in coll.objects)

    rug = builder._make_exact_supplier_proxy_in_aabb(
        item={"id": "rug"},
        aabb=aabb(0, 2.0, 0, 1.0, 0, 0.005),
        collection=coll,
        name="RugProxy",
        group="rug",
    )
    assert rug is not None
    assert rug["cgs_placement_confidence"] == "proxy"

    monkeypatch.setattr(builder, "_try_procedural_requirement_factory_mesh", lambda **_kwargs: None)
    for role in ("toilet", "sink", "shower", "bath", "table", "headboard", "bed"):
        root = builder._make_procedural_requirement_mesh(
            item={"id": role, "category": role, "asset": {"kind": "procedural_placeholder"}},
            aabb=aabb(0, 1.2, 0, 0.8, 0, 1.2),
            rotation_deg_engine=15,
            collection=coll,
            name=f"{role}_proxy",
        )
        assert root is not None
        assert root["cgs_procedural_requirement"] == role

    curtain = builder._make_curtain_proxy_mesh(
        item={"id": "curtain", "size_m": [1.6, 0.08, 2.2]},
        aabb=aabb(0, 1.6, 0, 0.08, 0, 2.2),
        rotation_deg_engine=0,
        texture_path=None,
        collection=coll,
        name="CurtainProxy",
    )
    assert curtain is not None
    assert curtain["cgs_procedural_proxy"] == "curtain"
    assert any(obj.name == "CurtainProxy_cloth" for obj in objects)


def test_catalog_proxy_uses_factory_candidate_and_cleanup(builder, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")
    built_obj = FakeObject("factory_built_mesh")
    objects.append(built_obj)
    captured = {}

    fake_factory = types.ModuleType("src.Plasement.procedural_object_factory_blender")

    def fake_build_from_catalog_item(candidate, **_kwargs):
        captured["candidate"] = dict(candidate)
        return [built_obj]

    fake_factory.build_from_catalog_item = fake_build_from_catalog_item
    monkeypatch.setitem(sys.modules, "src.Plasement.procedural_object_factory_blender", fake_factory)

    def fake_place(**kwargs):
        captured["place"] = kwargs
        root = FakeObject(kwargs["parent_name"], obj_type="EMPTY")
        objects.append(root)
        return root

    monkeypatch.setattr(builder, "place_in_aabb", fake_place)
    root = builder._build_procedural_catalog_proxy_in_aabb(
        item={
            "id": "soap",
            "category": "soap_dispenser",
            "asset": {"base_type": "soap_dispenser", "taxonomy_subclass": "pump", "material": "ceramic", "color": "#eeeeee"},
            "meta": {"supplier_candidate": {"semantic_group": "soap_dispenser", "_source": "catalog", "unique_key": "u1"}},
        },
        aabb=aabb(0, 0.2, 0, 0.2, 0, 0.5),
        rotation_deg_engine=30,
        collection=coll,
        name="SoapProxy",
        fit_mode="fit",
    )
    assert root is not None
    assert captured["candidate"]["base_type"] == "soap_dispenser"
    assert captured["candidate"]["width_cm"] == pytest.approx(20.0)
    assert "_source" not in captured["candidate"]
    assert captured["place"]["lock_rotation"] is True


def test_build_scene_orchestrates_items_with_mocked_blender_operations(builder, tmp_path, monkeypatch):
    class LinkedObjects(list):
        def link(self, obj):
            self.append(obj)
            obj.users_collection.append(collection)

        def unlink(self, obj):
            if obj in self:
                self.remove(obj)

    collection = types.SimpleNamespace(name="Items", objects=LinkedObjects())
    room_collection = types.SimpleNamespace(name="Room", objects=LinkedObjects())
    collections = {"Room": room_collection, "Items": collection, "BBoxOverlay": collection}

    def fake_ensure_collection(name):
        return collections.setdefault(name, types.SimpleNamespace(name=name, objects=LinkedObjects()))

    def make_root(name, box):
        obj = FakeObject(name, obj_type="EMPTY")
        obj.location = FakeVector((0.0, 0.0, 0.0))
        obj.rotation_euler = [0.0, 0.0, 0.0]
        obj["_aabb"] = dict(box)
        builder.bpy.data.objects.append(obj)
        collection.objects.link(obj)
        return obj

    def make_for_item(*, item, aabb, name, **_kwargs):
        return make_root(name, aabb)

    scene_path = tmp_path / "scene.json"
    mesh = tmp_path / "chair.glb"
    mesh.write_bytes(b"glb")
    scene = {
        "room": {
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 4}, {"x": 0, "y": 4}],
            "ceiling_height": 3.0,
        },
        "items": [
            {"id": "flat_light", "name": "flat light", "category": "ceiling_light", "asset": {"kind": "procedural_flat_ceiling_light"}, "aabb": aabb(1, 2, 1, 2, 2.9, 3.0), "constraints": {"mount_type": "ceiling"}},
            {"id": "toilet", "name": "compact toilet", "category": "toilet", "asset": {"kind": "procedural_placeholder"}, "aabb": aabb(0.2, 0.8, 0.2, 0.8, 0, 0.7)},
            {"id": "curtain", "name": "window curtain", "category": "curtain", "asset": {"kind": "procedural_curtain_proxy"}, "aabb": aabb(0, 2, 0, 0.1, 0, 2.4)},
            {"id": "table", "name": "generated table", "category": "table", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(2, 3, 2, 3, 0, 0.75), "meta": {"supplier_binding_applied": True}},
            {"id": "vase", "name": "vase on table", "category": "vase", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(2.2, 2.5, 2.2, 2.5, 0, 0.3), "meta": {"supplier_binding_applied": True, "supplier_support_reanchored": True, "supplier_support_anchor_target_id": "table", "supplier_support_mode": "top"}},
            {"id": "chair", "name": "imported chair", "category": "chair", "asset": {"mesh_path": str(mesh), "kind": "supplier_model"}, "aabb": aabb(3, 3.5, 1, 1.5, 0, 0.9), "meta": {"supplier_binding_applied": True, "supplier_candidate": {"unique_key": "chair-key", "semantic_group": "chair"}}},
            {"id": "lamp", "name": "desk lamp", "category": "DeskLampFactory", "aabb": aabb(4, 4.2, 1, 1.2, 0.75, 1.2), "meta": {"supplier_binding_applied": True}},
            {"id": "missing", "name": "missing replacement", "category": "wardrobe", "aabb": aabb(4.2, 5.4, 1, 2, 0, 2.2), "source": {"asset_source": "supplier_catalog_local_asset"}, "meta": {"supplier_binding_applied": True, "supplier_candidate": {"unique_key": "wardrobe-key", "semantic_group": "wardrobe"}}},
        ],
    }
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    calls = {"bbox": 0, "labels": 0, "lights": 0, "textures": 0, "support_moves": 0}
    monkeypatch.setattr(builder, "ensure_collection", fake_ensure_collection)
    monkeypatch.setattr(builder, "build_room_from_spec", lambda **_kwargs: ([], FakeVector((0, 0, 0)), FakeVector((5, 4, 3))))
    monkeypatch.setattr(builder, "_make_procedural_flat_ceiling_light_mesh", make_for_item)
    monkeypatch.setattr(builder, "_make_procedural_requirement_mesh", make_for_item)
    monkeypatch.setattr(builder, "_make_curtain_proxy_mesh", make_for_item)
    monkeypatch.setattr(builder, "_build_procedural_catalog_proxy_in_aabb", make_for_item)
    monkeypatch.setattr(builder, "_aabb_from_object_family_root", lambda obj: obj.get("_aabb"))
    monkeypatch.setattr(builder, "_safe_import_supported_mesh", lambda _path: ([FakeObject("mesh_body")], None))
    monkeypatch.setattr(builder, "_remove_or_hide_non_mesh_import_objects", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "_filter_imported_mesh_outliers", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "_drop_material_preview_meshes", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "_select_single_import_variant_mesh", lambda objs, **_kwargs: (objs, []))
    monkeypatch.setattr(builder, "_keep_primary_import_cluster", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "place_in_aabb", lambda objs, aabb, collection, **_kwargs: make_root("placed_mesh", aabb))
    monkeypatch.setattr(builder, "_export_import_group_glb_cache", lambda *_args, **_kwargs: str(tmp_path / "cache.glb"))
    monkeypatch.setattr(builder, "_ensure_textures", lambda **_kwargs: calls.__setitem__("textures", calls["textures"] + 1))
    monkeypatch.setattr(builder, "_make_renderable_bbox_box", lambda *_args, **_kwargs: calls.__setitem__("bbox", calls["bbox"] + 1))
    monkeypatch.setattr(builder, "_add_aabb_box", lambda *_args, **_kwargs: calls.__setitem__("bbox", calls["bbox"] + 1))
    monkeypatch.setattr(builder, "_add_aabb_label", lambda *_args, **_kwargs: calls.__setitem__("labels", calls["labels"] + 1))
    monkeypatch.setattr(builder, "_extract_support_planes_from_object_family", lambda _root: [{"x_min": 2, "x_max": 3, "y_min": 2, "y_max": 3, "z": 0.75, "area": 1.0, "clearance_height": 2.0}])
    monkeypatch.setattr(
        builder,
        "_move_object_family_to_exact_aabb",
        lambda root, _old, new: root.update({"_aabb": dict(new)}) or calls.__setitem__("support_moves", calls["support_moves"] + 1) or dict(new),
    )
    monkeypatch.setattr(builder, "_add_functional_light_for_item", lambda *_args, **_kwargs: calls.__setitem__("lights", calls["lights"] + 1) or 1)
    monkeypatch.setattr(builder, "_frame_camera_on_bounds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_add_basic_lights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_force_material_preview_if_ui", lambda: None)

    report = builder.build_scene(
        str(scene_path),
        draw_aabb=True,
        force_tint=False,
        keep_existing_mats=True,
        verbose=True,
        env_textures_dir=str(tmp_path),
        style_text=None,
        seed=1,
        bbox_fallback_missing_mesh=True,
        highlight_item_ids={"missing"},
    )

    assert report["schema"] == "blender_scene_builder_report/v1"
    assert report["functional_light_count"] >= 1
    assert calls["textures"] >= 1
    assert calls["support_moves"] == 1
    assert calls["bbox"] >= 1
    assert any(getattr(obj, "name", "") == "vase on table" for obj in builder.bpy.data.objects)


def test_material_node_relink_mtl_and_texture_assignment_paths(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    obj_path = asset_dir / "model.obj"
    mtl_path = asset_dir / "model.mtl"
    base = asset_dir / "model_diffuse.jpg"
    normal = asset_dir / "model_normal.png"
    rough = asset_dir / "model_roughness.jpg"
    metallic = asset_dir / "model_metallic.jpg"
    ao = asset_dir / "model_ao.jpg"
    height = asset_dir / "model_height.jpg"
    emissive = asset_dir / "model_emissive.jpg"
    opacity = asset_dir / "model_opacity.png"
    for path in (base, normal, rough, metallic, ao, height, emissive, opacity):
        path.write_bytes(b"image-data")
    obj_path.write_text("mtllib model.mtl\n", encoding="utf-8")
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl Fabric",
                "map_Kd model_diffuse.jpg",
                "bump model_normal.png",
                "map_Pr model_roughness.jpg",
                "map_Pm model_metallic.jpg",
                "map_d model_opacity.png",
            ]
        ),
        encoding="utf-8",
    )

    maps = builder.PBRMaps(
        basecolor=str(base),
        normal=str(normal),
        roughness=str(rough),
        metallic=str(metallic),
        ao=str(ao),
        height=str(height),
        emissive=str(emissive),
        opacity=str(opacity),
    )
    pbr = builder._make_pbr_material("Mat_Full", maps, tint_rgb=(0.2, 0.35, 0.5), tex_scale=2.0)
    assert pbr.use_nodes is True
    assert getattr(pbr, "blend_method", "") == "HASHED"
    assert any(getattr(node, "image", None) is not None for node in pbr.node_tree.nodes)

    tint_mat = builder._make_pbr_material("Mat_Tint", builder.PBRMaps(), tint_rgb=None, tex_scale=1.0)
    bsdf = next(node for node in tint_mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    assert builder._apply_tint_to_material_nodes(tint_mat, (0.1, 0.2, 0.3), strength=0.5)
    assert bsdf.inputs["Base Color"].default_value[:3] == pytest.approx((0.3, 0.35, 0.4))

    base_img = builder.bpy.data.images.load(str(base), check_existing=True)
    normal_img = builder.bpy.data.images.load(str(normal), check_existing=True)
    opacity_img = builder.bpy.data.images.load(str(opacity), check_existing=True)
    builder._apply_image_to_material_nodes(tint_mat, basecolor_img=base_img, normal_img=normal_img, opacity_img=opacity_img)
    assert getattr(tint_mat, "shadow_method", "") == "HASHED"

    parent = FakeObject("ImportedRoot", obj_type="EMPTY")
    child = FakeObject("ImportedMesh", parent=parent)
    fabric_mat = builder._make_pbr_material("Fabric", builder.PBRMaps(), tint_rgb=None, tex_scale=1.0)
    child.data.materials = [fabric_mat]
    idx = builder.build_image_index([str(asset_dir)])
    assert builder._apply_mtl_map_kd_to_existing_mats(parent, str(obj_path), idx, verbose=True)

    missing_node = FakeNode("ShaderNodeTexImage")
    missing_node.label = "model_diffuse.jpg"
    missing_node.image = types.SimpleNamespace(
        name="Map #1",
        filepath="",
        filepath_raw="",
        source="FILE",
        packed_file=None,
        size=(0, 0),
        has_data=False,
    )
    fabric_mat.node_tree.nodes.append(missing_node)
    fixed, used = builder._relink_missing_images(parent, idx)
    assert fixed >= 1
    assert str(base.resolve()) in used

    child.data.materials = []
    builder._ensure_textures(
        parent=parent,
        mesh_path=str(obj_path),
        mesh_texture_dirs=[str(asset_dir)],
        texture_path=str(base),
        texture_files=[str(normal), "model_roughness.jpg"],
        texture_scale=1.5,
        tint_rgb=(0.2, 0.3, 0.4),
        keep_existing_mats=False,
        verbose=True,
    )
    assert child.data.materials

    child.data.materials = [fabric_mat]
    monkeypatch.setattr(builder, "_has_loaded_textures", lambda _parent: False)
    assert builder._fallback_apply_largest_image_existing_mats(parent, idx, verbose=True)
    assert builder._fallback_apply_flat_tint_existing_mats(parent, (0.1, 0.2, 0.3), verbose=True)


def test_ensure_textures_keep_existing_branch_matrix(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    mesh_path = tmp_path / "model.obj"
    mesh_path.write_text("o Mesh\n", encoding="utf-8")
    tex = tmp_path / "basecolor.jpg"
    tex.write_bytes(b"img")

    def make_parent(name="Parent"):
        parent = FakeObject(name, obj_type="EMPTY")
        child = FakeObject(f"{name}_mesh", parent=parent)
        child.data.materials = [types.SimpleNamespace(name="old")]
        objects.extend([parent, child])
        return parent

    monkeypatch.setattr(builder, "build_search_dirs", lambda *_args, **_kwargs: [str(tmp_path)])
    monkeypatch.setattr(builder, "build_image_index", lambda *_args, **_kwargs: {"basecolor.jpg": str(tex)})
    monkeypatch.setattr(builder, "_apply_tint_to_existing_materials", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(builder, "_log", lambda *_args, **_kwargs: None)

    parent = make_parent("AlreadyTextured")
    monkeypatch.setattr(builder, "_has_loaded_textures", lambda _parent: True)
    builder._ensure_textures(parent, str(mesh_path), [], None, [], 1.0, (0.2, 0.3, 0.4), True, True)

    parent = make_parent("Relinked")
    calls = {"loaded": 0}

    def loaded_after_relink(_parent):
        calls["loaded"] += 1
        return calls["loaded"] >= 2

    monkeypatch.setattr(builder, "_has_loaded_textures", loaded_after_relink)
    monkeypatch.setattr(builder, "_relink_missing_images", lambda *_args, **_kwargs: (1, {str(tex)}))
    builder._ensure_textures(parent, str(mesh_path), [], None, [], 1.0, (0.2, 0.3, 0.4), True, True)

    parent = make_parent("MtlRestored")
    calls["loaded"] = 0
    monkeypatch.setattr(builder, "_has_loaded_textures", loaded_after_relink)
    monkeypatch.setattr(builder, "_relink_missing_images", lambda *_args, **_kwargs: (0, set()))
    monkeypatch.setattr(builder, "_apply_mtl_map_kd_to_existing_mats", lambda *_args, **_kwargs: True)
    builder._ensure_textures(parent, str(mesh_path), [], None, [], 1.0, (0.2, 0.3, 0.4), True, True)

    parent = make_parent("LargestFallback")
    calls["loaded"] = 0
    monkeypatch.setattr(builder, "_has_loaded_textures", loaded_after_relink)
    monkeypatch.setattr(builder, "_apply_mtl_map_kd_to_existing_mats", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(builder, "_object_has_any_material_slots", lambda _parent: True)
    monkeypatch.setattr(builder, "_fallback_apply_largest_image_existing_mats", lambda *_args, **_kwargs: True)
    builder._ensure_textures(parent, str(mesh_path), [], None, [], 1.0, (0.2, 0.3, 0.4), True, True)

    parent = make_parent("FlatTintFallback")
    monkeypatch.setattr(builder, "_has_loaded_textures", lambda _parent: False)
    monkeypatch.setattr(builder, "_fallback_apply_largest_image_existing_mats", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(builder, "_fallback_apply_flat_tint_existing_mats", lambda *_args, **_kwargs: True)
    builder._ensure_textures(parent, str(mesh_path), [], None, [], 1.0, (0.2, 0.3, 0.4), True, True)

    parent = make_parent("NoFallback")
    monkeypatch.setattr(builder, "_fallback_apply_flat_tint_existing_mats", lambda *_args, **_kwargs: False)
    builder._ensure_textures(parent, str(mesh_path), [], None, [], 1.0, (0.2, 0.3, 0.4), True, True)

    parent = make_parent("BuildFromTextureFiles")
    parent.children[0].data.materials = []
    monkeypatch.setattr(builder, "parse_obj_mtl_files", lambda _path: [])
    monkeypatch.setattr(builder, "_guess_maps_from_scan", lambda *_args, **_kwargs: builder.PBRMaps(roughness=str(tex)))
    builder._ensure_textures(
        parent,
        str(mesh_path),
        [],
        None,
        ["basecolor.jpg", str(tex), "missing_normal.png"],
        2.0,
        (0.2, 0.3, 0.4),
        False,
        True,
    )
    assert parent.children[0].data.materials


def test_procedural_light_factory_requirements_and_builder_main(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")
    created_lights = []

    class LightStore:
        def new(self, name, light_type):
            data = types.SimpleNamespace(name=name, type=light_type, energy=0.0, size=0.0)
            created_lights.append(data)
            return data

    def cylinder_add(**kwargs):
        obj = FakeObject(f"Cylinder{len(objects)}")
        obj.location = FakeVector(kwargs.get("location", (0, 0, 0)))
        objects.append(obj)
        builder.bpy.context.object = obj

    builder.bpy.data.lights = LightStore()
    builder.bpy.ops.mesh = types.SimpleNamespace(primitive_cylinder_add=cylinder_add)

    root = builder._make_procedural_flat_ceiling_light_mesh(
        item={"id": "light1"},
        aabb=aabb(0, 1, 0, 1, 2.85, 2.95),
        collection=coll,
        name="CeilingLight",
    )
    assert root is not None
    assert root["cgs_procedural_lighting"] == "flat_ceiling"
    assert created_lights and created_lights[0].energy == 220.0

    factory_roots = []

    def fake_catalog_proxy(**kwargs):
        obj = FakeObject(kwargs["name"], obj_type="EMPTY")
        factory_roots.append((kwargs, obj))
        return obj

    monkeypatch.setattr(builder, "_build_procedural_catalog_proxy_in_aabb", fake_catalog_proxy)
    for role in ("toilet", "sink", "shower", "bath"):
        made = builder._try_procedural_requirement_factory_mesh(
            item={"id": role, "name": f"compact {role}", "asset": {}},
            aabb=aabb(0, 1, 0, 1, 0, 1),
            rotation_deg_engine=0,
            collection=coll,
            name=f"{role}_factory",
            role=role,
        )
        assert made is not None
        assert made["cgs_procedural_requirement_factory"] is True
    assert factory_roots[0][0]["item"]["asset"]["base_type"] == "toilet"

    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"room": {}, "items": []}), encoding="utf-8")
    report_path = tmp_path / "report.json"
    blend_path = tmp_path / "scene.blend"
    png_path = tmp_path / "render.png"
    turntable_dir = tmp_path / "turntable"

    calls = {"build": 0, "render": 0, "turntable": 0, "save": 0, "pack": 0, "cleanup": 0}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "blender",
            "--",
            "--json",
            str(scene_path),
            "--project-root",
            str(tmp_path),
            "--build-report",
            str(report_path),
            "--save-blend",
            str(blend_path),
            "--render",
            str(png_path),
            "--turntable-render-dir",
            str(turntable_dir),
            "--turntable-frames",
            "3",
            "--turntable-frame-index",
            "1",
            "--hide-room-shell",
            "--render-layer",
            "kitchen",
            "--highlight-item-ids",
            "a,b",
            "--rebuild-materials",
            "--force-tint",
            "--draw-aabb",
        ],
    )
    monkeypatch.setattr(builder, "build_scene", lambda **kwargs: calls.__setitem__("build", calls["build"] + 1) or {"schema": "report", "json": kwargs["json_path"]})
    monkeypatch.setattr(builder, "_hide_room_shell_objects", lambda: 2)
    monkeypatch.setattr(builder, "_apply_render_layer_visibility", lambda layer: 3)
    monkeypatch.setattr(builder, "_cleanup_final_visual_helpers", lambda: calls.__setitem__("cleanup", calls["cleanup"] + 1) or {"ok": True})
    monkeypatch.setattr(builder, "_pack_assets_best_effort", lambda: calls.__setitem__("pack", calls["pack"] + 1))
    monkeypatch.setattr(builder, "_configure_fast_render", lambda scene: setattr(scene.render, "configured_fast", True))
    monkeypatch.setattr(builder, "_configure_turntable_render", lambda scene: setattr(scene.render, "configured_turntable", True))
    monkeypatch.setattr(builder, "_scene_room_bounds", lambda: (FakeVector((0, 0, 0)), FakeVector((4, 3, 3))))
    monkeypatch.setattr(builder, "_visible_mesh_bounds", lambda default_min, default_max: (default_min, default_max))
    monkeypatch.setattr(builder, "_render_turntable_sequence", lambda *args, **kwargs: calls.__setitem__("turntable", calls["turntable"] + 1))
    builder.bpy.context.scene.render.image_settings = types.SimpleNamespace(file_format="")
    builder.bpy.ops.wm.save_as_mainfile = lambda filepath: calls.__setitem__("save", calls["save"] + 1)
    builder.bpy.ops.render = types.SimpleNamespace(render=lambda write_still=True: calls.__setitem__("render", calls["render"] + 1))

    builder.main()

    assert calls == {"build": 1, "render": 1, "turntable": 1, "save": 1, "pack": 1, "cleanup": 1}
    assert json.loads(report_path.read_text(encoding="utf-8"))["final_visual_cleanup"] == {"ok": True}


def test_bounds_camera_turntable_import_and_light_duplication_paths(builder, tmp_path, monkeypatch):
    objects, scene_collection = _install_geometry_backend(builder)

    mesh = FakeMesh("bounds_mesh")
    mesh.vertices = [
        types.SimpleNamespace(co=FakeVector((-1, 0, 0))),
        types.SimpleNamespace(co=FakeVector((2, 3, 4))),
    ]
    mesh_obj = FakeObject("BoundsMesh")
    mesh_obj.data = mesh
    non_mesh = FakeObject("CameraHelper", obj_type="CAMERA")
    objects.extend([mesh_obj, non_mesh])

    bmin, bmax = builder._world_bounds_mesh_objects([non_mesh, mesh_obj])
    assert bmin == FakeVector((-1, 0, 0))
    assert bmax == FakeVector((2, 3, 4))
    single_min, single_max = builder._world_bounds_single_mesh_object(mesh_obj)
    assert single_min == bmin
    assert single_max == bmax
    zero_min, zero_max = builder._world_bounds_single_mesh_object(non_mesh)
    assert zero_min == zero_max == FakeVector((0, 0, 0))

    class CameraStore:
        def new(self, name):
            return types.SimpleNamespace(
                name=name,
                type="PERSP",
                lens=0,
                clip_start=0,
                clip_end=0,
                dof=types.SimpleNamespace(use_dof=True),
            )

    builder.bpy.data.cameras = CameraStore()
    cam = builder._ensure_scene_camera(FakeVector((0, 0, 0)), FakeVector((2, 2, 2)))
    assert cam.name == "CGS_TurntableCamera"
    assert builder.bpy.context.scene.camera is cam

    rendered_paths = []
    builder.bpy.ops.render = types.SimpleNamespace(render=lambda write_still=True: rendered_paths.append(builder.bpy.context.scene.render.filepath))
    builder._render_turntable_sequence(
        tmp_path / "turntable",
        frame_count=4,
        bb_min=FakeVector((0, 0, 0)),
        bb_max=FakeVector((2, 2, 2)),
        elevation_deg=25,
        frame_index=2,
    )
    assert len(rendered_paths) == 1
    assert rendered_paths[0].endswith("frame_002.png")
    with pytest.raises(ValueError, match="out of range"):
        builder._render_turntable_sequence(tmp_path / "bad", 2, FakeVector((0, 0, 0)), FakeVector((1, 1, 1)), frame_index=3)

    linked = []
    scene_collection.objects.link = lambda obj: linked.append(obj)
    root = FakeObject("PendantRoot", obj_type="EMPTY")
    fixture = FakeObject("PendantLight", obj_type="LIGHT", parent=root)
    fixture.data = types.SimpleNamespace(name="light_data", copy=lambda: types.SimpleNamespace(name="light_data_copy"))
    shade = FakeObject("shade lamp mesh", parent=fixture)
    duplicated = builder._duplicate_light_objects_from_family(root)
    assert duplicated >= 1
    assert any(obj.name.endswith("__kept") for obj in linked)

    imported = FakeObject("ImportedObj")
    builder.bpy.context.mode = "EDIT"

    class TempOverride:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    builder.bpy.context.temp_override = lambda **_kwargs: TempOverride()
    builder.bpy.ops.object.mode_set = lambda mode="OBJECT": setattr(builder.bpy.context, "mode", mode)

    def obj_import(filepath):
        assert filepath.endswith(".obj")
        objects.append(imported)

    builder.bpy.ops.wm.obj_import = obj_import
    assert builder.import_supported_mesh(str(tmp_path / "model.obj")) == [imported]

    selected = []
    builder.bpy.context.selected_objects = [mesh_obj]
    builder.bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set = lambda value: selected.append((mesh_obj.name, value))
    second = FakeObject("SecondMesh")
    objects.append(second)
    second.select_set = lambda value: selected.append((second.name, value))
    builder.bpy.ops.export_scene = types.SimpleNamespace(
        gltf=lambda **kwargs: Path(kwargs["filepath"]).write_bytes(b"glb")
    )
    exported = builder._export_import_group_glb_cache([mesh_obj, second], str(tmp_path / "source.fbx"), verbose=True)
    assert exported and exported.endswith("source.cgs_group.glb")


def test_mesh_support_faces_world_pack_and_reset_paths(builder, tmp_path, monkeypatch, capsys):
    objects, _scene_collection = _install_geometry_backend(builder)

    mesh = FakeMesh("support_mesh")
    mesh.vertices = [
        types.SimpleNamespace(co=FakeVector((0, 0, 0))),
        types.SimpleNamespace(co=FakeVector((1, 0, 0))),
        types.SimpleNamespace(co=FakeVector((1, 1, 0))),
        types.SimpleNamespace(co=FakeVector((0, 1, 0))),
        types.SimpleNamespace(co=FakeVector((1, 0, 0.01))),
        types.SimpleNamespace(co=FakeVector((2, 0, 0.01))),
        types.SimpleNamespace(co=FakeVector((2, 1, 0.01))),
        types.SimpleNamespace(co=FakeVector((1, 1, 0.01))),
    ]
    mesh.polygons = [
        types.SimpleNamespace(area=1.0, normal=FakeVector((0, 0, 1)), vertices=(0, 1, 2, 3)),
        types.SimpleNamespace(area=1.0, normal=FakeVector((0, 0, 1)), vertices=(4, 5, 6, 7)),
        types.SimpleNamespace(area=1.0, normal=FakeVector((1, 0, 0)), vertices=(0, 1, 4)),
    ]
    obj = FakeObject("SupportMesh")
    obj.data = mesh
    planes = builder._mesh_face_support_plane_candidates(obj)
    assert len(planes) == 1
    assert planes[0]["x_min"] == pytest.approx(0.0)
    assert planes[0]["x_max"] == pytest.approx(2.0)

    missing = types.SimpleNamespace(name="missing", filepath=str(tmp_path / "missing.jpg"), filepath_raw="", source="FILE", packed_file=None)
    generated = types.SimpleNamespace(name="generated", filepath="", filepath_raw="", source="GENERATED", packed_file=None)
    existing_path = tmp_path / "existing.jpg"
    existing_path.write_bytes(b"img")
    existing = types.SimpleNamespace(name="existing", filepath=str(existing_path), filepath_raw="", source="FILE", packed_file=None)
    removed = []

    class ImageStore(list):
        def remove(self, img, do_unlink=True):
            removed.append(img.name)
            if img in self:
                super().remove(img)

    builder.bpy.data.images = ImageStore([missing, generated, existing])
    assert builder._prune_missing_images_before_pack() == 1
    assert removed == ["missing"]

    calls = {"relative": 0, "pack": 0}
    builder.bpy.ops.file.make_paths_relative = lambda: calls.__setitem__("relative", calls["relative"] + 1)
    builder.bpy.ops.file.pack_all = lambda: calls.__setitem__("pack", calls["pack"] + 1)
    builder._pack_assets_best_effort()
    assert calls == {"relative": 1, "pack": 1}

    scene = builder.bpy.context.scene
    scene.cycles = types.SimpleNamespace(samples=999, preview_samples=999)
    builder._configure_fast_render(scene)
    assert scene.render.engine == "CYCLES"
    assert scene.cycles.samples == 64
    scene.eevee = types.SimpleNamespace(taa_render_samples=999, taa_samples=999, use_gtao=False)
    builder._configure_turntable_render(scene)
    assert scene.render.resolution_percentage == 100

    made_collections = {}
    builder.bpy.data.collections = types.SimpleNamespace(
        get=lambda name: made_collections.get(name),
        new=lambda name: made_collections.setdefault(name, types.SimpleNamespace(name=name, objects=FakeCollectionObjects(FakeCollection(name)))),
    )
    coll = builder.ensure_collection("Fresh")
    assert coll.name == "Fresh"
    doomed = FakeObject("Doomed")
    coll.objects.link(doomed)
    builder.bpy.data.objects.append(doomed)
    builder.clear_collection_objects(coll)
    assert doomed not in builder.bpy.data.objects

    builder.reset_scene()
    assert builder.bpy.context.scene.unit_settings.system == "METRIC"


def test_scene_setup_bounds_visibility_and_light_edge_paths(builder, monkeypatch):
    objects, scene_collection = _install_geometry_backend(builder)

    with pytest.raises(SystemExit):
        builder._parse_argv(["blender"])

    builder.bpy.app.build_options = {"BLENDER_EEVEE_NEXT"}
    removable = FakeObject("old_obj")
    objects.append(removable)
    builder.bpy.data.meshes = FakeMeshStore()
    builder.bpy.data.meshes.append(FakeMesh("old_mesh"))
    builder.reset_scene()
    assert builder.bpy.context.scene.render.engine == "BLENDER_EEVEE_NEXT"
    assert removable not in objects

    empty_mesh = FakeObject("empty")
    empty_mesh.data = FakeMesh("empty_mesh")
    objects.append(empty_mesh)
    assert builder._world_bounds_mesh_objects([]) == (FakeVector((0, 0, 0)), FakeVector((0, 0, 0)))
    assert builder._world_bounds_single_mesh_object(empty_mesh) == (FakeVector((0, 0, 0)), FakeVector((0, 0, 0)))

    mesh_one = FakeObject("mesh_one")
    assert builder._filter_imported_mesh_outliers([mesh_one]) == ([mesh_one], [])
    assert builder._drop_material_preview_meshes([mesh_one]) == ([mesh_one], [])
    assert builder._select_single_import_variant_mesh([mesh_one], "chair", aabb()) == ([mesh_one], [])
    assert builder._keep_primary_import_cluster([mesh_one]) == ([mesh_one], [])

    root = FakeObject("family_root", obj_type="EMPTY")
    child = FakeObject("family_child", parent=root)
    objects.extend([root, child])
    builder._remove_object_family(root)
    assert root not in objects and child not in objects
    builder._remove_object_family(None)

    bound_root = FakeObject("BoundRoot", obj_type="EMPTY")
    bound_root.bound_box = [(-1, -2, 0), (2, 3, 4)]
    objects.append(bound_root)
    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", lambda _family: (FakeVector((0, 0, 0)), FakeVector((0, 0, 0))))
    assert builder._aabb_from_blend_object_name("BoundRoot")["x_min"] == -1.0
    assert builder._aabb_from_object_family_root(bound_root)["z_max"] == 4.0
    assert builder._aabb_from_blend_object_name("") is None

    light_root = FakeObject("OldLightRoot", obj_type="LIGHT")
    light_root.data = types.SimpleNamespace(name="light_data", copy=lambda: types.SimpleNamespace(name="light_data_copy"))
    light_root.hide_render = True
    objects.append(light_root)
    linked = []
    scene_collection.objects.link = lambda obj: linked.append(obj)
    restored = builder._restore_hidden_reference_light_fixtures([light_root])
    assert restored >= 1
    assert any(obj.name.endswith("__kept") for obj in linked)

    door = FakeObject("Cube")
    door.dimensions = FakeVector((0.08, 0.9, 2.1))
    assert builder._looks_like_orphan_architectural_door_panel(door)
    objects[:] = [door]
    assert builder._hide_architectural_door_objects() >= 1
    assert door.hide_render

    keep_bbox = FakeObject("debug_AABB")
    keep_bbox["cgs_keep_bbox_fallback"] = True
    drop_bbox = FakeObject("old_AABB")
    objects[:] = [keep_bbox, drop_bbox]
    assert builder._remove_bbox_helper_objects() == 1
    assert keep_bbox in objects and drop_bbox not in objects

    created_lights = []

    class LightStore:
        def new(self, name, light_type):
            data = types.SimpleNamespace(name=name, type=light_type, energy=0.0, size=0.0, shadow_soft_size=0.0)
            created_lights.append(data)
            return data

    class CameraStore:
        def new(self, name):
            return types.SimpleNamespace(name=name, type="PERSP", lens=0, clip_start=0, clip_end=0)

    builder.bpy.data.lights = LightStore()
    builder.bpy.data.cameras = CameraStore()
    builder._add_basic_lights(FakeVector((0, 0, 0)), FakeVector((2, 2, 2)))
    assert [light.type for light in created_lights] == ["SUN", "AREA"]

    kitchen_obj = FakeObject("kitchen_marker")
    kitchen_obj["cgs_procedural_assembly"] = "kitchen"
    objects[:] = [kitchen_obj]
    cam = builder._frame_camera_on_bounds(FakeVector((0, 0, 0)), FakeVector((2, 2, 2)))
    assert builder.bpy.context.scene.camera.name == "Camera"

    assert builder._add_functional_light_for_item({"id": "x", "category": "chair"}, aabb()) == 0
    assert builder._add_functional_light_for_item({"id": "ceil", "category": "CeilingLightFactory"}, aabb(0, 2, 0, 2, 2.8, 3.0)) == 1
    assert builder._add_functional_light_for_item({"id": "ceil", "category": "CeilingLightFactory"}, aabb(0, 2, 0, 2, 2.8, 3.0)) == 0


def test_material_geometry_proxy_uv_and_import_error_branches(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")

    mat1 = builder._make_bbox_wire_material("MAT_SHARED", (0.1, 0.2, 0.3, 1.0))
    mat2 = builder._make_bbox_wire_material("MAT_SHARED", (0.5, 0.4, 0.3, 1.0))
    assert mat1 is mat2

    img = tmp_path / "glass.png"
    img.write_bytes(b"image")
    glass = builder._make_glass_material("GlassWithImage", str(img))
    assert glass.use_nodes is True
    assert any(getattr(node, "image", None) is not None for node in glass.node_tree.nodes)

    chair = builder._make_generated_chair_placeholder(aabb(0, 1, 0, 1, 0, 1.2), "GeneratedChair", coll, yaw_deg=30)
    assert chair is not None
    assert any(obj.name.startswith("GeneratedChair_leg") for obj in coll.objects)

    mesh = FakeMesh("flip_mesh")
    mesh.vertices = [types.SimpleNamespace(co=FakeVector((0, 0, 0))), types.SimpleNamespace(co=FakeVector((0, 0, 2)))]
    flip_obj = FakeObject("FlipObj")
    flip_obj.data = mesh
    builder._flip_mesh_objects_local_z([flip_obj, FakeObject("Empty", obj_type="EMPTY")])
    assert [vertex.co.z for vertex in mesh.vertices] == [2.0, 0.0]

    curtain_root = builder._make_curtain_proxy_mesh(
        item={"id": "curtain", "size_m": [1.0, 0.08, 1.0]},
        aabb=aabb(0, 1, 0, 0.08, 0, 1),
        rotation_deg_engine=0,
        texture_path=None,
        collection=coll,
        name="UV_Curtain",
    )
    cloth = objects.get("UV_Curtain_cloth")
    if cloth not in curtain_root.children:
        curtain_root.children.append(cloth)
    builder._apply_curtain_planar_uv(parent=curtain_root, aabb=aabb(0, 1, 0, 0.08, 0, 1), rotation_deg_engine=0, mirror_repeat=False)
    assert cloth.data.uv_layers.active.name == "ShtorystorePlanarUV"

    bbox_obj = builder._make_renderable_bbox_box(aabb(), "Renderable", coll, thickness=0.01)
    assert bbox_obj["cgs_keep_bbox_fallback"] is True

    curve_store = types.SimpleNamespace(
        new=lambda name, type: types.SimpleNamespace(name=name, type=type, body="", align_x="", size=0, materials=[])
    )
    builder.bpy.data.curves = curve_store
    label = builder._add_aabb_label(aabb(), "ID42", coll)
    assert label is not None and label.name == "ID42_Label"
    assert builder._add_aabb_label(aabb(), "", coll) is None

    with pytest.raises(RuntimeError, match="Unsupported mesh format"):
        builder.import_supported_mesh(str(tmp_path / "model.unsupported"))

    created_before_error = FakeObject("CreatedBeforeError")

    def fail_import(_path):
        objects.append(created_before_error)
        raise RuntimeError("bad import")

    monkeypatch.setattr(builder, "import_supported_mesh", fail_import)
    imported, error = builder._safe_import_supported_mesh(str(tmp_path / "bad.obj"))
    assert imported == []
    assert "RuntimeError" in error
    assert created_before_error not in objects


def test_support_solver_and_obj_import_fallback_paths(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    solver = builder.MLSupportSolver(room_floor_z=0.0)
    assert solver.solve(item_aabb=aabb(), anchor_aabb=aabb(), planes=[], occupied_aabbs=[], mode="near") is None
    assert solver._candidate_centers((2.0, 2.0), {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}, (0.5, 0.5)) == []

    candidate = aabb(0, 1, 0, 1, 0, 1)
    penalty = solver._collision_penalty(candidate, [aabb(0.5, 1.5, 0.5, 1.5, 0.5, 1.5)])
    assert penalty > 1000.0

    assert builder._choose_support_plane(aabb(10, 11, 10, 11, 0, 1), [], mode="top") is None
    fallback_plane = builder._choose_support_plane(
        aabb(0.1, 0.2, 0.1, 0.2, 2.0, 2.1),
        [{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z": 0.5, "area": 1.0, "clearance_height": 10.0}],
        mode="near",
    )
    assert fallback_plane is not None

    root = FakeObject("root", obj_type="EMPTY")
    anchor = FakeObject("anchor", obj_type="EMPTY")
    monkeypatch.setattr(builder, "_extract_support_planes_from_object_family", lambda _root: [])
    assert builder._snap_object_family_to_support_plane(root, aabb(), anchor, mode="top") is None

    imported = FakeObject("LegacyObj")

    class TempOverride:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    builder.bpy.context.mode = "EDIT"
    builder.bpy.context.temp_override = lambda **_kwargs: TempOverride()
    builder.bpy.ops.object.mode_set = lambda mode="OBJECT": setattr(builder.bpy.context, "mode", mode)

    def failing_obj_import(**_kwargs):
        raise RuntimeError("wm obj failed")

    def legacy_obj_import(**_kwargs):
        objects.append(imported)

    builder.bpy.ops.wm.obj_import = failing_obj_import
    builder.bpy.ops.import_scene = types.SimpleNamespace(obj=legacy_obj_import)
    assert builder.import_obj(str(tmp_path / "legacy.obj")) == [imported]

    delattr(builder.bpy.ops.wm, "obj_import")
    second = FakeObject("LegacyObj2")
    builder.bpy.ops.import_scene = types.SimpleNamespace(obj=lambda **_kwargs: objects.append(second))
    assert builder.import_obj(str(tmp_path / "legacy2.obj")) == [second]

    builder.bpy.ops.import_scene = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="No OBJ importer"):
        builder.import_obj(str(tmp_path / "missing_importer.obj"))


def test_build_scene_reference_import_proxy_overlay_and_kitchen_paths(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)

    class LinkedObjects(list):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def link(self, obj):
            if obj not in self:
                self.append(obj)
            if self.owner not in obj.users_collection:
                obj.users_collection.append(self.owner)

        def unlink(self, obj):
            if obj in self:
                self.remove(obj)
            if self.owner in obj.users_collection:
                obj.users_collection.remove(self.owner)

    class LinkedCollection:
        def __init__(self, name):
            self.name = name
            self.objects = LinkedObjects(self)

    collections = {name: LinkedCollection(name) for name in ("Room", "Items", "BBoxOverlay")}
    monkeypatch.setattr(builder, "ensure_collection", lambda name: collections.setdefault(name, LinkedCollection(name)))

    source_chair = FakeObject("SourceChair", obj_type="EMPTY")
    source_keep = FakeObject("SourceKeep", obj_type="EMPTY")
    source_replacement = FakeObject("SourceReplacement", obj_type="EMPTY")
    reference_light = FakeObject("ReferenceCeilingLight", obj_type="LIGHT")
    reference_light.data = types.SimpleNamespace(name="light_data", copy=lambda: types.SimpleNamespace(name="light_data_copy"))
    objects.extend([source_chair, source_keep, source_replacement, reference_light])

    mesh_ok = tmp_path / "ok.glb"
    mesh_fail = tmp_path / "fail.glb"
    mesh_force = tmp_path / "force.glb"
    mesh_curtain = tmp_path / "curtain.glb"
    texture = tmp_path / "curtain.jpg"
    for path in (mesh_ok, mesh_fail, mesh_force, mesh_curtain, texture):
        path.write_bytes(b"asset")

    calls = {
        "clear": 0,
        "hide": 0,
        "show": 0,
        "restore_lights": 0,
        "remove_bbox": 0,
        "floor_overlay": 0,
        "wall_overlay": 0,
        "wall_sync": 0,
        "textures": 0,
        "flip": 0,
        "uv": 0,
        "image_mat": 0,
        "tint_mat": 0,
        "bbox": 0,
        "label": 0,
        "functional": 0,
        "force_preview": 0,
    }

    monkeypatch.setattr(builder, "clear_collection_objects", lambda _coll: calls.__setitem__("clear", calls["clear"] + 1))
    monkeypatch.setattr(builder, "_collect_reference_light_fixture_roots", lambda: [reference_light])
    monkeypatch.setattr(builder, "_hide_object_family", lambda _root: calls.__setitem__("hide", calls["hide"] + 1) or 1)
    monkeypatch.setattr(builder, "_show_object_family", lambda _root: calls.__setitem__("show", calls["show"] + 1) or 1)
    monkeypatch.setattr(builder, "_restore_hidden_reference_light_fixtures", lambda _roots: calls.__setitem__("restore_lights", calls["restore_lights"] + 1) or 1)
    monkeypatch.setattr(builder, "_remove_bbox_helper_objects", lambda: calls.__setitem__("remove_bbox", calls["remove_bbox"] + 1) or 1)
    monkeypatch.setattr(builder, "_infer_reference_floor_z", lambda *_args, **_kwargs: 0.12)

    def overlay_obj(name, box):
        obj = FakeObject(name, obj_type="MESH")
        obj["_aabb"] = dict(box)
        objects.append(obj)
        collections["Room"].objects.link(obj)
        return obj

    monkeypatch.setattr(
        builder,
        "add_supplier_floor_overlay_from_spec",
        lambda *_args, **_kwargs: calls.__setitem__("floor_overlay", calls["floor_overlay"] + 1) or overlay_obj("floor_overlay", aabb(0, 5, 0, 4, 0.12, 0.13)),
    )
    monkeypatch.setattr(
        builder,
        "apply_supplier_wall_material_to_reference_scene",
        lambda *_args, **_kwargs: calls.__setitem__("wall_sync", calls["wall_sync"] + 1) or 2,
    )
    monkeypatch.setattr(
        builder,
        "add_supplier_wall_overlay_from_spec",
        lambda *_args, **_kwargs: calls.__setitem__("wall_overlay", calls["wall_overlay"] + 1) or [overlay_obj("wall_overlay", aabb(0, 0.05, 0, 4, 0, 3))],
    )
    monkeypatch.setattr(builder, "_get_scene_source_object", lambda name: objects.get(name) if name else None)
    monkeypatch.setattr(builder, "_move_object_family_to_target_aabb", lambda root, box, align_bottom=True: root.update({"_aabb": dict(box)}) or True)
    monkeypatch.setattr(builder, "_move_object_family_to_exact_aabb", lambda root, _old, new: root.update({"_aabb": dict(new)}) or dict(new))
    monkeypatch.setattr(builder, "_extract_support_planes_from_object_family", lambda _root: [])
    monkeypatch.setattr(builder, "_infer_support_planes_from_anchor_item", lambda _item, box: [{"x_min": box["x_min"], "x_max": box["x_max"], "y_min": box["y_min"], "y_max": box["y_max"], "z": box["z_max"], "area": 1.0, "clearance_height": 2.0}])
    monkeypatch.setattr(builder, "_add_functional_light_for_item", lambda *_args, **_kwargs: calls.__setitem__("functional", calls["functional"] + 1) or 1)
    monkeypatch.setattr(builder, "_frame_camera_on_bounds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_add_basic_lights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_force_material_preview_if_ui", lambda: calls.__setitem__("force_preview", calls["force_preview"] + 1))

    def fake_aabb_from_root(root):
        if root is None:
            return None
        return root.get("_aabb") or aabb(0, 1, 0, 1, 0, 1)

    monkeypatch.setattr(builder, "_aabb_from_object_family_root", fake_aabb_from_root)
    monkeypatch.setattr(builder, "_aabb_from_blend_object_name", lambda name: aabb(0.25, 0.75, 0.25, 0.75, 0, 1) if name else None)

    def make_root(name, box):
        root = FakeObject(name, obj_type="EMPTY")
        root["_aabb"] = dict(box)
        objects.append(root)
        collections["Items"].objects.link(root)
        return root

    monkeypatch.setattr(builder, "_make_procedural_flat_ceiling_light_mesh", lambda *, item, aabb, collection, name: make_root(name, aabb))
    monkeypatch.setattr(builder, "_make_procedural_requirement_mesh", lambda *, item, aabb, rotation_deg_engine, collection, name: make_root(name, aabb))
    monkeypatch.setattr(builder, "_build_procedural_catalog_proxy_in_aabb", lambda *, item, aabb, rotation_deg_engine, collection, name, fit_mode: make_root(name, aabb))

    created_kitchen_child = FakeObject("kitchen_child")
    fake_kitchen_module = types.ModuleType("src.suppliers.kitchen.kitchen_blender_builder")
    fake_kitchen_module.build_kitchen_assembly_in_blender = lambda *_args, **_kwargs: [created_kitchen_child]
    monkeypatch.setitem(sys.modules, "src.suppliers.kitchen.kitchen_blender_builder", fake_kitchen_module)

    imported_by_path = {}

    def fake_safe_import(path):
        if str(path).endswith("fail.glb"):
            return [], "RuntimeError: failed"
        obj = FakeObject(f"imported_{Path(path).stem}")
        objects.append(obj)
        imported_by_path[str(path)] = obj
        return [obj], None

    def fake_place(objs, aabb, collection, parent_name, semantic_group="", **_kwargs):
        root = make_root(parent_name, aabb)
        root["cgs_placement_score"] = 0.0
        root["cgs_placement_confidence"] = "low" if "low" in parent_name.lower() else "high"
        for obj in objs:
            obj.parent = root
            if obj not in root.children:
                root.children.append(obj)
            collection.objects.link(obj)
        return root

    monkeypatch.setattr(builder, "_discover_mesh_import_candidates", lambda path: [str(path), str(tmp_path / "ignored.txt")])
    monkeypatch.setattr(builder, "_safe_import_supported_mesh", fake_safe_import)
    monkeypatch.setattr(builder, "_remove_or_hide_non_mesh_import_objects", lambda objs: (objs, []))
    monkeypatch.setattr(builder, "place_in_aabb", fake_place)
    monkeypatch.setattr(builder, "_flip_mesh_objects_local_z", lambda _objs: calls.__setitem__("flip", calls["flip"] + 1))
    monkeypatch.setattr(builder, "_apply_curtain_planar_uv", lambda **_kwargs: calls.__setitem__("uv", calls["uv"] + 1))
    monkeypatch.setattr(builder, "_make_image_material", lambda **_kwargs: calls.__setitem__("image_mat", calls["image_mat"] + 1) or types.SimpleNamespace(name="image_mat"))
    monkeypatch.setattr(builder, "_assign_material_to_object", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_make_pbr_material", lambda **_kwargs: calls.__setitem__("tint_mat", calls["tint_mat"] + 1) or types.SimpleNamespace(name=_kwargs.get("name"), diffuse_color=None))
    monkeypatch.setattr(builder, "_ensure_textures", lambda **_kwargs: calls.__setitem__("textures", calls["textures"] + 1))
    monkeypatch.setattr(builder, "_make_renderable_bbox_box", lambda *_args, **_kwargs: calls.__setitem__("bbox", calls["bbox"] + 1) or FakeObject("bbox"))
    monkeypatch.setattr(builder, "_add_aabb_box", lambda *_args, **_kwargs: calls.__setitem__("bbox", calls["bbox"] + 1) or FakeObject("aabb"))
    monkeypatch.setattr(builder, "_add_aabb_label", lambda *_args, **_kwargs: calls.__setitem__("label", calls["label"] + 1) or FakeObject("label"))

    scene = {
        "room": {
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 4}, {"x": 0, "y": 4}],
            "floor_z": 0.0,
            "ceiling_height": 3.0,
        },
        "items": [
            {"id": "flat", "name": "flat ceiling", "category": "ceiling_light", "asset": {"kind": "procedural_flat_ceiling_light"}, "aabb": aabb(1, 2, 1, 2, 2.8, 3.0), "constraints": {"mount_type": "ceiling"}},
            {"id": "kitchen", "name": "kitchen set", "category": "kitchen_set", "asset": {"kind": "procedural_kitchen"}, "aabb": aabb(0, 2, 0, 1, 0, 2)},
            {"id": "ref_keep", "name": "reference plant", "category": "plant", "source": {"blend_object_name": "SourceKeep"}, "aabb": aabb(0.2, 0.7, 3.0, 3.5, 0, 1), "rotation": [0, 0, 45]},
            {"id": "curtain", "name": "curtain", "category": "curtain", "asset": {"kind": "curtain_fbx_textured", "vertical_flip": True, "mesh_path": str(mesh_curtain), "texture_tiling": {"tile_size_m": 0.8, "mode": "repeat"}}, "texture_path": str(texture), "aabb": aabb(0, 2, 0, 0.1, 0, 2.5)},
            {"id": "table_lamp", "name": "desk lamp", "category": "DeskLampFactory", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(1.0, 1.3, 1.0, 1.3, 0.7, 1.2), "meta": {"supplier_binding_applied": True}},
            {"id": "force_low", "name": "force low", "category": "chair", "mesh_path": str(mesh_force), "aabb": aabb(2.1, 2.6, 1.1, 1.6, 0, 1), "meta": {"supplier_binding_applied": True, "supplier_candidate": {"unique_key": "primary", "semantic_group": "chair"}}},
            {"id": "missing_repl", "name": "missing replacement", "category": "wardrobe", "mesh_path": str(mesh_fail), "source": {"blend_object_name": "SourceReplacement", "asset_source": "supplier_catalog_local_asset"}, "aabb": aabb(4, 5.4, 1, 2, 0, 2.2), "meta": {"supplier_binding_applied": True, "supplier_candidate": {"unique_key": "ward", "semantic_group": "wardrobe"}}},
            {"id": "dup_replacement", "name": "replacement chair", "category": "chair", "source": {"blend_object_name": "SourceChair", "asset_source": "supplier_catalog_local_asset"}, "asset": {"kind": "procedural_proxy"}, "aabb": aabb(3, 3.5, 2, 2.5, 0, 1), "meta": {"supplier_binding_applied": True}},
            {"id": "dup_source", "name": "old chair", "category": "chair", "source": {"blend_object_name": "SourceChair"}, "aabb": aabb(3, 3.5, 2, 2.5, 0, 1)},
        ],
    }
    scene_path = tmp_path / "scene.reference.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    report = builder.build_scene(
        str(scene_path),
        draw_aabb=False,
        force_tint=True,
        keep_existing_mats=False,
        verbose=True,
        env_textures_dir=str(tmp_path),
        style_text="modern",
        seed=3,
        reference_blend=str(tmp_path / "reference.blend"),
    )

    assert report["reference_blend"].endswith("reference.blend")
    assert report["hidden_reference_light_fixture_count"] >= 1
    assert report["hidden_skipped_duplicate_reference_count"] >= 1
    assert "dup_source" in report["skipped_duplicate_item_ids"]
    assert calls["floor_overlay"] == 1
    assert calls["wall_overlay"] == 1
    assert calls["wall_sync"] == 1
    assert calls["uv"] == 1
    assert calls["image_mat"] == 1
    assert calls["tint_mat"] >= 1
    assert calls["show"] >= 1
    assert calls["hide"] >= 1
    assert calls["functional"] >= 1
    assert "force_low" in report["item_issues"]
    assert "supplier_proxy_fallback_after_import_failure" in report["item_issues"]["missing_repl"]
    assert imported_by_path[str(mesh_curtain)].parent is not None
    assert created_kitchen_child.get("cgs_procedural_assembly") == "kitchen"

    overlay_scene = {
        "room": scene["room"],
        "placements": [
            {"id": "highlight", "name": "highlight", "category": "chair", "source": {"blend_object_name": "SourceKeep"}, "aabb": aabb(0, 1, 0, 1, 0, 1), "meta": {"placeholder_bbox": True}},
            {"id": "label", "name": "label", "category": "chair", "aabb": aabb(1, 2, 1, 2, 0, 1), "meta": {"placeholder_bbox": True}},
        ],
    }
    overlay_path = tmp_path / "scene.overlay.json"
    overlay_path.write_text(json.dumps(overlay_scene), encoding="utf-8")
    overlay_report = builder.build_scene(
        str(overlay_path),
        draw_aabb=True,
        force_tint=False,
        keep_existing_mats=True,
        verbose=False,
        env_textures_dir=str(tmp_path),
        style_text=None,
        seed=4,
        reference_blend=str(tmp_path / "reference.blend"),
        overlay_bbox_only=True,
        highlight_item_ids={"highlight"},
    )
    assert overlay_report["schema"] == "blender_scene_builder_report/v1"
    assert calls["clear"] >= 1
    assert calls["bbox"] >= 2
    assert calls["label"] >= 1


def test_build_scene_supplier_reject_reference_and_diagnostic_edges(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    collections = {}
    monkeypatch.setattr(builder, "ensure_collection", lambda name: collections.setdefault(name, FakeCollection(name)))
    monkeypatch.setattr(builder, "_find_duplicate_render_item_ids", lambda _items: [])
    monkeypatch.setattr(builder, "build_room_from_spec", lambda **_kwargs: ([], FakeVector((0, 0, 0.5)), FakeVector((2, 2, 2.5))))
    monkeypatch.setattr(builder, "_collect_reference_light_fixture_roots", lambda: [])
    monkeypatch.setattr(builder, "_remove_bbox_helper_objects", lambda: 0)
    monkeypatch.setattr(builder, "_infer_reference_floor_z", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(builder, "add_supplier_floor_overlay_from_spec", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "apply_supplier_wall_material_to_reference_scene", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(builder, "add_supplier_wall_overlay_from_spec", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(builder, "_frame_camera_on_bounds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_add_basic_lights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "_force_material_preview_if_ui", lambda: None)
    monkeypatch.setattr(builder, "_add_functional_light_for_item", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(builder, "_duplicate_light_objects_from_family", lambda _root: 1)

    source_import = FakeObject("SourceImport", obj_type="EMPTY")
    source_import["_aabb"] = aabb(0, 0.5, 0, 0.5, 0.5, 1.0)
    source_ref = FakeObject("SourceRef", obj_type="EMPTY")
    source_ref["_aabb"] = aabb(0.5, 1.0, 0, 0.5, 0.5, 1.0)
    source_disabled = FakeObject("SourceDisabled", obj_type="EMPTY")
    source_disabled["_aabb"] = aabb(1.0, 1.5, 0, 0.5, 0.5, 1.0)
    source_move = FakeObject("SourceMove", obj_type="EMPTY")
    source_move["_aabb"] = aabb(1.5, 1.9, 0, 0.4, 0.5, 1.0)
    objects.extend([source_import, source_ref, source_disabled, source_move])
    monkeypatch.setattr(builder, "_get_scene_source_object", lambda name: objects.get(name) if name else None)

    mesh_dir = tmp_path / "meshes"
    tex_dir = tmp_path / "textures"
    mesh_dir.mkdir()
    tex_dir.mkdir()
    paths = {}
    for name in ("reject.fbx", "empty.fbx", "good.fbx", "bad_place.fbx", "fail_ref.fbx", "fail_disabled.fbx"):
        path = mesh_dir / name
        path.write_bytes(b"mesh")
        paths[name] = path

    calls = {"hide": 0, "show": 0, "remove": 0, "bbox": 0, "label": 0, "moves": 0, "mat": 0}
    monkeypatch.setattr(builder, "_hide_object_family", lambda _root: calls.__setitem__("hide", calls["hide"] + 1) or 1)
    monkeypatch.setattr(builder, "_show_object_family", lambda _root: calls.__setitem__("show", calls["show"] + 1) or 1)
    monkeypatch.setattr(builder, "_remove_object_family", lambda _root: calls.__setitem__("remove", calls["remove"] + 1))
    monkeypatch.setattr(builder, "_make_renderable_bbox_box", lambda *_args, **_kwargs: calls.__setitem__("bbox", calls["bbox"] + 1) or FakeObject("invalid_bbox"))
    monkeypatch.setattr(builder, "_add_aabb_box", lambda *_args, **_kwargs: calls.__setitem__("bbox", calls["bbox"] + 1) or FakeObject("bbox"))
    monkeypatch.setattr(builder, "_add_aabb_label", lambda *_args, **_kwargs: calls.__setitem__("label", calls["label"] + 1) or FakeObject("label"))
    monkeypatch.setattr(builder, "_discover_mesh_import_candidates", lambda path: [str(path), str(tmp_path / "skip.txt")])
    monkeypatch.setattr(builder, "_remove_or_hide_non_mesh_import_objects", lambda objs: (objs, [{"name": o.name, "type": o.type} for o in objs if o.type != "MESH"]))
    monkeypatch.setattr(builder, "_export_import_group_glb_cache", lambda *_args, **_kwargs: str(tmp_path / "group_cache.glb"))
    monkeypatch.setattr(builder, "_supplier_candidate_reuse_reject_reason", lambda _item, cand, _index: "reused_supplier_asset_for_different_target:key" if cand.get("unique_key") == "primary" else None)
    monkeypatch.setattr(builder, "_make_pbr_material", lambda **_kwargs: calls.__setitem__("mat", calls["mat"] + 1) or types.SimpleNamespace(name=_kwargs.get("name")))
    monkeypatch.setattr(builder, "_ensure_textures", lambda **_kwargs: None)

    def fake_import(path):
        stem = Path(path).stem
        if "fail" in stem:
            return [], "RuntimeError: import failed"
        if "empty" in stem:
            obj = FakeObject(f"empty_{stem}", obj_type="EMPTY")
            objects.append(obj)
            return [obj], None
        obj = FakeObject(f"mesh_{stem}", obj_type="MESH")
        obj.data.materials = [types.SimpleNamespace(name="old_material")]
        objects.append(obj)
        return [obj], None

    monkeypatch.setattr(builder, "_safe_import_supported_mesh", fake_import)

    def make_root(name, box, *, no_actual=False):
        root = FakeObject(name, obj_type="EMPTY")
        if not no_actual:
            root["_aabb"] = dict(box)
        objects.append(root)
        collections.setdefault("Items", FakeCollection("Items")).objects.link(root)
        return root

    def fake_proxy(*, item, aabb, collection, name, **_kwargs):
        if item.get("id") == "repl_disabled":
            return None
        return make_root(name, aabb, no_actual=item.get("id") == "support_no_actual")

    def fake_place(objs, aabb, collection, parent_name, **_kwargs):
        if "bad place" in parent_name.lower():
            return None
        root = make_root(parent_name, aabb)
        root["cgs_placement_score"] = 2.0 if "good imported" in parent_name.lower() else 0.1
        root["cgs_placement_confidence"] = "low" if "good imported" in parent_name.lower() else "high"
        for obj in objs:
            obj.parent = root
            if obj not in root.children:
                root.children.append(obj)
            collection.objects.link(obj)
        return root

    monkeypatch.setattr(builder, "_build_procedural_catalog_proxy_in_aabb", fake_proxy)
    monkeypatch.setattr(builder, "place_in_aabb", fake_place)

    def fake_aabb_from_root(root):
        if getattr(root, "name", "") == "support without actual":
            return None
        return root.get("_aabb")

    monkeypatch.setattr(builder, "_aabb_from_object_family_root", fake_aabb_from_root)
    monkeypatch.setattr(builder, "_extract_support_planes_from_object_family", lambda _root: [])
    monkeypatch.setattr(
        builder,
        "_infer_support_planes_from_anchor_item",
        lambda _item, box: [{"x_min": box["x_min"], "x_max": box["x_max"], "y_min": box["y_min"], "y_max": box["y_max"], "z": box["z_max"], "area": 1.0, "clearance_height": 2.0}],
    )

    class FakeSupportSolver:
        def __init__(self, room_floor_z):
            self.room_floor_z = room_floor_z

        def solve(self, item_aabb, anchor_aabb, planes, occupied_aabbs, mode):
            solved = dict(item_aabb)
            dz = float(anchor_aabb["z_max"]) - float(item_aabb["z_min"])
            solved["z_min"] += dz
            solved["z_max"] += dz
            return solved

    monkeypatch.setattr(builder, "MLSupportSolver", FakeSupportSolver)

    def fake_move_exact(root, _old, new):
        if getattr(root, "name", "") == "support no move":
            return None
        root["_aabb"] = dict(new)
        calls["moves"] += 1
        return dict(new)

    monkeypatch.setattr(builder, "_move_object_family_to_exact_aabb", fake_move_exact)
    monkeypatch.setattr(builder, "_move_object_family_to_target_aabb", lambda root, box, **_kwargs: root.update({"_aabb": dict(box)}) or True)

    original_is_replacement = builder._is_replacement_render_item
    monkeypatch.setattr(
        builder,
        "_is_replacement_render_item",
        lambda item: False if item.get("id") == "ref_fallback" else original_is_replacement(item),
    )

    scene = {
        "room": {"bounds": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2, "z_min": 0.5, "z_max": 2.5}},
        "items": [
            {"id": "missing_support", "name": "missing support root", "meta": {"supplier_support_reanchored": True, "supplier_support_anchor_target_id": "anchor"}},
            {"id": "no_aabb", "name": "no aabb item"},
            {"id": "floor_low", "name": "floor low", "category": "chair", "rotation": [45], "asset": {"kind": "procedural_proxy"}, "aabb": aabb(0.1, 0.4, 0.1, 0.4, -0.2, 0.3)},
            {"id": "empty_rot", "name": "empty rotation", "category": "decor_vase", "rotation": [], "asset": {"kind": "procedural_proxy"}, "aabb": aabb(0.4, 0.7, 0.1, 0.4, 0.5, 0.8)},
            {
                "id": "good_imported",
                "name": "good imported",
                "category": "chair",
                "mesh_fit_mode": "uniform",
                "mesh_texture_dirs": [str(tex_dir)],
                "source": {"blend_object_name": "SourceImport"},
                "aabb": aabb(0.2, 0.7, 0.8, 1.3, 0.5, 1.2),
                "meta": {
                    "supplier_binding_applied": True,
                    "preserve_imported_group": True,
                    "supplier_candidate": {"unique_key": "primary", "semantic_group": "chair", "asset_local_path": str(paths["reject.fbx"]), "dimensions_cm": {"width": 50, "depth": 50, "height": 80}},
                    "supplier_candidate_pool": [
                        {"unique_key": "empty", "semantic_group": "chair", "asset_local_path": str(paths["empty.fbx"])},
                        {"unique_key": "alt", "semantic_group": "chair", "asset_local_path": str(paths["good.fbx"]), "dimensions_cm": {"width": 60, "depth": 55, "height": 85}},
                    ],
                },
            },
            {"id": "bad_place", "name": "bad place", "category": "chair", "mesh_path": str(paths["bad_place.fbx"]), "aabb": aabb(1.1, 1.5, 0.8, 1.2, 0.5, 1.1), "meta": {"supplier_binding_applied": True, "supplier_candidate": {"unique_key": "bad", "semantic_group": "chair"}}},
            {"id": "ref_fallback", "name": "reference fallback", "category": "custom_reference", "mesh_path": str(paths["fail_ref.fbx"]), "source": {"blend_object_name": "SourceRef"}, "aabb": aabb(0.6, 1.0, 0.2, 0.6, 0.5, 1.1), "meta": {"supplier_binding_applied": True, "supplier_candidate": {"unique_key": "ref", "semantic_group": "chair"}}},
            {"id": "repl_disabled", "name": "disabled fallback", "category": "wardrobe", "mesh_path": str(paths["fail_disabled.fbx"]), "source": {"blend_object_name": "SourceDisabled", "asset_source": "supplier_catalog_local_asset"}, "aabb": aabb(1.0, 1.5, 0.2, 0.7, 0.5, 1.4), "meta": {"supplier_binding_applied": True, "supplier_candidate": {"unique_key": "disabled", "semantic_group": "wardrobe"}}},
            {"id": "source_move", "name": "source move", "category": "custom_reference", "source": {"blend_object_name": "SourceMove"}, "aabb": aabb(1.5, 1.9, 0.2, 0.6, 0.5, 1.0), "meta": {"supplier_support_reanchored": True, "supplier_support_anchor_target_id": "anchor"}},
            {"id": "chair_overlap", "name": "chair overlap", "category": "chair", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(0.2, 0.8, 1.4, 1.9, 0.5, 1.0), "meta": {"supplier_binding_applied": True, "affordance": "table_chair", "target_table_id": "anchor"}},
            {"id": "anchor", "name": "anchor table", "category": "table", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(0.2, 0.8, 1.4, 1.9, 0.5, 1.0), "meta": {"supplier_binding_applied": True}},
            {"id": "support_ok", "name": "support ok", "category": "decor_vase", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(0.25, 0.35, 1.45, 1.55, 0.5, 0.7), "meta": {"supplier_support_reanchored": True, "supplier_support_anchor_target_id": "anchor", "supplier_support_mode": "top"}},
            {"id": "support_no_move", "name": "support no move", "category": "decor_vase", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(0.3, 0.4, 1.5, 1.6, 0.5, 0.7), "meta": {"supplier_support_reanchored": True, "supplier_support_anchor_target_id": "anchor", "supplier_support_mode": "top"}},
            {"id": "support_no_actual", "name": "support without actual", "category": "decor_vase", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(0.35, 0.45, 1.55, 1.65, 0.5, 0.7), "meta": {"supplier_support_reanchored": True, "supplier_support_anchor_target_id": "anchor", "supplier_support_mode": "top"}},
            {"id": "diag_oob", "name": "diag out of bounds", "category": "chair", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(-0.5, 0.1, 0.0, 0.4, 0.5, 1.0), "meta": {"supplier_binding_applied": True, "placeholder_bbox": True}},
            {"id": "diag_a", "name": "diag a", "category": "chair", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(1.2, 1.8, 1.2, 1.8, 0.5, 1.0), "meta": {"supplier_binding_applied": True}},
            {"id": "diag_b", "name": "diag b", "category": "chair", "asset": {"kind": "procedural_proxy"}, "aabb": aabb(1.4, 1.9, 1.4, 1.9, 0.5, 1.0), "meta": {"supplier_binding_applied": True}},
        ],
    }
    scene_path = tmp_path / "scene.edge.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    report = builder.build_scene(
        str(scene_path),
        draw_aabb=True,
        force_tint=True,
        keep_existing_mats=False,
        verbose=True,
        env_textures_dir=str(tmp_path),
        style_text=None,
        seed=7,
        reference_blend=str(tmp_path / "reference.blend"),
        bbox_fallback_missing_mesh=True,
    )

    assert report["removed_non_mesh_import_objects"]
    assert report["rejected_supplier_candidates"]
    assert "low_confidence_replacement" in report["item_issues"]["good_imported"]
    assert "supplier_reference_fallback_disabled" in report["item_issues"]["repl_disabled"]
    assert "out_of_bounds" in report["item_issues"]["diag_oob"]
    assert "collision:diag_b" in report["item_issues"]["diag_a"]
    assert calls["hide"] >= 2
    assert calls["show"] >= 2
    assert calls["moves"] >= 1
    assert calls["mat"] >= 1


def test_actual_room_mesh_material_overlay_and_cleanup_edges(builder, tmp_path, monkeypatch, capsys):
    objects, _scene_collection = _install_geometry_backend(builder)
    _install_fake_bmesh_backend(builder)
    coll = FakeCollection("RoomActual")

    floor_tex = tmp_path / "floor_supplier.jpg"
    wall_tex = tmp_path / "wall_supplier.jpg"
    door_tex = tmp_path / "door_oak.jpg"
    for path in (floor_tex, wall_tex, door_tex):
        path.write_bytes(b"img")

    room = {
        "room": {
            "type": "bathroom",
            "floor_z": 0.2,
            "z_max": 3.0,
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 0, "y": 3}, {"x": 4, "y": 3}, {"x": 4, "y": 0}],
            "doors": [
                {"id": "bad_door", "wall_id": "missing", "s": 0.1, "width": 0.8, "height": 2.0},
                {"id": "narrow_door", "wall_id": "w0", "s": 0.2, "width": 0.01, "height": 2.0},
                {"id": "entry", "wall_id": "w0", "s": 0.5, "width": 0.8, "height": 2.0},
            ],
            "windows": [
                {"id": "bad_window", "wall_id": "missing", "s": 0.5, "width": 1.0, "sill_height": 0.9, "height": 1.0},
                {"id": "slit", "wall_id": "w1", "s": 0.4, "width": 0.01, "sill_height": 0.9, "height": 1.0},
                {"id": "main", "wall_id": "w1", "s": 0.4, "width": 1.0, "sill_height": 0.9, "height": 1.0},
            ],
        }
    }

    objs, bmin, bmax = builder.build_room_from_spec(room, coll, "", None, 0, verbose=True)
    assert bmin.z == pytest.approx(0.2)
    assert bmax.z == pytest.approx(3.0)
    assert any(obj.name == "Room_Floor" for obj in objs)
    assert any(obj.name.startswith("Room_Door_entry") for obj in objs)
    assert any(obj.name.startswith("Room_Window_main") for obj in objs)
    assert any(name.startswith("MAT_ROOM_FLOOR_SANITARY") for name in builder.bpy.data.materials.created)
    assert "walls missing -> synthesized" in capsys.readouterr().out

    supplier_room = {
        "room": {
            "floor_z": 0.0,
            "ceiling_height": 2.7,
            "floor_polygon": [{"x": 0, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 2}, {"x": 0, "y": 2}],
            "floor_material": {"texture_path": str(floor_tex), "texture_tiling": {"tile_size_m": 0.75, "mode": "mirrored"}},
            "wall_material": {"texture_path": str(wall_tex), "wall_tiling": {"tile_size_m": 0.5}},
            "doors": [{"id": "entry", "wall_id": "w0", "s": 0.2, "width": 0.7, "height": 2.0}],
            "windows": [{"id": "main", "wall_id": "w2", "s": 0.4, "width": 0.8, "sill_height": 0.8, "height": 1.0}],
        }
    }
    floor_overlay = builder.add_supplier_floor_overlay_from_spec(supplier_room, coll, floor_z_override=0.3, verbose=True)
    wall_overlays = builder.add_supplier_wall_overlay_from_spec(supplier_room, coll, floor_z_override=0.3, verbose=True)
    assert floor_overlay is not None
    assert wall_overlays

    wall_obj = FakeObject("reference_wall_mesh")
    window_obj = FakeObject("reference_window_wall_mesh")
    empty_obj = FakeObject("not_a_mesh", obj_type="EMPTY")
    objects.extend([wall_obj, window_obj, empty_obj])
    assert builder.apply_supplier_wall_material_to_reference_scene(supplier_room, verbose=True) >= 1
    assert wall_obj.data.materials
    assert not window_obj.data.materials

    assert builder.add_supplier_floor_overlay_from_spec({"room": {}}, coll) is None
    assert builder.add_supplier_floor_overlay_from_spec({"room": {"floor_material": {"texture_path": str(floor_tex)}}}, coll) is None
    assert builder.add_supplier_wall_overlay_from_spec({"room": {"wall_material": {"texture_path": str(wall_tex)}}}, coll) == []
    assert builder.apply_supplier_wall_material_to_reference_scene({"room": {"wall_material": {}}}) == 0

    mat = builder._make_procedural_tile_material(
        "MAT_TEST_TILE",
        tile_rgb=(0.1, 0.2, 0.3),
        tile_alt_rgb=(0.2, 0.3, 0.4),
        grout_rgb=(0.01, 0.02, 0.03),
        brick_width=0.5,
        row_height=0.25,
        mortar_size=0.02,
    )
    assert mat.diffuse_color == (0.1, 0.2, 0.3, 1.0)
    plain_node = types.SimpleNamespace(inputs={})
    builder._set_node_input(plain_node, "missing", 1.0)

    emitter = FakeObject("CGS_FunctionalLight_lamp_Emitter")
    functional = FakeObject("CGS_FunctionalLight_lamp")
    functional["cgs_functional_light"] = True
    objects.extend([emitter, functional])
    monkeypatch.setattr(builder, "_replace_missing_texture_materials", lambda objects=None: 0)
    cleanup = builder._cleanup_final_visual_helpers()
    assert cleanup["removed_light_markers"] >= 1
    assert cleanup["hidden_functional_lights"] >= 1

    class BrokenLights:
        def __init__(self):
            self.calls = 0

        def new(self, name, light_type):
            self.calls += 1
            if light_type == "AREA":
                raise RuntimeError("area failed")
            return types.SimpleNamespace(name=name, type=light_type, energy=0.0, size=0.0)

    builder.bpy.data.lights = BrokenLights()
    builder._add_basic_lights(FakeVector((0, 0, 0)), FakeVector((1, 1, 1)))


def test_remaining_builder_bounds_filters_and_source_edges(builder, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Edges")

    assert builder._median([]) == 0.0
    assert builder._world_bounds_single_mesh_object(FakeObject("empty", obj_type="EMPTY")) == (
        FakeVector((0.0, 0.0, 0.0)),
        FakeVector((0.0, 0.0, 0.0)),
    )
    assert builder._world_bounds_mesh_objects([FakeObject("empty", obj_type="EMPTY")]) == (
        FakeVector((0.0, 0.0, 0.0)),
        FakeVector((0.0, 0.0, 0.0)),
    )

    sizes = {
        "zero": (0.0, 0.0, 0.0),
        "main_a": (1.0, 1.0, 1.0),
        "main_b": (1.1, 1.0, 0.9),
        "helper": (20.0, 20.0, 0.01),
        "mirror 600x900": (0.6, 0.04, 0.9),
        "mirror 1200x600": (1.2, 0.04, 0.6),
        "tub_good": (1.7, 0.75, 0.55),
        "tub_bad": (0.4, 0.4, 0.4),
    }

    def fake_bounds(obj):
        sx, sy, sz = sizes[getattr(obj, "name", "main_a")]
        return FakeVector((0, 0, 0)), FakeVector((sx, sy, sz))

    monkeypatch.setattr(builder, "_world_bounds_single_mesh_object", fake_bounds)
    outlier_objs = [FakeObject(name) for name in ("zero", "main_a", "main_b", "helper")]
    filtered, dropped = builder._filter_imported_mesh_outliers(outlier_objs)
    assert [obj.name for obj in filtered] == ["main_a", "main_b"]
    assert dropped and dropped[0]["name"] == "helper"

    preview_objs = [FakeObject("MAT_preview"), FakeObject("RealMesh"), FakeObject("Empty", obj_type="EMPTY")]
    filtered_preview, dropped_preview = builder._drop_material_preview_meshes(preview_objs)
    assert [obj.name for obj in filtered_preview] == ["RealMesh", "Empty"]
    assert dropped_preview == ["MAT_preview"]
    assert builder._drop_material_preview_meshes([FakeObject("MAT_only"), FakeObject("swatch_chip")])[1] == []

    mirror_objs = [FakeObject("mirror 1200x600"), FakeObject("mirror 600x900")]
    selected_mirror, dropped_mirror = builder._select_single_import_variant_mesh(mirror_objs, "mirror", aabb(0, 0.6, 0, 0.05, 0, 0.9))
    assert [obj.name for obj in selected_mirror] == ["mirror 600x900"]
    assert dropped_mirror == ["mirror 1200x600"]
    tub_objs = [FakeObject("tub_bad"), FakeObject("tub_good")]
    selected_tub, dropped_tub = builder._select_single_import_variant_mesh(tub_objs, "bathtub", aabb(0, 1.7, 0, 0.75, 0, 0.55))
    assert [obj.name for obj in selected_tub] == ["tub_good"]
    assert dropped_tub == ["tub_bad"]

    child = FakeObject("child")
    root = FakeObject("root")
    root.children.append(child)
    builder._translate_object_family(root, FakeVector((1, 2, 3)))
    assert root.location == FakeVector((1, 2, 3))
    assert child.location == FakeVector((1, 2, 3))

    assert builder._aabb_from_blend_object_name("") is None
    assert builder._aabb_from_blend_object_name("missing") is None
    bbox_obj = FakeObject("bbox_source", obj_type="EMPTY")
    bbox_obj.bound_box = [(0, 0, 0), (1, 2, 3)]
    objects.append(bbox_obj)
    assert builder._aabb_from_blend_object_name("bbox_source") == aabb(0, 1, 0, 2, 0, 3)
    assert builder._aabb_from_object_family_root(bbox_obj) == aabb(0, 1, 0, 2, 0, 3)

    source = FakeObject("Source.001")
    base = FakeObject("BaseName.004")
    objects.extend([source, base])
    assert builder._get_scene_source_object("") is None
    assert builder._get_scene_source_object("Source") is source
    assert builder._get_scene_source_object("BaseName.999") is base

    removable = FakeObject("removable", collections=[coll])
    coll.objects.link(removable)
    builder._unlink_from_all_collections(removable)
    assert removable.users_collection == []


def test_remaining_builder_image_material_and_texture_edges(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    mat_store = builder.bpy.data.materials
    real_image = tmp_path / "real_basecolor.png"
    real_image.write_bytes(b"img")

    assert builder._supplier_candidate_tint_rgb({"color": [0.1, 0.2, 0.3]}, "fallback") == (0.1, 0.2, 0.3)
    assert builder._supplier_candidate_tint_rgb({"color": ["bad", 0, 0], "asset": {"color": "green"}}, "fallback") == (0.33, 0.55, 0.36)
    assert builder._should_apply_tint_rgb(None) is False
    assert builder._should_apply_tint_rgb(("bad", 0, 0)) is False
    assert builder._should_apply_tint_rgb((0.2, 0.2, 0.2)) is True
    assert builder._blend_rgba((0.0, 0.0, 0.0, 0.5), (1.0, 0.5, 0.0), 0.25) == pytest.approx((0.25, 0.125, 0.0, 0.5))

    img = types.SimpleNamespace(
        name="Map #1",
        filepath="",
        filepath_raw="",
        packed_file=None,
        source="FILE",
        size=(0, 0),
        has_data=False,
        colorspace_settings=types.SimpleNamespace(name=""),
    )
    assert builder._is_placeholder_image(None)
    assert builder._is_placeholder_image(img)
    img.name = "real"
    img.filepath = str(real_image)
    assert builder._image_has_real_pixels(img)
    assert not builder._image_is_missing_file(img)
    normal_image = tmp_path / "normal_map.png"
    normal_image.write_bytes(b"img")
    img.filepath = str(normal_image)
    assert builder._image_looks_non_color_texture(img)
    generated = types.SimpleNamespace(name="generated", filepath="", filepath_raw="", packed_file=None, source="GENERATED", size=(8, 8), has_data=False)
    assert builder._image_has_real_pixels(generated)
    missing = types.SimpleNamespace(name="missing", filepath=str(tmp_path / "missing.png"), filepath_raw="", packed_file=None, source="FILE", size=(0, 0), has_data=False)
    assert builder._image_is_missing_file(missing)

    assert not builder._socket_chain_has_real_image(None)
    assert not builder._socket_chain_has_real_color_image(None)
    image_node = FakeNode("ShaderNodeTexImage")
    image_node.image = img
    image_node.inputs = []
    base_socket = FakeNodeSocket("Base Color")
    link = types.SimpleNamespace(from_node=image_node)
    base_socket.links = [link]
    base_socket.is_linked = True
    assert builder._socket_chain_has_real_image(base_socket)
    assert not builder._socket_chain_has_real_color_image(base_socket)
    img.filepath = str(real_image)
    assert builder._socket_chain_has_real_color_image(base_socket)

    mat = mat_store.new("MAT_WITH_IMAGE")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].links = [link]
    bsdf.inputs["Base Color"].is_linked = True
    assert builder._material_has_effective_basecolor_texture(mat)
    image_node.image = missing
    assert builder._material_has_missing_basecolor_texture(mat)
    magenta = types.SimpleNamespace(diffuse_color=(1.0, 0.1, 1.0, 1.0))
    assert builder._material_looks_magenta_missing(magenta)

    root = FakeObject("root")
    mesh_child = FakeObject("mesh_child", parent=root)
    mesh_child.data.materials = [mat]
    assert builder._object_has_any_material_slots(root)
    changed = builder._replace_missing_texture_materials([root])
    assert changed == 1
    assert mesh_child.data.materials[0].name == "MAT_CGS_MISSING_TEXTURE_NEUTRAL_GRAY"

    assert builder._as_float_or_none("") is None
    assert builder._as_float_or_none("bad") is None
    assert builder._floor_material_texture_info({}) == (None, 1.0, False)
    assert builder._floor_material_texture_info({"floor_material": {"texture_path": ""}}) == (None, 1.0, False)
    assert builder._floor_material_texture_info({"floor_material": {"texture_path": str(tmp_path / "missing.jpg")}}) == (None, 1.0, False)
    floor_info = builder._floor_material_texture_info({"floor_material": {"texture_path": str(real_image), "texture_tiling": {"tile_size_m": "0.1", "mode": "mirror"}}})
    assert floor_info == (str(real_image), 4.0, True)
    assert builder._make_supplier_floor_material_for_room({"floor_material": {"texture_path": ""}}) == (None, 1.0, False)
    assert builder._wall_material_texture_info({}) == (None, 1.0)
    assert builder._wall_material_texture_info({"wall_material": {"texture_path": "https://example.test/wall.jpg"}}) == (None, 1.0)
    wall_info = builder._wall_material_texture_info({"wall_material": {"texture_path": str(real_image), "wall_tiling": {"tile_size_m": "9"}}})
    assert wall_info == (str(real_image), pytest.approx(1.0 / 3.0))

    non_mesh_obj = FakeObject("non_mesh", obj_type="EMPTY")
    builder._assign_material_to_object(non_mesh_obj, mat)
    obj_with_empty_materials = FakeObject("assign")
    obj_with_empty_materials.data.materials = []
    builder._assign_material_to_object(obj_with_empty_materials, mat)
    assert obj_with_empty_materials.data.materials == [mat]

    assert builder._procedural_requirement_role({"category": "sink"}) == "sink"
    assert builder._procedural_requirement_role({"name": "душевая зона"}) == "shower"
    assert builder._procedural_requirement_role({"name": "ванна"}) == "bath"
    assert builder._procedural_requirement_role({"name": "стол"}) == "table"
    assert builder._procedural_requirement_role({"name": "unknown"}) == ""

    assert builder._make_emissive_marker_material().name == "MAT_CGS_FUNCTIONAL_LIGHT_MARKER"
    assert builder._make_emissive_marker_material().name == "MAT_CGS_FUNCTIONAL_LIGHT_MARKER"


def test_builder_defensive_edge_branches_without_blender_runtime(builder, tmp_path, monkeypatch, capsys):
    objects, scene_collection = _install_geometry_backend(builder)

    class RaiseOnSet:
        def __setattr__(self, _name, _value):
            raise RuntimeError("set failed")

    class RaisingObjects(FakeObjects):
        def remove(self, obj, do_unlink=True):
            raise RuntimeError("remove failed")

    old_scene = builder.bpy.context.scene
    old_objects = builder.bpy.data.objects
    old_meshes = builder.bpy.data.meshes
    builder.bpy.context.scene.unit_settings = RaiseOnSet()
    builder.bpy.context.scene.render = RaiseOnSet()
    builder.bpy.app.build_options = {"BLENDER_EEVEE_NEXT"}
    builder.bpy.data.objects = RaisingObjects([FakeObject("bad_remove")])
    builder.bpy.data.meshes = RaisingObjects([FakeMesh("bad_mesh_remove")])
    builder.reset_scene()
    builder.bpy.context.scene = old_scene
    builder.bpy.data.objects = old_objects
    builder.bpy.data.meshes = old_meshes

    doomed = FakeObject("doomed")
    coll = FakeCollection("BrokenCollection")
    coll.objects.link(doomed)
    builder.bpy.data.objects = RaisingObjects([doomed])
    builder.clear_collection_objects(coll)
    assert doomed in builder.bpy.data.objects
    builder.bpy.data.objects = objects

    bad_coll = types.SimpleNamespace(objects=types.SimpleNamespace(unlink=lambda _obj: (_ for _ in ()).throw(RuntimeError("unlink"))))
    linked = FakeObject("linked", collections=[bad_coll])
    builder._unlink_from_all_collections(linked)

    class NoMeshObject(FakeObject):
        def to_mesh(self):
            return None

    no_mesh = NoMeshObject("no_mesh")
    assert builder._world_bounds_mesh_objects([no_mesh]) == (FakeVector((0, 0, 0)), FakeVector((0, 0, 0)))
    assert builder._world_bounds_single_mesh_object(no_mesh) == (FakeVector((0, 0, 0)), FakeVector((0, 0, 0)))

    monkeypatch.setattr(builder, "_world_bounds_single_mesh_object", lambda _obj: (FakeVector((0, 0, 0)), FakeVector((1, 1, 1))))
    same_size = [FakeObject("main1"), FakeObject("main2"), FakeObject("main3")]
    assert builder._filter_imported_mesh_outliers(same_size) == (same_size, [])
    non_variant_objs = [FakeObject("a"), FakeObject("b")]
    assert builder._select_single_import_variant_mesh(non_variant_objs, "chair", aabb()) == (non_variant_objs, [])

    parent = FakeObject("parent", obj_type="EMPTY")
    child = FakeObject("child", parent=parent)
    builder.bpy.data.objects[:] = [parent, child]
    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", lambda _family: (FakeVector((0, 0, 0)), FakeVector((0, 0, 0))))
    assert builder._move_object_family_to_target_aabb(parent, aabb()) is False

    class BadLocationObject(FakeObject):
        def __setattr__(self, name, value):
            if name == "location" and hasattr(self, "location"):
                raise RuntimeError("location failed")
            super().__setattr__(name, value)

    bad_loc = BadLocationObject("bad_loc")
    builder._translate_object_family(bad_loc, FakeVector((1, 2, 3)))

    class BrokenReference:
        @property
        def name(self):
            raise ReferenceError("gone")

    assert builder._restore_hidden_reference_light_fixtures([BrokenReference()]) == 0
    missing_root = FakeObject("missing_ref", obj_type="LIGHT")
    assert builder._restore_hidden_reference_light_fixtures([missing_root]) == 0
    hidden_root = FakeObject("hidden_ref", obj_type="LIGHT")
    hidden_root.hide_render = True
    builder.bpy.data.objects[:] = [hidden_root]
    monkeypatch.setattr(builder, "_duplicate_light_objects_from_family", lambda _root: (_ for _ in ()).throw(ReferenceError("gone")))
    assert builder._restore_hidden_reference_light_fixtures([hidden_root]) == 0

    class HideRaises(FakeObject):
        def hide_get(self):
            raise RuntimeError("hide failed")

    assert builder._object_or_parent_hidden(HideRaises("hide_raises")) is False

    class ParentRaises:
        hide_render = False

        def hide_get(self):
            return False

        @property
        def parent(self):
            raise ReferenceError("parent gone")

    assert builder._object_or_parent_hidden(ParentRaises()) is False

    empty_semantic = {"category": "unknown"}
    assert builder._infer_support_planes_from_anchor_item(empty_semantic, aabb()) == []
    assert builder._infer_support_planes_from_anchor_item({"category": "cabinet"}, aabb(0, 1, 0, 1, 0, 0.3))
    assert builder._choose_support_plane(aabb(5, 6, 5, 6, 0, 1), [{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z": 0.5, "area": 1.0}], mode="top") is None
    monkeypatch.setattr(builder, "_extract_support_planes_from_object_family", lambda _root: [{"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z": 0.5, "area": 1.0}])
    assert builder._snap_object_family_to_support_plane(parent, aabb(5, 6, 5, 6, 0, 1), parent, mode="top") is None
    assert builder._snap_object_family_to_support_plane(parent, aabb(0, 0.5, 0, 0.5, 0.504, 0.8), parent, mode="top")["z_min"] == pytest.approx(0.504)

    solver = builder.MLSupportSolver(room_floor_z=1.0)
    tight_plane = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z": 0.2, "area": 1.0, "clearance_height": 0.1}
    assert solver.solve(item_aabb=aabb(0, 0.2, 0, 0.2, 0, 0.4), anchor_aabb=aabb(), planes=[tight_plane], occupied_aabbs=[], mode="top") is None

    assert builder._parse_id_set(None) == set()
    builder._store_scene_room_bounds(FakeVector((0, 0, 0)), FakeVector((1, 1, 1)))
    assert builder._scene_room_bounds() is not None
    builder.bpy.context.scene["cgs_room_bounds"] = {"x_min": "bad"}
    assert builder._scene_room_bounds() is None
    builder.bpy.context.scene["cgs_room_bounds"] = {"x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}
    assert builder._scene_room_bounds() is None
    builder.bpy.context.scene._store.clear()
    assert builder._nearest_room_wall_context(aabb()) is None

    assert builder._item_semantic_group({"name": "настольная лампа"}) == "lamp_table"
    assert builder._item_semantic_group({"name": "подвесная люстра"}) == "lamp_ceiling"
    assert builder._item_semantic_group({"name": "tv stand oak"}) == "tv_stand"
    assert builder._item_semantic_group({"name": "TV panel"}) == "tv_projector_screen"
    assert builder._item_semantic_group({"name": "soft sofa"}) == "sofa"
    assert builder._item_mount_mode({"constraints": {"touch_floor": {"side": "bottom"}}}) == "floor"

    assert builder._matches_room_shell_name("") is False
    assert builder._looks_like_room_wrapper_name("") is False
    assert builder._looks_like_room_wrapper_name("ceiling_light_diffuser") is False
    assert builder._looks_like_overlay_helper_name("") is False
    assert builder._looks_like_bbox_helper_name("") is False
    assert builder._looks_like_architectural_door_name("") is False

    class BadHideObject(FakeObject):
        def hide_set(self, _value):
            raise RuntimeError("hide_set failed")

        def __setattr__(self, name, value):
            if name in {"hide_viewport", "hide_render"} and hasattr(self, "hide_viewport"):
                raise RuntimeError("hide attr failed")
            super().__setattr__(name, value)

    assert builder._hide_single_object(BadHideObject("bad_hide")) >= 1
    wall = BadHideObject("Room_Wall_w0")
    builder.bpy.data.objects[:] = [wall]
    assert builder._hide_room_shell_objects() == 1
    overlay = BadHideObject("item_AABB")
    builder.bpy.data.objects[:] = [overlay]
    assert builder._set_overlay_helpers_render_visibility(False) == 0
    assert builder._apply_render_layer_visibility("all") == 0
    assert builder._object_matches_render_layer(FakeObject("anything"), "all")
    assert builder._object_matches_render_layer(FakeObject("anything"), "non_kitchen")
    assert builder._object_matches_render_layer(FakeObject("wall object"), "room_shell")
    assert builder._object_matches_render_layer(FakeObject("shtora curtain"), "curtains")
    assert builder._object_matches_render_layer(FakeObject("unknown"), "custom_layer")

    class BadDoor(FakeObject):
        @property
        def dimensions(self):
            raise RuntimeError("dims failed")

    assert not builder._looks_like_orphan_architectural_door_panel(BadDoor("Cube"))
    keep = FakeObject("keep_AABB")
    drop = FakeObject("drop_AABB")
    builder.bpy.data.objects = RaisingObjects([drop])
    assert builder._remove_bbox_helper_objects() == 1
    builder.bpy.data.objects = objects
    objects[:] = [FakeObject("CGS_FunctionalLight_x_Emitter"), FakeObject("CGS_FunctionalLight_y")]
    monkeypatch.setattr(builder, "_replace_missing_texture_materials", lambda objects=None: 0)
    monkeypatch.setattr(builder, "_hide_architectural_door_objects", lambda: 0)
    cleanup = builder._cleanup_final_visual_helpers()
    assert cleanup["hidden_functional_lights"] >= 0

    assert builder._functional_light_location(aabb(), "unknown")[2] == pytest.approx(0.65)
    builder.bpy.data.objects = FakeObjectStore()
    builder.bpy.context.scene.collection.objects = FakeCollection("SceneLights").objects

    class LightStore:
        def new(self, name, light_type):
            return types.SimpleNamespace(name=name, type=light_type, energy=0.0, shadow_soft_size=0.0)

    builder.bpy.data.lights = LightStore()
    floor_lamp = {"id": "floor_lamp", "category": "FloorLampFactory", "meta": {"supplier_binding_applied": True}}
    assert builder._add_functional_light_for_item(floor_lamp, aabb(0, 1, 0, 1, 0, 2)) == 1
    builder.bpy.data.lights = types.SimpleNamespace(new=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("light failed")))
    assert builder._add_functional_light_for_item({"id": "bad", "category": "DeskLampFactory", "meta": {"supplier_binding_applied": True}}, aabb()) == 0

    class ImageStore(list):
        def remove(self, _img, do_unlink=True):
            raise RuntimeError("image remove failed")

    existing = tmp_path / "existing.png"
    existing.write_bytes(b"img")
    builder.bpy.data.images = ImageStore(
        [
            types.SimpleNamespace(name="packed", packed_file=object(), source="FILE", filepath=""),
            types.SimpleNamespace(name="generated", packed_file=None, source="GENERATED", filepath=""),
            types.SimpleNamespace(name="existing", packed_file=None, source="FILE", filepath=str(existing)),
            types.SimpleNamespace(name="missing", packed_file=None, source="FILE", filepath=str(tmp_path / "missing.png")),
        ]
    )
    assert builder._prune_missing_images_before_pack() == 0
    builder.bpy.ops.file.make_paths_relative = lambda: (_ for _ in ()).throw(RuntimeError("relative"))
    builder.bpy.ops.file.pack_all = lambda: (_ for _ in ()).throw(RuntimeError("pack"))
    builder._pack_assets_best_effort()

    bad_scene = types.SimpleNamespace(render=RaiseOnSet(), cycles=RaiseOnSet())
    builder._configure_fast_render(bad_scene)
    builder._configure_turntable_render(bad_scene)

    assert builder._resolve_path_maybe(tmp_path, "  ") is None
    assert builder._item_mesh_yaw_offset_deg({"meta": {"asset_yaw_offset_deg": "bad"}, "asset_yaw_offset_deg": 17}) == 17.0
    assert builder._supplier_candidate_dimensions_m(None) is None
    assert not builder._sizes_materially_different((0, 1, 1), (0, 1, 1))
    assert builder._build_supplier_reuse_size_index([None, {"meta": {"supplier_candidate": {"unique_key": "x"}}}]) == {}
    assert builder._supplier_candidate_reuse_reject_reason({"category": "rug"}, {}, {}) is None
    assert builder._supplier_candidate_reuse_reject_reason({"category": "rug"}, {"unique_key": "x"}, {"x": [("a", "rug", (1, 1, 1)), ("b", "rug", (1, 1, 1))]}) is None
    assert builder._aabb_center_size_from_dict({"x_min": "bad"}) is None
    assert not builder._aabb_nearly_identical({"x_min": "bad"}, aabb())
    assert not builder._aabb_nearly_identical(aabb(0, 1, 0, 1, 0, 1), aabb(10, 11, 10, 11, 10, 11))
    assert builder._is_replacement_bed_item({"category": "bed"}) is False
    assert builder._is_bed_companion_item({"meta": {"supplier_binding_applied": True}}) is False
    assert builder._aabb_xy_intersection_ratio({}, aabb()) == 0.0
    assert builder._aabb_intersection_ratio({}, aabb()) == 0.0
    assert builder._duplicate_semantics_compatible({"category": "chair"}, {"category": "chair"})
    duplicate_items = [
        {"id": "", "category": "chair"},
        {"id": "bad_aabb", "category": "chair", "aabb": []},
        {"id": "one", "category": "chair", "aabb": aabb(), "meta": {"supplier_binding_applied": True}},
        {"id": "two", "category": "chair", "aabb": aabb()},
        {"id": "three", "category": "chair", "aabb": aabb(0, 1, 0, 1, 0, 1)},
    ]
    assert "three" in builder._find_duplicate_render_item_ids(duplicate_items)

    builder.bpy.context.window_manager = types.SimpleNamespace(windows=[types.SimpleNamespace(screen=None)])
    assert builder._build_obj_import_override() is None
    area = types.SimpleNamespace(type="OUTLINER", regions=[], spaces=types.SimpleNamespace(active=None))
    builder.bpy.context.window_manager = types.SimpleNamespace(windows=[types.SimpleNamespace(screen=types.SimpleNamespace(areas=[area]))])
    assert builder._build_obj_import_override() is None

    assert builder._discover_mesh_import_candidates(str(tmp_path / "missing.obj")) == [str((tmp_path / "missing.obj").resolve())]
    source_glb = tmp_path / "source.glb"
    source_glb.write_bytes(b"glb")
    assert builder._export_import_group_glb_cache([FakeObject("a"), FakeObject("b")], str(source_glb)) is None
    assert builder._export_import_group_glb_cache([FakeObject("a")], str(tmp_path / "source.fbx")) is None
    cached = tmp_path / "source.cgs_group.glb"
    cached.write_bytes(b"glb")
    assert builder._export_import_group_glb_cache([FakeObject("a"), FakeObject("b")], str(tmp_path / "source.fbx")) == str(cached)

    builder.bpy.ops.object.select_all = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("select all"))
    obj_select = FakeObject("select_me")
    obj_select.select_set = lambda _value: (_ for _ in ()).throw(RuntimeError("select"))
    builder.bpy.data.objects[:] = [obj_select]
    builder.bpy.context.view_layer.objects = types.SimpleNamespace(active=None)
    builder._restore_object_selection({"select_me"}, "select_me")

    assert not builder._is_hashlike_name("")
    assert not builder._is_hashlike_name("short")
    assert not builder._is_hashlike_name("hash name with spaces")
    assert not builder._is_hashlike_name("абвdef1234567890")
    assert list(builder._walk_images_limited(str(tmp_path / "missing_dir"), 1, 10)) == []
    deep = tmp_path / "deep"
    (deep / "a" / "b").mkdir(parents=True)
    (deep / "a" / "b" / "tex.jpg").write_bytes(b"img")
    assert list(builder._walk_images_limited(str(deep), max_depth=0, max_files=10)) == []

    mtl = tmp_path / "bad.mtl"
    mtl.write_text("map_Kd\nnewmtl\nmap_Bump normal.png\n", encoding="utf-8")
    assert builder._strip_mtl_opts("") == ""
    assert builder.parse_mtl_refs(str(mtl)).get("bump_ref") == "normal.png"
    valid_mtl = tmp_path / "valid.mtl"
    valid_mtl.write_text("newmtl Mat\nmap_Bump normal.png\n", encoding="utf-8")
    assert builder.parse_mtl_materials(str(valid_mtl))
    assert builder.resolve_texture_ref("missing.png", tmp_path, {}) is None
    assert builder.resolve_texture_ref("/no/such/file.png", tmp_path, {}) is None


def test_remaining_import_filter_cluster_aabb_and_family_edges(builder, monkeypatch):
    def mesh_box(name, xmin, xmax, ymin, ymax, zmin, zmax):
        obj = FakeObject(name)
        obj.data.vertices = [
            types.SimpleNamespace(co=FakeVector((x, y, z)))
            for x in (xmin, xmax)
            for y in (ymin, ymax)
            for z in (zmin, zmax)
        ]
        return obj

    zero_meshes = [
        mesh_box("zero_a", 0, 0, 0, 0, 0, 0),
        mesh_box("zero_b", 0, 0, 0, 0, 0, 0),
        mesh_box("zero_c", 0, 0, 0, 0, 0, 0),
    ]
    assert builder._filter_imported_mesh_outliers(zero_meshes) == (zero_meshes, [])
    assert builder._keep_primary_import_cluster(zero_meshes) == (zero_meshes, [])

    preview = FakeObject("mat_preview")
    chair = FakeObject("chair_body")
    filtered, dropped = builder._drop_material_preview_meshes([preview, chair])
    assert filtered == [chair]
    assert dropped == ["mat_preview"]

    one_variant = [mesh_box("mirror_800x600", 0, 0.8, 0, 0.05, 0, 0.6)]
    assert builder._select_single_import_variant_mesh(one_variant, "mirror", aabb()) == (one_variant, [])
    same_variant = mesh_box("same_800x600", 0, 0.8, 0, 0.05, 0, 0.6)
    assert builder._select_single_import_variant_mesh([same_variant, same_variant], "mirror", aabb(0, 0.8, 0, 0.05, 0, 0.6)) == (
        [same_variant, same_variant],
        [],
    )

    normal_a = mesh_box("normal_a", 0, 1, 0, 1, 0, 1)
    normal_b = mesh_box("normal_b", 0, 1, 0, 1, 0, 1)
    helper_plane = mesh_box("huge_flat_helper", 0, 100, 0, 100, 0, 0.001)
    non_mesh = FakeObject("curve", obj_type="CURVE")
    filtered, dropped = builder._filter_imported_mesh_outliers([normal_a, normal_b, helper_plane, non_mesh])
    assert helper_plane not in filtered
    assert dropped[0]["name"] == "huge_flat_helper"

    close_cluster = [
        mesh_box("cluster_a", 0, 1, 0, 1, 0, 1),
        mesh_box("cluster_b", 0.5, 1.5, 0, 1, 0, 1),
        mesh_box("cluster_c", 1.0, 2.0, 0, 1, 0, 1),
    ]
    assert builder._keep_primary_import_cluster(close_cluster) == (close_cluster, [])

    far_cluster = [
        mesh_box("main_a", 0, 1, 0, 1, 0, 1),
        mesh_box("main_b", 0.5, 1.5, 0, 1, 0, 1),
        mesh_box("far_piece", 100, 101, 100, 101, 0, 1),
    ]
    filtered, dropped = builder._keep_primary_import_cluster(far_cluster)
    assert far_cluster[2] not in filtered
    assert dropped[0]["name"] == "far_piece"

    class RaisingObjects(FakeObjects):
        def remove(self, _obj, do_unlink=True):
            raise RuntimeError("remove failed")

    doomed_parent = FakeObject("doomed_parent", obj_type="EMPTY")
    doomed_child = FakeObject("doomed_child", parent=doomed_parent)
    builder.bpy.data.objects = RaisingObjects([doomed_parent, doomed_child])
    builder._remove_object_family(doomed_parent)

    empty_source = FakeObject("empty_source", obj_type="EMPTY")
    builder.bpy.data.objects = FakeObjects([empty_source])
    assert builder._aabb_from_blend_object_name("empty_source") is None
    assert builder._aabb_from_object_family_root(empty_source) is None
    assert builder._get_scene_source_object("missing_source") is None

    suffixed = FakeObject("SourceName.004")
    builder.bpy.data.objects = FakeObjects([suffixed])
    assert builder._get_scene_source_object("SourceName") is suffixed

    class BadHideCopy(FakeObject):
        def hide_set(self, _value):
            raise RuntimeError("hide copy failed")

    class CopyToBadHide(FakeObject):
        def copy(self):
            copied = BadHideCopy(f"{self.name}_copy", obj_type=self.type)
            copied.data = self.data.copy() if hasattr(self.data, "copy") else self.data
            copied.matrix_world = self.matrix_world.copy()
            copied.matrix_parent_inverse = self.matrix_parent_inverse.copy()
            return copied

    fixture_root = CopyToBadHide("floorlamp_root", obj_type="EMPTY")
    fixture_root.data = types.SimpleNamespace(name="root_data", copy=lambda: types.SimpleNamespace(name="root_data_copy"))
    fixture_child = CopyToBadHide("fixture_light", obj_type="LIGHT", parent=fixture_root)
    fixture_child.data = types.SimpleNamespace(name="light_data", copy=lambda: types.SimpleNamespace(name="light_data_copy"))
    plain = FakeObject("plain_mesh")
    builder.bpy.context.scene.collection.objects = FakeCollection("Scene").objects
    builder.bpy.data.objects = FakeObjects([fixture_root, fixture_child, plain])
    assert builder._duplicate_light_objects_from_family(fixture_root) == 2
    assert builder._collect_reference_light_fixture_roots() == [fixture_root]

    class HideRenderReferenceError:
        @property
        def hide_render(self):
            raise ReferenceError("gone")

    class HideGetReferenceError(FakeObject):
        def hide_get(self):
            raise ReferenceError("gone")

    assert builder._object_or_parent_hidden(HideRenderReferenceError()) is False
    assert builder._object_or_parent_hidden(HideGetReferenceError("hide_get_gone")) is False

    monkeypatch.setattr(builder, "_object_or_parent_hidden", lambda _root: (_ for _ in ()).throw(ReferenceError("gone")))
    builder.bpy.data.objects = FakeObjects([fixture_root])
    assert builder._restore_hidden_reference_light_fixtures([fixture_root]) == 0

    class BadMoveObject(FakeObject):
        def __setattr__(self, name, value):
            if name == "location" and hasattr(self, "location"):
                raise RuntimeError("move failed")
            super().__setattr__(name, value)

    bad_move = BadMoveObject("bad_move")
    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", lambda _family: (FakeVector((0, 0, 0)), FakeVector((1, 1, 1))))
    assert builder._move_object_family_to_target_aabb(bad_move, aabb(1, 2, 1, 2, 0, 1))

    class BadFamilyObject(FakeObject):
        def hide_set(self, _value):
            raise RuntimeError("hide_set failed")

        def __setattr__(self, name, value):
            if name in {"hide_viewport", "hide_render"} and hasattr(self, "hide_viewport"):
                raise RuntimeError("hide attr failed")
            super().__setattr__(name, value)

    bad_family = BadFamilyObject("bad_family")
    object.__setattr__(bad_family, "hide_viewport", True)
    object.__setattr__(bad_family, "hide_render", True)
    assert builder._hide_object_family(bad_family) == 0
    assert builder._show_object_family(bad_family) == 1


def test_additional_builder_texture_camera_and_filter_edges(builder, tmp_path, monkeypatch):
    objects, _scene_collection = _install_geometry_backend(builder)
    coll = FakeCollection("Items")
    builder.bpy.data.materials = FakeMaterialStore()

    loaded_images = []

    def load_image(path, check_existing=True):
        img = types.SimpleNamespace(
            name=Path(path).name,
            filepath=str(path),
            filepath_raw=str(path),
            source="FILE",
            packed_file=None,
            size=(16, 16),
            has_data=True,
            colorspace_settings=types.SimpleNamespace(name=""),
        )
        loaded_images.append(img)
        return img

    builder.bpy.data.images = types.SimpleNamespace(load=load_image)

    single = [FakeObject("single")]
    assert builder._drop_material_preview_meshes(single) == (single, [])
    assert builder._select_single_import_variant_mesh([FakeObject("chair")], "chair", aabb())[1] == []
    assert builder._select_single_import_variant_mesh([FakeObject("mirror")], "mirror", aabb())[1] == []
    zero_a = FakeObject("zero_a")
    zero_b = FakeObject("zero_b")
    zero_c = FakeObject("zero_c")
    monkeypatch.setattr(builder, "_world_bounds_single_mesh_object", lambda _obj: (FakeVector((0, 0, 0)), FakeVector((0, 0, 0))))
    assert builder._keep_primary_import_cluster([zero_a, zero_b, zero_c])[1] == []

    builder.bpy.data.objects = FakeObjectStore()
    builder.bpy.data.cameras = types.SimpleNamespace(new=lambda name: types.SimpleNamespace(name=name, lens=0))
    builder.bpy.context.scene.collection.objects = FakeCollection("Scene").objects
    builder._frame_camera_on_bounds(FakeVector((0, 0, 0)), FakeVector((2, 2, 2)))
    assert builder.bpy.context.scene.camera.name == "Camera"
    assert builder.bpy.context.scene.camera.location.x == pytest.approx(-2.2)

    builder.bpy.data.objects.new("visible_mesh", FakeMesh("visible_mesh_data"))
    monkeypatch.setattr(builder, "_world_bounds_mesh_objects", lambda _objs: (FakeVector((1, 1, 1)), FakeVector((1, 1, 1))))
    default_min = FakeVector((-1, -1, 0))
    default_max = FakeVector((1, 1, 2))
    assert builder._visible_mesh_bounds(default_min, default_max) == (default_min, default_max)

    assert builder._apply_tint_to_material_nodes(None, (0.1, 0.2, 0.3)) is False
    no_tree = types.SimpleNamespace(use_nodes=True, node_tree=None)
    assert builder._apply_tint_to_material_nodes(no_tree, (0.1, 0.2, 0.3)) is False
    no_bsdf = builder.bpy.data.materials.new("no_bsdf")
    no_bsdf.use_nodes = True
    assert builder._apply_tint_to_material_nodes(no_bsdf, (0.1, 0.2, 0.3)) is False
    no_base = builder.bpy.data.materials.new("no_base")
    no_base.use_nodes = True
    bsdf_no_base = no_base.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    del bsdf_no_base.inputs["Base Color"]
    assert builder._apply_tint_to_material_nodes(no_base, (0.1, 0.2, 0.3)) is False

    mat = builder.bpy.data.materials.new("image_nodes")
    base_img = load_image(tmp_path / "base.jpg")
    normal_img = load_image(tmp_path / "normal.jpg")
    alpha_img = load_image(tmp_path / "alpha.png")
    builder._apply_image_to_material_nodes(mat, basecolor_img=base_img, normal_img=normal_img, opacity_img=alpha_img)
    assert mat.blend_method == "HASHED"
    assert any(node.type == "TEX_IMAGE" for node in mat.node_tree.nodes)

    mat2 = builder.bpy.data.materials.new("base_only")
    builder._ensure_principled_basecolor_image(mat2, base_img)
    assert any(node.type == "TEX_IMAGE" and node.image is base_img for node in mat2.node_tree.nodes)

    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    obj_path = asset_dir / "chair.obj"
    mtl_path = asset_dir / "chair.mtl"
    diffuse = asset_dir / "fabric_diffuse.jpg"
    ambient = asset_dir / "fabric_ambient.jpg"
    normal = asset_dir / "fabric_normal.png"
    alpha = asset_dir / "fabric_alpha.png"
    small = asset_dir / "small.jpg"
    big = asset_dir / "big_color.jpg"
    for path, payload in (
        (diffuse, b"123"),
        (ambient, b"1234"),
        (normal, b"12"),
        (alpha, b"1"),
        (small, b"1"),
        (big, b"123456789"),
    ):
        path.write_bytes(payload)
    obj_path.write_text("mtllib chair.mtl\n", encoding="utf-8")
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl FabricMain",
                "map_Ka fabric_ambient.jpg",
                "map_Bump fabric_normal.png",
                "map_d fabric_alpha.png",
            ]
        ),
        encoding="utf-8",
    )
    parent = FakeObject("TexturedParent")
    parent_mat = builder.bpy.data.materials.new("FabricMain")
    parent.data.materials = [parent_mat]
    idx = builder.build_image_index([str(asset_dir)])
    assert builder._apply_mtl_map_kd_to_existing_mats(parent, str(obj_path), idx, verbose=True)
    assert any(node.type == "TEX_IMAGE" for node in parent_mat.node_tree.nodes)

    missing_mat = builder.bpy.data.materials.new("MissingMat")
    missing_mat.use_nodes = True
    missing_node = missing_mat.node_tree.nodes.new("ShaderNodeTexImage")
    missing_node.name = "fabric_diffuse.jpg"
    missing_node.image = types.SimpleNamespace(name="missing", filepath="", filepath_raw="", source="FILE", packed_file=None, size=(0, 0), has_data=False)
    parent.data.materials = [missing_mat]
    fixed, used = builder._relink_missing_images(parent, idx)
    assert fixed == 1
    assert used[0].endswith("fabric_diffuse.jpg")

    plain_mat = builder.bpy.data.materials.new("PlainMat")
    parent.data.materials = [plain_mat]
    assert builder._fallback_apply_largest_image_existing_mats(parent, idx, verbose=True)
    empty_mats = FakeObject("EmptyMats")
    empty_mats.data.materials = []
    assert builder._fallback_apply_flat_tint_existing_mats(empty_mats, (0.2, 0.3, 0.4), verbose=True)
    assert len(empty_mats.data.materials) == 1

    maps = builder.PBRMaps(
        basecolor=str(diffuse),
        normal=str(normal),
        roughness=str(small),
        metallic=str(small),
        ao=str(ambient),
        height=str(small),
        emissive=str(diffuse),
        opacity=str(alpha),
    )
    full_mat = builder._make_pbr_material("full_pbr", maps, tint_rgb=(0.2, 0.25, 0.3), tex_scale=2.0)
    assert full_mat.blend_method == "HASHED"
    assert len(loaded_images) >= 10

    source_img = types.SimpleNamespace(name="source_color.jpg", filepath=str(diffuse), filepath_raw="", source="FILE", packed_file=None, size=(8, 8), has_data=True)
    source = types.SimpleNamespace(type="TEX_IMAGE", image=source_img, inputs=[], as_pointer=lambda: id(source_img))
    middle_socket = FakeNodeSocket("middle")
    middle_socket.is_linked = True
    middle_socket.links = [types.SimpleNamespace(from_node=source)]
    middle = types.SimpleNamespace(type="MIX", inputs=[middle_socket], as_pointer=lambda: id(middle_socket))
    root_socket = FakeNodeSocket("root")
    root_socket.is_linked = True
    root_socket.links = [types.SimpleNamespace(from_node=middle)]
    assert builder._socket_chain_has_real_image(root_socket)
    assert builder._socket_chain_has_real_color_image(root_socket)

    curtain_root = FakeObject("curtain_root", obj_type="EMPTY")
    curtain_mesh = FakeMesh("curtain_mesh")
    curtain_mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [], [(0, 1, 2, 3)])
    curtain_obj = FakeObject("curtain_obj", parent=curtain_root)
    curtain_obj.data = curtain_mesh
    builder._apply_curtain_planar_uv(
        parent=curtain_root,
        aabb=aabb(0, 1, 0, 0.1, 0, 1),
        rotation_deg_engine=0,
        mirror_repeat=False,
    )
    assert curtain_mesh.uv_layers.active.name == "ShtorystorePlanarUV"

    bbox_obj = builder._make_renderable_bbox_box(aabb(), "DebugBBox", coll)
    assert bbox_obj["cgs_keep_bbox_fallback"] is True
    chair = builder._make_generated_chair_placeholder(aabb(0, 1, 0, 1, 0, 1), "GeneratedChair", coll, yaw_deg=20)
    assert chair.name.endswith("GeneratedChairRoot")
    builder.bpy.ops.mesh = types.SimpleNamespace(
        primitive_cylinder_add=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cylinder failed"))
    )
    builder.bpy.data.lights = types.SimpleNamespace(new=lambda name, light_type: types.SimpleNamespace(name=name, type=light_type, energy=0, size=0))
    flat = builder._make_procedural_flat_ceiling_light_mesh(
        item={"id": "flat"},
        aabb=aabb(0, 0.4, 0, 0.4, 2.4, 2.5),
        collection=coll,
        name="FlatLight",
    )
    assert flat is not None
    glass = builder._make_glass_material("GlassWithImage", str(diffuse))
    assert glass.use_nodes is True
