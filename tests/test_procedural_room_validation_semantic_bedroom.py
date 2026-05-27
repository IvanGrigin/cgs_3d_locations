from __future__ import annotations

import random
from types import SimpleNamespace

from src.pipeline.procedural_rooms import bedroom_generator as bedroom
from src.pipeline.procedural_rooms import semantic_polish as sem
from src.pipeline.procedural_rooms.object_specs import BEDROOM_SPECS, ObjectSpec
from src.pipeline.procedural_rooms.placement_engine import PlacementEngine
from src.pipeline.procedural_rooms.room_context import build_room_context
from src.pipeline.procedural_rooms.semantic_polish import (
    _physical_role,
    annotate_layout_contracts,
    apply_procedural_semantic_polish,
    enforce_bedroom_functional_clearances,
    enforce_category_limits,
    enforce_surface_limits,
    normalize_item_semantics,
    on_top_items,
    remove_invalid_wardrobe_top_items,
    repair_solid_floor_overlaps,
    repair_wall_mounted_overlaps,
    solid_floor_items,
    wall_mounted_items,
)
from src.pipeline.procedural_rooms.validation import (
    _aabb_contains_xy,
    _aabb_from_item,
    _collision_margin_for_pair,
    _gap_between,
    _sample_access_points,
    is_solid_floor_obstacle,
    validate_placements,
)


def room_scene(room_type: str = "bedroom", width: float = 6.0, depth: float = 4.0) -> dict:
    return {
        "room": {
            "id": "room_1",
            "room_type": room_type,
            "width_m": width,
            "depth_m": depth,
            "area_m2": width * depth,
            "ceiling_height_m": 2.8,
            "walls": [
                {"id": "w0", "name": "entrance wall"},
                {"id": "w1", "name": "right wall"},
                {"id": "w2", "name": "far bed wall"},
                {"id": "w3", "name": "window wall"},
            ],
            "doors": [{"wall_id": "w0", "s": 0.45, "width": 0.8}],
            "windows": [{"wall_id": "w3", "s": 1.0, "width": 1.2}],
        }
    }


def placement(
    item_id: str,
    category: str,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float = 0.0,
    z_max: float = 0.6,
    role: str | None = "solid_floor",
    mount_type: str | None = None,
    parent_id: str | None = None,
    wall_id: str | None = None,
    procedural: bool = True,
    position_delta: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    cx = (x_min + x_max) * 0.5 + position_delta[0]
    cy = (y_min + y_max) * 0.5 + position_delta[1]
    cz = (z_min + z_max) * 0.5 + position_delta[2]
    meta: dict = {"procedural": procedural}
    if role:
        meta["physical_role"] = role
    if wall_id:
        meta["wall_id"] = wall_id
    constraints: dict = {}
    if parent_id:
        constraints["parent_id"] = parent_id
    return {
        "id": item_id,
        "category": category,
        "name": category.title(),
        "position_m": [cx, cy, cz],
        "size_m": [x_max - x_min, y_max - y_min, z_max - z_min],
        "yaw_deg": 0.0,
        "mount_type": mount_type or "floor",
        "constraints": constraints,
        "meta": meta,
        "aabb": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        },
    }


def test_validation_reports_scene_contract_and_geometry_problems():
    ctx = build_room_context(room_scene("bedroom", 3.0, 3.0))
    bed = placement("bed", "bed", x_min=0.2, x_max=1.5, y_min=0.2, y_max=1.8, position_delta=(0.2, 0.0, 0.0))
    wardrobe = placement("wardrobe", "wardrobe", x_min=0.8, x_max=1.8, y_min=0.4, y_max=1.4)
    outside = placement("outside", "desk", x_min=2.6, x_max=3.4, y_min=1.0, y_max=1.8)
    too_tall = placement("too_tall", "dresser", x_min=1.9, x_max=2.4, y_min=2.0, y_max=2.5, z_max=3.2)
    wall_art = placement("art", "wall_art", x_min=2.1, x_max=2.6, y_min=2.1, y_max=2.6, role="wall_mounted", mount_type="wall")

    report = validate_placements(ctx, [bed, wardrobe, outside, too_tall, wall_art])
    assert report["collisions"]
    assert report["outside_room"]
    assert report["aabb_bounds_violations"]
    assert report["vertical_bounds_violations"]
    assert report["door_clearance_violations"]
    assert report["aabb_center_mismatches"]
    assert report["orientation_contract_missing"]
    assert report["clearance_contract_missing"]
    assert report["functional_clearance_violations"]
    assert not report["accessibility_ok"]

    bathroom = build_room_context(room_scene("bathroom", 2.4, 2.2))
    bath_mat = placement("mat", "bath_mat", x_min=0.5, x_max=1.4, y_min=0.7, y_max=1.5, z_max=0.03, role="soft_floor")
    toilet = placement("toilet", "toilet", x_min=0.8, x_max=1.2, y_min=0.9, y_max=1.5)
    bathroom_report = validate_placements(bathroom, [bath_mat, toilet])
    assert bathroom_report["soft_floor_solid_overlaps"]
    assert {"sink", "bathtub_or_shower"} <= {item["category"] for item in bathroom_report["required_missing"]}

    missing_living = validate_placements(build_room_context(room_scene("living_room", 4, 4)), [])
    assert missing_living["required_missing"] == [{"category": "sofa", "reason": "required_sofa_missing"}]


def test_validation_low_level_helpers_and_solid_classification():
    solid = placement("desk", "desk", x_min=0, x_max=1, y_min=0, y_max=1)
    rug = placement("rug", "rug", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    wall = placement("mirror", "mirror", x_min=0, x_max=1, y_min=0, y_max=1, role=None, mount_type="wall")
    compact_a = placement("a", "toilet", x_min=0, x_max=1, y_min=0, y_max=1)
    compact_b = placement("b", "sink", x_min=0, x_max=1, y_min=0, y_max=1)
    compact_a["meta"]["compact_bathroom_template"] = True
    compact_b["meta"]["compact_bathroom_template"] = True

    aabb = _aabb_from_item(solid)
    assert aabb is not None
    assert _aabb_contains_xy(aabb, 0.5, 0.5)
    assert len(_sample_access_points(aabb, clearance=0.3, step=0.5)) >= 8
    assert round(_gap_between(aabb, _aabb_from_item(placement("other", "chair", x_min=1.2, x_max=1.7, y_min=0, y_max=1))), 3) == 0.2
    assert is_solid_floor_obstacle(solid)
    assert not is_solid_floor_obstacle(rug)
    assert not is_solid_floor_obstacle(wall)
    assert _collision_margin_for_pair(compact_a, compact_b) == 0.0
    assert _collision_margin_for_pair(
        {"meta": {"door_swing_assumption": "outward_or_sliding"}},
        {"meta": {"door_swing_assumption": "outward_or_sliding"}},
    ) == 0.0

    assert not is_solid_floor_obstacle(placement("soft", "decor_vase", x_min=0, x_max=1, y_min=0, y_max=1, role=None, mount_type="on_top"))
    assert is_solid_floor_obstacle(placement("generated", "box", x_min=0, x_max=1, y_min=0, y_max=1, role=None, mount_type="floor", procedural=True))
    allowed = placement("allowed", "box", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    allowed["meta"]["allow_collision"] = True
    assert not is_solid_floor_obstacle(allowed)
    child = placement("child", "box", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    child["meta"]["support_relation"] = "on_top"
    assert not is_solid_floor_obstacle(child)
    soft_layer = placement("soft_layer", "box", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    soft_layer["meta"]["density_layer"] = "decor"
    assert not is_solid_floor_obstacle(soft_layer)
    ignored_light = placement("floor_lamp", "floor_lamp", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    ignored_light["meta"]["density_layer"] = "lighting"
    assert not is_solid_floor_obstacle(ignored_light)
    assert is_solid_floor_obstacle(placement("plain", "box", x_min=0, x_max=1, y_min=0, y_max=1, role=None))

    diagonal_gap = _gap_between(
        _aabb_from_item(placement("a", "chair", x_min=0, x_max=1, y_min=0, y_max=1)),
        _aabb_from_item(placement("b", "chair", x_min=2, x_max=3, y_min=2, y_max=3)),
    )
    assert diagonal_gap == 1


def test_validation_required_and_accessibility_edge_cases():
    no_door_ctx = build_room_context({"room": {"room_type": "bedroom", "width_m": 2.0, "depth_m": 2.0, "area_m2": 4.0}})
    reachable = validate_placements(no_door_ctx, [])
    assert reachable["required_missing"] == [
        {"category": "bed", "reason": "required_bed_missing"},
        {"category": "wardrobe_or_nightstand", "reason": "required_storage_or_nightstand_missing"},
    ]

    tiny_ctx = build_room_context(room_scene("bedroom", 2.0, 3.0))
    wide_bed = placement("wide_bed", "bed", x_min=0.15, x_max=1.85, y_min=0.7, y_max=2.0)
    nightstand = placement("nightstand", "nightstand", x_min=0.1, x_max=0.4, y_min=2.2, y_max=2.5)
    tiny_report = validate_placements(tiny_ctx, [wide_bed, nightstand])
    assert any(item["reason"] == "bed_splits_tiny_room_width" for item in tiny_report["functional_clearance_violations"])

    blocked_ctx = build_room_context(room_scene("bedroom", 2.2, 2.2))
    blocker = placement("blocker", "bed", x_min=0.0, x_max=2.2, y_min=0.0, y_max=2.2)
    wardrobe = placement("wardrobe", "wardrobe", x_min=0.4, x_max=0.8, y_min=0.4, y_max=0.8)
    blocked_report = validate_placements(blocked_ctx, [blocker, wardrobe])
    assert any(item["reason"] == "no_access_point_reachable" for item in blocked_report["functional_clearance_violations"])

    toilet_report = validate_placements(build_room_context(room_scene("toilet", 1.5, 2.0)), [])
    assert toilet_report["required_missing"] == [{"category": "toilet", "reason": "required_toilet_missing"}]

    toilet_ctx = build_room_context(room_scene("toilet", 1.5, 2.0))
    toilet = placement("toilet", "toilet", x_min=0.0, x_max=1.5, y_min=0.0, y_max=2.0)
    exempt = placement("exempt", "cabinet", x_min=0.0, x_max=1.5, y_min=0.0, y_max=2.0)
    exempt["meta"]["door_clearance_exempt"] = True
    door_report = validate_placements(toilet_ctx, [toilet, exempt])
    assert door_report["door_clearance_violations"] == []


def test_semantic_polish_normalizes_contracts_and_removes_invalid_items():
    bed = placement("bed", "bed", x_min=0.0, x_max=1.8, y_min=0.0, y_max=2.0, wall_id="w1")
    wardrobe = placement("wardrobe", "wardrobe", x_min=1.9, x_max=2.8, y_min=0.0, y_max=0.6, z_max=2.2, wall_id="w1")
    vase = placement("vase", "decor_vase", x_min=2.0, x_max=2.1, y_min=0.1, y_max=0.2, role=None, parent_id="wardrobe")
    pillow = placement("pillow", "pillow", x_min=0.2, x_max=0.5, y_min=0.2, y_max=0.3, role=None, parent_id="bed")
    wall_art_1 = placement("art1", "wall_art", x_min=1.9, x_max=2.8, y_min=0.0, y_max=0.6, z_min=1.0, z_max=1.5, role=None, mount_type="wall", wall_id="w1")
    wall_art_2 = placement("art2", "wall_art", x_min=2.9, x_max=3.4, y_min=0.0, y_max=0.4, z_min=1.0, z_max=1.5, role=None, mount_type="wall", wall_id="w1")
    wall_art_3 = placement("art3", "wall_art", x_min=3.5, x_max=4.0, y_min=0.0, y_max=0.4, z_min=1.0, z_max=1.5, role=None, mount_type="wall", wall_id="w1")
    nightstand = placement("nightstand", "nightstand", x_min=3.0, x_max=3.4, y_min=1.0, y_max=1.4)
    children = [
        placement(f"book{i}", "decor_books", x_min=3.0, x_max=3.1, y_min=1.0, y_max=1.1, role=None, parent_id="nightstand")
        for i in range(5)
    ]
    chair = placement("chair", "chair", x_min=0.1, x_max=1.0, y_min=0.1, y_max=1.0)

    normalize_item_semantics(wall_art_1)
    normalize_item_semantics(pillow)
    assert wall_art_1["mount_type"] == "wall"
    assert wall_art_1["meta"]["physical_role"] == "wall_mounted"
    assert pillow["mount_type"] == "on_top"
    assert pillow["meta"]["physical_role"] == "soft_on_object"
    assert _physical_role(bed) == "solid_floor"

    kept, removed = remove_invalid_wardrobe_top_items([wardrobe, vase, pillow])
    assert [item["reason"] for item in removed] == ["invalid_wardrobe_top_item"]
    assert vase not in kept

    kept, removed = enforce_category_limits([wall_art_1, wall_art_2, wall_art_3], room_type="bedroom", size_class="medium", density="normal")
    assert len(kept) == 2
    assert removed[0]["reason"] == "category_limit"

    kept, removed = enforce_surface_limits([nightstand, *children])
    assert len(removed) == 2
    assert {item["reason"] for item in removed} == {"surface_limit"}

    kept, removed = repair_solid_floor_overlaps([bed, chair])
    assert kept == [bed]
    assert removed[0]["id"] == "chair"

    kept, removed = enforce_bedroom_functional_clearances([bed, wardrobe], room={"room_type": "bedroom", "area_m2": 6.5})
    assert removed and removed[0]["reason"] in {"wardrobe_access_gap", "small_bedroom_floor_clutter"}

    kept, removed = repair_wall_mounted_overlaps([wardrobe, wall_art_1], room={"room_type": "bedroom"})
    assert removed and removed[0]["reason"] == "wall_mounted_overlap"

    scene = {"room": {"room_type": "bedroom", "area_m2": 6.5}, "placements": [bed, wardrobe, vase, pillow, wall_art_1, wall_art_2, wall_art_3, nightstand, *children, chair]}
    polished, report = apply_procedural_semantic_polish(scene, room_type="bedroom", size_class="small", density="normal")
    assert report["schema"] == "procedural_semantic_polish/v1"
    assert report["removed_count"] >= 4
    assert len(polished["placements"]) < len(scene["placements"])
    assert solid_floor_items(polished["placements"])
    assert wall_mounted_items(polished["placements"])
    assert on_top_items(polished["placements"])

    missing_scene, missing_report = apply_procedural_semantic_polish({"room": {}}, room_type="bedroom", size_class="small", density="normal")
    assert missing_scene == {"room": {}}
    assert missing_report["skipped"] is True


def test_semantic_contract_annotation_for_role_specific_items():
    items = [
        placement("bed", "bed", x_min=0, x_max=1.8, y_min=0, y_max=2, wall_id="w1"),
        placement("sink", "sink", x_min=2, x_max=2.5, y_min=0, y_max=0.6, wall_id="w0"),
        placement("mirror", "mirror", x_min=2, x_max=2.6, y_min=0, y_max=0.1, z_min=1.0, z_max=1.8, role=None, mount_type="wall", wall_id="w0"),
        placement("light", "ceiling_light", x_min=1, x_max=1.3, y_min=1, y_max=1.3, z_min=2.5, z_max=2.7, role=None, mount_type="ceiling"),
        placement("bench", "bench", x_min=0.1, x_max=1.1, y_min=2.1, y_max=2.4),
        placement("desk", "desk", x_min=2.7, x_max=3.5, y_min=1.0, y_max=1.5),
    ]
    items[4]["meta"]["anchor_id"] = "bed"
    items[4]["meta"]["placement_relation"] = "near"
    items[5]["meta"]["front_target"] = "bed"
    for item in items:
        normalize_item_semantics(item)

    annotate_layout_contracts(items, room_type="bathroom")
    assert items[0]["orientation_rule"]["type"] == "headboard_against_wall"
    assert items[1]["clearance_rule"]["min_clearance_m"] == 0.32
    assert items[2]["orientation_rule"]["type"] == "mounted_on_wall"
    assert items[3]["orientation_rule"]["type"] == "ceiling_mounted"
    assert items[4]["clearance_rule"]["type"] == "decorative_adjacent_anchor"
    assert items[5]["orientation_rule"]["target_id"] == "bed"


def test_semantic_polish_remaining_edge_branches():
    bad_meta = {"id": "bad", "category": "unknown", "meta": "not-dict"}
    assert sem._meta(bad_meta) == {}
    assert bad_meta["meta"] == {}
    assert sem._to_float(None, 7.0) == 7.0
    assert sem._to_float("bad", 3.0) == 3.0

    rug = placement("rug", "rug", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    chair = placement("chair", "chair", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    normalize_item_semantics(rug)
    normalize_item_semantics(chair)
    assert rug["meta"]["physical_role"] == "soft_floor"
    assert rug["meta"]["allow_collision"] is True
    assert chair["meta"]["physical_role"] == "solid_floor"
    assert sem._replace_with_supplier_for_role("custom", "mystery") is False

    invalid_aabb = placement("bad_aabb", "chair", x_min=0, x_max=1, y_min=0, y_max=1)
    invalid_aabb["aabb"]["x_min"] = "bad"
    assert sem._aabb(invalid_aabb) is None
    assert sem._access_min_clearance("bench", "bedroom") == 0.15
    assert sem._intersects_z({"z_min": 0, "z_max": 1}, {"z_min": 2, "z_max": 3}) is False
    assert sem._axis_gap({"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}, {"x_min": 0.2, "x_max": 0.8, "y_min": 2, "y_max": 3}) == 1
    assert sem._axis_gap({"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}, {"x_min": 2, "x_max": 3, "y_min": 2, "y_max": 3}) == 1

    door_toilet = placement("toilet", "toilet", x_min=0, x_max=0.5, y_min=0, y_max=0.7, role=None)
    door_toilet["meta"]["front_target"] = "door"
    normalize_item_semantics(door_toilet)
    annotate_layout_contracts([door_toilet], room_type="toilet")
    assert door_toilet["access_target"] == "door"
    assert door_toilet["orientation_rule"]["type"] == "face_target"

    generic_target = placement("generic", "chair", x_min=0, x_max=1, y_min=0, y_max=1, role=None)
    generic_target["meta"]["front_target"] = "chair"
    other_chair = placement("other_chair", "chair", x_min=2, x_max=3, y_min=2, y_max=3, role=None)
    for item in [generic_target, other_chair]:
        normalize_item_semantics(item)
    annotate_layout_contracts([generic_target, other_chair], room_type="bedroom")
    assert generic_target["orientation_rule"]["target_id"] == "other_chair"

    unchanged, removed = enforce_category_limits([generic_target], room_type="office", size_class="medium", density="normal")
    assert unchanged == [generic_target]
    assert removed == []

    headboard = placement("headboard", "headboard", x_min=0, x_max=1, y_min=0, y_max=0.1, z_min=1, z_max=1.5, role=None, mount_type="wall", wall_id="w1")
    curtain = placement("curtain", "curtain", x_min=0, x_max=1, y_min=0, y_max=0.1, z_min=1, z_max=1.5, role=None, mount_type="wall", wall_id="w1")
    for item in [headboard, curtain]:
        normalize_item_semantics(item)
    kept, removed = repair_wall_mounted_overlaps([headboard, curtain], room={})
    assert kept == [headboard, curtain]
    assert removed == []

    plant = placement("plant", "plant", x_min=0, x_max=1, y_min=0, y_max=1)
    bed = placement("bed2", "bed", x_min=0, x_max=1, y_min=0, y_max=1)
    kept, removed = repair_solid_floor_overlaps([plant, bed])
    assert kept == [bed]
    assert removed[0]["id"] == "plant"

    on_top_parent = placement("parent", "nightstand", x_min=0, x_max=0.4, y_min=0, y_max=0.4)
    many_on_top = [
        placement(f"decor{i}", "decor_books", x_min=0, x_max=0.1, y_min=0, y_max=0.1, role=None, parent_id="parent")
        for i in range(7)
    ]
    _kept, removed = enforce_bedroom_functional_clearances(
        [on_top_parent, *many_on_top],
        room={"room_type": "bedroom", "area_m2": 0},
    )
    assert any(item["reason"] == "small_bedroom_on_top_limit" for item in removed)

    own = placement("own", "chair", x_min=0, x_max=1, y_min=0, y_max=1)
    other = placement("other", "chair", x_min=2, x_max=3, y_min=0, y_max=1)
    by_id = {"own": own, "other": other}
    by_category = {"chair": [own, other]}
    assert sem._target_id("", own, by_id, by_category) is None
    assert sem._target_id("door", own, by_id, by_category) == "door"
    assert sem._target_id("other", own, by_id, by_category) == "other"
    assert sem._target_id("chair", own, by_id, by_category) == "other"
    assert sem._target_id("missing", own, by_id, by_category) == "missing"

    no_limit_items = [placement("one", "wall_art", x_min=0, x_max=1, y_min=0, y_max=1, role=None, mount_type="wall")]
    assert enforce_category_limits(no_limit_items, room_type="kitchen", size_class="large", density="normal") == (no_limit_items, [])
    hooks = [
        placement(f"hook{i}", "wall_hooks", x_min=i, x_max=i + 0.1, y_min=0, y_max=0.1, role=None, mount_type="wall")
        for i in range(3)
    ]
    kept, removed = enforce_category_limits(hooks, room_type="corridor", size_class="small", density="normal")
    assert len(kept) == 1
    assert [item["reason"] for item in removed] == ["category_limit", "category_limit"]

    orphan_child = placement("orphan", "decor_books", x_min=0, x_max=0.1, y_min=0, y_max=0.1, role=None, parent_id="missing")
    bed = placement("bed_parent", "bed", x_min=0, x_max=2, y_min=0, y_max=2)
    bed_child = placement("bed_child", "decor_books", x_min=0, x_max=0.1, y_min=0, y_max=0.1, role=None, parent_id="bed_parent")
    assert enforce_surface_limits([orphan_child, bed, bed_child]) == ([orphan_child, bed, bed_child], [])

    low_priority = placement("plant", "plant", x_min=0, x_max=1, y_min=0, y_max=1)
    high_priority = placement("sofa", "sofa", x_min=0.2, x_max=1.2, y_min=0.2, y_max=1.2)
    kept, removed = repair_solid_floor_overlaps([low_priority, high_priority])
    assert kept == [high_priority]
    assert removed[0]["id"] == "plant"

    headboard = placement("headboard", "headboard", x_min=0, x_max=2, y_min=0, y_max=0.1, z_min=0.5, z_max=1.4, role=None, mount_type="wall", wall_id="w1")
    art = placement("art", "wall_art", x_min=0.5, x_max=1.5, y_min=0, y_max=0.1, z_min=0.7, z_max=1.2, role=None, mount_type="wall", wall_id="w1")
    kept, removed = repair_wall_mounted_overlaps([headboard, art], room={})
    assert removed == []
    assert kept == [headboard, art]

    non_bedroom_items = [placement("wardrobe", "wardrobe", x_min=0, x_max=1, y_min=0, y_max=1)]
    assert enforce_bedroom_functional_clearances(non_bedroom_items, room={"room_type": "living_room"}) == (non_bedroom_items, [])

    nightstand = placement("nightstand", "nightstand", x_min=0, x_max=0.4, y_min=0, y_max=0.4)
    on_top = [
        placement(f"decor{i}", "decor_books", x_min=0, x_max=0.1, y_min=0, y_max=0.1, role=None, parent_id="nightstand")
        for i in range(7)
    ]
    clutter = [
        placement("floor_lamp", "floor_lamp", x_min=1, x_max=1.2, y_min=0, y_max=0.2),
        placement("dresser", "dresser", x_min=1.3, x_max=1.8, y_min=0, y_max=0.4),
    ]
    kept, removed = enforce_bedroom_functional_clearances([nightstand, *on_top, *clutter], room={"room_type": "bedroom", "area_m2": 6.0})
    reasons = [item["reason"] for item in removed]
    assert reasons.count("small_bedroom_on_top_limit") == 2
    assert reasons.count("small_bedroom_floor_clutter") == 2


def test_bedroom_generator_helpers_and_public_generation():
    ctx = build_room_context(room_scene("bedroom", 6.0, 4.0))
    tiny_ctx = build_room_context(room_scene("bedroom", 2.0, 3.0))
    wall = ctx.walls[1]
    queen = BEDROOM_SPECS["queen_bed"]

    assert bedroom._bed_side_margin(ctx, 1.8) == 0.25
    assert bedroom._bed_side_margin(tiny_ctx, 1.4) == 0.08
    assert bedroom._bed_foot_clearance(tiny_ctx) == 0.35
    assert "queen_bed" in bedroom._bed_size_options(ctx)
    assert bedroom._bed_size_options(tiny_ctx)[-1] == "single_bed"
    assert bedroom._wall_name(ctx, "w2") == "far bed wall"
    assert bedroom._door_reference_point(ctx) is not None
    assert bedroom._preferred_bed_wall(ctx, 1.4) is not None
    assert bedroom._tiny_bed_wall(tiny_ctx, 0.9) is not None
    assert bedroom._wall_room_depth(ctx, wall) > 0

    aabb = bedroom._wall_aligned_aabb(ctx, wall, queen, wall.length * 0.5)
    assert bedroom._bed_clearance_ok(ctx, aabb)
    assert bedroom._bed_fits_wall(ctx, wall, queen)
    assert bedroom._select_bed_along(ctx, wall, queen) >= queen.size_m[0] * 0.5
    selected_spec, selected_wall, selected_along = bedroom._select_bed_plan(ctx)
    assert selected_spec.category == "bed"
    assert selected_wall is not None
    assert selected_along is not None

    resized = bedroom._spec_with_size(queen, (1.2, 2.0, 0.5), name="Test bed")
    assert isinstance(resized, ObjectSpec)
    assert resized.name == "Test bed"
    assert bedroom._nightstand_plan_for_space(ctx, 0.2) is None
    assert bedroom._nightstand_plan_for_space(ctx, 0.6)[0].category == "nightstand"
    assert len(bedroom._bed_variants()) == 5
    assert len(bedroom._nightstand_variants()) == 2
    assert len(bedroom._wardrobe_variants()) == 4
    assert len(bedroom._desk_variants()) == 3
    assert len(bedroom._shelf_variants()) == 3
    assert len(bedroom._armchair_variants()) == 2
    assert len(bedroom._plant_variants()) == 2
    assert bedroom._far_bed_wall(ctx) is not None

    placements, report = bedroom.generate_bedroom(ctx, density="very_high", seed=1)
    categories = {item["category"] for item in placements}
    assert {"bed", "nightstand", "wardrobe", "desk", "chair", "wall_art", "ceiling_light"} <= categories
    assert report["generator"] == "bedroom_generator"
    assert report["greedy_algorithm"]["bed_width_m"] >= 0.9
    assert report["bed_wall_id"]


def test_bedroom_generator_edge_branches_and_greedy_rollbacks(monkeypatch):
    tiny = build_room_context(room_scene("bedroom", 2.0, 2.8))
    medium = build_room_context(room_scene("bedroom", 2.5, 3.4))
    regular = build_room_context(room_scene("bedroom", 4.0, 3.0))
    wall = tiny.walls[1]
    single = BEDROOM_SPECS["single_bed"]

    assert bedroom._bed_side_margin(regular, 1.4) == 0.14
    assert bedroom._bed_foot_clearance(medium) == 0.45
    assert len(bedroom._bed_along_candidates(tiny, wall, single)) == 3
    assert bedroom._bed_fits_wall(tiny, wall, BEDROOM_SPECS["queen_bed"]) is False

    blocked = bedroom._wall_aligned_aabb(tiny, wall, single, wall.length * 0.5)
    tiny_with_window = build_room_context(room_scene("bedroom", 2.0, 2.8))
    tiny_with_window.window_clearance_zones.append(blocked)
    assert bedroom._bed_clearance_ok(tiny_with_window, blocked) is False

    no_valid_door = build_room_context({"room": {**room_scene("bedroom", 4.0, 3.0)["room"], "doors": [{"wall_id": "missing", "s": 1.0}]}})
    assert bedroom._door_reference_point(no_valid_door) is None
    assert bedroom._preferred_bed_wall(no_valid_door, 1.0) is not None
    assert bedroom._tiny_bed_wall(SimpleNamespace(walls=[]), 0.9) is None
    assert bedroom._wall_name(regular, "missing") == ""

    engine = PlacementEngine(ctx=regular, rng=random.Random(1), source_name="unit", generator_name="bedroom_generator", archetype="unit")
    pouf_count_before = len(engine.placements)
    bedroom._add_foreground_pouf(engine, regular)
    assert len(engine.placements) == pouf_count_before + 1
    console = bedroom._add_small_console(engine, regular, "w2")
    assert console is not None

    assert bedroom._item_intersects_window_clearance(regular, None) is False
    assert bedroom._item_intersects_window_clearance(regular, {"aabb": {"x_min": "bad"}}) is False
    zone = regular.window_clearance_zones[0]
    assert bedroom._item_intersects_window_clearance(
        regular,
        {"aabb": {"x_min": zone.x_min, "x_max": zone.x_max, "y_min": zone.y_min, "y_max": zone.y_max}},
    )
    removable = {"id": "remove-me"}
    engine.placements.append(removable)
    bedroom._remove_placement(engine, removable)
    assert removable not in engine.placements
    bedroom._remove_placement(engine, None)

    class RejectingEngine:
        def __init__(self):
            self.placements = [{"id": "snapshot"}]

        def add_wall_aligned(self, *args, **kwargs):
            return None

    rejecting = RejectingEngine()
    assert bedroom._place_short_wall_group(
        rejecting,
        regular.walls[1],
        bed_spec=BEDROOM_SPECS["single_bed"],
        nightstand_spec=BEDROOM_SPECS["nightstand"],
        wardrobe_spec=None,
        order=["bed", "nightstand"],
        from_start=True,
    ) is None
    assert rejecting.placements == [{"id": "snapshot"}]

    class FakeEngine:
        def __init__(self):
            self.placements = []
            self.wall_calls = 0
            self.near_calls = 0
            self.corner_calls = []
            self.on_top_calls = []

        def add_wall_aligned(self, spec, wall_id, along, **kwargs):
            del spec, wall_id, along, kwargs
            self.wall_calls += 1
            if self.wall_calls == 1:
                item = {"id": "desk", "aabb": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}}
                self.placements.append(item)
                return item
            return None

        def add_near(self, *args, **kwargs):
            del args, kwargs
            self.near_calls += 1
            return None

        def add_corner_object(self, spec, preferred_index=0, category=None):
            self.corner_calls.append((spec.category, preferred_index, category))
            item = {"id": f"corner_{len(self.corner_calls)}", "category": category or spec.category}
            self.placements.append(item)
            return item

        def add_on_top(self, *args, **kwargs):
            self.on_top_calls.append((args, kwargs))

    fake = FakeEngine()
    desk, chair = bedroom._fill_long_wall(fake, regular, "w2")
    assert desk is None and chair is None
    assert fake.near_calls == 1
    assert fake.corner_calls

    fake_extra = FakeEngine()
    monkeypatch.setattr(bedroom, "_try_wall_item", lambda *_args, **_kwargs: None)
    bedroom._add_extra_furniture(fake_extra, regular, "w2")
    assert fake_extra.corner_calls[0][2] == "armchair"

    no_wall_ctx = SimpleNamespace(walls=[])
    assert bedroom._far_bed_wall(no_wall_ctx) is None
    assert bedroom._generate_bedroom_greedy(no_wall_ctx, density="normal", seed=1)[1]["status"] == "no_wall"
