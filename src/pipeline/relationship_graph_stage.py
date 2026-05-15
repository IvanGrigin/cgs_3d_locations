#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/pipeline/relationship_graph_stage.py

Relationship graph stage for scene.v1 / placement.v1 compatible room scenes.

Purpose
-------
This file adds a deterministic relationship layer on top of an existing scene.v1
or placement.v1 JSON without breaking backward compatibility.

It does NOT require old pipeline stages to change their schema. It only:
- reads existing items / placements;
- infers semantic groups and zones from current fields;
- builds relationship_graph.nodes / relationship_graph.edges;
- adds rule-based relations such as:
  mug on_top_of desk,
  chair faces desk,
  coffee_table in_front_of sofa,
  pillow on_top_of bed;
- resolves relations to object ids;
- generates anchors;
- optionally applies light relation-aware placement repair to JSON AABBs;
- validates support/orientation/proximity relations;
- writes a new JSON with the original root schema preserved.

Typical CLI usage
-----------------
python3 src/pipeline/relationship_graph_stage.py \
  --input out/.../scene_supplier.optimal.v1.json \
  --out out/.../scene_supplier.optimal.relationships.v1.json \
  --prompt "Современная спальня студента с рабочим местом" \
  --apply-placement \
  --validate

Pipeline integration
--------------------
Use maybe_apply_relationship_graph_stage(...) after scene.v1 is built and before
final Blender render. The returned JSON is still a normal scene.v1 / placement.v1,
only enriched with additional fields.

Compatibility contract
----------------------
- Existing fields are preserved.
- Existing items list key is preserved: items or placements.
- Existing item ids are preserved.
- Existing aabb/bbox format is preserved and updated only if --apply-placement is enabled.
- Added metadata is stored under:
  root["relationship_graph"],
  root["relationship_stage"],
  item["meta"]["relationship_graph"].
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA = "relationship_graph_stage/v1"
GRAPH_SCHEMA = "object_relationship_graph/v1"

RELATION_TYPES = {
    "on_top_of",
    "inside",
    "under",
    "near",
    "next_to",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "faces",
    "against_wall",
    "mounted_on_wall",
    "centered_on",
    "aligned_with",
    "grouped_with",
    "around",
    "above",
    "below",
    "visible_from",
}

RELATION_CLASSES = {
    "support",
    "containment",
    "proximity",
    "orientation",
    "wall",
    "group",
    "semantic",
}

CONSTRAINT_LEVELS = {"hard", "soft", "decorative"}

FLOOR_PLACEMENT_GROUPS = {
    "bed",
    "desk",
    "dining_table",
    "kitchen_table",
    "coffee_table",
    "side_table",
    "nightstand",
    "dresser",
    "wardrobe",
    "shelf",
    "bookcase",
    "tv_stand",
    "chair",
    "office_chair",
    "dining_chair",
    "stool",
    "armchair",
    "sofa",
    "floor_lamp",
    "plant",
    "rug",
    "toilet",
    "sink",
    "bathtub",
    "shower",
    "kitchen_set",
}

TABLETOP_ACCESSORY_GROUPS = {
    "laptop",
    "computer",
    "monitor",
    "keyboard",
    "mouse",
    "mug",
    "cup",
    "water_bottle",
    "bottle",
    "book",
    "notebook",
    "phone",
    "remote",
    "plate",
    "bowl",
    "vase",
    "desk_lamp",
    "table_lamp",
    "desk_organizer",
    "decor_tray",
    "decor_box",
    "decor_books",
    "small_decor",
    "soap_dispenser",
    "toothbrush_cup",
}

BED_TOP_GROUPS = {
    "pillow",
    "blanket",
    "throw_blanket",
    "bedspread",
    "comforter",
}

WALL_MOUNTED_GROUPS = {
    "tv",
    "tv_projector_screen",
    "mirror",
    "wall_art",
    "wall_light",
    "shelf_wall",
    "wall_shelf",
}

CHAIR_GROUPS = {"chair", "office_chair", "dining_chair", "stool"}
SEAT_GROUPS = CHAIR_GROUPS | {"armchair", "sofa"}
TABLE_GROUPS = {"desk", "dining_table", "kitchen_table", "coffee_table", "side_table", "nightstand"}
STORAGE_GROUPS = {"wardrobe", "dresser", "shelf", "bookcase", "cabinet", "tv_stand"}
SUPPORT_SURFACE_GROUPS = TABLE_GROUPS | STORAGE_GROUPS | {"bed", "sofa", "sink"}

DEFAULT_FRONT_AXIS_BY_GROUP = {
    "chair": "+Y",
    "office_chair": "+Y",
    "dining_chair": "+Y",
    "stool": "+Y",
    "armchair": "+Y",
    "sofa": "+Y",
    "desk": "+Y",
    "dining_table": "+Y",
    "kitchen_table": "+Y",
    "coffee_table": "+Y",
    "side_table": "+Y",
    "nightstand": "+Y",
    "bed": "+Y",
    "wardrobe": "+Y",
    "dresser": "+Y",
    "shelf": "+Y",
    "bookcase": "+Y",
    "tv_stand": "+Y",
    "tv": "-Y",
    "tv_projector_screen": "-Y",
    "mirror": "-Y",
}

ZONE_ALIASES = {
    "work": "work_zone",
    "workspace": "work_zone",
    "office": "work_zone",
    "рабоч": "work_zone",
    "study": "work_zone",
    "sleep": "sleeping_zone",
    "bed": "sleeping_zone",
    "спаль": "sleeping_zone",
    "прикров": "sleeping_zone",
    "storage": "storage_zone",
    "хран": "storage_zone",
    "wardrobe": "storage_zone",
    "living": "living_zone",
    "гостин": "living_zone",
    "sofa": "living_zone",
    "tv": "living_zone",
    "dining": "dining_zone",
    "обеден": "dining_zone",
    "kitchen": "kitchen_zone",
    "кух": "kitchen_zone",
    "bath": "bathroom_zone",
    "wc": "bathroom_zone",
    "toilet": "bathroom_zone",
    "сануз": "bathroom_zone",
    "ванн": "bathroom_zone",
    "entry": "entry_zone",
    "corridor": "entry_zone",
    "прихож": "entry_zone",
}

SEMANTIC_RULES: list[tuple[str, str]] = [
    (r"office[_\s-]*chair|computer[_\s-]*chair|кресло компьютер|офисн.*кресл|рабоч.*кресл", "office_chair"),
    (r"dining[_\s-]*chair|обеден.*стул", "dining_chair"),
    (r"\bchair\b|стул", "chair"),
    (r"\bstool\b|табурет|барн.*стул", "stool"),
    (r"armchair|кресло", "armchair"),
    (r"\bsofa\b|couch|диван", "sofa"),
    (r"coffee[_\s-]*table|журнальн.*стол", "coffee_table"),
    (r"side[_\s-]*table|приставн.*стол|столик", "side_table"),
    (r"dining[_\s-]*table|обеден.*стол", "dining_table"),
    (r"kitchen[_\s-]*table|кухон.*стол", "kitchen_table"),
    (r"\bdesk\b|work[_\s-]*desk|computer[_\s-]*desk|письменн.*стол|рабоч.*стол|компьютерн.*стол", "desk"),
    (r"\btable\b|\bстол\b", "dining_table"),
    (r"nightstand|bedside|прикроват.*тумб|тумб.*прикров", "nightstand"),
    (r"dresser|комод", "dresser"),
    (r"wardrobe|closet|шкаф", "wardrobe"),
    (r"bookcase|shelving|rack|стеллаж|полк", "shelf"),
    (r"tv[_\s-]*stand|тумб.*телевиз|тумб.*tv", "tv_stand"),
    (r"\btv\b|television|телевизор|projector[_\s-]*screen", "tv_projector_screen"),
    (r"monitor|imac|computer[_\s-]*screen|монитор", "monitor"),
    (r"laptop|notebook computer|ноутбук|macbook", "laptop"),
    (r"keyboard|клавиатур", "keyboard"),
    (r"mouse|мыш", "mouse"),
    (r"mug|coffee[_\s-]*cup|кружк", "mug"),
    (r"\bcup\b|чашк", "cup"),
    (r"water[_\s-]*bottle|бутыл.*вод|бутыл", "water_bottle"),
    (r"\bbook\b|книг", "book"),
    (r"notebook|тетрад|блокнот", "notebook"),
    (r"phone|smartphone|телефон", "phone"),
    (r"remote|пульт", "remote"),
    (r"desk[_\s-]*lamp|table[_\s-]*lamp|настольн.*ламп", "table_lamp"),
    (r"floor[_\s-]*lamp|торшер", "floor_lamp"),
    (r"wall[_\s-]*light|sconce|бра|настенн.*свет", "wall_light"),
    (r"ceiling[_\s-]*light|pendant|chandelier|люстр|потолочн.*свет", "ceiling_light"),
    (r"\bbed\b|кровать", "bed"),
    (r"pillow|подуш", "pillow"),
    (r"blanket|comforter|bedspread|одеял|плед|покрывал", "blanket"),
    (r"\brug\b|carpet|ков[её]р", "rug"),
    (r"mirror|зеркал", "mirror"),
    (r"wall[_\s-]*art|painting|poster|картина|постер|панно", "wall_art"),
    (r"plant|potted[_\s-]*plant|растен|цветок", "plant"),
    (r"vase|ваза", "vase"),
    (r"plate|тарел", "plate"),
    (r"bowl|миска|чаша", "bowl"),
    (r"toilet|унитаз", "toilet"),
    (r"sink|basin|раков|умыв", "sink"),
    (r"bathtub|bath|ванн", "bathtub"),
    (r"shower|душ", "shower"),
    (r"soap[_\s-]*dispenser|дозатор", "soap_dispenser"),
    (r"toothbrush|щетк", "toothbrush_cup"),
    (r"kitchen[_\s-]*set|кухон.*гарнитур|procedural_kitchen", "kitchen_set"),
]

ROOM_CENTER_ID = "__room_center__"
WALL_TARGET_ID = "__wall__"


@dataclass
class AABB:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_any(cls, value: Any) -> Optional["AABB"]:
        if not isinstance(value, dict):
            return None
        keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        if not all(k in value for k in keys):
            return None
        try:
            return cls(*(float(value[k]) for k in keys))
        except Exception:
            return None

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def depth(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def height(self) -> float:
        return max(0.0, self.z_max - self.z_min)

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
            0.5 * (self.z_min + self.z_max),
        )

    @property
    def center_xy(self) -> tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    @property
    def area_xy(self) -> float:
        return self.width * self.depth

    @property
    def volume(self) -> float:
        return self.width * self.depth * self.height

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": round(self.x_min, 6),
            "x_max": round(self.x_max, 6),
            "y_min": round(self.y_min, 6),
            "y_max": round(self.y_max, 6),
            "z_min": round(self.z_min, 6),
            "z_max": round(self.z_max, 6),
        }

    def moved_center_xy(self, x: float, y: float) -> "AABB":
        w = self.width
        d = self.depth
        return AABB(
            x_min=float(x) - 0.5 * w,
            x_max=float(x) + 0.5 * w,
            y_min=float(y) - 0.5 * d,
            y_max=float(y) + 0.5 * d,
            z_min=self.z_min,
            z_max=self.z_max,
        )

    def moved_bottom_z(self, z_min: float) -> "AABB":
        h = self.height
        return AABB(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=float(z_min),
            z_max=float(z_min) + h,
        )

    def translated(self, dx: float, dy: float, dz: float = 0.0) -> "AABB":
        return AABB(
            self.x_min + dx,
            self.x_max + dx,
            self.y_min + dy,
            self.y_max + dy,
            self.z_min + dz,
            self.z_max + dz,
        )

    def contains_xy(self, other: "AABB", margin: float = 0.0) -> bool:
        return (
            other.x_min >= self.x_min - margin
            and other.x_max <= self.x_max + margin
            and other.y_min >= self.y_min - margin
            and other.y_max <= self.y_max + margin
        )

    def overlap_xy_area(self, other: "AABB") -> float:
        ox = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        oy = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        return ox * oy

    def overlap_volume(self, other: "AABB") -> float:
        ox = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        oy = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        oz = max(0.0, min(self.z_max, other.z_max) - max(self.z_min, other.z_min))
        return ox * oy * oz


@dataclass
class ItemRef:
    object_id: str
    raw: dict[str, Any]
    index: int
    zone_id: str
    semantic_group: str
    role: str
    aabb: Optional[AABB]
    yaw_deg: float
    front_axis_local: str
    placement_type: str
    source: str = "scene"


@dataclass
class RelationEdge:
    from_object_id: str
    relation_type: str
    to_object_id: str
    relation_class: str
    constraint_level: str = "hard"
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "rule_based"
    reason: str = ""

    def edge_id(self) -> str:
        base = f"rel_{self.from_object_id}_{self.relation_type}_{self.to_object_id}"
        return _safe_id(base)

    def key(self) -> tuple[str, str, str]:
        return (self.from_object_id, self.relation_type, self.to_object_id)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.edge_id(),
            "from_object_id": self.from_object_id,
            "relation_type": self.relation_type,
            "relation_class": self.relation_class,
            "to_object_id": self.to_object_id,
            "constraint_level": self.constraint_level,
            "required": self.constraint_level == "hard",
            "weight": float(self.weight),
            "params": self.params,
            "source": self.source,
        }
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass
class StageOptions:
    apply_placement: bool = False
    validate: bool = True
    repair: bool = True
    add_missing_supports: bool = False
    preserve_existing_aabbs: bool = False
    min_relation_score: float = 0.75
    verbose: bool = False


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _safe_id(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace(".", "_")
    text = re.sub(r"[^0-9A-Za-zА-Яа-я_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "id"


def _norm_text(value: Any) -> str:
    text = str(value or "").lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я_\-]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:[\.,]\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except Exception:
        return None


def _deep_text_values(value: Any, limit: int = 80) -> list[str]:
    out: list[str] = []

    def walk(v: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, (str, int, float)) and not isinstance(v, bool):
            s = str(v).strip()
            if s:
                out.append(s)

    walk(value)
    return out


def _item_text(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    asset = item.get("asset") if isinstance(item.get("asset"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    parts: list[Any] = [
        item.get("id"),
        item.get("name"),
        item.get("category"),
        item.get("semantic_group"),
        item.get("type"),
        item.get("subclass"),
        item.get("label_ru"),
        item.get("label_en"),
        candidate.get("title"),
        candidate.get("semantic_group"),
        candidate.get("category_norm"),
        candidate.get("category_raw"),
        asset.get("kind"),
        asset.get("semantic_group"),
        source.get("asset_source"),
    ]
    return _norm_text(" ".join(str(x or "") for x in parts))


def _semantic_group_from_item(item: dict[str, Any]) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidate = meta.get("supplier_candidate") if isinstance(meta.get("supplier_candidate"), dict) else {}
    explicit = str(
        item.get("semantic_group")
        or item.get("subclass")
        or candidate.get("semantic_group")
        or ""
    ).strip().lower()
    explicit = explicit.replace("-", "_").replace(" ", "_")
    explicit_aliases = {
        "lamp_table": "table_lamp",
        "lamp_floor": "floor_lamp",
        "lamp_ceiling": "ceiling_light",
        "tv_projector_screen": "tv_projector_screen",
        "computer_chair": "office_chair",
        "work_chair": "office_chair",
        "bedside_table": "nightstand",
        "bookshelf": "shelf",
    }
    if explicit in explicit_aliases:
        return explicit_aliases[explicit]
    if explicit and explicit not in {"furniture", "decor", "decor_accessory", "other", "unknown", "none"}:
        return explicit

    text = _item_text(item)
    for pattern, group in SEMANTIC_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return group
    return "unknown"


def _role_from_item(item: dict[str, Any], group: str) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    raw = str(item.get("role") or meta.get("role") or meta.get("required_role") or "").strip().lower()
    if raw in {"main", "primary", "secondary", "accessory", "decor", "surface"}:
        return {"primary": "main", "decor": "accessory"}.get(raw, raw)
    if group in {"bed", "desk", "dining_table", "kitchen_table", "sofa", "wardrobe", "kitchen_set", "toilet", "sink"}:
        return "main"
    if group in TABLETOP_ACCESSORY_GROUPS or group in BED_TOP_GROUPS or group in WALL_MOUNTED_GROUPS:
        return "accessory"
    if group in {"nightstand", "chair", "office_chair", "dining_chair", "coffee_table", "tv_stand", "dresser", "floor_lamp"}:
        return "secondary"
    return "secondary"


def _yaw_deg_from_item(item: dict[str, Any]) -> float:
    for key in ("yaw_deg", "rotation_deg", "rotation"):
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            if len(value) >= 3:
                return float(value[2] or 0.0) % 360.0
            if value:
                return float(value[0] or 0.0) % 360.0
        value_float = _float_or_none(value)
        if value_float is not None:
            return value_float % 360.0
    return 0.0


def _set_yaw_deg_on_item(item: dict[str, Any], yaw_deg: float) -> None:
    yaw = float(yaw_deg or 0.0) % 360.0
    if isinstance(item.get("rotation"), list):
        rot = list(item.get("rotation") or [])
        while len(rot) < 3:
            rot.append(0.0)
        rot[2] = round(yaw, 6)
        item["rotation"] = rot
    elif "yaw_deg" in item:
        item["yaw_deg"] = round(yaw, 6)
    elif "rotation_deg" in item:
        item["rotation_deg"] = round(yaw, 6)
    else:
        item["yaw_deg"] = round(yaw, 6)


def _placement_type_from_item(item: dict[str, Any], group: str, role: str) -> str:
    raw = str(item.get("placement_type") or "").strip().lower()
    if raw:
        return raw
    constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
    mount = str(constraints.get("mount_type") or "").strip().lower()
    if mount == "wall" or group in WALL_MOUNTED_GROUPS:
        return "wall"
    if mount == "ceiling" or group == "ceiling_light":
        return "ceiling"
    if group in TABLETOP_ACCESSORY_GROUPS or group in BED_TOP_GROUPS:
        return "on_furniture"
    if group == "rug":
        return "floor"
    if group in FLOOR_PLACEMENT_GROUPS or role in {"main", "secondary"}:
        return "floor"
    return "floor"


def _zone_from_item(item: dict[str, Any], group: str, prompt: str = "") -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    raw_values = [
        item.get("zone_id"),
        item.get("zone"),
        meta.get("zone_id"),
        meta.get("zone"),
        meta.get("room_zone"),
        item.get("room_zone"),
    ]
    text = _norm_text(" ".join(str(x or "") for x in raw_values))
    for token, zone in ZONE_ALIASES.items():
        if token in text:
            return zone

    if group in {"desk", "office_chair", "laptop", "monitor", "keyboard", "mouse", "desk_organizer"}:
        return "work_zone"
    if group in {"bed", "nightstand", "pillow", "blanket"}:
        return "sleeping_zone"
    if group in {"wardrobe", "dresser", "shelf", "bookcase"}:
        return "storage_zone"
    if group in {"sofa", "coffee_table", "tv_projector_screen", "tv", "tv_stand", "remote", "armchair"}:
        return "living_zone"
    if group in {"dining_table", "kitchen_table", "dining_chair", "plate", "bowl", "cup", "vase"}:
        prompt_norm = _norm_text(prompt)
        if "кух" in prompt_norm or "kitchen" in prompt_norm:
            return "kitchen_zone"
        return "dining_zone"
    if group in {"kitchen_set"}:
        return "kitchen_zone"
    if group in {"toilet", "sink", "bathtub", "shower", "soap_dispenser", "toothbrush_cup"}:
        return "bathroom_zone"
    return "decor_zone"


def _front_axis_for_group(group: str) -> str:
    return DEFAULT_FRONT_AXIS_BY_GROUP.get(group, "+Y")


def _items_key(data: dict[str, Any]) -> str:
    if isinstance(data.get("items"), list):
        return "items"
    if isinstance(data.get("placements"), list):
        return "placements"
    data.setdefault("items", [])
    return "items"


def _aabb_from_item(item: dict[str, Any]) -> Optional[AABB]:
    return AABB.from_any(item.get("aabb")) or AABB.from_any(item.get("bbox"))


def _write_aabb_to_item(item: dict[str, Any], aabb: AABB) -> None:
    key = "aabb" if isinstance(item.get("aabb"), dict) else "bbox" if isinstance(item.get("bbox"), dict) else "aabb"
    item[key] = aabb.to_dict()
    if key != "aabb" and isinstance(item.get("aabb"), dict):
        item["aabb"] = aabb.to_dict()

    cx, cy, cz = aabb.center
    if isinstance(item.get("position"), list):
        pos = list(item.get("position") or [])
        while len(pos) < 3:
            pos.append(0.0)
        pos[0] = round(cx, 6)
        pos[1] = round(cy, 6)
        pos[2] = round(aabb.z_min, 6)
        item["position"] = pos
    elif isinstance(item.get("position"), dict):
        pos = dict(item.get("position") or {})
        pos["x"] = round(cx, 6)
        pos["y"] = round(cy, 6)
        pos["z"] = round(aabb.z_min, 6)
        item["position"] = pos


def _room_polygon(data: dict[str, Any]) -> list[tuple[float, float]]:
    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    poly = room.get("floor_polygon") or room.get("floor_polygon_xz") or []
    out: list[tuple[float, float]] = []
    if isinstance(poly, list):
        for p in poly:
            if isinstance(p, dict):
                x = _float_or_none(p.get("x"))
                y = _float_or_none(p.get("y", p.get("z")))
                if x is not None and y is not None:
                    out.append((x, y))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x = _float_or_none(p[0])
                y = _float_or_none(p[1])
                if x is not None and y is not None:
                    out.append((x, y))
    return out


def _room_bounds_from_data(data: dict[str, Any], items: list[ItemRef]) -> AABB:
    poly = _room_polygon(data)
    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    z_min = _float_or_none(room.get("floor_z", room.get("z_min")))
    if z_min is None:
        z_min = 0.0
    z_max = _float_or_none(room.get("z_max"))
    if z_max is None:
        height = _float_or_none(room.get("ceiling_height", room.get("ceiling_height_m"))) or 2.8
        z_max = z_min + height

    if len(poly) >= 3:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return AABB(min(xs), max(xs), min(ys), max(ys), z_min, z_max)

    item_aabbs = [it.aabb for it in items if it.aabb is not None]
    if item_aabbs:
        return AABB(
            min(a.x_min for a in item_aabbs),
            max(a.x_max for a in item_aabbs),
            min(a.y_min for a in item_aabbs),
            max(a.y_max for a in item_aabbs),
            z_min,
            z_max,
        )
    return AABB(0.0, 5.0, 0.0, 5.0, z_min, z_max)


def _distance_xy(a: AABB, b: AABB) -> float:
    ax, ay = a.center_xy
    bx, by = b.center_xy
    return math.hypot(ax - bx, ay - by)


def _angle_deg_from_to(a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b_xy[1] - a_xy[1], b_xy[0] - a_xy[0])) % 360.0


def _axis_offset_deg(front_axis: str) -> float:
    axis = str(front_axis or "+Y").strip().upper()
    return {
        "+X": 0.0,
        "+Y": 90.0,
        "-X": 180.0,
        "-Y": 270.0,
    }.get(axis, 90.0)


def _yaw_to_face(source: ItemRef, target_xy: tuple[float, float]) -> float:
    if source.aabb is None:
        return source.yaw_deg
    angle = _angle_deg_from_to(source.aabb.center_xy, target_xy)
    return (angle - _axis_offset_deg(source.front_axis_local)) % 360.0


def _angle_delta_abs(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def _unit_vec_from_yaw(yaw_deg: float, front_axis: str = "+Y") -> tuple[float, float]:
    angle = math.radians((float(yaw_deg) + _axis_offset_deg(front_axis)) % 360.0)
    return (math.cos(angle), math.sin(angle))


# -----------------------------------------------------------------------------
# Scene item collection
# -----------------------------------------------------------------------------

def collect_items(data: dict[str, Any], prompt: str = "") -> list[ItemRef]:
    key = _items_key(data)
    raw_items = data.get(key) or []
    if not isinstance(raw_items, list):
        return []

    out: list[ItemRef] = []
    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        object_id = str(raw.get("id") or raw.get("object_id") or raw.get("name") or f"obj_{idx + 1:03d}").strip()
        object_id = _safe_id(object_id)
        raw["id"] = object_id
        group = _semantic_group_from_item(raw)
        role = _role_from_item(raw, group)
        zone = _zone_from_item(raw, group, prompt=prompt)
        aabb = _aabb_from_item(raw)
        yaw_deg = _yaw_deg_from_item(raw)
        front_axis = str(raw.get("front_axis_local") or _front_axis_for_group(group))
        placement_type = _placement_type_from_item(raw, group, role)
        out.append(
            ItemRef(
                object_id=object_id,
                raw=raw,
                index=idx,
                zone_id=zone,
                semantic_group=group,
                role=role,
                aabb=aabb,
                yaw_deg=yaw_deg,
                front_axis_local=front_axis,
                placement_type=placement_type,
            )
        )
    return out


def _items_by_id(items: list[ItemRef]) -> dict[str, ItemRef]:
    return {it.object_id: it for it in items}


def _items_by_group(items: list[ItemRef], zone_id: Optional[str] = None) -> dict[str, list[ItemRef]]:
    out: dict[str, list[ItemRef]] = {}
    for item in items:
        if zone_id and item.zone_id != zone_id:
            continue
        out.setdefault(item.semantic_group, []).append(item)
    return out


def _find_items(
    items: list[ItemRef],
    groups: Iterable[str],
    *,
    zone_id: Optional[str] = None,
    role_preference: Optional[str] = None,
) -> list[ItemRef]:
    group_set = set(groups)
    candidates = [it for it in items if it.semantic_group in group_set and (zone_id is None or it.zone_id == zone_id)]
    if role_preference:
        preferred = [it for it in candidates if it.role == role_preference]
        if preferred:
            candidates = preferred
    return candidates


def _best_target_for(
    source: ItemRef,
    items: list[ItemRef],
    target_groups: Iterable[str],
    *,
    allow_cross_zone: bool = False,
) -> Optional[ItemRef]:
    groups = set(target_groups)
    candidates = [it for it in items if it.object_id != source.object_id and it.semantic_group in groups]
    same_zone = [it for it in candidates if it.zone_id == source.zone_id]
    if same_zone:
        candidates = same_zone
    elif not allow_cross_zone:
        return None
    if not candidates:
        return None

    def score(target: ItemRef) -> tuple[int, int, float, int]:
        role_rank = {"main": 0, "secondary": 1, "accessory": 2}.get(target.role, 3)
        has_aabb = 0 if target.aabb is not None else 1
        dist = _distance_xy(source.aabb, target.aabb) if source.aabb and target.aabb else 9999.0
        return (role_rank, has_aabb, dist, target.index)

    return min(candidates, key=score)


def _dedupe_edges(edges: list[RelationEdge]) -> list[RelationEdge]:
    by_key: dict[tuple[str, str, str], RelationEdge] = {}
    for edge in edges:
        if edge.relation_type not in RELATION_TYPES:
            continue
        if edge.relation_class not in RELATION_CLASSES:
            edge.relation_class = _class_for_relation(edge.relation_type)
        if edge.constraint_level not in CONSTRAINT_LEVELS:
            edge.constraint_level = "hard"
        key = edge.key()
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = edge
            continue
        if _constraint_rank(edge.constraint_level) < _constraint_rank(prev.constraint_level):
            prev.constraint_level = edge.constraint_level
        prev.weight = max(float(prev.weight), float(edge.weight))
        prev.params = {**prev.params, **edge.params}
        if edge.source not in prev.source:
            prev.source = "llm_and_rule" if "llm" in (edge.source + prev.source) else f"{prev.source}+{edge.source}"
    return list(by_key.values())


def _constraint_rank(level: str) -> int:
    return {"hard": 0, "soft": 1, "decorative": 2}.get(level, 3)


def _class_for_relation(relation_type: str) -> str:
    if relation_type in {"on_top_of", "under", "above", "below"}:
        return "support"
    if relation_type == "inside":
        return "containment"
    if relation_type in {"faces", "visible_from"}:
        return "orientation"
    if relation_type in {"against_wall", "mounted_on_wall"}:
        return "wall"
    if relation_type in {"grouped_with", "around", "aligned_with", "centered_on"}:
        return "group"
    return "proximity"


def _edge(
    source: ItemRef,
    relation_type: str,
    target: ItemRef | str,
    *,
    level: str = "hard",
    weight: float = 1.0,
    params: Optional[dict[str, Any]] = None,
    source_name: str = "rule_based",
    reason: str = "",
) -> RelationEdge:
    target_id = target.object_id if isinstance(target, ItemRef) else str(target)
    return RelationEdge(
        from_object_id=source.object_id,
        relation_type=relation_type,
        relation_class=_class_for_relation(relation_type),
        to_object_id=target_id,
        constraint_level=level,
        weight=weight,
        params=params or {},
        source=source_name,
        reason=reason,
    )


# -----------------------------------------------------------------------------
# Rule-based relationship augmentation
# -----------------------------------------------------------------------------

def build_rule_based_edges(items: list[ItemRef], room_bounds: AABB) -> list[RelationEdge]:
    edges: list[RelationEdge] = []

    for item in items:
        group = item.semantic_group

        if group in {"office_chair", "chair"} and item.zone_id == "work_zone":
            target = _best_target_for(item, items, {"desk"})
            if target:
                edges.append(
                    _edge(
                        item,
                        "in_front_of",
                        target,
                        params={"distance_m": {"min": 0.15, "max": 0.55}, "side_preference": "front"},
                        reason="Work chair must be placed in front of a desk.",
                    )
                )
                edges.append(
                    _edge(
                        item,
                        "faces",
                        target,
                        params={"tolerance_deg": 25},
                        reason="Work chair must face the desk.",
                    )
                )

        if group in {"dining_chair", "chair", "stool"} and item.zone_id in {"dining_zone", "kitchen_zone"}:
            target = _best_target_for(item, items, {"dining_table", "kitchen_table"})
            if target:
                edges.append(
                    _edge(
                        item,
                        "around",
                        target,
                        params={"distribution": "evenly_around", "faces_center": True},
                        reason="Dining/kitchen chairs must be arranged around the table.",
                    )
                )
                edges.append(
                    _edge(
                        item,
                        "faces",
                        target,
                        params={"tolerance_deg": 30},
                        reason="Dining/kitchen chair must face table center.",
                    )
                )

        if group in TABLETOP_ACCESSORY_GROUPS:
            target_groups = _support_targets_for_accessory(item)
            target = _best_target_for(item, items, target_groups, allow_cross_zone=False)
            if target:
                edges.append(
                    _edge(
                        item,
                        "on_top_of",
                        target,
                        params={"surface": "top", "placement_area": _default_placement_area(item, target)},
                        reason="Small accessory must be supported by a tabletop or furniture surface.",
                    )
                )

        if group in BED_TOP_GROUPS:
            target = _best_target_for(item, items, {"bed"})
            if target:
                edges.append(
                    _edge(
                        item,
                        "on_top_of",
                        target,
                        params={"surface": "top", "placement_area": _default_placement_area(item, target)},
                        reason="Bed textile must lie on a bed.",
                    )
                )

        if group == "nightstand":
            target = _best_target_for(item, items, {"bed"})
            if target:
                side = _infer_side_preference(item, target)
                edges.append(
                    _edge(
                        item,
                        "next_to",
                        target,
                        params={"distance_m": {"min": 0.05, "max": 0.35}, "side_preference": side},
                        reason="Nightstand should be beside the bed near the headboard.",
                    )
                )

        if group == "coffee_table":
            target = _best_target_for(item, items, {"sofa", "armchair"}, allow_cross_zone=False)
            if target:
                edges.append(
                    _edge(
                        item,
                        "in_front_of",
                        target,
                        params={"distance_m": {"min": 0.35, "max": 0.9}, "side_preference": "front"},
                        reason="Coffee table belongs in front of a sofa or armchair.",
                    )
                )
                edges.append(
                    _edge(item, "centered_on", target, level="soft", weight=0.55)
                )

        if group == "sofa":
            target = _sofa_orientation_target(item, items)
            if target:
                edges.append(
                    _edge(
                        item,
                        "faces",
                        target,
                        params={"tolerance_deg": 35},
                        reason="Sofa should face TV, fireplace, coffee table, or room center.",
                    )
                )
            else:
                edges.append(
                    _edge(
                        item,
                        "faces",
                        ROOM_CENTER_ID,
                        params={"target_xy": list(room_bounds.center_xy), "tolerance_deg": 45},
                        reason="No TV or coffee table found; sofa faces room center.",
                    )
                )

        if group in {"tv_projector_screen", "tv"}:
            target = _best_target_for(item, items, {"sofa", "armchair"}, allow_cross_zone=True)
            edges.append(
                _edge(
                    item,
                    "mounted_on_wall",
                    WALL_TARGET_ID,
                    params={"preferred_height_m": 1.2},
                    reason="TV is normally wall-mounted or placed against a wall.",
                )
            )
            if target:
                edges.append(
                    _edge(
                        item,
                        "visible_from",
                        target,
                        level="soft",
                        weight=0.8,
                        params={"tolerance_deg": 45},
                    )
                )

        if group == "tv_stand":
            edges.append(_edge(item, "against_wall", WALL_TARGET_ID, params={"clearance_front_m": 0.6}))
            target = _best_target_for(item, items, {"tv_projector_screen", "tv"}, allow_cross_zone=True)
            if target:
                edges.append(_edge(target, "above", item, level="soft", weight=0.7))

        if group in {"wardrobe", "dresser", "shelf", "bookcase"}:
            edges.append(
                _edge(
                    item,
                    "against_wall",
                    WALL_TARGET_ID,
                    params={"clearance_front_m": 0.65 if group == "wardrobe" else 0.45},
                    reason="Storage furniture should stand against a wall.",
                )
            )

        if group == "mirror":
            target = _best_target_for(item, items, {"dresser", "sink"}, allow_cross_zone=False)
            if target:
                edges.append(_edge(item, "above", target, params={"vertical_gap_m": {"min": 0.05, "max": 0.45}}))
            else:
                edges.append(_edge(item, "mounted_on_wall", WALL_TARGET_ID, params={"preferred_height_m": 1.45}))

        if group == "wall_art":
            target = _best_target_for(item, items, {"bed", "sofa", "desk", "dresser"}, allow_cross_zone=False)
            if target:
                edges.append(_edge(item, "above", target, level="soft", weight=0.65))
            edges.append(_edge(item, "mounted_on_wall", WALL_TARGET_ID, level="hard", params={"preferred_height_m": 1.55}))

        if group == "rug":
            target = _best_target_for(item, items, {"bed", "dining_table", "kitchen_table", "coffee_table", "sofa"}, allow_cross_zone=False)
            if target:
                edges.append(
                    _edge(
                        item,
                        "under",
                        target,
                        level="soft",
                        weight=0.7,
                        params={"coverage": "partial_or_group"},
                    )
                )

        if group == "plant":
            edges.append(
                _edge(
                    item,
                    "near",
                    WALL_TARGET_ID,
                    level="decorative",
                    weight=0.25,
                    params={"preference": "window_or_free_corner"},
                    reason="Plants look natural near a window or free corner.",
                )
            )

        if group == "toilet":
            edges.append(_edge(item, "against_wall", WALL_TARGET_ID, params={"clearance_front_m": 0.55}))
        if group == "sink":
            edges.append(_edge(item, "against_wall", WALL_TARGET_ID, params={"clearance_front_m": 0.45}))
        if group == "soap_dispenser":
            target = _best_target_for(item, items, {"sink"}, allow_cross_zone=False)
            if target:
                edges.append(_edge(item, "on_top_of", target, params={"surface": "top", "placement_area": "side"}))
        if group == "toothbrush_cup":
            target = _best_target_for(item, items, {"sink"}, allow_cross_zone=False)
            if target:
                edges.append(_edge(item, "on_top_of", target, params={"surface": "top", "placement_area": "back"}))

    return _dedupe_edges(edges)


def _support_targets_for_accessory(item: ItemRef) -> set[str]:
    group = item.semantic_group
    if item.zone_id == "work_zone":
        if group in {"laptop", "monitor", "keyboard", "mouse", "mug", "cup", "water_bottle", "notebook", "book", "phone", "desk_lamp", "table_lamp", "desk_organizer"}:
            return {"desk"}
    if item.zone_id in {"dining_zone", "kitchen_zone"}:
        if group in {"plate", "bowl", "cup", "mug", "vase", "water_bottle", "bottle", "book"}:
            return {"dining_table", "kitchen_table", "coffee_table"}
    if item.zone_id == "living_zone":
        if group in {"mug", "cup", "book", "remote", "vase", "decor_tray", "water_bottle"}:
            return {"coffee_table", "side_table", "tv_stand"}
    if item.zone_id == "sleeping_zone":
        if group in {"table_lamp", "desk_lamp", "book", "phone", "mug", "water_bottle"}:
            return {"nightstand", "dresser", "desk"}
    if item.zone_id == "storage_zone":
        if group in {"book", "decor_books", "decor_box", "vase", "small_decor"}:
            return {"shelf", "bookcase", "dresser", "wardrobe"}
    if item.zone_id == "bathroom_zone":
        if group in {"soap_dispenser", "toothbrush_cup", "cup"}:
            return {"sink"}
    return set(TABLE_GROUPS | STORAGE_GROUPS)


def _default_placement_area(item: ItemRef, target: ItemRef) -> str:
    group = item.semantic_group
    target_group = target.semantic_group
    if target_group == "desk":
        mapping = {
            "laptop": "center",
            "monitor": "back_center",
            "keyboard": "front_center",
            "mouse": "right_front",
            "mug": "right_front",
            "water_bottle": "back_right",
            "desk_lamp": "back_left",
            "table_lamp": "back_left",
            "notebook": "left_front",
            "book": "left_front",
            "phone": "right_front",
        }
        return mapping.get(group, "center")
    if target_group == "nightstand":
        mapping = {
            "table_lamp": "center",
            "desk_lamp": "center",
            "book": "front",
            "phone": "front",
            "mug": "side",
            "water_bottle": "side",
        }
        return mapping.get(group, "center")
    if target_group == "bed":
        mapping = {
            "pillow": "pillow_area",
            "blanket": "center",
            "book": "side",
        }
        return mapping.get(group, "center")
    if target_group in {"dining_table", "kitchen_table"}:
        mapping = {
            "vase": "center",
            "plate": "front",
            "bowl": "front",
            "cup": "side",
            "mug": "side",
        }
        return mapping.get(group, "center")
    if target_group == "coffee_table":
        mapping = {
            "remote": "center",
            "book": "left_front",
            "mug": "right_front",
            "vase": "center",
        }
        return mapping.get(group, "center")
    return "center"


def _infer_side_preference(item: ItemRef, target: ItemRef) -> str:
    if not item.aabb or not target.aabb:
        return "auto"
    ix, iy = item.aabb.center_xy
    tx, ty = target.aabb.center_xy
    dx = ix - tx
    dy = iy - ty
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "front" if dy > 0 else "back"


def _sofa_orientation_target(item: ItemRef, items: list[ItemRef]) -> Optional[ItemRef]:
    for groups in ({"tv_projector_screen", "tv"}, {"fireplace"}, {"coffee_table"}, {"dining_table", "kitchen_table"}):
        target = _best_target_for(item, items, groups, allow_cross_zone=True)
        if target:
            return target
    return None


# -----------------------------------------------------------------------------
# Anchors
# -----------------------------------------------------------------------------

def generate_anchors_for_item(item: ItemRef) -> dict[str, dict[str, float]]:
    a = item.aabb
    if a is None:
        return {}
    w = a.width
    d = a.depth
    h = a.height

    def p(x: float, y: float, z: float) -> dict[str, float]:
        return {"x": round(float(x), 6), "y": round(float(y), 6), "z": round(float(z), 6)}

    anchors: dict[str, dict[str, float]] = {
        "center": p(0.0, 0.0, h * 0.5),
        "top.center": p(0.0, 0.0, h),
        "front.center": p(0.0, d * 0.5 + 0.25, 0.0),
        "back.center": p(0.0, -d * 0.5 - 0.10, 0.0),
        "left.center": p(-w * 0.5 - 0.10, 0.0, 0.0),
        "right.center": p(w * 0.5 + 0.10, 0.0, 0.0),
    }

    top_margin_x = min(max(w * 0.18, 0.06), max(w * 0.42, 0.06))
    top_margin_y = min(max(d * 0.18, 0.06), max(d * 0.42, 0.06))
    anchors.update(
        {
            "top.left_front": p(-w * 0.5 + top_margin_x, d * 0.5 - top_margin_y, h),
            "top.right_front": p(w * 0.5 - top_margin_x, d * 0.5 - top_margin_y, h),
            "top.back_left": p(-w * 0.5 + top_margin_x, -d * 0.5 + top_margin_y, h),
            "top.back_right": p(w * 0.5 - top_margin_x, -d * 0.5 + top_margin_y, h),
            "top.front_center": p(0.0, d * 0.5 - top_margin_y, h),
            "top.back_center": p(0.0, -d * 0.5 + top_margin_y, h),
        }
    )

    if item.semantic_group == "bed":
        anchors.update(
            {
                "top.pillow_left": p(-w * 0.25, -d * 0.34, h),
                "top.pillow_right": p(w * 0.25, -d * 0.34, h),
                "top.blanket_center": p(0.0, d * 0.12, h),
                "left_side.nightstand": p(-w * 0.5 - 0.32, -d * 0.28, 0.0),
                "right_side.nightstand": p(w * 0.5 + 0.32, -d * 0.28, 0.0),
                "foot.rug": p(0.0, d * 0.55, 0.0),
            }
        )
    elif item.semantic_group == "sofa":
        anchors.update(
            {
                "front.coffee_table": p(0.0, d * 0.5 + 0.55, 0.0),
                "left_side.side_table": p(-w * 0.5 - 0.35, 0.0, 0.0),
                "right_side.side_table": p(w * 0.5 + 0.35, 0.0, 0.0),
                "left_side.floor_lamp": p(-w * 0.5 - 0.45, -d * 0.10, 0.0),
                "right_side.floor_lamp": p(w * 0.5 + 0.45, -d * 0.10, 0.0),
            }
        )
    elif item.semantic_group in {"desk", "dining_table", "kitchen_table", "coffee_table", "side_table", "nightstand"}:
        anchors.update(
            {
                "chair.front": p(0.0, d * 0.5 + 0.36, 0.0),
                "chair.left": p(-w * 0.5 - 0.32, 0.0, 0.0),
                "chair.right": p(w * 0.5 + 0.32, 0.0, 0.0),
                "chair.back": p(0.0, -d * 0.5 - 0.32, 0.0),
            }
        )
    return anchors


def _local_to_world_anchor(item: ItemRef, local: dict[str, float]) -> tuple[float, float, float]:
    if item.aabb is None:
        return (0.0, 0.0, 0.0)
    cx, cy, cz = item.aabb.center
    yaw = math.radians(item.yaw_deg)
    lx = float(local.get("x", 0.0))
    ly = float(local.get("y", 0.0))
    lz = float(local.get("z", 0.0))
    wx = cx + lx * math.cos(yaw) - ly * math.sin(yaw)
    wy = cy + lx * math.sin(yaw) + ly * math.cos(yaw)
    wz = item.aabb.z_min + lz
    return (wx, wy, wz)


def _anchor_name_for_placement_area(area: str, target: ItemRef, source: Optional[ItemRef] = None) -> str:
    area = str(area or "center").strip().lower()
    mapping = {
        "center": "top.center",
        "top": "top.center",
        "left_front": "top.left_front",
        "right_front": "top.right_front",
        "front": "top.front_center",
        "back": "top.back_center",
        "back_left": "top.back_left",
        "back_right": "top.back_right",
        "left": "top.left_front",
        "right": "top.right_front",
        "side": "top.right_front",
        "back_center": "top.back_center",
        "front_center": "top.front_center",
        "pillow_area": "top.pillow_left",
    }
    if target.semantic_group == "bed" and source is not None:
        if source.semantic_group == "pillow":
            return "top.pillow_left"
        if source.semantic_group == "blanket":
            return "top.blanket_center"
    return mapping.get(area, "top.center")


# -----------------------------------------------------------------------------
# Placement repair
# -----------------------------------------------------------------------------

def apply_relation_aware_placement(
    data: dict[str, Any],
    items: list[ItemRef],
    edges: list[RelationEdge],
    room_bounds: AABB,
    options: StageOptions,
) -> dict[str, Any]:
    if not options.apply_placement:
        return {"applied": False, "changed_count": 0, "changes": []}

    by_id = _items_by_id(items)
    anchors = {item.object_id: generate_anchors_for_item(item) for item in items}
    changed: list[dict[str, Any]] = []

    floor_main = [it for it in items if it.role == "main" and it.placement_type == "floor" and it.aabb is not None]
    for item in floor_main:
        _clamp_item_to_room(item, room_bounds, changed, reason="main_floor_inside_room")

    ordered_edges = sorted(edges, key=lambda e: _placement_edge_order(e, by_id))
    for edge in ordered_edges:
        source = by_id.get(edge.from_object_id)
        target = by_id.get(edge.to_object_id)
        if source is None or source.aabb is None:
            continue

        if edge.relation_type == "on_top_of" and target is not None and target.aabb is not None:
            area = str((edge.params or {}).get("placement_area") or "center")
            anchor_name = _anchor_name_for_placement_area(area, target, source)
            anchor = anchors.get(target.object_id, {}).get(anchor_name) or anchors.get(target.object_id, {}).get("top.center")
            if anchor:
                wx, wy, wz = _local_to_world_anchor(target, anchor)
                new_aabb = source.aabb.moved_center_xy(wx, wy).moved_bottom_z(wz + 0.003)
                _update_item_aabb(source, new_aabb, changed, reason=f"relation:{edge.edge_id()}")

        elif edge.relation_type in {"in_front_of", "next_to", "left_of", "right_of", "behind"} and target is not None and target.aabb is not None:
            new_aabb = _aabb_for_proximity_relation(source, target, edge)
            if new_aabb is not None:
                new_aabb = _clamped_aabb_to_room(new_aabb, room_bounds)
                _update_item_aabb(source, new_aabb, changed, reason=f"relation:{edge.edge_id()}")

        elif edge.relation_type == "around" and target is not None and target.aabb is not None:
            new_aabb = _aabb_for_around_relation(source, target, edge, items)
            if new_aabb is not None:
                new_aabb = _clamped_aabb_to_room(new_aabb, room_bounds)
                _update_item_aabb(source, new_aabb, changed, reason=f"relation:{edge.edge_id()}")

        elif edge.relation_type == "under" and target is not None and target.aabb is not None:
            new_aabb = source.aabb.moved_center_xy(*target.aabb.center_xy).moved_bottom_z(room_bounds.z_min + 0.002)
            _update_item_aabb(source, new_aabb, changed, reason=f"relation:{edge.edge_id()}")

        elif edge.relation_type in {"above", "mounted_on_wall"} and target is not None and target.aabb is not None:
            if source.placement_type == "wall" or source.semantic_group in WALL_MOUNTED_GROUPS:
                new_aabb = _aabb_above_target(source, target, edge, room_bounds)
                _update_item_aabb(source, new_aabb, changed, reason=f"relation:{edge.edge_id()}")

        if edge.relation_type == "faces":
            target_xy = None
            if target is not None and target.aabb is not None:
                target_xy = target.aabb.center_xy
            elif edge.to_object_id == ROOM_CENTER_ID:
                raw_xy = (edge.params or {}).get("target_xy")
                if isinstance(raw_xy, list) and len(raw_xy) >= 2:
                    target_xy = (float(raw_xy[0]), float(raw_xy[1]))
                else:
                    target_xy = room_bounds.center_xy
            if target_xy is not None:
                old_yaw = source.yaw_deg
                new_yaw = _yaw_to_face(source, target_xy)
                if _angle_delta_abs(old_yaw, new_yaw) > 1.0:
                    source.yaw_deg = new_yaw
                    _set_yaw_deg_on_item(source.raw, new_yaw)
                    changed.append(
                        {
                            "object_id": source.object_id,
                            "change": "yaw",
                            "old_yaw_deg": round(old_yaw, 6),
                            "new_yaw_deg": round(new_yaw, 6),
                            "reason": f"relation:{edge.edge_id()}",
                        }
                    )

    for item in items:
        if item.aabb is not None and item.placement_type == "floor":
            _clamp_item_to_room(item, room_bounds, changed, reason="final_inside_room")

    return {
        "applied": True,
        "changed_count": len(changed),
        "changes": changed,
    }


def _placement_edge_order(edge: RelationEdge, by_id: dict[str, ItemRef]) -> tuple[int, int]:
    source = by_id.get(edge.from_object_id)
    if edge.relation_type in {"against_wall", "mounted_on_wall"}:
        base = 1
    elif source and source.role == "main":
        base = 2
    elif edge.relation_type in {"in_front_of", "near", "next_to", "around", "faces"}:
        base = 3
    elif edge.relation_type in {"on_top_of", "inside", "under", "above"}:
        base = 4
    else:
        base = 5
    return (base, source.index if source else 999999)


def _update_item_aabb(item: ItemRef, new_aabb: AABB, changed: list[dict[str, Any]], reason: str) -> None:
    old = item.aabb
    if old is None:
        item.aabb = new_aabb
        _write_aabb_to_item(item.raw, new_aabb)
        return
    delta = math.hypot(old.center_xy[0] - new_aabb.center_xy[0], old.center_xy[1] - new_aabb.center_xy[1]) + abs(old.z_min - new_aabb.z_min)
    if delta < 1e-5:
        return
    item.aabb = new_aabb
    _write_aabb_to_item(item.raw, new_aabb)
    changed.append(
        {
            "object_id": item.object_id,
            "change": "aabb",
            "old_aabb": old.to_dict(),
            "new_aabb": new_aabb.to_dict(),
            "reason": reason,
        }
    )


def _clamped_aabb_to_room(aabb: AABB, room: AABB, margin: float = 0.03) -> AABB:
    w = aabb.width
    d = aabb.depth
    x = min(max(aabb.center_xy[0], room.x_min + margin + 0.5 * w), room.x_max - margin - 0.5 * w)
    y = min(max(aabb.center_xy[1], room.y_min + margin + 0.5 * d), room.y_max - margin - 0.5 * d)
    z = max(aabb.z_min, room.z_min)
    return aabb.moved_center_xy(x, y).moved_bottom_z(z)


def _clamp_item_to_room(item: ItemRef, room: AABB, changed: list[dict[str, Any]], reason: str) -> None:
    if item.aabb is None:
        return
    clamped = _clamped_aabb_to_room(item.aabb, room)
    _update_item_aabb(item, clamped, changed, reason=reason)


def _aabb_for_proximity_relation(source: ItemRef, target: ItemRef, edge: RelationEdge) -> Optional[AABB]:
    if source.aabb is None or target.aabb is None:
        return None
    relation = edge.relation_type
    params = edge.params or {}
    distance_raw = params.get("distance_m") if isinstance(params.get("distance_m"), dict) else {}
    min_dist = float(distance_raw.get("min", 0.08) or 0.08)
    max_dist = float(distance_raw.get("max", 0.45) or 0.45)
    gap = 0.5 * (min_dist + max_dist)
    side = str(params.get("side_preference") or "auto").strip().lower()

    if relation == "left_of":
        side = "left"
    elif relation == "right_of":
        side = "right"
    elif relation == "behind":
        side = "back"
    elif relation == "in_front_of":
        side = "front"
    elif relation == "next_to" and side == "auto":
        side = _infer_side_preference(source, target)
    elif side == "auto":
        side = "front"

    tx, ty = target.aabb.center_xy
    yaw = math.radians(target.yaw_deg)
    front = (math.cos(yaw + math.radians(_axis_offset_deg(target.front_axis_local))), math.sin(yaw + math.radians(_axis_offset_deg(target.front_axis_local))))
    right = (front[1], -front[0])

    if side == "front":
        dist = 0.5 * target.aabb.depth + 0.5 * source.aabb.depth + gap
        x = tx + front[0] * dist
        y = ty + front[1] * dist
    elif side == "back":
        dist = 0.5 * target.aabb.depth + 0.5 * source.aabb.depth + gap
        x = tx - front[0] * dist
        y = ty - front[1] * dist
    elif side == "left":
        dist = 0.5 * target.aabb.width + 0.5 * source.aabb.width + gap
        x = tx - right[0] * dist
        y = ty - right[1] * dist
    elif side == "right":
        dist = 0.5 * target.aabb.width + 0.5 * source.aabb.width + gap
        x = tx + right[0] * dist
        y = ty + right[1] * dist
    else:
        dist = 0.5 * target.aabb.depth + 0.5 * source.aabb.depth + gap
        x = tx + front[0] * dist
        y = ty + front[1] * dist
    return source.aabb.moved_center_xy(x, y).moved_bottom_z(target.aabb.z_min if source.placement_type == "floor" else source.aabb.z_min)


def _aabb_for_around_relation(source: ItemRef, target: ItemRef, edge: RelationEdge, items: list[ItemRef]) -> Optional[AABB]:
    if source.aabb is None or target.aabb is None:
        return None
    siblings = [
        it
        for it in items
        if it.object_id != target.object_id
        and it.semantic_group in CHAIR_GROUPS
        and it.zone_id == source.zone_id
        and it.aabb is not None
    ]
    siblings.sort(key=lambda it: it.index)
    try:
        idx = [it.object_id for it in siblings].index(source.object_id)
    except ValueError:
        idx = 0
    n = max(1, len(siblings))
    directions = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    if n <= 4:
        dx, dy = directions[idx % len(directions)]
    else:
        angle = 2.0 * math.pi * idx / n
        dx, dy = math.cos(angle), math.sin(angle)
    tx, ty = target.aabb.center_xy
    dist_x = 0.5 * target.aabb.width + 0.5 * source.aabb.width + 0.28
    dist_y = 0.5 * target.aabb.depth + 0.5 * source.aabb.depth + 0.28
    x = tx + dx * dist_x
    y = ty + dy * dist_y
    return source.aabb.moved_center_xy(x, y).moved_bottom_z(target.aabb.z_min)


def _aabb_above_target(source: ItemRef, target: ItemRef, edge: RelationEdge, room_bounds: AABB) -> AABB:
    assert source.aabb is not None and target.aabb is not None
    x, y = target.aabb.center_xy
    gap = 0.18
    params = edge.params or {}
    gap_raw = params.get("vertical_gap_m")
    if isinstance(gap_raw, dict):
        gap = 0.5 * (float(gap_raw.get("min", 0.05)) + float(gap_raw.get("max", 0.45)))
    z_min = min(room_bounds.z_max - source.aabb.height - 0.05, target.aabb.z_max + gap)
    z_min = max(room_bounds.z_min + 0.5, z_min)
    return source.aabb.moved_center_xy(x, y).moved_bottom_z(z_min)


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_relationship_graph(
    items: list[ItemRef],
    edges: list[RelationEdge],
    room_bounds: AABB,
) -> dict[str, Any]:
    by_id = _items_by_id(items)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    relation_scores = {
        "support": [],
        "orientation": [],
        "proximity": [],
        "wall": [],
        "group": [],
    }

    support_parent: dict[str, str] = {}
    hard_support_count: dict[str, int] = {}

    for edge in edges:
        source = by_id.get(edge.from_object_id)
        target = by_id.get(edge.to_object_id)
        edge_id = edge.edge_id()

        if source is None:
            errors.append({"relation_id": edge_id, "problem": "from_object_missing", "from_object_id": edge.from_object_id})
            continue
        if edge.to_object_id not in {ROOM_CENTER_ID, WALL_TARGET_ID} and target is None:
            errors.append({"relation_id": edge_id, "problem": "to_object_missing", "to_object_id": edge.to_object_id})
            continue

        if edge.relation_type == "on_top_of":
            if target is None:
                errors.append({"relation_id": edge_id, "problem": "support_target_missing"})
                continue
            if target.semantic_group not in SUPPORT_SURFACE_GROUPS:
                errors.append(
                    {
                        "relation_id": edge_id,
                        "problem": "invalid_support_target",
                        "target_group": target.semantic_group,
                    }
                )
                continue
            support_parent[source.object_id] = target.object_id
            if edge.constraint_level == "hard":
                hard_support_count[source.object_id] = hard_support_count.get(source.object_id, 0) + 1
            score = _score_on_top_of(source, target)
            relation_scores["support"].append(score)
            if edge.constraint_level == "hard" and score < 0.65:
                errors.append({"relation_id": edge_id, "problem": "support_relation_not_satisfied", "score": round(score, 4)})
            elif score < 0.65:
                warnings.append({"relation_id": edge_id, "problem": "support_relation_weak", "score": round(score, 4)})

        elif edge.relation_type == "faces":
            if edge.to_object_id == ROOM_CENTER_ID:
                target_xy = tuple((edge.params or {}).get("target_xy") or list(room_bounds.center_xy))
            elif target and target.aabb:
                target_xy = target.aabb.center_xy
            else:
                target_xy = None
            if target_xy is None:
                warnings.append({"relation_id": edge_id, "problem": "orientation_target_has_no_aabb"})
                continue
            score = _score_faces(source, target_xy, edge)
            relation_scores["orientation"].append(score)
            if edge.constraint_level == "hard" and score < 0.65:
                errors.append({"relation_id": edge_id, "problem": "faces_relation_not_satisfied", "score": round(score, 4)})
            elif score < 0.65:
                warnings.append({"relation_id": edge_id, "problem": "faces_relation_weak", "score": round(score, 4)})

        elif edge.relation_type in {"near", "next_to", "in_front_of", "left_of", "right_of", "behind", "around"}:
            if target is None:
                continue
            score = _score_proximity(source, target, edge)
            relation_scores["proximity"].append(score)
            if edge.constraint_level == "hard" and score < 0.45:
                errors.append({"relation_id": edge_id, "problem": "proximity_relation_not_satisfied", "score": round(score, 4)})
            elif score < 0.45:
                warnings.append({"relation_id": edge_id, "problem": "proximity_relation_weak", "score": round(score, 4)})

        elif edge.relation_type in {"against_wall", "mounted_on_wall"}:
            score = _score_wall_relation(source, room_bounds)
            relation_scores["wall"].append(score)
            if edge.constraint_level == "hard" and score < 0.45:
                warnings.append({"relation_id": edge_id, "problem": "wall_relation_weak", "score": round(score, 4)})

    for object_id, count in hard_support_count.items():
        if count > 1:
            errors.append({"object_id": object_id, "problem": "multiple_hard_support_relations", "count": count})

    for item in items:
        if item.role == "accessory" and item.semantic_group not in WALL_MOUNTED_GROUPS:
            has_relation = any(edge.from_object_id == item.object_id and edge.relation_type in {"on_top_of", "inside", "under", "near", "mounted_on_wall", "above"} for edge in edges)
            if not has_relation:
                warnings.append({"object_id": item.object_id, "problem": "accessory_without_support_or_context_relation", "semantic_group": item.semantic_group})
        if item.semantic_group in CHAIR_GROUPS:
            has_faces = any(edge.from_object_id == item.object_id and edge.relation_type == "faces" for edge in edges)
            if not has_faces:
                warnings.append({"object_id": item.object_id, "problem": "chair_without_faces_relation"})
        if item.semantic_group == "sofa":
            has_faces = any(edge.from_object_id == item.object_id and edge.relation_type == "faces" for edge in edges)
            if not has_faces:
                warnings.append({"object_id": item.object_id, "problem": "sofa_without_orientation_target"})

    cycle = _detect_support_cycle(support_parent)
    if cycle:
        errors.append({"problem": "support_cycle", "cycle": cycle})

    score_by_class = {
        key: (round(sum(values) / len(values), 4) if values else None)
        for key, values in relation_scores.items()
    }
    numeric_scores = [v for v in score_by_class.values() if isinstance(v, (int, float))]
    base_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 1.0
    penalty = min(0.65, 0.08 * len(errors) + 0.025 * len(warnings))
    final_score = max(0.0, min(1.0, base_score - penalty))

    return {
        "schema": "relationship_validation/v1",
        "is_valid": not errors,
        "score": round(final_score, 4),
        "errors": errors,
        "warnings": warnings,
        "relation_scores": score_by_class,
        "counts": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "edge_count": len(edges),
            "item_count": len(items),
        },
    }


def _score_on_top_of(source: ItemRef, target: ItemRef) -> float:
    if source.aabb is None or target.aabb is None:
        return 0.0
    footprint = max(source.aabb.area_xy, 1e-9)
    overlap = source.aabb.overlap_xy_area(target.aabb) / footprint
    z_gap = abs(source.aabb.z_min - target.aabb.z_max)
    z_score = max(0.0, 1.0 - z_gap / 0.18)
    return max(0.0, min(1.0, 0.72 * min(overlap, 1.0) + 0.28 * z_score))


def _score_faces(source: ItemRef, target_xy: tuple[float, float], edge: RelationEdge) -> float:
    if source.aabb is None:
        return 0.0
    desired = _yaw_to_face(source, target_xy)
    delta = _angle_delta_abs(source.yaw_deg, desired)
    tolerance = float((edge.params or {}).get("tolerance_deg", 30) or 30)
    return max(0.0, min(1.0, 1.0 - delta / max(180.0, tolerance * 3.0)))


def _score_proximity(source: ItemRef, target: ItemRef, edge: RelationEdge) -> float:
    if source.aabb is None or target.aabb is None:
        return 0.0
    dist = _distance_xy(source.aabb, target.aabb)
    params = edge.params or {}
    raw = params.get("distance_m") if isinstance(params.get("distance_m"), dict) else {}
    if raw:
        min_d = float(raw.get("min", 0.0) or 0.0)
        max_d = float(raw.get("max", 1.0) or 1.0)
        if min_d <= dist <= max_d + max(source.aabb.width, source.aabb.depth, target.aabb.width, target.aabb.depth):
            return 1.0
        ideal = 0.5 * (min_d + max_d)
        return max(0.0, min(1.0, 1.0 - abs(dist - ideal) / max(ideal + 1.0, 1e-6)))
    ideal = max(source.aabb.width, source.aabb.depth, target.aabb.width, target.aabb.depth) * 0.7
    return max(0.0, min(1.0, 1.0 - abs(dist - ideal) / max(ideal + 1.0, 1e-6)))


def _score_wall_relation(source: ItemRef, room_bounds: AABB) -> float:
    if source.aabb is None:
        return 0.0
    distances = [
        abs(source.aabb.x_min - room_bounds.x_min),
        abs(source.aabb.x_max - room_bounds.x_max),
        abs(source.aabb.y_min - room_bounds.y_min),
        abs(source.aabb.y_max - room_bounds.y_max),
    ]
    d = min(distances)
    return max(0.0, min(1.0, 1.0 - d / 0.65))


def _detect_support_cycle(parent: dict[str, str]) -> list[str]:
    for start in parent:
        seen: set[str] = set()
        path: list[str] = []
        cur = start
        while cur in parent:
            if cur in seen:
                idx = path.index(cur) if cur in path else 0
                return path[idx:] + [cur]
            seen.add(cur)
            path.append(cur)
            cur = parent[cur]
    return []


# -----------------------------------------------------------------------------
# Graph building and stage execution
# -----------------------------------------------------------------------------

def build_relationship_graph(
    data: dict[str, Any],
    *,
    prompt: str = "",
    llm_relations: Optional[list[dict[str, Any]]] = None,
    options: Optional[StageOptions] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    options = options or StageOptions()
    data = deepcopy(data)
    items = collect_items(data, prompt=prompt)
    room_bounds = _room_bounds_from_data(data, items)

    edges: list[RelationEdge] = []
    if llm_relations:
        edges.extend(_resolve_llm_relations(llm_relations, items))
    edges.extend(build_rule_based_edges(items, room_bounds))
    edges = _dedupe_edges(edges)

    placement_info = apply_relation_aware_placement(data, items, edges, room_bounds, options)
    if options.apply_placement:
        items = collect_items(data, prompt=prompt)
        room_bounds = _room_bounds_from_data(data, items)
        edges = _dedupe_edges(edges)

    anchors = {item.object_id: generate_anchors_for_item(item) for item in items}
    validation = validate_relationship_graph(items, edges, room_bounds) if options.validate else None

    graph = {
        "schema": GRAPH_SCHEMA,
        "nodes": [
            {
                "object_id": item.object_id,
                "zone_id": item.zone_id,
                "semantic_group": item.semantic_group,
                "role": item.role,
                "placement_type": item.placement_type,
                "front_axis_local": item.front_axis_local,
                "has_aabb": item.aabb is not None,
            }
            for item in items
        ],
        "edges": [edge.to_dict() for edge in edges],
        "anchors_local": anchors,
        "room_targets": {
            ROOM_CENTER_ID: {"type": "point", "xy": list(room_bounds.center_xy)},
            WALL_TARGET_ID: {"type": "wall_set", "room_bounds": room_bounds.to_dict()},
        },
        "meta": {
            "source": "relationship_graph_stage",
            "edge_count": len(edges),
            "node_count": len(items),
            "room_bounds": room_bounds.to_dict(),
        },
    }

    data["relationship_graph"] = graph
    data["relationship_stage"] = {
        "schema": SCHEMA,
        "status": "success" if not validation or validation.get("is_valid") else "partial_success",
        "options": dataclasses.asdict(options),
        "placement": placement_info,
        "validation": validation,
        "summary": {
            "item_count": len(items),
            "edge_count": len(edges),
            "hard_edge_count": sum(1 for edge in edges if edge.constraint_level == "hard"),
            "soft_edge_count": sum(1 for edge in edges if edge.constraint_level == "soft"),
            "decorative_edge_count": sum(1 for edge in edges if edge.constraint_level == "decorative"),
        },
    }

    _annotate_items_with_relationship_meta(data, items, edges)
    return data, data["relationship_stage"]


def _resolve_llm_relations(raw_relations: list[dict[str, Any]], items: list[ItemRef]) -> list[RelationEdge]:
    edges: list[RelationEdge] = []
    for raw in raw_relations:
        if not isinstance(raw, dict):
            continue
        relation_type = str(raw.get("relation_type") or raw.get("relation") or "").strip()
        if relation_type not in RELATION_TYPES:
            continue
        level = str(raw.get("constraint_level") or raw.get("priority") or "hard").strip().lower()
        if level not in CONSTRAINT_LEVELS:
            level = "hard"
        weight = _float_or_none(raw.get("weight")) or (1.0 if level == "hard" else 0.6)
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        reason = str(raw.get("reason") or "")

        from_id = str(raw.get("from_object_id") or raw.get("from_object") or "").strip()
        to_id = str(raw.get("to_object_id") or raw.get("to_object") or "").strip()
        from_subclass = str(raw.get("from_subclass") or raw.get("from_group") or "").strip().lower()
        to_subclass = str(raw.get("to_subclass") or raw.get("to_group") or "").strip().lower()

        source_items: list[ItemRef] = []
        target_items: list[ItemRef] = []
        if from_id:
            source_items = [it for it in items if it.object_id == from_id]
        elif from_subclass:
            source_items = [it for it in items if it.semantic_group == from_subclass]

        if to_id in {ROOM_CENTER_ID, WALL_TARGET_ID}:
            target_items = []
        elif to_id:
            target_items = [it for it in items if it.object_id == to_id]
        elif to_subclass:
            target_items = [it for it in items if it.semantic_group == to_subclass]

        for source in source_items:
            if to_id in {ROOM_CENTER_ID, WALL_TARGET_ID}:
                edges.append(
                    _edge(
                        source,
                        relation_type,
                        to_id,
                        level=level,
                        weight=weight,
                        params=params,
                        source_name="llm",
                        reason=reason,
                    )
                )
                continue
            targets = [t for t in target_items if t.object_id != source.object_id]
            same_zone = [t for t in targets if t.zone_id == source.zone_id]
            if same_zone:
                targets = same_zone
            if not targets:
                continue
            target = min(targets, key=lambda t: (0 if t.role == "main" else 1, t.index))
            edges.append(
                _edge(
                    source,
                    relation_type,
                    target,
                    level=level,
                    weight=weight,
                    params=params,
                    source_name="llm",
                    reason=reason,
                )
            )
    return edges


def _annotate_items_with_relationship_meta(data: dict[str, Any], items: list[ItemRef], edges: list[RelationEdge]) -> None:
    outgoing: dict[str, list[dict[str, Any]]] = {}
    incoming: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        edge_dict = edge.to_dict()
        outgoing.setdefault(edge.from_object_id, []).append(edge_dict)
        incoming.setdefault(edge.to_object_id, []).append(edge_dict)

    for item in items:
        meta = item.raw.setdefault("meta", {})
        if not isinstance(meta, dict):
            item.raw["meta"] = {}
            meta = item.raw["meta"]
        meta["relationship_graph"] = {
            "zone_id": item.zone_id,
            "semantic_group": item.semantic_group,
            "role": item.role,
            "placement_type": item.placement_type,
            "front_axis_local": item.front_axis_local,
            "outgoing_relation_ids": [edge["id"] for edge in outgoing.get(item.object_id, [])],
            "incoming_relation_ids": [edge["id"] for edge in incoming.get(item.object_id, [])],
        }
        item.raw.setdefault("semantic_group", item.semantic_group)
        item.raw.setdefault("zone_id", item.zone_id)
        item.raw.setdefault("role", item.role)
        item.raw.setdefault("front_axis_local", item.front_axis_local)


# -----------------------------------------------------------------------------
# Pipeline integration
# -----------------------------------------------------------------------------

def add_relationship_graph_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("relationship graph stage")
    group.add_argument("--relationship-graph", action="store_true", help="Enable object relationship graph postprocess stage")
    group.add_argument("--relationship-graph-apply-placement", action="store_true", help="Apply relation-aware AABB/yaw repair to JSON")
    group.add_argument("--relationship-graph-no-validate", action="store_true", help="Disable relationship validation")
    group.add_argument("--relationship-graph-no-repair", action="store_true", help="Disable relation repair flags; graph is still generated")
    group.add_argument("--relationship-graph-add-missing-supports", action="store_true", help="Reserved: allow adding missing support objects in future versions")
    group.add_argument("--relationship-graph-min-score", type=float, default=0.75)


def maybe_apply_relationship_graph_stage(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scene_json_path: Path,
    prompt_text: str,
    tag: str = "base",
) -> tuple[Path, dict[str, Any] | None]:
    enabled = bool(getattr(args, "relationship_graph", False))
    if not enabled:
        return scene_json_path, None
    scene_json_path = Path(scene_json_path).expanduser().resolve()
    if not scene_json_path.is_file():
        return scene_json_path, {"skipped_reason": "scene_json_missing", "input_scene_json": str(scene_json_path)}

    options = StageOptions(
        apply_placement=bool(getattr(args, "relationship_graph_apply_placement", False)),
        validate=not bool(getattr(args, "relationship_graph_no_validate", False)),
        repair=not bool(getattr(args, "relationship_graph_no_repair", False)),
        add_missing_supports=bool(getattr(args, "relationship_graph_add_missing_supports", False)),
        min_relation_score=float(getattr(args, "relationship_graph_min_score", 0.75) or 0.75),
    )
    data = read_json(scene_json_path)
    if not isinstance(data, dict):
        return scene_json_path, {"skipped_reason": "input_json_not_object", "input_scene_json": str(scene_json_path)}

    out_path = (run_dir / f"{scene_json_path.stem}.relationship_graph_{tag}.v1.json").resolve()
    out_data, info = build_relationship_graph(data, prompt=prompt_text, options=options)
    write_json(out_path, out_data)
    info = dict(info)
    info["input_scene_json"] = str(scene_json_path)
    info["output_scene_json"] = str(out_path)
    info["tag"] = tag
    return out_path, info


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build object relationship graph for scene.v1 / placement.v1 JSON")
    parser.add_argument("--input", required=True, help="Input scene.v1 / placement.v1 JSON")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--prompt", default="", help="Original room prompt for zone inference")
    parser.add_argument("--prompt-file", default=None, help="Optional prompt file")
    parser.add_argument("--llm-relations-json", default=None, help="Optional JSON with LLM-proposed relations")
    parser.add_argument("--apply-placement", action="store_true", help="Apply relation-aware AABB/yaw repair")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--add-missing-supports", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _read_prompt_from_cli(args: argparse.Namespace) -> str:
    if args.prompt_file:
        path = Path(args.prompt_file).expanduser().resolve()
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return str(args.prompt or "")


def _load_llm_relations(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    data = read_json(p)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("relations", "edges", "relationship_edges"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise RuntimeError("LLM relations JSON must be a list or object with relations/edges")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    data = read_json(input_path)
    if not isinstance(data, dict):
        raise RuntimeError("Input JSON root must be an object")

    options = StageOptions(
        apply_placement=bool(args.apply_placement),
        validate=not bool(args.no_validate),
        repair=not bool(args.no_repair),
        add_missing_supports=bool(args.add_missing_supports),
        verbose=bool(args.verbose),
    )
    prompt = _read_prompt_from_cli(args)
    llm_relations = _load_llm_relations(args.llm_relations_json)
    out_data, info = build_relationship_graph(
        data,
        prompt=prompt,
        llm_relations=llm_relations,
        options=options,
    )
    write_json(out_path, out_data)

    summary = info.get("summary") if isinstance(info, dict) else {}
    validation = info.get("validation") if isinstance(info, dict) else {}
    print(
        "relationship_graph_stage: "
        f"items={summary.get('item_count')} "
        f"edges={summary.get('edge_count')} "
        f"valid={validation.get('is_valid') if isinstance(validation, dict) else None} "
        f"out={out_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"relationship_graph_stage failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
