from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def round6(v: float) -> float:
    return round(float(v), 6)


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def canonical_room_type(name: str) -> str:
    s = as_str(name).strip().lower()
    if s.startswith("processed_"):
        s = s[len("processed_"):]
    if s.endswith("_augmented"):
        s = s[: -len("_augmented")]
    return s


def room_name_from_folder(folder_name: str) -> str:
    if "_" not in folder_name:
        return folder_name
    return folder_name.split("_", 1)[1]


def house_id_from_folder(folder_name: str) -> str:
    if "_" not in folder_name:
        return folder_name
    return folder_name.split("_", 1)[0]


def split_by_house(
    house_ids: List[str],
    seed: int,
    frac_train: float = 0.78,
    frac_val: float = 0.10,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(house_ids))
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))
    train_h = set(uniq[:n_train])
    val_h = set(uniq[n_train:n_train + n_val])
    test_h = set(uniq[n_train + n_val:])
    out = {"train": [], "val": [], "test": []}
    for idx, house_id in enumerate(house_ids):
        if house_id in train_h:
            out["train"].append(idx)
        elif house_id in val_h:
            out["val"].append(idx)
        else:
            out["test"].append(idx)
    return out


def top_counter(counter: Counter[str], limit: int = 100) -> List[Dict[str, Any]]:
    return [{"name": name, "count": int(count)} for name, count in counter.most_common(limit)]


def polygon_from_corners(arr: np.ndarray) -> List[List[float]]:
    pts = np.asarray(arr)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return []
    if pts.shape[1] == 2:
        return [[round6(p[0]), round6(p[1])] for p in pts]
    if pts.shape[1] >= 3:
        return [[round6(p[0]), round6(p[2])] for p in pts]
    return []


def polygon_bounds(poly: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    zs = [p[1] for p in poly]
    return min(xs), min(zs), max(xs), max(zs)


def rect_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    dx = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    dz = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return dx * dz


def compute_support_relations(objects: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {obj["id"]: None for obj in objects}
    for obj in objects:
        a = obj["aabb"]
        best_id: Optional[str] = None
        best_area = 0.0
        for cand in objects:
            if cand["id"] == obj["id"]:
                continue
            b = cand["aabb"]
            touching = abs(float(a["z_min"]) - float(b["z_max"])) <= 0.08
            if not touching:
                continue
            overlap = rect_intersection_area(
                (float(a["x_min"]), float(a["x_max"]), float(a["y_min"]), float(a["y_max"])),
                (float(b["x_min"]), float(b["x_max"]), float(b["y_min"]), float(b["y_max"])),
            )
            if overlap > best_area and overlap > 1e-4:
                best_area = overlap
                best_id = cand["id"]
        result[obj["id"]] = best_id
    return result


def infer_mount_type(
    category: str,
    super_category: str,
    z_center: float,
    z_min: float,
    z_max: float,
    room_height: float,
) -> str:
    c = category.lower()
    sc = super_category.lower()
    if "pendant lamp" in c or "ceiling lamp" in c:
        return "ceiling"
    if "lighting" in sc and z_center >= 0.75 * room_height:
        return "ceiling"
    if z_min <= 0.08:
        return "floor"
    if z_max >= 0.85 * room_height:
        return "wall"
    return "support"


def load_model_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    arr = load_json(path)
    if not isinstance(arr, list):
        raise RuntimeError(f"{path.name} must be a list")
    out: Dict[str, Dict[str, Any]] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        model_id = as_str(item.get("model_id")).strip()
        if not model_id:
            continue
        out[model_id] = item
    return out


def load_split_maps(dataset_root: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for csv_path in sorted(dataset_root.glob("*_threed_front_splits.csv")):
        room_type = csv_path.name.replace("_threed_front_splits.csv", "").strip().lower()
        mapping: Dict[str, str] = {}
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                room_name = as_str(row[0]).strip()
                split = as_str(row[1]).strip().lower()
                if room_name and split in {"train", "val", "test"}:
                    mapping[room_name] = split
        out[room_type] = mapping
    return out


@dataclass
class SceneObject:
    object_id: str
    uid: str
    jid: str
    category: str
    super_category: str
    style: Optional[str]
    theme: Optional[str]
    material: Optional[str]
    semantics_source: str
    class_id: int
    class_logits: List[float]
    position: Tuple[float, float, float]
    size_full: Tuple[float, float, float]
    yaw_rad: float
    yaw_deg: float


@dataclass
class RoomRecord:
    room_id: str
    room_name: str
    house_id: str
    room_type: str
    source_subset: str
    source_dir: str
    split: Optional[str]
    room_polygon_xz: List[List[float]]
    room_bounds: Tuple[float, float, float, float]
    room_center: Tuple[float, float, float]
    room_half_extents: Tuple[float, float, float]
    objects: List[SceneObject]


def gather_boxes_paths(dataset_root: Path, include_augmented: bool) -> List[Path]:
    out: List[Path] = []
    for subdir in sorted(dataset_root.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if not name.startswith("processed_"):
            continue
        if not include_augmented and name.endswith("_augmented"):
            continue
        out.extend(sorted(subdir.glob("*/boxes.npz")))
    return out


def build_room_record(
    boxes_path: Path,
    prepared_index: Dict[str, Dict[str, Any]],
    model_index: Dict[str, Dict[str, Any]],
    split_maps: Dict[str, Dict[str, str]],
) -> RoomRecord:
    z = np.load(boxes_path, allow_pickle=False)

    jids = z["jids"]
    uids = z["uids"]
    class_labels = np.asarray(z["class_labels"], dtype=np.float32)
    translations = np.asarray(z["translations"], dtype=np.float32)
    sizes_half = np.asarray(z["sizes"], dtype=np.float32)
    angles = np.asarray(z["angles"], dtype=np.float32).reshape(-1)

    room_folder = boxes_path.parent.name
    source_subset = boxes_path.parent.parent.name
    room_type = canonical_room_type(source_subset)
    room_name = room_name_from_folder(room_folder)
    house_id = house_id_from_folder(room_folder)
    split = split_maps.get(room_type, {}).get(room_name)

    polygon = polygon_from_corners(np.asarray(z["floor_plan_ordered_corners"]))
    if len(polygon) < 3:
        polygon = polygon_from_corners(np.asarray(z["floor_plan_vertices"]))
    if len(polygon) < 3:
        raise RuntimeError(f"Room polygon is missing for {boxes_path}")

    min_x, min_z, max_x, max_z = polygon_bounds(polygon)
    cx = 0.5 * (min_x + max_x)
    cz = 0.5 * (min_z + max_z)
    hx = max(0.5 * (max_x - min_x), 1e-6)
    hz = max(0.5 * (max_z - min_z), 1e-6)

    objs: List[SceneObject] = []
    top_height = 0.0
    for idx in range(len(jids)):
        jid = as_str(jids[idx]).strip()
        uid = as_str(uids[idx]).strip()
        prepared_info = prepared_index.get(jid, {})
        model_info = model_index.get(jid, {})
        info = prepared_info or model_info

        class_vec = class_labels[idx].astype(float).tolist()
        class_id = int(np.argmax(class_labels[idx])) if class_labels.shape[1] > 0 else 0

        category = as_str(info.get("category")).strip() or f"class_{class_id}"
        super_category = as_str(info.get("super-category")).strip() or "Unknown"
        style = as_str(info.get("style")).strip() or None
        theme = as_str(info.get("theme")).strip() or None
        material = as_str(info.get("material")).strip() or None
        semantics_source = "prepared_model_info" if prepared_info else ("model_info" if model_info else "class_index")

        pos = tuple(float(v) for v in translations[idx].tolist())
        half = tuple(float(v) for v in sizes_half[idx].tolist())
        size_full = (2.0 * half[0], 2.0 * half[1], 2.0 * half[2])
        yaw_rad = float(angles[idx])
        yaw_deg = math.degrees(yaw_rad)
        top_height = max(top_height, pos[1] + half[1])

        objs.append(
            SceneObject(
                object_id=f"obj_{idx:04d}",
                uid=uid or f"uid_{idx:04d}",
                jid=jid or f"jid_{idx:04d}",
                category=category,
                super_category=super_category,
                style=style,
                theme=theme,
                material=material,
                semantics_source=semantics_source,
                class_id=class_id,
                class_logits=class_vec,
                position=pos,
                size_full=size_full,
                yaw_rad=yaw_rad,
                yaw_deg=yaw_deg,
            )
        )

    hy = max(top_height, 1e-3)
    return RoomRecord(
        room_id=room_folder,
        room_name=room_name,
        house_id=house_id,
        room_type=room_type,
        source_subset=source_subset,
        source_dir=str(boxes_path.parent),
        split=split,
        room_polygon_xz=polygon,
        room_bounds=(min_x, min_z, max_x, max_z),
        room_center=(cx, 0.0, cz),
        room_half_extents=(hx, hy, hz),
        objects=objs,
    )


def object_to_export_dict(
    obj: SceneObject,
    room_height: float,
) -> Dict[str, Any]:
    px, py, pz = obj.position
    sx, sy, sz = obj.size_full
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    aabb = {
        "x_min": round6(px - hx),
        "x_max": round6(px + hx),
        "y_min": round6(pz - hz),
        "y_max": round6(pz + hz),
        "z_min": round6(py - hy),
        "z_max": round6(py + hy),
    }
    mount_type = infer_mount_type(
        category=obj.category,
        super_category=obj.super_category,
        z_center=py,
        z_min=py - hy,
        z_max=py + hy,
        room_height=room_height,
    )
    return {
        "id": obj.object_id,
        "name": obj.category,
        "category": obj.category,
        "position_m": [round6(px), round6(pz), round6(py)],
        "size_m": [round6(sx), round6(sz), round6(sy)],
        "rotation_deg": int(round(obj.yaw_deg)) % 360,
        "yaw_deg": round6(obj.yaw_deg),
        "yaw_rad": round6(obj.yaw_rad),
        "aabb": aabb,
        "mount_type": mount_type,
        "constraints": {},
        "asset": {
            "source": "3dfront_65347_boxes",
            "model_id": obj.jid,
            "uid": obj.uid,
            "mesh_path": None,
            "mesh_fit_mode": "bbox_from_boxes_npz",
            "mesh_texture_dirs": [],
        },
        "recognition": {
            "prepared_recognized": obj.semantics_source in {"prepared_model_info", "model_info"},
            "reason": (
                "matched_prepared_model_info"
                if obj.semantics_source == "prepared_model_info"
                else ("matched_model_info" if obj.semantics_source == "model_info" else "fallback_class_index")
            ),
        },
        "source": {
            "placement_source": "3dfront_65347_boxes",
        },
        "meta": {
            "super_category": obj.super_category,
            "style": obj.style,
            "theme": obj.theme,
            "material": obj.material,
            "semantics_source": obj.semantics_source,
            "class_id": obj.class_id,
            "class_logits": [round6(v) for v in obj.class_logits],
            "raw_uid": obj.uid,
            "raw_jid": obj.jid,
        },
        "color": [0.7, 0.7, 0.7],
    }


def write_room_exports(room: RoomRecord, out_dir: Path) -> Dict[str, Any]:
    room_dir = out_dir / room.room_id
    room_dir.mkdir(parents=True, exist_ok=True)
    room_json = {
        "schema": "room.v1",
        "id": room.room_id,
        "name": room.room_name,
        "room_type": room.room_type,
        "floor_polygon_xz": room.room_polygon_xz,
        "bounds_xz": {
            "x_min": round6(room.room_bounds[0]),
            "z_min": round6(room.room_bounds[1]),
            "x_max": round6(room.room_bounds[2]),
            "z_max": round6(room.room_bounds[3]),
        },
        "meta": {
            "house_id": room.house_id,
            "source_subset": room.source_subset,
            "source_dir": room.source_dir,
            "split": room.split,
        },
    }

    exports = [object_to_export_dict(obj, room_height=2.0 * room.room_half_extents[1]) for obj in room.objects]
    support_parent = compute_support_relations(exports)

    objects_v1 = {
        "schema": "objects.v1",
        "seed": 0,
        "objects": [],
        "meta": {
            "source": {
                "dataset": "3DFRONT_65347",
                "subset": room.source_subset,
                "room_id": room.room_id,
            },
            "export_policy": {
                "keep_all_objects": True,
                "source": "boxes.npz + prepared_model_info.json",
            },
        },
    }
    placements = []

    for item in exports:
        category = item["category"]
        super_category = item["meta"]["super_category"]
        support_id = support_parent.get(item["id"])
        objects_v1["objects"].append(
            {
                "id": item["id"],
                "name": item["name"],
                "category": category,
                "size_m": item["size_m"],
                "size_min_m": item["size_m"],
                "size_max_m": item["size_m"],
                "color": item["color"],
                "constraints": {"supported_by": support_id} if support_id else {},
                "asset": item["asset"],
                "recognition": item["recognition"],
                "meta": {
                    "super_category": super_category,
                    "style": item["meta"]["style"],
                    "theme": item["meta"]["theme"],
                    "material": item["meta"]["material"],
                    "semantics_source": item["meta"]["semantics_source"],
                    "class_id": item["meta"]["class_id"],
                    "class_logits": item["meta"]["class_logits"],
                    "raw_uid": item["meta"]["raw_uid"],
                    "raw_jid": item["meta"]["raw_jid"],
                },
            }
        )
        placements.append(item | {"constraints": {"supported_by": support_id} if support_id else {}})

    scene_gt_v1 = {
        "schema": "scene_gt.v1",
        "room": room_json,
        "placements": placements,
        "meta": {
            "source": {
                "dataset": "3DFRONT_65347",
                "subset": room.source_subset,
                "room_id": room.room_id,
            },
            "export_policy": {
                "keep_all_objects": True,
            },
        },
    }

    save_json(room_dir / "room.json", room_json)
    save_json(room_dir / "objects.v1.json", objects_v1)
    save_json(room_dir / "scene_gt.v1.json", scene_gt_v1)

    return {
        "room_json": str((room_dir / "room.json").resolve()),
        "objects_v1_json": str((room_dir / "objects.v1.json").resolve()),
        "scene_gt_v1_json": str((room_dir / "scene_gt.v1.json").resolve()),
    }


def build_canonical_arrays(
    rooms: List[RoomRecord],
    cat2id: Dict[str, int],
    nmax: int,
) -> Dict[str, np.ndarray]:
    m = len(rooms)
    pos_gt_xz = np.zeros((m, nmax, 2), dtype=np.float32)
    size_room = np.zeros((m, nmax, 3), dtype=np.float32)
    cat_id = np.zeros((m, nmax), dtype=np.int64)
    mask = np.zeros((m, nmax), dtype=np.uint8)
    room_h = np.ones((m, 3), dtype=np.float32)
    room_h_world = np.zeros((m, 3), dtype=np.float32)
    room_c_world = np.zeros((m, 3), dtype=np.float32)
    room_axes_world = np.zeros((m, 9), dtype=np.float32)
    yaw_rad = np.zeros((m, nmax), dtype=np.float32)

    for i, room in enumerate(rooms):
        cx, cy, cz = room.room_center
        hx, hy, hz = room.room_half_extents
        room_h_world[i] = np.array([hx, hy, hz], dtype=np.float32)
        room_c_world[i] = np.array([cx, cy, cz], dtype=np.float32)
        room_axes_world[i] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        for j, obj in enumerate(room.objects[:nmax]):
            px, py, pz = obj.position
            sx, sy, sz = obj.size_full
            pos_gt_xz[i, j, 0] = np.clip((px - cx) / max(hx, 1e-6), -1.25, 1.25)
            pos_gt_xz[i, j, 1] = np.clip((pz - cz) / max(hz, 1e-6), -1.25, 1.25)
            size_room[i, j, 0] = sx / max(hx, 1e-6)
            size_room[i, j, 1] = sy / max(hy, 1e-6)
            size_room[i, j, 2] = sz / max(hz, 1e-6)
            cat_id[i, j] = int(cat2id.get(obj.category, 0))
            mask[i, j] = 1
            yaw_rad[i, j] = float(obj.yaw_rad)

    return {
        "pos_gt_xz": pos_gt_xz,
        "size_room": size_room,
        "cat_id": cat_id,
        "mask": mask,
        "room_h": room_h,
        "room_h_world": room_h_world,
        "room_c_world": room_c_world,
        "room_axes_world": room_axes_world,
        "yaw_rad": yaw_rad,
    }


def assign_splits(
    rooms: List[RoomRecord],
    seed: int,
) -> Tuple[Dict[str, List[int]], int]:
    known = {"train": [], "val": [], "test": []}
    unresolved_idx: List[int] = []
    unresolved_houses: List[str] = []
    for idx, room in enumerate(rooms):
        if room.split in known:
            known[room.split].append(idx)
        else:
            unresolved_idx.append(idx)
            unresolved_houses.append(room.house_id)

    if not unresolved_idx:
        return known, 0

    fallback = split_by_house(unresolved_houses, seed=seed)
    for split_name, local_indices in fallback.items():
        for local_idx in local_indices:
            known[split_name].append(unresolved_idx[local_idx])
    for split_name in known:
        known[split_name] = sorted(known[split_name])
    return known, len(unresolved_idx)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full 3D-FRONT preprocessing for repair_v1 from 3DFRONT_65347 boxes.npz")
    ap.add_argument("--dataset-root", required=True, help="Path to /workspace/3DFRONT_65347")
    ap.add_argument("--prepared-info", default=None, help="Optional prepared_model_info.json for jid -> semantic mapping")
    ap.add_argument("--model-info", default=None, help="Optional model_info.json fallback for jid -> semantic mapping")
    ap.add_argument("--out-dir", required=True, help="Output root for repair_v1 dataset")
    ap.add_argument("--room-types", default="", help="Comma-separated canonical room types: bedroom,livingroom,diningroom,library")
    ap.add_argument("--include-augmented", action="store_true", help="Include processed_*_augmented subsets")
    ap.add_argument("--nmax", default="auto", help="'auto' to keep all objects, or integer cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of rooms for smoke tests")
    ap.add_argument("--skip-room-jsons", action="store_true", help="Do not export room.json / objects.v1.json / scene_gt.v1.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_index = load_model_index(Path(args.prepared_info).expanduser().resolve() if args.prepared_info else None)
    model_index = load_model_index(Path(args.model_info).expanduser().resolve() if args.model_info else None)
    split_maps = load_split_maps(dataset_root)
    boxes_paths = gather_boxes_paths(dataset_root, include_augmented=bool(args.include_augmented))
    if not boxes_paths:
        raise RuntimeError(f"No boxes.npz found under {dataset_root}")

    allowed_room_types = {
        canonical_room_type(x)
        for x in as_str(args.room_types).split(",")
        if x.strip()
    }

    rooms: List[RoomRecord] = []
    for boxes_path in boxes_paths:
        source_subset = boxes_path.parent.parent.name
        room_type = canonical_room_type(source_subset)
        if allowed_room_types and room_type not in allowed_room_types:
            continue
        rooms.append(build_room_record(boxes_path, prepared_index, model_index, split_maps))
        if args.limit > 0 and len(rooms) >= int(args.limit):
            break

    if not rooms:
        raise RuntimeError("No rooms survived filtering")

    max_objects = max(len(room.objects) for room in rooms)
    if as_str(args.nmax).strip().lower() == "auto":
        nmax = max_objects
    else:
        nmax = max(1, int(args.nmax))

    clipped_rooms = sum(1 for room in rooms if len(room.objects) > nmax)
    clipped_objects = sum(max(0, len(room.objects) - nmax) for room in rooms)

    cat_counter = Counter(obj.category for room in rooms for obj in room.objects)
    super_counter = Counter(obj.super_category for room in rooms for obj in room.objects)
    room_counter = Counter(room.room_type for room in rooms)
    subset_counter = Counter(room.source_subset for room in rooms)
    recognized_objects = sum(1 for room in rooms for obj in room.objects if obj.semantics_source != "class_index")
    total_objects = sum(len(room.objects) for room in rooms)

    categories = sorted(cat_counter.keys())
    cat2id = {cat: idx + 1 for idx, cat in enumerate(categories)}
    arrays = build_canonical_arrays(rooms=rooms, cat2id=cat2id, nmax=nmax)
    splits, unresolved_splits = assign_splits(rooms=rooms, seed=args.seed)

    canonical_dir = out_dir / "canonical"
    rooms_dir = out_dir / "rooms"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_room_jsons:
        rooms_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(canonical_dir / "repair_v1.npz", **arrays)

    rooms_meta: List[Dict[str, Any]] = []
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for room in rooms:
            export_paths = {}
            if not args.skip_room_jsons:
                export_paths = write_room_exports(room, rooms_dir)
            row = {
                "room_id": room.room_id,
                "room_name": room.room_name,
                "room_type": room.room_type,
                "house_id": room.house_id,
                "source_subset": room.source_subset,
                "source_dir": room.source_dir,
                "split": room.split,
                "object_count": len(room.objects),
                "room_center": [round6(x) for x in room.room_center],
                "room_half_extents": [round6(x) for x in room.room_half_extents],
                "room_polygon_xz": room.room_polygon_xz,
                "objects": [
                    {
                        "id": obj.object_id,
                        "uid": obj.uid,
                        "jid": obj.jid,
                        "category": obj.category,
                        "super_category": obj.super_category,
                        "style": obj.style,
                        "theme": obj.theme,
                        "material": obj.material,
                        "semantics_source": obj.semantics_source,
                        "class_id": obj.class_id,
                        "position_xyz": [round6(x) for x in obj.position],
                        "size_full_xyz": [round6(x) for x in obj.size_full],
                        "yaw_rad": round6(obj.yaw_rad),
                        "yaw_deg": round6(obj.yaw_deg),
                    }
                    for obj in room.objects
                ],
            } | export_paths
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rooms_meta.append(
                {
                    "room_id": room.room_id,
                    "room_name": room.room_name,
                    "room_type": room.room_type,
                    "house_id": room.house_id,
                    "source_subset": room.source_subset,
                    "split": row["split"],
                    "object_count": len(room.objects),
                }
            )

    meta = {
        "schema": "3dfront_repair_v1.meta",
        "dataset_root": str(dataset_root),
        "prepared_info": str(Path(args.prepared_info).expanduser().resolve()) if args.prepared_info else None,
        "model_info": str(Path(args.model_info).expanduser().resolve()) if args.model_info else None,
        "nmax": nmax,
        "max_objects_full": max_objects,
        "categories": ["<PAD>"] + categories,
        "cat2id": cat2id,
        "rooms": rooms_meta,
        "note": "pos_gt_xz and size_room are normalized by room half-extents derived from floor polygon bounds. size_room stores full object size / room half extent.",
    }
    save_json(canonical_dir / "repair_v1.meta.json", meta)
    save_json(canonical_dir / "repair_v1.splits.json", splits)

    stats = {
        "schema": "3dfront_repair_v1.stats",
        "rooms_total": len(rooms),
        "rooms_with_zero_objects": sum(1 for room in rooms if not room.objects),
        "objects_total_raw": total_objects,
        "objects_total_kept": total_objects - clipped_objects,
        "object_recall": 0.0 if total_objects <= 0 else round((total_objects - clipped_objects) / total_objects, 6),
        "recognized_objects": recognized_objects,
        "prepared_semantic_resolution_rate": 0.0 if total_objects <= 0 else round(recognized_objects / total_objects, 6),
        "max_objects_per_room": max_objects,
        "mean_objects_per_room": round(total_objects / max(len(rooms), 1), 6),
        "p95_objects_per_room": int(np.quantile(np.array([len(room.objects) for room in rooms], dtype=np.int32), 0.95)),
        "clipped_rooms_due_to_nmax": int(clipped_rooms),
        "clipped_objects_due_to_nmax": int(clipped_objects),
        "unresolved_split_rooms": int(unresolved_splits),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "room_type_counts": dict(room_counter),
        "source_subset_counts": dict(subset_counter),
        "top_categories": top_counter(cat_counter, limit=100),
        "top_super_categories": top_counter(super_counter, limit=30),
    }
    save_json(out_dir / "stats.json", stats)

    print(f"[repair_v1] rooms={len(rooms)} objects={total_objects} max_objects={max_objects} nmax={nmax}")
    print(f"[repair_v1] split_counts={stats['split_counts']} unresolved_split_rooms={unresolved_splits}")
    print(f"[repair_v1] clipped_rooms={clipped_rooms} clipped_objects={clipped_objects}")
    print(f"[repair_v1] wrote npz={canonical_dir / 'repair_v1.npz'}")
    print(f"[repair_v1] wrote meta={canonical_dir / 'repair_v1.meta.json'}")
    print(f"[repair_v1] wrote splits={canonical_dir / 'repair_v1.splits.json'}")
    print(f"[repair_v1] wrote stats={out_dir / 'stats.json'}")
    print(f"[repair_v1] wrote manifest={manifest_path}")


if __name__ == "__main__":
    main()
