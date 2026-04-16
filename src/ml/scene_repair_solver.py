#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from src.ml.local_diffusion_repair import repair_scene_dict
    from src.tools.evaluate_unified_scene import box_intersection_volume, parse_placements, parse_room, should_ignore_collision
    from src.tools.normalize_scene_format import build_aabb_from_center_size
except ImportError:
    from ml.local_diffusion_repair import repair_scene_dict
    from tools.evaluate_unified_scene import box_intersection_volume, parse_placements, parse_room, should_ignore_collision
    from tools.normalize_scene_format import build_aabb_from_center_size


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_pair(a: int, b: int) -> Tuple[int, int]:
    if a > b:
        a, b = b, a
    return (a, b)


def _aabb_overlap_3d(aabb_a: Dict[str, float], aabb_b: Dict[str, float], *, margin: float) -> bool:
    return (
        float(aabb_a["x_min"]) < float(aabb_b["x_max"]) - margin
        and float(aabb_a["x_max"]) > float(aabb_b["x_min"]) + margin
        and float(aabb_a["y_min"]) < float(aabb_b["y_max"]) - margin
        and float(aabb_a["y_max"]) > float(aabb_b["y_min"]) + margin
        and float(aabb_a["z_min"]) < float(aabb_b["z_max"]) - margin
        and float(aabb_a["z_max"]) > float(aabb_b["z_min"]) + margin
    )


def _collect_candidate_indices(scene: Dict[str, Any], scope: str) -> List[int]:
    items = scene.get("placements")
    if not isinstance(items, list):
        return []

    out: List[int] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if scope == "all":
            out.append(idx)
            continue
        meta = item.get("meta") or {}
        source = item.get("source") or {}
        if meta.get("supplier_binding_applied") or source.get("supplier_replaced"):
            out.append(idx)
    return out


def _build_ignore_pair_indices(scene: Dict[str, Any]) -> List[Tuple[int, int]]:
    items = scene.get("placements")
    if not isinstance(items, list):
        return []

    id_to_idx: Dict[str, int] = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if item_id:
            id_to_idx[item_id] = idx

    pairs: set[Tuple[int, int]] = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        meta = item.get("meta") or {}
        anchor_target_id = str(meta.get("supplier_support_anchor_target_id") or "").strip()
        if not anchor_target_id:
            continue
        anchor_idx = id_to_idx.get(anchor_target_id)
        if anchor_idx is None:
            continue
        pairs.add(_normalize_pair(idx, anchor_idx))
    return sorted(pairs)


def _item_issue_details(
    scene: Dict[str, Any],
    index: int,
    *,
    ignore_pair_indices: Sequence[Tuple[int, int]],
    room_margin: float,
    collision_margin: float,
) -> Dict[str, Any]:
    room = parse_room(scene)
    placements = parse_placements(scene)
    if index < 0 or index >= len(placements):
        raise IndexError(f"placement index out of range: {index}")

    ignore_pairs = {_normalize_pair(int(a), int(b)) for a, b in ignore_pair_indices}
    item = placements[index]
    room_height = float(room.z_max) - float(room.z_min)
    outside_room = (
        item.x_min < room.x_min - room_margin
        or item.x_max > room.x_max + room_margin
        or item.y_min < room.y_min - room_margin
        or item.y_max > room.y_max + room_margin
        or item.z_min < room.z_min - room_margin
        or (room_height > max(room_margin, 1e-6) and item.z_max > room.z_max + room_margin)
    )

    colliding_indices: List[int] = []
    overlap_volume_total = 0.0
    for other_idx, other in enumerate(placements):
        if other_idx == index:
            continue
        if _normalize_pair(index, other_idx) in ignore_pairs:
            continue
        if should_ignore_collision(item, other):
            continue
        if _aabb_overlap_3d(item.aabb, other.aabb, margin=collision_margin):
            colliding_indices.append(other_idx)
            overlap_volume_total += float(box_intersection_volume(item, other))

    return {
        "outside_room": bool(outside_room),
        "colliding_indices": colliding_indices,
        "collision_count": len(colliding_indices),
        "overlap_volume_total": float(overlap_volume_total),
    }


def _issue_key(issue: Dict[str, Any]) -> Tuple[int, int, float]:
    return (
        1 if bool(issue.get("outside_room")) else 0,
        int(issue.get("collision_count") or 0),
        round(float(issue.get("overlap_volume_total") or 0.0), 6),
    )


def _detect_invalid_candidate_indices(
    scene: Dict[str, Any],
    candidate_indices: Sequence[int],
    *,
    ignore_pair_indices: Sequence[Tuple[int, int]],
    room_margin: float,
    collision_margin: float,
) -> List[int]:
    bad: List[int] = []
    for index in candidate_indices:
        issue = _item_issue_details(
            scene,
            int(index),
            ignore_pair_indices=ignore_pair_indices,
            room_margin=room_margin,
            collision_margin=collision_margin,
        )
        if issue["outside_room"] or issue["collision_count"] > 0:
            bad.append(int(index))
    return bad


def _load_repair_meta(model_path: str | Path, meta_path: Optional[str | Path]) -> Dict[str, Any]:
    if meta_path:
        meta = load_json(meta_path)
        if isinstance(meta, dict) and isinstance(meta.get("cat2id"), dict):
            return meta
        raise RuntimeError("repair meta json must contain object field 'cat2id'")

    import torch

    checkpoint = torch.load(str(Path(model_path).expanduser().resolve()), map_location="cpu")
    cat_vocab = checkpoint.get("cat_vocab")
    num_categories = int(checkpoint.get("num_categories") or 0)
    if isinstance(cat_vocab, dict):
        if isinstance(cat_vocab.get("cat2id"), dict):
            return {"cat2id": {str(k): int(v) for k, v in cat_vocab["cat2id"].items()}}
        if all(isinstance(v, int) for v in cat_vocab.values()):
            return {"cat2id": {str(k): int(v) for k, v in cat_vocab.items()}}
    if isinstance(cat_vocab, list) and cat_vocab:
        start = 1 if len(cat_vocab) + 1 == num_categories else 0
        return {"cat2id": {str(cat): idx + start for idx, cat in enumerate(cat_vocab)}}
    raise RuntimeError("could not infer repair cat2id from checkpoint; pass --repair-meta explicitly")


def _resolve_default_trained_model(model_path: Optional[str | Path]) -> Optional[Path]:
    if model_path:
        return Path(model_path).expanduser().resolve()
    candidate = Path(__file__).resolve().parent / "models" / "diffusion_model_20260212.pt"
    if candidate.is_file():
        return candidate.resolve()
    return None


def _resolve_runtime_device(requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested in {"cpu", "mps", "cuda"}:
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _detect_checkpoint_kind(model_path: str | Path) -> str:
    import torch

    checkpoint = torch.load(str(Path(model_path).expanduser().resolve()), map_location="cpu")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"unsupported model checkpoint format: {model_path}")
    keys = list(state.keys())
    if any(str(k).startswith("room_enc.") for k in keys):
        return "layout_diffusion"
    if any(str(k).startswith("ctx_mlp.") or str(k).startswith("target_mlp.") for k in keys):
        return "repair_diffusion"
    raise RuntimeError(f"unsupported trained repair checkpoint kind: {model_path}")


def _scene_category_to_layout_label(item: Dict[str, Any]) -> Optional[str]:
    category = str(item.get("category") or "").strip()
    name = str(item.get("name") or "").lower()
    supplier_name = str((((item.get("meta") or {}).get("supplier_candidate") or {}).get("title")) or "").lower()
    text = f"{category} {name} {supplier_name}"

    if category == "BedFactory" or " bed" in text or "кровать" in text:
        return "Bed/Double Bed"
    if category in {"KitchenCabinetFactory", "SingleCabinetFactory"} or "cabinet" in text or "комод" in text or "sideboard" in text or "сервант" in text:
        return "Table/Sideboard / Side Cabinet / Console Table"
    if category in {"LargeShelfFactory", "SimpleBookcaseFactory"} or "shelf" in text or "полка" in text or "bookcase" in text:
        return "Cabinet/Shelf/Desk/Shelf"
    if category == "SimpleDeskFactory" or "desk" in text or "стол" in text:
        return "Table/Desk"
    if category == "CeilingLightFactory" or "ceiling lamp" in text or "chandelier" in text or "люстра" in text:
        return "Lighting/Ceiling Lamp"
    if category in {"FloorLampFactory", "DeskLampFactory"} or "floor lamp" in text or "lamp" in text or "торшер" in text:
        return "Lighting/Floor Lamp"
    if category == "LargePlantContainerFactory" or "plant" in text:
        return "Other"
    return None


def _is_layout_context_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    meta = item.get("meta") or {}
    if meta.get("supplier_support_anchor_target_id"):
        return False
    category = str(item.get("category") or "").strip()
    if category in {
        "BlanketFactory",
        "PillowFactory",
        "MattressFactory",
        "BookStackFactory",
        "BookColumnFactory",
        "NatureShelfTrinketsFactory",
        "MonitorFactory",
        "TowelFactory",
    }:
        return False
    return _scene_category_to_layout_label(item) is not None


def _collect_layout_model_indices(scene: Dict[str, Any], candidate_indices: Sequence[int]) -> List[int]:
    items = scene.get("placements")
    if not isinstance(items, list):
        return []

    candidate_set = {int(idx) for idx in candidate_indices}
    layout_indices: List[int] = []
    for idx, item in enumerate(items):
        if not _is_layout_context_item(item):
            continue
        if idx in candidate_set or str(item.get("category") or "").strip() == "BedFactory":
            layout_indices.append(idx)
    return layout_indices


def _build_layout_room_spec(scene: Dict[str, Any]) -> Dict[str, Any]:
    room = parse_room(scene)
    polygon = room.floor_polygon
    if polygon and len(polygon) >= 3:
        poly = [{"x": float(x), "z": float(y)} for x, y in polygon]
    else:
        poly = [
            {"x": float(room.x_min), "z": float(room.y_min)},
            {"x": float(room.x_max), "z": float(room.y_min)},
            {"x": float(room.x_max), "z": float(room.y_max)},
            {"x": float(room.x_min), "z": float(room.y_max)},
        ]
    return {"floor_polygon_xz": poly}


def _build_layout_model_items(scene: Dict[str, Any], layout_indices: Sequence[int]) -> List[Dict[str, Any]]:
    items = scene.get("placements")
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    for idx in layout_indices:
        item = items[int(idx)]
        label = _scene_category_to_layout_label(item) or "Other"
        size_m = item.get("size_m") or [0.6, 0.6, 0.6]
        sx = max(0.05, float(size_m[0]))
        sy = max(0.05, float(size_m[1]))
        sz = max(0.05, float(size_m[2]))
        out.append(
            {
                "name": label,
                "size_mm": [
                    int(round(sx * 1000.0)),
                    int(round(sz * 1000.0)),
                    int(round(sy * 1000.0)),
                ],
            }
        )
    return out


def _apply_layout_prediction_to_item(
    scene: Dict[str, Any],
    index: int,
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    room = parse_room(scene)
    out_scene = deepcopy(scene)
    item = out_scene["placements"][int(index)]
    size_m = item.get("size_m") or [0.6, 0.6, 0.6]
    half_x = 0.5 * max(0.0, float(size_m[0]))
    half_y = 0.5 * max(0.0, float(size_m[1]))
    x = max(float(room.x_min) + half_x, min(float(room.x_max) - half_x, float(prediction.get("x", 0.0))))
    y = max(float(room.y_min) + half_y, min(float(room.y_max) - half_y, float(prediction.get("z", 0.0))))
    z = float((item.get("position_m") or [0.0, 0.0, 0.0])[2])
    item["position_m"] = [x, y, z]
    item["aabb"] = build_aabb_from_center_size([x, y, z], item["size_m"])
    meta = item.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        item["meta"] = meta
    meta["scene_repair_ml_proposal"] = {
        "backend": "layout_diffusion",
        "pred_x": float(prediction.get("x", 0.0)),
        "pred_y": float(prediction.get("z", 0.0)),
        "pred_yaw_deg": float(prediction.get("yaw_deg", 0.0)),
    }
    return out_scene


def _run_layout_diffusion_solver(
    scene: Dict[str, Any],
    *,
    model_path: str | Path,
    device: str,
    infer_steps: int,
    rounds: int,
    max_bad: Optional[int],
    candidate_indices: Sequence[int],
    ignore_pair_indices: Sequence[Tuple[int, int]],
    room_margin: float,
    collision_margin: float,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        from src.ml.infer.diffusion_placer import DiffusionPlacer
    except ImportError:
        from ml.infer.diffusion_placer import DiffusionPlacer

    model_path = Path(model_path).expanduser().resolve()
    runtime_device = _resolve_runtime_device(device)
    placer = DiffusionPlacer.load(model_path=str(model_path), device=runtime_device)
    working = deepcopy(scene)
    history: List[Dict[str, Any]] = []
    layout_indices = _collect_layout_model_indices(working, candidate_indices)
    room_spec = _build_layout_room_spec(working)
    model_items = _build_layout_model_items(working, layout_indices)
    layout_index_to_slot = {scene_idx: slot for slot, scene_idx in enumerate(layout_indices)}

    if not model_items:
        raise RuntimeError("no layout objects could be mapped to the trained diffusion vocabulary")

    num_attempts = max(4, 8 * max(1, int(rounds)))
    for round_idx in range(max(1, int(rounds))):
        bad_indices = _detect_invalid_candidate_indices(
            working,
            candidate_indices,
            ignore_pair_indices=ignore_pair_indices,
            room_margin=room_margin,
            collision_margin=collision_margin,
        )
        if max_bad is not None:
            bad_indices = bad_indices[: max(0, int(max_bad))]
        if not bad_indices:
            break

        accepted = 0
        round_history: List[Dict[str, Any]] = []
        for target_index in bad_indices:
            slot = layout_index_to_slot.get(int(target_index))
            if slot is None:
                continue
            before_issue = _item_issue_details(
                working,
                int(target_index),
                ignore_pair_indices=ignore_pair_indices,
                room_margin=room_margin,
                collision_margin=collision_margin,
            )
            best_scene: Optional[Dict[str, Any]] = None
            best_issue = before_issue
            best_attempt: Optional[Dict[str, Any]] = None

            for attempt in range(num_attempts):
                sample_seed = int(seed + round_idx * 1000 + target_index * 37 + attempt)
                torch = __import__("torch")
                torch.manual_seed(sample_seed)
                np.random.seed(sample_seed)
                random.seed(sample_seed)
                preds = placer.predict(room_spec=room_spec, placed_items=model_items, steps=int(infer_steps))
                if slot >= len(preds):
                    continue
                candidate_scene = _apply_layout_prediction_to_item(working, int(target_index), preds[slot])
                after_issue = _item_issue_details(
                    candidate_scene,
                    int(target_index),
                    ignore_pair_indices=ignore_pair_indices,
                    room_margin=room_margin,
                    collision_margin=collision_margin,
                )
                if _issue_key(after_issue) < _issue_key(best_issue):
                    best_issue = after_issue
                    best_scene = candidate_scene
                    best_attempt = {
                        "seed": sample_seed,
                        "prediction": preds[slot],
                        "after_issue": after_issue,
                    }

            if best_scene is not None:
                accepted += 1
                working = best_scene
                round_history.append(
                    {
                        "index": int(target_index),
                        "before_issue": before_issue,
                        "best_attempt": best_attempt,
                    }
                )

            room_spec = _build_layout_room_spec(working)
            model_items = _build_layout_model_items(working, layout_indices)

        history.append(
            {
                "round": int(round_idx),
                "requested_indices": [int(idx) for idx in bad_indices],
                "accepted_count": int(accepted),
                "accepted": round_history,
            }
        )
        if accepted == 0:
            break

    report = {
        "used_mode": "trained",
        "backend": "layout_diffusion",
        "model_path": str(model_path),
        "device": runtime_device,
        "infer_steps": int(infer_steps),
        "rounds": int(rounds),
        "num_attempts_per_item": int(num_attempts),
        "layout_indices": [int(idx) for idx in layout_indices],
        "history": history,
    }
    return working, report


def _run_trained_solver(
    scene: Dict[str, Any],
    *,
    model_path: str | Path,
    meta_path: Optional[str | Path],
    device: str,
    infer_steps: int,
    rounds: int,
    max_bad: Optional[int],
    candidate_indices: Sequence[int],
    ignore_pair_indices: Sequence[Tuple[int, int]],
    room_margin: float,
    collision_margin: float,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checkpoint_kind = _detect_checkpoint_kind(model_path)
    if checkpoint_kind == "layout_diffusion":
        return _run_layout_diffusion_solver(
            scene,
            model_path=model_path,
            device=device,
            infer_steps=infer_steps,
            rounds=rounds,
            max_bad=max_bad,
            candidate_indices=candidate_indices,
            ignore_pair_indices=ignore_pair_indices,
            room_margin=room_margin,
            collision_margin=collision_margin,
            seed=seed,
        )

    try:
        from src.ml.infer.repair_diffusion import infer_one, load_model, pick_device
    except ImportError:
        from ml.infer.repair_diffusion import infer_one, load_model, pick_device

    requested_device = pick_device(device)
    meta = _load_repair_meta(model_path, meta_path)
    model, cfg, schedule = load_model(str(Path(model_path).expanduser().resolve()), requested_device)

    working = deepcopy(scene)
    history: List[Dict[str, Any]] = []

    for round_idx in range(max(1, int(rounds))):
        bad_indices = _detect_invalid_candidate_indices(
            working,
            candidate_indices,
            ignore_pair_indices=ignore_pair_indices,
            room_margin=room_margin,
            collision_margin=collision_margin,
        )
        if max_bad is not None:
            bad_indices = bad_indices[: max(0, int(max_bad))]
        if not bad_indices:
            break

        accepted_moves = 0
        for index in bad_indices:
            before_issue = _item_issue_details(
                working,
                index,
                ignore_pair_indices=ignore_pair_indices,
                room_margin=room_margin,
                collision_margin=collision_margin,
            )
            candidate_scene = infer_one(
                model=model,
                schedule=schedule,
                scene=working,
                meta=meta,
                index=index,
                device=requested_device,
                steps=infer_steps,
            )
            after_issue = _item_issue_details(
                candidate_scene,
                index,
                ignore_pair_indices=ignore_pair_indices,
                room_margin=room_margin,
                collision_margin=collision_margin,
            )
            if _issue_key(after_issue) >= _issue_key(before_issue):
                continue
            accepted_moves += 1
            history.append(
                {
                    "round": int(round_idx),
                    "index": int(index),
                    "before_issue": before_issue,
                    "after_issue": after_issue,
                }
            )
            working = candidate_scene

        if accepted_moves == 0:
            break

    report = {
        "used_mode": "trained",
        "backend": "repair_diffusion",
        "model_path": str(Path(model_path).expanduser().resolve()),
        "meta_path": str(Path(meta_path).expanduser().resolve()) if meta_path else None,
        "device": str(requested_device),
        "infer_steps": int(infer_steps),
        "rounds": int(rounds),
        "history": history,
        "model_cfg": cfg,
    }
    return working, report


def _run_local_solver(
    scene: Dict[str, Any],
    *,
    rounds: int,
    max_bad: Optional[int],
    candidate_indices: Sequence[int],
    ignore_pair_indices: Sequence[Tuple[int, int]],
    room_margin: float,
    collision_margin: float,
    local_steps: int,
    local_samples_per_step: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    working = deepcopy(scene)
    history: List[Dict[str, Any]] = []

    for round_idx in range(max(1, int(rounds))):
        bad_indices = _detect_invalid_candidate_indices(
            working,
            candidate_indices,
            ignore_pair_indices=ignore_pair_indices,
            room_margin=room_margin,
            collision_margin=collision_margin,
        )
        if max_bad is not None:
            bad_indices = bad_indices[: max(0, int(max_bad))]
        if not bad_indices:
            break

        repaired_scene, results, _ = repair_scene_dict(
            scene=working,
            bad_index=None,
            max_bad=None,
            steps=local_steps,
            samples_per_step=local_samples_per_step,
            room_margin=room_margin,
            seed=seed + round_idx,
            candidate_indices=bad_indices,
            ignore_pair_indices=ignore_pair_indices,
        )
        history.append(
            {
                "round": int(round_idx),
                "requested_indices": [int(idx) for idx in bad_indices],
                "results": [
                    {
                        "index": int(r.index),
                        "changed": bool(r.changed),
                        "success": bool(r.success),
                        "displacement_m": float(r.displacement_m),
                        "colliding_indices_before": [int(x) for x in r.colliding_indices_before],
                        "colliding_indices_after": [int(x) for x in r.colliding_indices_after],
                    }
                    for r in results
                ],
            }
        )
        working = repaired_scene
        if not any(r.success and r.changed for r in results):
            break

    report = {
        "used_mode": "local",
        "rounds": int(rounds),
        "local_steps": int(local_steps),
        "local_samples_per_step": int(local_samples_per_step),
        "seed": int(seed),
        "history": history,
    }
    return working, report


def repair_scene_dict_with_solver(
    scene: Dict[str, Any],
    *,
    mode: str = "auto",
    scope: str = "supplier",
    model_path: Optional[str | Path] = None,
    meta_path: Optional[str | Path] = None,
    device: str = "auto",
    infer_steps: int = 50,
    local_steps: int = 7,
    local_samples_per_step: int = 96,
    rounds: int = 2,
    max_bad: Optional[int] = None,
    room_margin: float = 0.02,
    collision_margin: float = 0.012,
    seed: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    requested_mode = str(mode or "auto").strip().lower()
    if requested_mode not in {"auto", "trained", "local"}:
        raise RuntimeError(f"unsupported repair mode: {requested_mode}")

    candidate_indices = _collect_candidate_indices(scene, scope=str(scope or "supplier").strip().lower())
    ignore_pair_indices = _build_ignore_pair_indices(scene)
    initial_bad = _detect_invalid_candidate_indices(
        scene,
        candidate_indices,
        ignore_pair_indices=ignore_pair_indices,
        room_margin=room_margin,
        collision_margin=collision_margin,
    )

    repair_error: Optional[str] = None
    used_mode: Optional[str] = None
    working = deepcopy(scene)
    stage_report: Dict[str, Any] = {}
    resolved_model_path = _resolve_default_trained_model(model_path)

    if requested_mode in {"auto", "trained"}:
        if not resolved_model_path:
            if requested_mode == "trained":
                raise RuntimeError("trained repair mode requires --repair-model or a default model in src/ml/models/diffusion_model_20260212.pt")
        else:
            try:
                working, stage_report = _run_trained_solver(
                    working,
                    model_path=resolved_model_path,
                    meta_path=meta_path,
                    device=device,
                    infer_steps=infer_steps,
                    rounds=rounds,
                    max_bad=max_bad,
                    candidate_indices=candidate_indices,
                    ignore_pair_indices=ignore_pair_indices,
                    room_margin=room_margin,
                    collision_margin=collision_margin,
                    seed=seed,
                )
                used_mode = "trained"
            except Exception as exc:
                repair_error = str(exc)
                if requested_mode == "trained":
                    raise

    if used_mode is None:
        working, stage_report = _run_local_solver(
            working,
            rounds=rounds,
            max_bad=max_bad,
            candidate_indices=candidate_indices,
            ignore_pair_indices=ignore_pair_indices,
            room_margin=room_margin,
            collision_margin=collision_margin,
            local_steps=local_steps,
            local_samples_per_step=local_samples_per_step,
            seed=seed,
        )
        used_mode = "local"

    final_bad = _detect_invalid_candidate_indices(
        working,
        candidate_indices,
        ignore_pair_indices=ignore_pair_indices,
        room_margin=room_margin,
        collision_margin=collision_margin,
    )

    meta = working.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        working["meta"] = meta
    meta["scene_repair_solver"] = {
        "requested_mode": requested_mode,
        "used_mode": used_mode,
        "scope": str(scope or "supplier").strip().lower(),
        "candidate_indices": [int(idx) for idx in candidate_indices],
        "ignored_pair_count": len(ignore_pair_indices),
        "initial_bad_indices": [int(idx) for idx in initial_bad],
        "final_bad_indices": [int(idx) for idx in final_bad],
        "room_margin": float(room_margin),
        "collision_margin": float(collision_margin),
        "repair_error": repair_error,
        "stage_report": stage_report,
    }

    report = {
        "requested_mode": requested_mode,
        "used_mode": used_mode,
        "scope": str(scope or "supplier").strip().lower(),
        "candidate_indices": [int(idx) for idx in candidate_indices],
        "initial_bad_indices": [int(idx) for idx in initial_bad],
        "final_bad_indices": [int(idx) for idx in final_bad],
        "ignored_pair_count": len(ignore_pair_indices),
        "repair_error": repair_error,
        "stage_report": stage_report,
    }
    return working, report


def repair_scene_file(
    scene_path: str | Path,
    out_path: str | Path,
    *,
    mode: str = "auto",
    scope: str = "supplier",
    model_path: Optional[str | Path] = None,
    meta_path: Optional[str | Path] = None,
    device: str = "auto",
    infer_steps: int = 50,
    local_steps: int = 7,
    local_samples_per_step: int = 96,
    rounds: int = 2,
    max_bad: Optional[int] = None,
    room_margin: float = 0.02,
    collision_margin: float = 0.012,
    seed: int = 0,
) -> Tuple[Path, Dict[str, Any]]:
    scene = load_json(scene_path)
    repaired_scene, report = repair_scene_dict_with_solver(
        scene,
        mode=mode,
        scope=scope,
        model_path=model_path,
        meta_path=meta_path,
        device=device,
        infer_steps=infer_steps,
        local_steps=local_steps,
        local_samples_per_step=local_samples_per_step,
        rounds=rounds,
        max_bad=max_bad,
        room_margin=room_margin,
        collision_margin=collision_margin,
        seed=seed,
    )
    out = Path(out_path).expanduser().resolve()
    save_json(out, repaired_scene)
    return out, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair supplier scene placements with learned diffusion model or explicit local fallback")
    ap.add_argument("--scene", required=True, help="Input scene.v1-like json")
    ap.add_argument("--out", required=True, help="Output repaired scene json")
    ap.add_argument("--mode", choices=["auto", "trained", "local"], default="auto")
    ap.add_argument("--scope", choices=["supplier", "all"], default="supplier")
    ap.add_argument("--model", default=None, help="Optional trained repair checkpoint; defaults to src/ml/models/diffusion_model_20260212.pt when present")
    ap.add_argument("--meta", default=None, help="Optional repair meta json with cat2id")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--infer-steps", type=int, default=50)
    ap.add_argument("--local-steps", type=int, default=7)
    ap.add_argument("--local-samples-per-step", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-bad", type=int, default=None)
    ap.add_argument("--room-margin", type=float, default=0.02)
    ap.add_argument("--collision-margin", type=float, default=0.012)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path, report = repair_scene_file(
        scene_path=args.scene,
        out_path=args.out,
        mode=args.mode,
        scope=args.scope,
        model_path=args.model,
        meta_path=args.meta,
        device=args.device,
        infer_steps=args.infer_steps,
        local_steps=args.local_steps,
        local_samples_per_step=args.local_samples_per_step,
        rounds=args.rounds,
        max_bad=args.max_bad,
        room_margin=args.room_margin,
        collision_margin=args.collision_margin,
        seed=args.seed,
    )
    print(
        "[scene_repair_solver] "
        f"used_mode={report['used_mode']} "
        f"scope={report['scope']} "
        f"initial_bad={len(report['initial_bad_indices'])} "
        f"final_bad={len(report['final_bad_indices'])} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()
