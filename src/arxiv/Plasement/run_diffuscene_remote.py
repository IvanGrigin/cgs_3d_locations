#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/run_diffuscene_remote.py

import argparse
import json
import math
import os
import re
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


CATEGORY_TO_DIFFUSCENE = {
    "armchair": "armchair",
    "bookcase / jewelry armoire": "bookshelf",
    "ceiling lamp": "ceiling_lamp",
    "children cabinet": "children_cabinet",
    "coffee table": "coffee_table",
    "corner/side table": "nightstand",
    "desk": "desk",
    "dining chair": "chair",
    "dining table": "table",
    "drawer chest / corner cabinet": "cabinet",
    "dressing chair": "dressing_chair",
    "dressing table": "dressing_table",
    "footstool / sofastool / bed end stool / stool": "stool",
    "kids bed": "kids_bed",
    "king-size bed": "double_bed",
    "single bed": "single_bed",
    "nightstand": "nightstand",
    "pendant lamp": "pendant_lamp",
    "round end table": "nightstand",
    "shelf": "shelf",
    "sideboard / side cabinet / console table": "cabinet",
    "tv stand": "tv_stand",
    "wardrobe": "wardrobe",
    "wine cabinet": "cabinet",
    "chaise longue sofa": "sofa",
    "l-shaped sofa": "sofa",
    "lazy sofa": "sofa",
    "loveseat sofa": "sofa",
    "three-seat / multi-seat sofa": "sofa",
    "lounge chair / cafe chair / office chair": "chair",
    "classic chinese chair": "chair",
    "barstool": "stool",
}

TMP_ROOT = "out/tmp"


def _norm_cat(s: Any) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def deep_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def run_live(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    print("▶", " ".join(cmd))
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def make_run_dir(run_dir_arg: Optional[str], run_name: str) -> Tuple[Path, bool]:
    if run_dir_arg:
        p = Path(run_dir_arg).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p, True

    name = run_name.strip() if run_name.strip() else f"run_{secrets.token_hex(6)}"
    p = (Path(TMP_ROOT) / name).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p, False


def extract_class_name(obj: Dict[str, Any]) -> str:
    for key in ("class_name", "class", "type"):
        v = obj.get(key)
        if v:
            return str(v)

    asset_meta = obj.get("asset_meta") or {}
    candidates = [
        asset_meta.get("category"),
        asset_meta.get("super-category"),
        asset_meta.get("super_category"),
        obj.get("category"),
        obj.get("name"),
    ]

    for cand in candidates:
        norm = _norm_cat(cand)
        if norm in CATEGORY_TO_DIFFUSCENE:
            return CATEGORY_TO_DIFFUSCENE[norm]

    raise ValueError(
        "Не удалось определить class_name для объекта. "
        f"category={asset_meta.get('category')!r}, "
        f"super_category={asset_meta.get('super-category') or asset_meta.get('super_category')!r}, "
        f"name={obj.get('name')!r}"
    )


def extract_jid(obj: Dict[str, Any]) -> str:
    for key in ("jid", "model_jid", "future_jid"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    asset_meta = obj.get("asset_meta") or {}
    v = asset_meta.get("model_id")
    if isinstance(v, str) and v.strip():
        return v.strip()

    return ""


def extract_size_m(obj: Dict[str, Any]) -> List[float]:
    for key in ("size_m", "bbox_size_m", "size"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 3:
            return [float(v[0]), float(v[1]), float(v[2])]

    for key in ("min_size_mm", "max_size_mm"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 3:
            return [float(v[0]) / 1000.0, float(v[1]) / 1000.0, float(v[2]) / 1000.0]

    asset_meta = obj.get("asset_meta") or {}
    if all(k in asset_meta for k in ("size_x", "size_y", "size_z")):
        return [
            float(asset_meta["size_x"]),
            float(asset_meta["size_y"]),
            float(asset_meta["size_z"]),
        ]

    raise ValueError(f"Не удалось определить size_m для объекта: {obj}")


def extract_translation_m(obj: Dict[str, Any]) -> List[float]:
    for key in ("translation_m", "position_m", "position"):
        v = obj.get(key)
        if isinstance(v, list) and len(v) == 3:
            return [float(v[0]), float(v[1]), float(v[2])]
    return [0.0, 0.0, 0.0]


def extract_yaw_rad(obj: Dict[str, Any]) -> float:
    for key in ("yaw_rad", "rotation_yaw_rad", "angle_rad"):
        v = obj.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def convert_local_objects_to_server_input(
    src_objects: Dict[str, Any],
    domain: str,
) -> Dict[str, Any]:
    src_items = src_objects.get("objects") or src_objects.get("items") or src_objects.get("placements") or []
    if not isinstance(src_items, list):
        raise ValueError("objects.json: поле objects/items/placements должно быть списком")

    out_items = []
    for obj in src_items:
        class_name = extract_class_name(obj)
        jid = extract_jid(obj)
        size_m = extract_size_m(obj)
        translation_m = extract_translation_m(obj)
        yaw_rad = extract_yaw_rad(obj)

        item = {
            "class_name": class_name,
            "size_m": size_m,
            "yaw_rad": yaw_rad,
            "translation_m": translation_m,
        }
        if jid:
            item["jid"] = jid
        out_items.append(item)

    return {
        "version": "1.0",
        "domain": domain,
        "objects": out_items,
    }


def quantize_rot_0_90_180_270(deg: float) -> float:
    a = float(deg or 0.0) % 360.0
    allowed = (0.0, 90.0, 180.0, 270.0)
    return min(allowed, key=lambda t: (abs(((a - t + 180.0) % 360.0) - 180.0), t))


def make_aabb_from_room_pos_and_size(
    position_room_xy_m,
    size_m,
    yaw_rad: float,
    z_floor_m: float = 0.0,
):
    cx = float(position_room_xy_m[0])
    cy = float(position_room_xy_m[1])

    sx = float(size_m[0])
    sy = float(size_m[1])
    sz = float(size_m[2])

    yaw_deg = math.degrees(float(yaw_rad))
    rot_deg = quantize_rot_0_90_180_270(yaw_deg)

    box_x, box_y = sx, sy
    if rot_deg in (90.0, 270.0):
        box_x, box_y = sy, sx

    z0 = float(z_floor_m)

    return {
        "x_min": cx - box_x / 2.0,
        "x_max": cx + box_x / 2.0,
        "y_min": cy - box_y / 2.0,
        "y_max": cy + box_y / 2.0,
        "z_min": z0,
        "z_max": z0 + sz,
    }, rot_deg, yaw_deg


def build_local_placement_from_server_result(
    local_objects: Dict[str, Any],
    server_placements: Dict[str, Any],
) -> Dict[str, Any]:
    src_items = local_objects.get("objects") or local_objects.get("items") or local_objects.get("placements") or []
    result_items = server_placements.get("items") or server_placements.get("placements") or []

    if len(result_items) != len(src_items):
        raise ValueError(
            f"Количество объектов не совпадает: local={len(src_items)}, server={len(result_items)}"
        )

    placements = []
    for i, (src_obj, pred) in enumerate(zip(src_items, result_items)):
        pos_xy = pred["position_room_xy_m"]
        size_m = pred["size_m"]
        yaw_rad = float(pred["yaw_rad"])
        z_floor_m = float(pred.get("z_floor_m", 0.0))

        aabb, rot_deg, yaw_deg = make_aabb_from_room_pos_and_size(
            position_room_xy_m=pos_xy,
            size_m=size_m,
            yaw_rad=yaw_rad,
            z_floor_m=z_floor_m,
        )

        item = dict(src_obj)
        item["placement_source"] = "diffuscene_remote"
        item["aabb"] = aabb
        item["bbox"] = dict(aabb)
        item["rotation"] = rot_deg
        item["position_room_xy_m"] = [float(pos_xy[0]), float(pos_xy[1])]
        item["z_floor_m"] = z_floor_m
        item["size_m"] = [float(size_m[0]), float(size_m[1]), float(size_m[2])]
        item["yaw_rad"] = yaw_rad
        item["yaw_deg"] = yaw_deg
        item["server_class_name"] = pred.get("class_name")
        item["server_index"] = pred.get("i", i)

        placements.append(item)

    return {
        "placer": "diffuscene_remote",
        "placements": placements,
        "server_raw": server_placements,
    }


def detect_download_dir_from_output(output_text: str) -> Optional[Path]:
    m = re.search(r"Done\.\s*Results in:\s*(.+)", output_text)
    if not m:
        return None
    return Path(m.group(1).strip()).expanduser().resolve()


def find_results_dir(run_dir: Path, run_name: str, output_text: str) -> Path:
    candidates = []

    parsed = detect_download_dir_from_output(output_text)
    if parsed is not None:
        candidates.append(parsed)

    candidates.append(run_dir)
    candidates.append((Path.cwd() / run_name).resolve())
    candidates.append((Path.cwd() / "out" / "tmp" / run_name).resolve())

    seen = set()
    uniq = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq.append(c)

    for c in uniq:
        if (c / "placements_room.json").is_file():
            return c

    checked = "\n".join(str(c) for c in uniq)
    raise FileNotFoundError(
        "Не найден placements_room.json. Проверены каталоги:\n" + checked
    )


def copy_results_into_run_dir(src_dir: Path, run_dir: Path) -> None:
    if src_dir.resolve() == run_dir.resolve():
        return

    for name in (
        "placements_room.json",
        "placements_room_check.json",
        "pred_bbox.json",
        "pred_bbox_metric.json",
    ):
        src = src_dir / name
        dst = run_dir / name
        if not src.is_file():
            continue
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)


def safe_copy_if_needed(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        pass

    shutil.copy2(src, dst)


def resolve_remote_value(
    cli_value: Optional[str],
    cfg: Dict[str, Any],
    keys: List[str],
    default: Optional[str] = None,
) -> Optional[str]:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value)
    val = deep_get(cfg, keys, default)
    if val is None:
        return default
    return str(val)


def resolve_remote_port(
    cli_value: Optional[int],
    cfg: Dict[str, Any],
    keys: List[str],
    default: int = 22,
) -> int:
    if cli_value is not None:
        return int(cli_value)
    val = deep_get(cfg, keys, default)
    return int(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="Локальный room.json")
    ap.add_argument("--objects", required=True, help="Локальный objects.json")
    ap.add_argument("--out", required=True, help="Итоговый placement_result.json")
    ap.add_argument("--run-dir", default=None, help="Каталог запуска")
    ap.add_argument("--run-name", default="", help="Имя запуска")
    ap.add_argument("--remote-runner", default="./src/run_room_layout_remote.sh")
    ap.add_argument("--paths-config", default=None, help="Путь к YAML-конфигу путей")
    ap.add_argument(
        "--domain",
        default="bedroom",
        choices=["bedroom", "livingroom", "diningroom", "library"],
        help="Домен DiffuScene",
    )

    # Эти аргументы раньше ломали argparse. Теперь поддерживаются.
    ap.add_argument("--remote-host", default=None, help="SSH host удалённого сервера")
    ap.add_argument("--remote-port", type=int, default=None, help="SSH port удалённого сервера")
    ap.add_argument("--remote-user", default=None, help="SSH user удалённого сервера")
    ap.add_argument("--remote-key", default=None, help="SSH private key")

    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.paths_config:
        cfg = load_yaml(Path(args.paths_config).expanduser().resolve())

    room_path = Path(args.room).expanduser().resolve()
    objects_path = Path(args.objects).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not room_path.is_file():
        raise FileNotFoundError(room_path)
    if not objects_path.is_file():
        raise FileNotFoundError(objects_path)

    tmp_root_cfg = deep_get(cfg, ["local", "output", "tmp_root"], None)
    global TMP_ROOT
    if tmp_root_cfg:
        TMP_ROOT = str(Path(tmp_root_cfg).expanduser().resolve())

    remote_host = resolve_remote_value(args.remote_host, cfg, ["remote", "ssh", "host"])
    remote_user = resolve_remote_value(args.remote_user, cfg, ["remote", "ssh", "user"])
    remote_key = resolve_remote_value(args.remote_key, cfg, ["remote", "ssh", "key"])
    remote_port = resolve_remote_port(args.remote_port, cfg, ["remote", "ssh", "port"], 22)

    run_name = args.run_name.strip() if args.run_name.strip() else f"run_{secrets.token_hex(6)}"
    run_dir, _ = make_run_dir(args.run_dir, run_name)

    print(f"📁 run_dir: {run_dir}")

    local_objects = load_json(objects_path)
    server_input = convert_local_objects_to_server_input(local_objects, domain=args.domain)

    local_room_copy = run_dir / "room.json"
    local_server_objects = run_dir / "objects_for_server.json"
    local_manifest = run_dir / "run_manifest.json"

    shutil.copy2(room_path, local_room_copy)
    save_json(local_server_objects, server_input)

    save_json(local_manifest, {
        "room": str(room_path),
        "objects": str(objects_path),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "remote_runner": args.remote_runner,
        "paths_config": args.paths_config,
        "domain": args.domain,
        "remote_host": remote_host,
        "remote_port": remote_port,
        "remote_user": remote_user,
        "remote_key": remote_key,
    })

    runner = Path(args.remote_runner).expanduser()
    runner_cmd = str(runner.resolve()) if runner.exists() else args.remote_runner

    env = os.environ.copy()

    # Прокидываем SSH-параметры shell-раннеру через environment.
    # Это не ломает старый сценарий и даёт run_room_layout_remote.sh доступ к конфигу.
    if remote_host:
        env["DIFFUSCENE_REMOTE_HOST"] = remote_host
    if remote_user:
        env["DIFFUSCENE_REMOTE_USER"] = remote_user
    if remote_key:
        env["DIFFUSCENE_REMOTE_KEY"] = str(Path(remote_key).expanduser())
    env["DIFFUSCENE_REMOTE_PORT"] = str(remote_port)
    env["DIFFUSCENE_RUN_NAME"] = run_name
    env["DIFFUSCENE_RUN_DIR"] = str(run_dir)
    env["DIFFUSCENE_DOMAIN"] = args.domain
    if args.paths_config:
        env["DIFFUSCENE_PATHS_CONFIG"] = str(Path(args.paths_config).expanduser().resolve())

    proc = run_live(
        [
            runner_cmd,
            str(local_room_copy),
            str(local_server_objects),
            run_name,
        ],
        capture_output=True,
        env=env,
    )

    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")

    output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    results_dir = find_results_dir(run_dir=run_dir, run_name=run_name, output_text=output_text)
    print(f"📥 results_dir: {results_dir}")

    copy_results_into_run_dir(results_dir, run_dir)

    placements_path = run_dir / "placements_room.json"
    if not placements_path.is_file():
        raise FileNotFoundError(f"Не найден placements_room.json в {run_dir}")

    server_placements = load_json(placements_path)
    placement_result = build_local_placement_from_server_result(local_objects, server_placements)

    save_json(out_path, placement_result)

    extra_map = {
        "placements_room.json": out_path.with_name("placements_room.json"),
        "placements_room_check.json": out_path.with_name("placements_room_check.json"),
        "pred_bbox.json": out_path.with_name("pred_bbox.json"),
        "pred_bbox_metric.json": out_path.with_name("pred_bbox_metric.json"),
    }
    for src_name, dst_path in extra_map.items():
        src = run_dir / src_name
        safe_copy_if_needed(src, dst_path)

    print(f"OK: saved placement -> {out_path}")
    print(f"OK: run artifacts -> {run_dir}")


if __name__ == "__main__":
    main()