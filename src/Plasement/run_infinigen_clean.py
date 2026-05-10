#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/run_infinigen_clean.py

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from src.prompt_compiler.inventory_mapping import factory_to_semantic, is_core_furniture_factory, is_technical_factory_name
except ModuleNotFoundError:
    _FALLBACK_FACTORY_TO_SEMANTIC = {
        "BedFactory": "Bed",
        "Lighting": "Lighting",
        "LampFactory": "Lighting",
        "DeskLampFactory": "Lighting",
        "FloorLampFactory": "Lighting",
        "CeilingLightFactory": "CeilingLight",
        "SingleCabinetFactory": "Storage",
        "SimpleBookcaseFactory": "Storage",
        "LargeShelfFactory": "Storage",
        "CellShelfFactory": "Storage",
        "KitchenCabinetFactory": "Storage",
        "StandingSinkFactory": "Sink",
        "SinkFactory": "Sink",
        "ToiletFactory": "Toilet",
        "BathtubFactory": "Bathtub",
        "ShowerFactory": "Shower",
        "SideTableFactory": "SideTable",
        "SidetableDeskFactory": "SideTable",
        "MirrorFactory": "WallDecoration",
        "WallArtFactory": "WallDecoration",
        "BookStackFactory": "Decor",
        "BookColumnFactory": "Decor",
        "NatureShelfTrinketsFactory": "Decor",
        "LargePlantContainerFactory": "LargePlant",
        "RugFactory": "Rug",
        "ChairFactory": "Chair",
        "ArmChairFactory": "Chair",
    }
    _FALLBACK_NON_CORE_FACTORIES = {
        "BookStackFactory",
        "BookColumnFactory",
        "NatureShelfTrinketsFactory",
    }

    def factory_to_semantic(factory_name: str) -> str | None:
        return _FALLBACK_FACTORY_TO_SEMANTIC.get(str(factory_name or "").strip())

    def is_core_furniture_factory(factory_name: str) -> bool:
        name = str(factory_name or "").strip()
        if not name or name in _FALLBACK_NON_CORE_FACTORIES:
            return False
        semantic = factory_to_semantic(name)
        return semantic not in {None, "", "Decor"}

    def is_technical_factory_name(name: str) -> bool:
        low = str(name or "").strip().lower()
        return low.startswith("hoof_parent_temp") or low.startswith("beziercurve")

MAX_NUMPY_SEED = 2**32 - 1
REMOTE_MIN_FREE_KB = 8 * 1024 * 1024


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


TECHNICAL_NAME_PREFIXES = {"cube", "plane", "mesh", "beziercurve", "curve"}


def _looks_technical_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if low in TECHNICAL_NAME_PREFIXES:
        return True
    return low.startswith("cube.") or low.startswith("plane.") or low.startswith("mesh.")


def clean_placement_payload(placement_data: Dict[str, Any]) -> Dict[str, Any]:
    placements = placement_data.get("placements") or []
    cleaned: list[Dict[str, Any]] = []
    for item in placements:
        name = str(item.get("name") or "")
        category = str(item.get("category") or "")
        if (
            _looks_technical_name(name)
            or _looks_technical_name(category)
            or is_technical_factory_name(name)
            or is_technical_factory_name(category)
        ):
            continue
        cleaned.append(item)
    out = dict(placement_data)
    out["placements"] = cleaned
    meta = dict(out.get("meta") or {})
    meta["object_count"] = len(cleaned)
    meta["filtered_technical_objects"] = len(placements) - len(cleaned)
    out["meta"] = meta
    return out


def build_inventory_from_placement(placement_data: Dict[str, Any]) -> Dict[str, Any]:
    placements = placement_data.get("placements") or []
    inventory_items: list[Dict[str, Any]] = []
    factory_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    core_factory_counts: Counter[str] = Counter()
    core_semantic_counts: Counter[str] = Counter()
    for item in placements:
        factory_name = str(item.get("name") or item.get("category") or "").strip()
        semantic = factory_to_semantic(factory_name) or str(item.get("category") or factory_name).strip()
        inventory_items.append(
            {
                "id": str(item.get("id") or ""),
                "factory_name": factory_name,
                "semantic": semantic,
                "position_m": item.get("position_m"),
                "size_m": item.get("size_m"),
            }
        )
        if factory_name:
            factory_counts[factory_name] += 1
        if semantic:
            semantic_counts[semantic] += 1
        if is_core_furniture_factory(factory_name):
            if factory_name:
                core_factory_counts[factory_name] += 1
            if semantic:
                core_semantic_counts[semantic] += 1
    return {
        "items": inventory_items,
        "summary": {
            "raw_real_object_count": len(inventory_items),
            "real_object_count": sum(core_factory_counts.values()),
            "factory_counts": dict(factory_counts),
            "semantic_counts": dict(semantic_counts),
            "core_factory_counts": dict(core_factory_counts),
            "core_semantic_counts": dict(core_semantic_counts),
        },
    }


def parse_solver_log(log_path: str | Path) -> Dict[str, Any]:
    p = Path(log_path).expanduser().resolve()
    if not p.is_file():
        return {
            "exists": False,
            "termination_status": "missing",
            "violations": {},
            "stage_markers": [],
            "warnings": [],
            "errors": [],
        }
    text = p.read_text(encoding="utf-8", errors="replace")
    violations: Dict[str, Any] = {}
    for match in re.finditer(r"violations=(\{.*?\})", text):
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, dict):
                violations = {str(k): v for k, v in parsed.items()}
        except Exception:
            continue
    stage_markers: list[str] = []
    for marker in [
        "solve_rooms",
        "solve_large",
        "solve_medium",
        "solve_small",
        "populate_assets",
        "room_doors",
        "room_windows",
        "room_walls",
        "room_floors",
        "room_ceilings",
    ]:
        if marker in text:
            stage_markers.append(marker)
    warnings = [line.strip() for line in text.splitlines() if "warning" in line.lower()]
    errors = [line.strip() for line in text.splitlines() if "error" in line.lower() or "traceback" in line.lower()]
    no_object_lines = [line.strip() for line in text.splitlines() if "No objects to be added for desc_full=" in line]
    termination_status = "success"
    if errors:
        termination_status = "error"
    elif "__OK__" not in text and "OK:" not in text and "populate_assets" not in text:
        termination_status = "unknown"
    return {
        "exists": True,
        "termination_status": termination_status,
        "violations": violations,
        "stage_markers": stage_markers,
        "warnings": warnings[-20:],
        "errors": errors[-20:],
        "no_object_events": no_object_lines,
        "empty_candidate_pool_detected": any(
            "on_floor_and_wall" in line or "on_floor_freestanding" in line for line in no_object_lines
        ),
    }


def write_inventory_artifacts(candidate_dir: str | Path, placement_path: str | Path) -> tuple[Path, Path]:
    candidate_path = Path(candidate_dir).expanduser().resolve()
    placement_data = load_json(placement_path)
    cleaned = clean_placement_payload(placement_data)
    placement_file = Path(placement_path).expanduser().resolve()
    placement_file.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory = build_inventory_from_placement(cleaned)
    inventory_path = candidate_path / "inventory.json"
    inventory_summary_path = candidate_path / "inventory_summary.json"
    inventory_path.write_text(json.dumps(inventory["items"], ensure_ascii=False, indent=2), encoding="utf-8")
    inventory_summary_path.write_text(json.dumps(inventory["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    return inventory_path, inventory_summary_path


def build_ssh_base(args: argparse.Namespace, *, allocate_tty: bool) -> list[str]:
    cmd = ["ssh"]
    if allocate_tty:
        cmd.append("-tt")
    cmd += [
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "StrictHostKeyChecking=no",
    ]
    if args.remote_port:
        cmd += ["-p", str(int(args.remote_port))]
    if args.remote_key:
        cmd += ["-i", str(Path(args.remote_key).expanduser())]
    cmd.append(f"{args.remote_user}@{args.remote_host}")
    return cmd


def wrap_remote(command: str) -> str:
    command = f"set -e; {command}; echo __OK__"
    return f"bash -lc {shlex.quote(command)}"


def wrap_remote_bash(command: str) -> str:
    return f"bash -lc {shlex.quote(command)}"


def ssh_run(args: argparse.Namespace, command: str) -> None:
    cmd = build_ssh_base(args, allocate_tty=True) + [wrap_remote(command)]
    print("▶", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ssh_upload_file(args: argparse.Namespace, local_path: Path, remote_path: str) -> None:
    payload = base64.b64encode(local_path.read_bytes())
    remote_inner = f"set -e; base64 -d > {shlex.quote(remote_path)}"
    cmd = build_ssh_base(args, allocate_tty=False) + [wrap_remote_bash(remote_inner)]
    print(f"▶ upload: {local_path} -> {remote_path}")
    subprocess.run(cmd, input=payload, check=True)


def ssh_download_file(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_inner = f"set -e; base64 {shlex.quote(remote_path)}"
    cmd = build_ssh_base(args, allocate_tty=False) + [wrap_remote_bash(remote_inner)]
    print(f"▶ download: {remote_path} -> {local_path}")
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    local_path.write_bytes(base64.b64decode(completed.stdout))


def ssh_capture(args: argparse.Namespace, command: str) -> subprocess.CompletedProcess[str]:
    cmd = build_ssh_base(args, allocate_tty=False) + [wrap_remote_bash(f"set -e; {command}")]
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )


def ensure_remote_free_space(args: argparse.Namespace, *, min_free_kb: int = REMOTE_MIN_FREE_KB) -> None:
    completed = ssh_capture(args, "df -Pk /workspace /workspace/tmp")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    free_values: list[int] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            free_values.append(int(parts[3]))
        except ValueError:
            continue
    if not free_values:
        return
    min_available = min(free_values)
    if min_available < int(min_free_kb):
        free_gb = min_available / (1024 * 1024)
        need_gb = int(min_free_kb) / (1024 * 1024)
        raise RuntimeError(
            f"REMOTE_DISK_FULL: remote /workspace has only {free_gb:.1f} GiB free, "
            f"need at least {need_gb:.1f} GiB before running Infinigen"
        )


def build_remote_preamble(args: argparse.Namespace) -> list[str]:
    parts: list[str] = []
    if args.remote_conda_env:
        parts += [
            "source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true",
            "source /opt/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true",
            "source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true",
            "command -v conda >/dev/null 2>&1 || { echo 'conda not found on remote host' >&2; exit 127; }",
            f"conda activate {shlex.quote(args.remote_conda_env)}",
        ]
    return parts


def parse_seed_value(seed_value: str | int) -> int:
    s = str(seed_value).strip().lower()
    if re.fullmatch(r"[0-9a-f]+", s) and re.search(r"[a-f]", s):
        return int(s, 16)
    return int(s)


def normalize_seed(seed_value: str | int) -> int:
    return parse_seed_value(seed_value) % (MAX_NUMPY_SEED + 1)


def normalize_seed_for_infinigen(seed_value: str | int) -> str:
    n = parse_seed_value(seed_value)
    n = (n % 0xFFFFFFFE) + 1
    return f"{n:x}"


def default_infinigen_src() -> Path:
    env = os.environ.get("INFINIGEN_SRC")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    candidates = [
        Path("/workspace/infinigen/src"),
        Path(__file__).resolve().parents[2] / "infinigen" / "src",
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise RuntimeError("Не найден infinigen/src. Укажи --infinigen-src или INFINIGEN_SRC")


def infer_room_semantic(room_data: Dict[str, Any]) -> str:
    room = room_data.get("room", room_data)
    explicit_map = {
        "bedroom": "bedroom",
        # Treat a studio as a room that must have a sleeping zone. The
        # requirements pass can add living/dining furniture later, but if the
        # base placement starts as a living room it can omit the bed entirely.
        "studio": "bedroom",
        "studio apartment": "bedroom",
        "living room": "living-room",
        "livingroom": "living-room",
        "dining room": "dining-room",
        "diningroom": "dining-room",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "toilet": "restroom",
        "wc": "restroom",
        "restroom": "restroom",
        "hallway": "hallway",
        "corridor": "hallway",
        "living room kitchen": "living-room",
        "loggia": "balcony",
        "balcony": "balcony",
    }
    for key in ("source_room_type", "type", "room_type"):
        explicit_room_type = " ".join(
            str(room.get(key, "")).strip().lower().replace("_", " ").replace("-", " ").split()
        ).strip()
        if explicit_room_type in explicit_map:
            return explicit_map[explicit_room_type]

    raw = " ".join(
        [
            str(room.get("name", "")),
            str(room.get("id", "")),
            str(room.get("type", "")),
            str(room.get("room_type", "")),
        ]
    ).lower()

    if "bed" in raw or "спаль" in raw:
        return "bedroom"
    if "living" in raw or "гостин" in raw:
        return "living-room"
    if "kitchen" in raw or "кух" in raw:
        return "kitchen"
    if "dining" in raw or "столов" in raw:
        return "dining-room"
    if "wc" in raw or "restroom" in raw or "toilet" in raw or "туал" in raw:
        return "restroom"
    if "bath" in raw or "ван" in raw or "сануз" in raw:
        return "bathroom"
    if "hallway" in raw or "hall" in raw or "corridor" in raw or "прихож" in raw or "корид" in raw:
        return "hallway"
    if "loggia" in raw or "balcony" in raw or "лодж" in raw or "балкон" in raw:
        return "balcony"

    return "bedroom"


def infer_room_semantic_from_style_profile(style_profile: Dict[str, Any] | None) -> str | None:
    if not isinstance(style_profile, dict):
        return None
    raw = str(style_profile.get("room_type") or "").strip().lower().replace("_", " ").replace("-", " ")
    raw = " ".join(raw.split())
    explicit_map = {
        "bedroom": "bedroom",
        "living room": "living-room",
        "livingroom": "living-room",
        "dining room": "dining-room",
        "diningroom": "dining-room",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "toilet": "restroom",
        "wc": "restroom",
        "restroom": "restroom",
        "hallway": "hallway",
        "corridor": "hallway",
        "balcony": "balcony",
        "loggia": "balcony",
    }
    return explicit_map.get(raw)


def has_source_restroom_type(room_data: Dict[str, Any]) -> bool:
    room = room_data.get("room", room_data)
    raw = str(room.get("source_room_type") or "").strip().lower().replace("_", " ").replace("-", " ")
    raw = " ".join(raw.split())
    return raw in {"toilet", "wc", "restroom"}


def infer_room_polygon(room_data: Dict[str, Any]) -> List[Tuple[float, float]]:
    room = room_data.get("room", room_data)
    poly = room.get("floor_polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        raise RuntimeError("room.json должен содержать room.floor_polygon")
    out: List[Tuple[float, float]] = []
    for pt in poly:
        out.append((float(pt["x"]), float(pt["y"])))
    return out


def infer_walls(room_data: Dict[str, Any], poly: List[Tuple[float, float]]) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
    room = room_data.get("room", room_data)
    walls = room.get("walls")
    out: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    if isinstance(walls, list) and walls:
        for i, w in enumerate(walls):
            wid = str(w.get("id", f"w{i}"))
            a = int(w["from_vertex"])
            b = int(w["to_vertex"])
            out[wid] = (poly[a], poly[b])
        return out

    # fallback: последовательные рёбра полигона
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        out[f"w{i}"] = (a, b)
    return out


def place_segment_on_wall(
    wall_a: Tuple[float, float],
    wall_b: Tuple[float, float],
    s: float,
    width: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ax, ay = wall_a
    bx, by = wall_b
    dx = bx - ax
    dy = by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return wall_a, wall_a

    ux = dx / length
    uy = dy / length

    s0 = max(0.0, min(float(s), length))
    s1 = max(0.0, min(float(s) + float(width), length))

    p0 = (ax + ux * s0, ay + uy * s0)
    p1 = (ax + ux * s1, ay + uy * s1)
    return p0, p1


def build_custom_floorplan_module_text(room_data: Dict[str, Any], semantic_name: str) -> str:
    poly = infer_room_polygon(room_data)
    walls = infer_walls(room_data, poly)
    room = room_data.get("room", room_data)

    room_key = f"{semantic_name}_0/0"

    doors_entries: List[str] = []
    for i, d in enumerate(room.get("doors", []) or []):
        wall_id = str(d.get("wall_id"))
        if wall_id not in walls:
            continue
        p0, p1 = place_segment_on_wall(
            walls[wall_id][0],
            walls[wall_id][1],
            float(d.get("s", 0.0)),
            float(d.get("width", 0.9)),
        )
        doors_entries.append(
            f'"door_{i}": {{"shape": shapely.LineString([{p0}, {p1}])}}'
        )

    windows_entries: List[str] = []
    for i, w in enumerate(room.get("windows", []) or []):
        wall_id = str(w.get("wall_id"))
        if wall_id not in walls:
            continue
        p0, p1 = place_segment_on_wall(
            walls[wall_id][0],
            walls[wall_id][1],
            float(w.get("s", 0.0)),
            float(w.get("width", 1.0)),
        )
        windows_entries.append(
            f'"window_{i}": {{"shape": shapely.LineString([{p0}, {p1}])}}'
        )

    return textwrap.dedent(
        f"""
        import shapely

        def example(factory_seed):
            return {{
                "rooms": {{
                    "{room_key}": {{
                        "shape": shapely.Polygon({poly})
                    }}
                }},
                "doors": {{{", ".join(doors_entries)}}},
                "opens": {{}},
                "interiors": {{}},
                "windows": {{{", ".join(windows_entries)}}},
                "entrance": {{}},
            }}
        """
    ).strip() + "\n"


def load_style_profile(path: str | Path | None) -> Dict[str, Any] | None:
    if not path:
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError("style profile must be a JSON object")
    return data


def style_profile_infinigen_patch(style_profile: Dict[str, Any] | None) -> tuple[Dict[str, Any], List[str], str]:
    if not isinstance(style_profile, dict):
        return {}, [], ""
    infinigen = style_profile.get("infinigen") or {}
    if not isinstance(infinigen, dict):
        raise RuntimeError("style profile infinigen block must be an object")
    params = infinigen.get("monkeypatch_params") or {}
    overrides = infinigen.get("overrides") or []
    if not isinstance(params, dict):
        raise RuntimeError("style profile infinigen.monkeypatch_params must be an object")
    if not isinstance(overrides, list):
        raise RuntimeError("style profile infinigen.overrides must be a list")
    return (
        {str(k): v for k, v in params.items()},
        [str(x).strip() for x in overrides if str(x).strip()],
        str(style_profile.get("style_label") or "").strip(),
    )


def styled_infinigen_runner_script_text() -> str:
    return textwrap.dedent(
        r"""
        import argparse
        import json
        import sys
        from pathlib import Path


        def _load_style_profile(path: str):
            data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise RuntimeError("style profile must be a JSON object")
            infinigen = data.get("infinigen") or {}
            if not isinstance(infinigen, dict):
                raise RuntimeError("style profile infinigen block must be an object")
            params = infinigen.get("monkeypatch_params") or {}
            overrides = infinigen.get("overrides") or []
            if not isinstance(params, dict):
                raise RuntimeError("style profile infinigen.monkeypatch_params must be an object")
            if not isinstance(overrides, list):
                raise RuntimeError("style profile infinigen.overrides must be a list")
            whitelist = [str(x).strip() for x in ((infinigen.get("effective_factory_whitelist") or infinigen.get("factory_whitelist") or [])) if str(x).strip()]
            blacklist = [str(x).strip() for x in ((infinigen.get("effective_factory_blacklist") or infinigen.get("factory_blacklist") or [])) if str(x).strip()]
            runtime = {
                "apply_child_restrictions": bool(infinigen.get("apply_child_restrictions", False)),
                "final_restrict_child_primary": list(infinigen.get("final_restrict_child_primary") or []),
                "final_restrict_child_secondary": list(infinigen.get("final_restrict_child_secondary") or []),
                "required_factory_coverage": dict(infinigen.get("required_factory_coverage") or {}),
                "stage_flags": dict(infinigen.get("stage_flags") or {}),
                "solver_steps": dict(infinigen.get("solver_steps") or {}),
                "max_counts": dict(infinigen.get("max_counts") or {}),
            }
            return data, params, [str(x).strip() for x in overrides if str(x).strip()], whitelist, blacklist, runtime


        def _semantic_name(value):
            if hasattr(value, "name"):
                return str(getattr(value, "name"))
            text = str(value)
            return text.rsplit(".", 1)[-1]


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--infinigen-src", required=True)
            parser.add_argument("--seed", required=True)
            parser.add_argument("--output-folder", required=True)
            parser.add_argument("--floor-plan-module", required=True)
            parser.add_argument("--style-profile", required=True)
            args = parser.parse_args()

            infinigen_src = Path(args.infinigen_src).expanduser().resolve()
            sys.path.insert(0, str(infinigen_src))

            style_profile, patch_params, extra_overrides, factory_whitelist, factory_blacklist, runtime = _load_style_profile(args.style_profile)

            from infinigen_examples.constraints import home as home_constraints
            from infinigen_examples.constraints import semantics as semantics_constraints
            from infinigen_examples import generate_indoors

            base_fn = home_constraints.sample_home_constraint_params
            base_asset_usage = semantics_constraints.home_asset_usage

            empty_required = sorted(
                [k for k, v in (runtime.get("required_factory_coverage") or {}).items() if not v]
            )
            if empty_required:
                raise RuntimeError(
                    "empty_candidate_pool_after_blacklist: missing required factory coverage for "
                    + ", ".join(empty_required)
                )

            def _patched_sample_home_constraint_params():
                base = dict(base_fn())
                base.update(patch_params)
                print(
                    "[style-profile] label=%s params=%s"
                    % (
                        str(style_profile.get("style_label") or "unknown"),
                        json.dumps(patch_params, ensure_ascii=False, sort_keys=True),
                    )
                )
                return base

            home_constraints.sample_home_constraint_params = _patched_sample_home_constraint_params

            cached_usage = None

            def _patched_home_asset_usage():
                nonlocal cached_usage
                usage = dict(base_asset_usage())
                whitelist = set(factory_whitelist)
                blacklist = set(factory_blacklist)
                if not whitelist and not blacklist:
                    cached_usage = usage
                    return usage
                filtered = {}
                for semantic, factories in usage.items():
                    original = set(factories)
                    kept = set()
                    for factory in original:
                        name = getattr(factory, "__name__", str(factory))
                        if blacklist and name in blacklist:
                            continue
                        if whitelist and name not in whitelist:
                            continue
                        kept.add(factory)
                    if not kept:
                        kept = {factory for factory in original if getattr(factory, "__name__", str(factory)) not in blacklist}
                    filtered[semantic] = kept or original
                print(
                    "[style-profile] factory_filter whitelist=%s blacklist=%s"
                    % (
                        json.dumps(sorted(whitelist), ensure_ascii=False),
                        json.dumps(sorted(blacklist), ensure_ascii=False),
                        )
                    )
                cached_usage = filtered
                return filtered

            semantics_constraints.home_asset_usage = _patched_home_asset_usage

            def _stage_candidate_pool_dump():
                usage = cached_usage or _patched_home_asset_usage()
                primary = set(runtime.get("final_restrict_child_primary") or [])
                secondary = set(runtime.get("final_restrict_child_secondary") or [])

                def _factories_for(semantic_names):
                    if semantic_names is not None and len(semantic_names) == 0:
                        return []
                    out = []
                    for semantic, factories in usage.items():
                        name = _semantic_name(semantic)
                        if semantic_names is not None and name not in semantic_names:
                            continue
                        out.extend(sorted(getattr(factory, "__name__", str(factory)) for factory in factories))
                    return sorted(set(out))

                dump = {
                    "primary_semantics": sorted(primary),
                    "secondary_semantics": sorted(secondary),
                    "stages": {
                        "on_floor_and_wall": _factories_for(primary),
                        "on_floor_freestanding": _factories_for(primary),
                        "on_wall": _factories_for(primary),
                        "on_ceiling": _factories_for(primary),
                        "side_obj": _factories_for(secondary),
                        "obj_ontop_obj": _factories_for(secondary),
                        "obj_on_support": _factories_for(secondary),
                    },
                }
                print("[candidate-pool] %s" % json.dumps(dump, ensure_ascii=False, sort_keys=True))
                candidate_pool_path = Path(args.output_folder).expanduser().resolve().parent / "candidate_pool.json"
                candidate_pool_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")

            overrides = [
                "compose_indoors.terrain_enabled=False",
                f'Solver.floor_plan="{args.floor_plan_module}"',
            ]
            overrides.extend(extra_overrides)
            _stage_candidate_pool_dump()
            print(
                "[runtime-config] %s"
                % json.dumps(
                    {
                        "effective_factory_whitelist": sorted(factory_whitelist),
                        "effective_factory_blacklist": sorted(factory_blacklist),
                        "apply_child_restrictions": runtime.get("apply_child_restrictions"),
                        "restrict_child_primary": runtime.get("final_restrict_child_primary"),
                        "restrict_child_secondary": runtime.get("final_restrict_child_secondary"),
                        "solver_steps": runtime.get("solver_steps"),
                        "stage_flags": runtime.get("stage_flags"),
                        "max_counts": runtime.get("max_counts"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            print(
                "[style-profile] overrides=%s"
                % json.dumps(overrides, ensure_ascii=False, sort_keys=False)
            )
            generate_args = argparse.Namespace(
                output_folder=Path(args.output_folder).expanduser().resolve(),
                input_folder=None,
                seed=args.seed,
                task=["coarse"],
                configs=["singleroom.gin", "fast_solve.gin"],
                overrides=overrides,
                task_uniqname=None,
                debug=None,
            )
            generate_indoors.main(generate_args)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"


def run_infinigen_generate(
    infinigen_src: Path,
    module_name: str,
    seed: str | int,
    output_folder: Path,
    *,
    run_dir: Path,
    style_profile: Dict[str, Any] | None = None,
    log_path: Path | None = None,
    task: str = "coarse",
    configs: List[str] | None = None,
) -> None:
    safe_seed_hex = normalize_seed_for_infinigen(seed)
    floor_plan_module = f"infinigen_examples.configs_indoor.floor_plans.custom.{module_name}.example"
    patch_params, extra_overrides, style_label = style_profile_infinigen_patch(style_profile)
    infinigen_block = (style_profile or {}).get("infinigen") if isinstance(style_profile, dict) else {}
    factory_whitelist = list((infinigen_block or {}).get("factory_whitelist") or [])
    factory_blacklist = list((infinigen_block or {}).get("factory_blacklist") or [])
    effective_task = str(task or "coarse").strip() or "coarse"
    effective_configs = list(configs or ["singleroom.gin", "fast_solve.gin"])
    if patch_params or extra_overrides or factory_whitelist or factory_blacklist:
        style_profile_source = ""
        if isinstance(style_profile, dict):
            style_profile_source = str(style_profile.get("__source_path__") or "").strip()
        if not style_profile_source:
            raise RuntimeError("style profile source path is required for styled Infinigen generation")
        helper_path = run_dir / "_run_infinigen_with_style.py"
        helper_path.write_text(styled_infinigen_runner_script_text(), encoding="utf-8")
        cmd = [
            sys.executable,
            str(helper_path.resolve()),
            "--infinigen-src",
            str(infinigen_src.resolve()),
            "--seed",
            safe_seed_hex,
            "--output-folder",
            str(output_folder.resolve()),
            "--floor-plan-module",
            floor_plan_module,
            "--style-profile",
            str(Path(style_profile_source).expanduser().resolve()),
        ]
        print(f"▶ Infinigen styled command [{style_label or 'custom'}]:\n ", " ".join(cmd))
        if log_path is None:
            subprocess.run(cmd, cwd=str(infinigen_src.resolve()), check=True)
        else:
            with Path(log_path).expanduser().resolve().open("w", encoding="utf-8") as fh:
                subprocess.run(
                    cmd,
                    cwd=str(infinigen_src.resolve()),
                    check=True,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        return

    cmd = [
        sys.executable,
        "-m",
        "infinigen_examples.generate_indoors",
        "--seed",
        safe_seed_hex,
        "--task",
        effective_task,
        "--output_folder",
        str(output_folder.resolve()),
        "-g",
        *effective_configs,
        "-p",
        "compose_indoors.terrain_enabled=False",
        f'Solver.floor_plan="{floor_plan_module}"',
    ]
    print("▶ Infinigen command:\n ", " ".join(cmd))
    if log_path is None:
        subprocess.run(cmd, cwd=str(infinigen_src.resolve()), check=True)
    else:
        with Path(log_path).expanduser().resolve().open("w", encoding="utf-8") as fh:
            subprocess.run(
                cmd,
                cwd=str(infinigen_src.resolve()),
                check=True,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
            )


def blender_extract_script_text() -> str:
    return r'''
import bpy
import json
import math
import os
import re
import sys

blend_path = sys.argv[-2]
out_json = sys.argv[-1]

def build_aabb(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return {
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "z_min": min(zs), "z_max": max(zs),
    }

def clean_name(name: str) -> str:
    m = re.search(r'([A-Za-z]+Factory)', name)
    if m:
        return m.group(1)
    s = re.sub(r'\(\d+\)', '', name)
    s = s.replace('.spawn_asset', '')
    s = s.replace('.bbox_placeholder', '')
    s = s.replace('.spawn_placeholder', '')
    return s.strip() or "object"

def should_skip(obj):
    if obj.type != "MESH":
        return True
    name = obj.name.lower()
    bad_name_parts = [
        "room_wall", "room_floor", "room_ceiling", "room_exterior",
        "door", "window", "skirting", "placeholder", "portal_cutters",
        "room_meshes", "room_shells", "cutter",
    ]
    if any(x in name for x in bad_name_parts):
        return True

    col_names = [c.name.lower() for c in obj.users_collection]
    bad_coll_parts = [
        "room_wall", "room_floor", "room_ceiling", "room_exterior",
        "doors", "windows", "skirting", "placeholders", "portal_cutters",
    ]
    if any(any(x in cn for x in bad_coll_parts) for cn in col_names):
        return True

    return False

bpy.ops.wm.open_mainfile(filepath=blend_path)

placements = []
idx = 1

depsgraph = bpy.context.evaluated_depsgraph_get()

for obj in bpy.data.objects:
    if should_skip(obj):
        continue

    try:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mw = obj_eval.matrix_world
        pts = [mw @ v.co for v in mesh.vertices]
        obj_eval.to_mesh_clear()

        if not pts:
            continue

        pts_xyz = [(float(p.x), float(p.y), float(p.z)) for p in pts]
        aabb = build_aabb(pts_xyz)
        pos = [
            0.5 * (aabb["x_min"] + aabb["x_max"]),
            0.5 * (aabb["y_min"] + aabb["y_max"]),
            0.5 * (aabb["z_min"] + aabb["z_max"]),
        ]
        size = [
            aabb["x_max"] - aabb["x_min"],
            aabb["y_max"] - aabb["y_min"],
            aabb["z_max"] - aabb["z_min"],
        ]

        euler = obj.matrix_world.to_euler('XYZ')
        yaw_rad = float(euler.z)
        yaw_deg = float(math.degrees(yaw_rad))

        clean = clean_name(obj.name)

        placements.append({
            "id": f"obj_{idx:04d}",
            "name": clean,
            "category": clean,
            "position_m": pos,
            "size_m": size,
            "rotation_deg": int(round(yaw_deg)) % 360,
            "yaw_deg": yaw_deg,
            "yaw_rad": yaw_rad,
            "aabb": aabb,
            "constraints": {},
            "asset": {},
            "source": {
                "placement_source": "infinigen_clean",
                "blend_object_name": obj.name,
            },
            "meta": {
                "collections": [c.name for c in obj.users_collection],
            },
            "color": [0.7, 0.7, 0.7],
        })
        idx += 1
    except Exception:
        continue

out = {
    "placer": "infinigen_clean",
    "mode": "infinigen_clean",
    "placements": placements,
    "meta": {
        "scene_blend": blend_path,
        "object_count": len(placements),
        "extracted_by": "bpy",
    }
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
'''


def extract_placement_from_blend(blend_path: Path, out_json: Path, run_dir: Path) -> None:
    helper = run_dir / "_extract_infinigen_blend.py"
    helper.write_text(blender_extract_script_text(), encoding="utf-8")

    cmd = [
        sys.executable,
        str(helper.resolve()),
        str(blend_path.resolve()),
        str(out_json.resolve()),
    ]
    print("▶ Extract placement from blend:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_local(args: argparse.Namespace) -> None:
    normalized_seed = normalize_seed(args.seed)
    infinigen_seed = normalize_seed_for_infinigen(args.seed)
    room_data = load_json(args.room)
    style_profile = load_style_profile(args.style_profile)
    if isinstance(style_profile, dict):
        style_profile["__source_path__"] = str(Path(args.style_profile).expanduser().resolve())
    if has_source_restroom_type(room_data):
        semantic_name = infer_room_semantic(room_data)
    else:
        semantic_name = infer_room_semantic_from_style_profile(style_profile) or infer_room_semantic(room_data)

    infinigen_src = Path(args.infinigen_src).expanduser().resolve() if args.infinigen_src else default_infinigen_src()
    custom_dir = infinigen_src / "infinigen_examples" / "configs_indoor" / "floor_plans" / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    module_name = f"_auto_fp_{semantic_name.replace('-', '_')}_{uuid.uuid4().hex[:10]}"
    module_path = custom_dir / f"{module_name}.py"
    module_path.write_text(build_custom_floorplan_module_text(room_data, semantic_name), encoding="utf-8")

    output_folder = run_dir / f"infinigen_scene_seed_{normalized_seed}"
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    try:
        run_infinigen_generate(
            infinigen_src=infinigen_src,
            module_name=module_name,
            seed=infinigen_seed,
            output_folder=output_folder,
            run_dir=run_dir,
            style_profile=style_profile,
            log_path=log_path,
            task=args.infinigen_task,
            configs=list(args.infinigen_configs or []),
        )

        blend_path = output_folder / "scene.blend"
        if not blend_path.is_file():
            raise RuntimeError(f"Infinigen не создал scene.blend: {blend_path}")

        copied_blend = run_dir / "infinigen_clean_scene.blend"
        copied_blend.write_bytes(blend_path.read_bytes())

        meta_path = run_dir / "infinigen_clean_meta.json"
        save_json(
            meta_path,
            {
                "infinigen_src": str(infinigen_src),
                "output_folder": str(output_folder),
                "scene_blend": str(copied_blend),
                "floor_plan_module": f"infinigen_examples.configs_indoor.floor_plans.custom.{module_name}.example",
                "seed": normalized_seed,
                "room_semantic": semantic_name,
                "style_label": str(style_profile.get("style_label") or "") if isinstance(style_profile, dict) else "",
                "style_profile": str(Path(args.style_profile).expanduser().resolve()) if args.style_profile else None,
                "style_infinigen_params": (
                    (style_profile.get("infinigen") or {}).get("monkeypatch_params")
                    if isinstance(style_profile, dict)
                    else None
                ),
            },
        )

        extract_placement_from_blend(
            blend_path=copied_blend,
            out_json=Path(args.out).expanduser().resolve(),
            run_dir=run_dir,
        )
        inventory_path, inventory_summary_path = write_inventory_artifacts(run_dir, Path(args.out).expanduser().resolve())
        solver_summary = parse_solver_log(log_path)
        inventory_summary = load_json(inventory_summary_path)
        core_semantic_counts = dict(inventory_summary.get("core_semantic_counts") or {})
        early_failure_reason = None
        prompt_driven_mode = bool(
            isinstance(style_profile, dict)
            and str(((style_profile.get("infinigen") or {}).get("compiled_policy_path") or "")).strip()
        )
        if prompt_driven_mode and semantic_name == "bedroom" and int(core_semantic_counts.get("Bed", 0)) <= 0:
            early_failure_reason = "missing_required_bed_early"
            print(f"[early-fail] {early_failure_reason}: bedroom candidate has no Bed in core inventory")
        save_json(run_dir / "solver_summary.json", solver_summary)
        if early_failure_reason:
            save_json(
                run_dir / "early_failure.json",
                {
                    "reason": early_failure_reason,
                    "room_semantic": semantic_name,
                    "core_semantic_counts": core_semantic_counts,
                    "solver_summary": solver_summary,
                },
            )

        print(f"OK: saved Infinigen placement -> {Path(args.out).expanduser().resolve()}")
        print(f"OK: saved Infinigen blend -> {copied_blend}")
        print(f"OK: saved inventory -> {inventory_path}")
        print(f"OK: saved inventory summary -> {inventory_summary_path}")

    finally:
        if module_path.exists():
            try:
                module_path.unlink()
            except Exception:
                pass


def maybe_download_remote_artifact(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    try:
        ssh_download_file(args, remote_path, local_path)
    except subprocess.CalledProcessError:
        pass


def run_remote(args: argparse.Namespace) -> None:
    if not args.remote_host or not args.remote_user:
        raise RuntimeError("Для remote Infinigen нужны remote-host и remote-user")

    ensure_remote_free_space(args)

    normalized_seed = normalize_seed(args.seed)
    infinigen_seed = normalize_seed_for_infinigen(args.seed)
    run_id = uuid.uuid4().hex[:8]
    remote_dir = f"/workspace/tmp/infinigen_clean_{run_id}"
    remote_room = f"{remote_dir}/room.json"
    remote_script = f"{remote_dir}/run_infinigen_clean.py"
    remote_style_profile = f"{remote_dir}/style_profile.json"
    remote_out = f"{remote_dir}/placement.json"
    remote_run_dir = f"{remote_dir}/run"

    local_out = Path(args.out).expanduser().resolve()
    local_run_dir = Path(args.run_dir).expanduser().resolve()
    local_run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as _tmp:
        local_script = Path(__file__).resolve()
        ssh_run(args, f"mkdir -p {shlex.quote(remote_dir)} {shlex.quote(remote_run_dir)}")
        ssh_upload_file(args, Path(args.room).expanduser().resolve(), remote_room)
        ssh_upload_file(args, local_script, remote_script)
        if args.style_profile:
            ssh_upload_file(args, Path(args.style_profile).expanduser().resolve(), remote_style_profile)

        remote_cmd_parts = build_remote_preamble(args)
        local_mode_cmd = [
            "python",
            shlex.quote(remote_script),
            "--room",
            shlex.quote(remote_room),
            "--seed",
            infinigen_seed,
            "--out",
            shlex.quote(remote_out),
            "--run-dir",
            shlex.quote(remote_run_dir),
            "--infinigen-task",
            shlex.quote(str(args.infinigen_task)),
        ]
        if list(args.infinigen_configs or []):
            local_mode_cmd += ["--infinigen-configs", *[shlex.quote(str(cfg_name)) for cfg_name in list(args.infinigen_configs or [])]]
        if args.style_profile:
            local_mode_cmd += ["--style-profile", shlex.quote(remote_style_profile)]
        remote_infinigen_src = str(args.remote_infinigen_src or "/workspace/infinigen/src").strip()
        if remote_infinigen_src:
            local_mode_cmd += ["--infinigen-src", shlex.quote(remote_infinigen_src)]
        remote_cmd_parts.append(" ".join(local_mode_cmd))
        ssh_run(args, "; ".join(remote_cmd_parts))

        ssh_download_file(args, remote_out, local_out)
        maybe_download_remote_artifact(args, f"{remote_run_dir}/infinigen_clean_meta.json", local_run_dir / "infinigen_clean_meta.json")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/infinigen_clean_scene.blend", local_run_dir / "infinigen_clean_scene.blend")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/run.log", local_run_dir / "run.log")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/inventory.json", local_run_dir / "inventory.json")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/inventory_summary.json", local_run_dir / "inventory_summary.json")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/solver_summary.json", local_run_dir / "solver_summary.json")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/candidate_pool.json", local_run_dir / "candidate_pool.json")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/early_failure.json", local_run_dir / "early_failure.json")


def run_from_compiled_policy(
    compiled_policy_path: str | Path,
    output_dir: str | Path,
    seed: int,
    *,
    remote_host: str | None = None,
    remote_port: int = 22,
    remote_user: str | None = None,
    remote_key: str | None = None,
    remote_conda_env: str | None = None,
    remote_infinigen_src: str = "/workspace/infinigen/src",
    infinigen_src: str | None = None,
) -> dict[str, Any]:
    from src.prompt_compiler.compile_to_infinigen import build_room_json, build_style_profile
    from src.prompt_compiler.schemas import CompiledPolicy

    compiled_policy = CompiledPolicy.load(compiled_policy_path)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    room_json_path = out_dir / "room.json"
    style_profile_path = out_dir / "style_profile.json"
    placement_path = out_dir / "placement.json"
    room_json_path.write_text(json.dumps(build_room_json(compiled_policy), ensure_ascii=False, indent=2), encoding="utf-8")
    style_profile_path.write_text(json.dumps(build_style_profile(compiled_policy), ensure_ascii=False, indent=2), encoding="utf-8")
    args = argparse.Namespace(
        room=str(room_json_path),
        seed=str(seed),
        out=str(placement_path),
        run_dir=str(out_dir),
        style_profile=str(style_profile_path),
        infinigen_src=infinigen_src,
        remote_host=remote_host,
        remote_port=remote_port,
        remote_user=remote_user,
        remote_key=remote_key,
        remote_conda_env=remote_conda_env,
        remote_infinigen_src=remote_infinigen_src,
    )
    if remote_host and remote_user:
        run_remote(args)
    else:
        run_local(args)
    return {
        "candidate_dir": str(out_dir),
        "placement": str(placement_path),
        "inventory": str(out_dir / "inventory.json"),
        "inventory_summary": str(out_dir / "inventory_summary.json"),
        "solver_summary": str(out_dir / "solver_summary.json"),
        "blend": str(out_dir / "infinigen_clean_scene.blend"),
        "run_log": str(out_dir / "run.log"),
    }


def run_screening_from_compiled_policy(
    compiled_policy_path: str | Path,
    screening_base_dir: str | Path,
    seeds: list[int],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    base_dir = Path(screening_base_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        candidate_dir = base_dir / f"seed_{index:03d}"
        result = run_from_compiled_policy(
            compiled_policy_path=compiled_policy_path,
            output_dir=candidate_dir,
            seed=int(seed),
            **kwargs,
        )
        results.append(result)
    return results


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wrapper для Infinigen -> placement-like JSON + .blend")
    p.add_argument("--room", default=None)
    p.add_argument("--seed", default="0")
    p.add_argument("--out", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--style-profile", default=None)
    p.add_argument("--compiled-policy", default=None)
    p.add_argument("--screening-seeds", default=None, help="comma-separated list of seeds for screening mode")
    p.add_argument("--screening-base-dir", default=None)
    p.add_argument("--infinigen-src", default=None)
    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=22)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)
    p.add_argument("--remote-conda-env", default=None)
    p.add_argument("--remote-infinigen-src", default="/workspace/infinigen/src")
    p.add_argument("--infinigen-task", default="coarse")
    p.add_argument("--infinigen-configs", nargs="+", default=["singleroom.gin", "fast_solve.gin"])
    return p


def main() -> None:
    args = build_cli().parse_args()
    if args.compiled_policy:
        compiled_policy_path = Path(args.compiled_policy).expanduser().resolve()
        if args.screening_seeds:
            seeds = [int(part.strip()) for part in str(args.screening_seeds).split(",") if part.strip()]
            screening_base_dir = args.screening_base_dir or (Path(args.run_dir or ".").expanduser().resolve() / "screening")
            run_screening_from_compiled_policy(
                compiled_policy_path=compiled_policy_path,
                screening_base_dir=screening_base_dir,
                seeds=seeds,
                remote_host=args.remote_host,
                remote_port=args.remote_port,
                remote_user=args.remote_user,
                remote_key=args.remote_key,
                remote_conda_env=args.remote_conda_env,
                remote_infinigen_src=args.remote_infinigen_src,
                infinigen_src=args.infinigen_src,
            )
            return
        output_dir = args.run_dir or args.out
        if not output_dir:
            raise RuntimeError("--run-dir or --out is required with --compiled-policy")
        run_from_compiled_policy(
            compiled_policy_path=compiled_policy_path,
            output_dir=output_dir,
            seed=int(args.seed),
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            remote_user=args.remote_user,
            remote_key=args.remote_key,
            remote_conda_env=args.remote_conda_env,
            remote_infinigen_src=args.remote_infinigen_src,
            infinigen_src=args.infinigen_src,
        )
        return
    if not args.room or not args.out or not args.run_dir:
        raise RuntimeError("--room, --out and --run-dir are required unless --compiled-policy is used")
    if args.remote_host and args.remote_user:
        run_remote(args)
        return
    run_local(args)


if __name__ == "__main__":
    main()
