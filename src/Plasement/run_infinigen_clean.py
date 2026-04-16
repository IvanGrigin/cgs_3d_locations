#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/run_infinigen_clean.py

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

MAX_NUMPY_SEED = 2**32 - 1


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    raw = " ".join(
        [
            str(room.get("name", "")),
            str(room.get("id", "")),
            str(room.get("room_type", "")),
        ]
    ).lower()

    if "bed" in raw or "спаль" in raw:
        return "bedroom"
    if "living" in raw or "гостин" in raw:
        return "living-room"
    if "dining" in raw or "столов" in raw:
        return "dining-room"
    if "bath" in raw or "ван" in raw or "toilet" in raw or "сануз" in raw:
        return "bathroom"
    if "kitchen" in raw or "кух" in raw:
        return "kitchen"

    return "bedroom"


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


def run_infinigen_generate(
    infinigen_src: Path,
    module_name: str,
    seed: str | int,
    output_folder: Path,
) -> None:
    safe_seed_hex = normalize_seed_for_infinigen(seed)
    cmd = [
        sys.executable,
        "-m",
        "infinigen_examples.generate_indoors",
        "--seed",
        safe_seed_hex,
        "--task",
        "coarse",
        "--output_folder",
        str(output_folder.resolve()),
        "-g",
        "singleroom.gin",
        "fast_solve.gin",
        "-p",
        "compose_indoors.terrain_enabled=False",
        f'Solver.floor_plan="infinigen_examples.configs_indoor.floor_plans.custom.{module_name}.example"',
    ]
    print("▶ Infinigen command:\n ", " ".join(cmd))
    subprocess.run(cmd, cwd=str(infinigen_src.resolve()), check=True)


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
    semantic_name = infer_room_semantic(room_data)

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

    try:
        run_infinigen_generate(
            infinigen_src=infinigen_src,
            module_name=module_name,
            seed=infinigen_seed,
            output_folder=output_folder,
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
            },
        )

        extract_placement_from_blend(
            blend_path=copied_blend,
            out_json=Path(args.out).expanduser().resolve(),
            run_dir=run_dir,
        )

        print(f"OK: saved Infinigen placement -> {Path(args.out).expanduser().resolve()}")
        print(f"OK: saved Infinigen blend -> {copied_blend}")

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

    normalized_seed = normalize_seed(args.seed)
    infinigen_seed = normalize_seed_for_infinigen(args.seed)
    run_id = uuid.uuid4().hex[:8]
    remote_dir = f"/workspace/tmp/infinigen_clean_{run_id}"
    remote_room = f"{remote_dir}/room.json"
    remote_script = f"{remote_dir}/run_infinigen_clean.py"
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
        ]
        remote_infinigen_src = str(args.remote_infinigen_src or "/workspace/infinigen/src").strip()
        if remote_infinigen_src:
            local_mode_cmd += ["--infinigen-src", shlex.quote(remote_infinigen_src)]
        remote_cmd_parts.append(" ".join(local_mode_cmd))
        ssh_run(args, "; ".join(remote_cmd_parts))

        ssh_download_file(args, remote_out, local_out)
        maybe_download_remote_artifact(args, f"{remote_run_dir}/infinigen_clean_meta.json", local_run_dir / "infinigen_clean_meta.json")
        maybe_download_remote_artifact(args, f"{remote_run_dir}/infinigen_clean_scene.blend", local_run_dir / "infinigen_clean_scene.blend")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wrapper для Infinigen -> placement-like JSON + .blend")
    p.add_argument("--room", required=True)
    p.add_argument("--seed", default="0")
    p.add_argument("--out", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--infinigen-src", default=None)
    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=22)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)
    p.add_argument("--remote-conda-env", default=None)
    p.add_argument("--remote-infinigen-src", default="/workspace/infinigen/src")
    return p


def main() -> None:
    args = build_cli().parse_args()
    if args.remote_host and args.remote_user:
        run_remote(args)
        return
    run_local(args)


if __name__ == "__main__":
    main()
