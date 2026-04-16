#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path

MAX_NUMPY_SEED = 2**32 - 1

# =========================
# SSH helpers
# =========================

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
    # КРИТИЧНО: echo в конце, иначе vast убивает соединение
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


def normalize_seed(seed: int) -> int:
    return int(seed) % (MAX_NUMPY_SEED + 1)

# =========================
# Remote worker
# =========================
def remote_worker_code() -> str:
    return r'''
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
def load_json(p):
    return json.loads(Path(p).read_text())
def save_json(p, d):
    Path(p).write_text(json.dumps(d, indent=2))
def room_center_xy(room_data):
    room = room_data.get("room") or {}
    poly = room.get("floor_polygon") or []
    pts = []
    for p in poly:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts.append((float(p["x"]), float(p["y"])))
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))
    return (0.0, 0.0)
def aabb(pos, size):
    cx, cy, cz = pos
    sx, sy, sz = size
    return {
        "x_min": cx - sx/2, "x_max": cx + sx/2,
        "y_min": cy - sy/2, "y_max": cy + sy/2,
        "z_min": cz - sz/2, "z_max": cz + sz/2,
    }
def import_demo(root):
    p = Path(root) / "gradio_demo.py"
    spec = importlib.util.spec_from_file_location("demo", p)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    spec.loader.exec_module(m)
    return m
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--room", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model-type", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    room_data = load_json(args.room)
    room_cx, room_cy = room_center_xy(room_data)
    demo = import_demo(Path(args.repo_root))
    if args.model_type == "autoregressive":
        gen = demo.M3DLayoutArGenerator(
            config_file=str(Path(args.repo_root)/"config/m3dlayout_autoregressive.yaml"),
            weight_file=str(Path(args.repo_root)/"weights/autoregressive_59000.pth"),
            max_boxes=64,
            default_seed=args.seed,
        )
    else:
        gen = demo.M3DLayoutDiffusionGenerator(
            config_file=str(Path(args.repo_root)/"config/m3dlayout_diffusion.yaml"),
            weight_file=str(Path(args.repo_root)/"weights/diffusion_30000.pth"),
            default_seed=args.seed,
        )
    res = gen.generate_scene_from_text(args.prompt, seed=args.seed)
    out = []
    for i, (c, t, s, a) in enumerate(zip(
        res["class_names"], res["translations"], res["sizes"], res["angles"]
    )):
        yaw = float(a[0] if isinstance(a, (list, tuple)) else a)
        yaw_deg = math.degrees(yaw)
        size = [float(s[0]), float(s[2]), float(s[1])]
        pos = [float(t[0]) + room_cx, float(t[2]) + room_cy, size[2]/2]
        out.append({
            "id": f"obj_{i:04d}",
            "name": c,
            "category": c,
            "position_m": pos,
            "size_m": size,
            "yaw_deg": yaw_deg,
            "yaw_rad": yaw,
            "rotation_deg": int(yaw_deg)%360,
            "aabb": aabb(pos, size),
        })
    save_json(args.out, {"placements": out})
if __name__ == "__main__":
    main()
'''
# =========================
# CLI
# =========================
def build_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--room", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--model-type", required=True, choices=["autoregressive", "diffusion"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--remote-host", required=True)
    p.add_argument("--remote-port", type=int, default=22)
    p.add_argument("--remote-user", required=True)
    p.add_argument("--remote-key")
    p.add_argument("--remote-repo-root", default="/workspace/M3DLayout-code")
    p.add_argument("--remote-python", default="python")
    p.add_argument("--remote-conda-env", default=None)
    return p
# =========================
# MAIN
# =========================
def main():
    args = build_cli().parse_args()
    normalized_seed = normalize_seed(args.seed)
    run_id = uuid.uuid4().hex[:8]
    remote_dir = f"/workspace/tmp/m3d_{run_id}"
    remote_room = f"{remote_dir}/room.json"
    remote_worker = f"{remote_dir}/worker.py"
    remote_out = f"{remote_dir}/out.json"
    local_out = Path(args.out).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        worker = Path(tmp) / "worker.py"
        worker.write_text(remote_worker_code(), encoding="utf-8")
        ssh_run(args, f"mkdir -p {shlex.quote(remote_dir)}")
        ssh_upload_file(args, Path(args.room).expanduser().resolve(), remote_room)
        ssh_upload_file(args, worker, remote_worker)
        cmd = " ".join(
            [
                shlex.quote(args.remote_python),
                shlex.quote(remote_worker),
                "--repo-root",
                shlex.quote(args.remote_repo_root),
                "--room",
                shlex.quote(remote_room),
                "--prompt",
                shlex.quote(args.prompt),
                "--model-type",
                shlex.quote(args.model_type),
                "--seed",
                str(normalized_seed),
                "--out",
                shlex.quote(remote_out),
            ]
        )
        remote_parts = build_remote_preamble(args)
        remote_parts.append(cmd)
        ssh_run(args, "; ".join(remote_parts))
        ssh_download_file(args, remote_out, local_out)
    print("✅ DONE:", local_out)
if __name__ == "__main__":
    main()
