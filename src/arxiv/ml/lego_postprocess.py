#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


DEFAULT_PATHS_CONFIG = "config/paths.yaml"


def load_yaml(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"YAML-конфиг не найден: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Некорректный YAML-конфиг: {p}")
    return data


def get_nested(cfg: dict, path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def project_root_from_config(cfg: dict, cfg_path: Path) -> Path:
    root = get_nested(cfg, "project.root", None)
    if root:
        return Path(root).expanduser().resolve()
    return cfg_path.parent.parent.resolve()


def resolve_local_path(value: Optional[str], base_dir: Path) -> Optional[str]:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remote LEGO-Net postprocess wrapper")

    p.add_argument("--paths-config", default=DEFAULT_PATHS_CONFIG)

    p.add_argument("--room-type", choices=["bedroom", "livingroom"], required=True)
    p.add_argument("--mode", required=True)

    p.add_argument("--room", required=True)
    p.add_argument("--in-placement-legacy", required=True)
    p.add_argument("--in-placement-v1", required=True)
    p.add_argument("--in-scene-v1", required=True)

    p.add_argument("--out-placement-legacy", required=True)
    p.add_argument("--out-placement-v1", required=True)
    p.add_argument("--out-scene-legacy", required=True)
    p.add_argument("--out-scene-v1", required=True)

    p.add_argument("--remote-host", default=None)
    p.add_argument("--remote-port", type=int, default=None)
    p.add_argument("--remote-user", default=None)
    p.add_argument("--remote-key", default=None)

    p.add_argument("--lego-repo", default=None)
    p.add_argument("--lego-python", default=None)
    p.add_argument("--lego-helper-script", default=None)
    p.add_argument("--lego-tmp-root", default=None)

    p.add_argument("--checkpoint", default=None)
    p.add_argument("--lego-checkpoint-bedroom", default=None)
    p.add_argument("--lego-checkpoint-livingroom", default=None)

    return p.parse_args()


def apply_config_defaults(args: argparse.Namespace, cfg: dict, cfg_base_dir: Path) -> None:
    if args.remote_host is None:
        args.remote_host = get_nested(cfg, "remote.ssh.host")
    if args.remote_port is None:
        args.remote_port = int(get_nested(cfg, "remote.ssh.port"))
    if args.remote_user is None:
        args.remote_user = get_nested(cfg, "remote.ssh.user")
    if args.remote_key is None:
        key = get_nested(cfg, "remote.ssh.key")
        args.remote_key = str(Path(key).expanduser()) if key else None

    if args.lego_repo is None:
        args.lego_repo = get_nested(cfg, "remote.lego_net.repo_root")
    if args.lego_python is None:
        args.lego_python = get_nested(cfg, "remote.lego_net.python")
    if args.lego_helper_script is None:
        args.lego_helper_script = get_nested(cfg, "remote.lego_net.helper_script")
    if args.lego_tmp_root is None:
        args.lego_tmp_root = get_nested(cfg, "remote.lego_net.tmp_root")

    if args.lego_checkpoint_bedroom is None:
        args.lego_checkpoint_bedroom = get_nested(cfg, "remote.lego_net.checkpoint_bedroom")
    if args.lego_checkpoint_livingroom is None:
        args.lego_checkpoint_livingroom = get_nested(cfg, "remote.lego_net.checkpoint_livingroom")


def choose_checkpoint(args: argparse.Namespace) -> str:
    if args.checkpoint:
        return str(Path(args.checkpoint).expanduser())

    if args.room_type == "bedroom":
        if not args.lego_checkpoint_bedroom:
            raise RuntimeError("Не задан remote bedroom checkpoint")
        return str(Path(args.lego_checkpoint_bedroom).expanduser())

    if args.room_type == "livingroom":
        if not args.lego_checkpoint_livingroom:
            raise RuntimeError("Не задан remote livingroom checkpoint")
        return str(Path(args.lego_checkpoint_livingroom).expanduser())

    raise RuntimeError(f"Unsupported room_type: {args.room_type}")


def ensure_file(path: str | Path, name: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{name} не найден: {p}")
    return p


def ensure_parent(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ssh_base_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["ssh", "-p", str(args.remote_port)]
    if args.remote_key:
        cmd += ["-i", str(Path(args.remote_key).expanduser())]
    cmd += [f"{args.remote_user}@{args.remote_host}"]
    return cmd


def scp_base_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["scp", "-P", str(args.remote_port)]
    if args.remote_key:
        cmd += ["-i", str(Path(args.remote_key).expanduser())]
    return cmd


def remote_quote(s: str) -> str:
    return shlex.quote(s)


def run_cmd(cmd: list[str]) -> None:
    print("▶", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def run_ssh(args: argparse.Namespace, script: str) -> None:
    cmd = ssh_base_cmd(args) + [script]
    run_cmd(cmd)


def scp_to_remote(args: argparse.Namespace, local_path: Path, remote_path: str) -> None:
    cmd = scp_base_cmd(args) + [str(local_path), f"{args.remote_user}@{args.remote_host}:{remote_path}"]
    run_cmd(cmd)


def scp_from_remote(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = scp_base_cmd(args) + [f"{args.remote_user}@{args.remote_host}:{remote_path}", str(local_path)]
    run_cmd(cmd)


def write_manifest(local_dir: Path, data: dict) -> None:
    p = local_dir / "lego_remote_manifest.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.paths_config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    cfg_base_dir = project_root_from_config(cfg, cfg_path)
    apply_config_defaults(args, cfg, cfg_base_dir)

    room_path = ensure_file(args.room, "room")
    in_placement_legacy = ensure_file(args.in_placement_legacy, "in-placement-legacy")
    in_placement_v1 = ensure_file(args.in_placement_v1, "in-placement-v1")
    in_scene_v1 = ensure_file(args.in_scene_v1, "in-scene-v1")

    out_placement_legacy = ensure_parent(args.out_placement_legacy)
    out_placement_v1 = ensure_parent(args.out_placement_v1)
    out_scene_legacy = ensure_parent(args.out_scene_legacy)
    out_scene_v1 = ensure_parent(args.out_scene_v1)

    if not args.remote_host or not args.remote_user or not args.remote_port:
        raise RuntimeError("Не заданы remote SSH параметры")

    if not args.lego_python:
        raise RuntimeError("Не задан remote.lego_net.python")
    if not args.lego_helper_script:
        raise RuntimeError("Не задан remote.lego_net.helper_script")
    if not args.lego_tmp_root:
        raise RuntimeError("Не задан remote.lego_net.tmp_root")

    checkpoint = choose_checkpoint(args)

    token = secrets.token_urlsafe(8).replace("-", "").replace("_", "").lower()
    remote_run_dir = f"{args.lego_tmp_root.rstrip('/')}/{args.room_type}_{args.mode}_{token}"

    remote_room = f"{remote_run_dir}/room.json"
    remote_in_placement_legacy = f"{remote_run_dir}/placement_input_legacy.json"
    remote_in_placement_v1 = f"{remote_run_dir}/placement_input.v1.json"
    remote_in_scene_v1 = f"{remote_run_dir}/scene_input.v1.json"

    remote_out_placement_legacy = f"{remote_run_dir}/placement_lego.json"
    remote_out_placement_v1 = f"{remote_run_dir}/placement_lego.v1.json"
    remote_out_scene_legacy = f"{remote_run_dir}/scene_lego.json"
    remote_out_scene_v1 = f"{remote_run_dir}/scene_lego.v1.json"

    local_meta_dir = out_scene_v1.parent
    write_manifest(local_meta_dir, {
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "remote_user": args.remote_user,
        "remote_run_dir": remote_run_dir,
        "room_type": args.room_type,
        "mode": args.mode,
        "checkpoint": checkpoint,
        "helper_script": args.lego_helper_script,
    })

    run_ssh(
        args,
        f"mkdir -p {remote_quote(remote_run_dir)}"
    )

    scp_to_remote(args, room_path, remote_room)
    scp_to_remote(args, in_placement_legacy, remote_in_placement_legacy)
    scp_to_remote(args, in_placement_v1, remote_in_placement_v1)
    scp_to_remote(args, in_scene_v1, remote_in_scene_v1)

    remote_cmd = " ".join([
        remote_quote(args.lego_python),
        remote_quote(args.lego_helper_script),
        "--room-type", remote_quote(args.room_type),
        "--mode", remote_quote(args.mode),
        "--room", remote_quote(remote_room),
        "--in-placement-legacy", remote_quote(remote_in_placement_legacy),
        "--in-placement-v1", remote_quote(remote_in_placement_v1),
        "--in-scene-v1", remote_quote(remote_in_scene_v1),
        "--out-placement-legacy", remote_quote(remote_out_placement_legacy),
        "--out-placement-v1", remote_quote(remote_out_placement_v1),
        "--out-scene-legacy", remote_quote(remote_out_scene_legacy),
        "--out-scene-v1", remote_quote(remote_out_scene_v1),
        "--checkpoint", remote_quote(checkpoint),
        "--lego-repo", remote_quote(args.lego_repo or ""),
    ])

    run_ssh(args, remote_cmd)

    scp_from_remote(args, remote_out_placement_legacy, out_placement_legacy)
    scp_from_remote(args, remote_out_placement_v1, out_placement_v1)
    scp_from_remote(args, remote_out_scene_legacy, out_scene_legacy)
    scp_from_remote(args, remote_out_scene_v1, out_scene_v1)

    print("✅ LEGO-Net remote postprocess completed")
    print(f"   placement_legacy: {out_placement_legacy}")
    print(f"   placement_v1:     {out_placement_v1}")
    print(f"   scene_legacy:     {out_scene_legacy}")
    print(f"   scene_v1:         {out_scene_v1}")


if __name__ == "__main__":
    main()