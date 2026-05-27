#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _room_files(bundle_dir: Path) -> list[Path]:
    files = sorted(p for p in bundle_dir.glob("room_*.json") if p.is_file())
    if not files:
        raise RuntimeError(f"Не найдены room_*.json в {bundle_dir}")
    return files


def _semantic_hint(room_path: Path) -> str:
    data = _load_json(room_path)
    room = data.get("room", data)
    return str(room.get("room_type") or room.get("name") or room_path.stem)


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serial batch-run clean Infinigen for apartment room bundle")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--out-root", default="out/apartment")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--remote-host", required=True)
    p.add_argument("--remote-port", type=int, default=22)
    p.add_argument("--remote-user", required=True)
    p.add_argument("--remote-key", required=True)
    p.add_argument("--remote-conda-env", default="infinigen")
    p.add_argument("--remote-infinigen-src", default="/workspace/infinigen/src")
    p.add_argument("--infinigen-task", default="coarse")
    p.add_argument("--infinigen-configs", nargs="+", default=["singleroom.gin"])
    return p


def main() -> None:
    args = build_cli().parse_args()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    run_id = f"{_stamp()}_{uuid.uuid4().hex[:8]}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "out_dir": str(out_dir),
        "remote_host": args.remote_host,
        "remote_port": int(args.remote_port),
        "remote_user": args.remote_user,
        "remote_conda_env": args.remote_conda_env,
        "remote_infinigen_src": args.remote_infinigen_src,
        "infinigen_task": args.infinigen_task,
        "infinigen_configs": list(args.infinigen_configs),
        "rooms": [],
    }

    room_files = _room_files(bundle_dir)
    for idx, room_path in enumerate(room_files, start=1):
        room_out = out_dir / room_path.stem
        room_out.mkdir(parents=True, exist_ok=True)
        seed = int(args.seed_base) + idx
        cmd = [
            sys.executable,
            str((Path(__file__).resolve().parents[1] / "Plasement" / "run_infinigen_clean.py").resolve()),
            "--room",
            str(room_path),
            "--seed",
            str(seed),
            "--out",
            str((room_out / "placement.json").resolve()),
            "--run-dir",
            str(room_out.resolve()),
            "--remote-host",
            str(args.remote_host),
            "--remote-port",
            str(int(args.remote_port)),
            "--remote-user",
            str(args.remote_user),
            "--remote-key",
            str(Path(args.remote_key).expanduser()),
            "--remote-conda-env",
            str(args.remote_conda_env),
            "--remote-infinigen-src",
            str(args.remote_infinigen_src),
            "--infinigen-task",
            str(args.infinigen_task),
            "--infinigen-configs",
            *list(args.infinigen_configs),
        ]
        print(f"[{idx}/{len(room_files)}] {room_path.name}  semantic_hint={_semantic_hint(room_path)}  seed={seed}")
        print("▶", " ".join(cmd))
        status = "ok"
        returncode = 0
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            returncode = int(exc.returncode)
            print(f"[warn] room failed: {room_path.name} returncode={returncode}", file=sys.stderr)
        summary["rooms"].append(
            {
                "room_file": str(room_path),
                "room_name": room_path.stem,
                "semantic_hint": _semantic_hint(room_path),
                "seed": seed,
                "status": status,
                "returncode": returncode,
                "out_dir": str(room_out),
            }
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: apartment batch out_dir={out_dir}")


if __name__ == "__main__":
    main()
