#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# src/tools/batch_make_processed_mini.py
#
# Пакетно прогоняет *.json из 3D-FRONT-processed через Blender background,
# вызывая ИМЕННО src/tools/make_scene_mini_blender.py
#
# Выход:
# - <out_dir>/<stem>.mini.json
# - <out_dir>/_logs/<stem>.blender.log
# - <out_dir>/_failed.txt
# - <out_dir>/_ok.txt
#

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_processed_jsons(processed_dir: Path) -> List[Path]:
    files = sorted([p for p in processed_dir.glob("*.json") if p.is_file()])
    return [p for p in files if p.name != "meta.json"]


def run_blender(
    blender_bin: Path,
    blender_script: Path,
    scene_json: Path,
    models_root: Path,
    out_json: Path,
    prefer_raw: int,
    resolve_collisions: int,
    collision_margin: float,
    wall_height: float,
    upright_idx: int,
    pretty: int,
    log_path: Path,
) -> int:
    cmd = [
        str(blender_bin),
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--scene_json",
        str(scene_json),
        "--models_root",
        str(models_root),
        "--out_json",
        str(out_json),
        "--prefer_raw",
        str(int(prefer_raw)),
        "--resolve_collisions",
        str(int(resolve_collisions)),
        "--collision_margin",
        str(float(collision_margin)),
        "--wall_height",
        str(float(wall_height)),
        "--upright_idx",
        str(int(upright_idx)),
        "--pretty",
        str(int(pretty)),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--blender_bin", required=True)

    ap.add_argument("--prefer_raw", type=int, default=1)
    ap.add_argument("--resolve_collisions", type=int, default=1)
    ap.add_argument("--collision_margin", type=float, default=0.02)
    ap.add_argument("--wall_height", type=float, default=2.7)

    # upright_idx=2 по умолчанию
    ap.add_argument("--upright_idx", type=int, default=2)

    ap.add_argument("--pretty", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = без лимита")
    ap.add_argument("--skip_existing", type=int, default=1)

    args = ap.parse_args()

    processed_dir = Path(args.processed_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    models_root = Path(args.models_root).expanduser().resolve()
    blender_bin = Path(args.blender_bin).expanduser().resolve()

    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")
    if not models_root.exists():
        raise FileNotFoundError(f"models_root not found: {models_root}")
    if not blender_bin.exists():
        raise FileNotFoundError(f"blender_bin not found: {blender_bin}")

    blender_script = Path(__file__).resolve().parent / "make_scene_mini_blender.py"
    if not blender_script.exists():
        raise FileNotFoundError(f"make_scene_mini_blender.py not found: {blender_script}")

    ensure_dir(out_dir)
    logs_dir = out_dir / "_logs"
    ensure_dir(logs_dir)

    failed_txt = out_dir / "_failed.txt"
    ok_txt = out_dir / "_ok.txt"
    if failed_txt.exists():
        failed_txt.unlink()
    if ok_txt.exists():
        ok_txt.unlink()

    files = list_processed_jsons(processed_dir)
    if int(args.limit) > 0:
        files = files[: int(args.limit)]

    ok = 0
    bad = 0
    total = len(files)

    for i, scene_json in enumerate(files, 1):
        stem = scene_json.stem
        out_json = out_dir / f"{stem}.mini.json"
        log_path = logs_dir / f"{stem}.blender.log"

        if int(args.skip_existing) == 1 and out_json.exists():
            print(f"[{i}/{total}] SKIP exists {scene_json.name}")
            continue

        code = run_blender(
            blender_bin=blender_bin,
            blender_script=blender_script,
            scene_json=scene_json,
            models_root=models_root,
            out_json=out_json,
            prefer_raw=int(args.prefer_raw),
            resolve_collisions=int(args.resolve_collisions),
            collision_margin=float(args.collision_margin),
            wall_height=float(args.wall_height),
            upright_idx=int(args.upright_idx),
            pretty=int(args.pretty),
            log_path=log_path,
        )

        # ВАЖНО: считаем успехом только наличие out_json и code==0
        if code == 0 and out_json.exists() and out_json.stat().st_size > 0:
            ok += 1
            print(f"[{i}/{total}] OK   file={scene_json.name}")
            with ok_txt.open("a", encoding="utf-8") as f:
                f.write(f"{scene_json.name}\n")
        else:
            bad += 1
            print(f"[{i}/{total}] FAIL code={code} file={scene_json.name}")
            with failed_txt.open("a", encoding="utf-8") as f:
                f.write(f"{scene_json.name}\tcode={code}\tlog={log_path.name}\n")

    print(f"DONE: ok={ok} bad={bad} out_dir={out_dir}")
    print(f"Logs: {logs_dir}")
    if bad > 0:
        print(f"See failures: {failed_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
