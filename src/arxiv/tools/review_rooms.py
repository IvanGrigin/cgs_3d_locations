#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/review_rooms.py
#
# Режим "одна комната + перебор 24 upright":
# - запускаем Blender 24 раза для ОДНОЙ комнаты
# - каждый запуск: build_front_scene.py получает --upright_idx k (0..23)
# - вы смотрите сцену, закрываете Blender, отвечаете y/n (или q)
# - результаты пишем в JSON (по каждому k)

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def _default_blender_path() -> str:
    return "/Applications/Blender.app/Contents/MacOS/Blender"


def _ask_yes_no(prompt: str) -> bool:
    while True:
        s = input(prompt).strip().lower()
        if s in ("y", "yes", "1", "+"):
            return True
        if s in ("n", "no", "0", "-"):
            return False
        if s in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        print("Введите y/n (или q для выхода).")


def _run_blender(
    blender: str,
    build_script: Path,
    scene_json: Path,
    models_root: Path,
    front_raw_json: Path | None,
    room_id: str,
    upright_idx: int,
):
    cmd = [
        blender,
        "--python", str(build_script),
        "--",
        "--scene_json", str(scene_json),
        "--models_root", str(models_root),
        "--room_id", room_id,
        "--upright_idx", str(int(upright_idx)),
    ]
    if front_raw_json is not None:
        cmd += ["--front_raw_json", str(front_raw_json)]

    print(f"\n=== Blender: room={room_id}, upright_idx={upright_idx}/23 ===\n")
    print("CMD:", " ".join(cmd), "\n")
    return subprocess.run(cmd, check=False).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=_default_blender_path(), help="Путь к Blender бинарнику")
    ap.add_argument("--build_script", default="src/tools/build_front_scene.py", help="Скрипт визуализации в Blender")
    ap.add_argument("--scene_json", required=True, help="prepared_scene_*.json")
    ap.add_argument("--models_root", required=True, help=".../3D-FUTURE-model")
    ap.add_argument("--front_raw_json", default=None, help="Исходный 3D-FRONT json для ref->jid")

    # строго одна комната
    ap.add_argument("--room_id", required=True, help="Ровно одна комната для проверки")

    ap.add_argument("--out", default="data/output/review_results_upright24.json", help="Куда сохранить результаты (json)")
    ap.add_argument("--start", type=int, default=0, help="С какого upright_idx начать (0..23)")
    ap.add_argument("--end", type=int, default=23, help="На каком upright_idx закончить (0..23)")

    args = ap.parse_args()

    blender = args.blender
    build_script = Path(args.build_script)
    scene_json = Path(args.scene_json)
    models_root = Path(args.models_root)
    front_raw_json = Path(args.front_raw_json) if args.front_raw_json else None

    if not Path(blender).exists():
        print(f"Ошибка: blender не найден по пути: {blender}", file=sys.stderr)
        sys.exit(2)
    if not build_script.exists():
        print(f"Ошибка: build_script не найден: {build_script}", file=sys.stderr)
        sys.exit(2)
    if not scene_json.exists():
        print(f"Ошибка: scene_json не найден: {scene_json}", file=sys.stderr)
        sys.exit(2)
    if not models_root.exists():
        print(f"Ошибка: models_root не найден: {models_root}", file=sys.stderr)
        sys.exit(2)
    if front_raw_json is not None and not front_raw_json.exists():
        print(f"Ошибка: front_raw_json не найден: {front_raw_json}", file=sys.stderr)
        sys.exit(2)

    start = max(0, min(23, int(args.start)))
    end = max(0, min(23, int(args.end)))
    if start > end:
        start, end = end, start

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "scene_json": str(scene_json),
        "models_root": str(models_root),
        "front_raw_json": str(front_raw_json) if front_raw_json else None,
        "room_id": args.room_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "items": [],
    }

    try:
        for k in range(start, end + 1):
            rc = _run_blender(
                blender=blender,
                build_script=build_script,
                scene_json=scene_json,
                models_root=models_root,
                front_raw_json=front_raw_json,
                room_id=args.room_id,
                upright_idx=k,
            )
            print(f"Blender завершился с кодом: {rc}")

            ok = _ask_yes_no(f"upright_idx={k}: всё верно? (y/n, q=выход): ")
            results["items"].append({"upright_idx": k, "ok": ok, "blender_returncode": rc})

            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Сохранено:", out_path)

    except KeyboardInterrupt:
        print("\nОстановлено пользователем. Промежуточные результаты сохранены:", out_path)

    results["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nГотово. Итоговый файл:", out_path)


if __name__ == "__main__":
    main()
