#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/BlenderVisualizePlacement.py

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_GLB = "src/data/input/room.glb"
DEFAULT_JSON = "src/data/output/placement_result.json"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUILDER = os.path.join(THIS_DIR, "blender_scene_builder.py")

DEFAULT_BLENDER_CANDIDATES = [
    os.environ.get("BLENDER_PATH"),  # можно указать руками export BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender
    "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
    "blender",  # если blender в PATH
]

def find_executable(candidates):
    for p in candidates:
        if not p:
            continue
        # подходит и абсолютный путь, и имя в PATH
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        if shutil.which(p):
            return shutil.which(p)
    raise FileNotFoundError(
        "Не найден исполняемый файл Blender. "
        "Задайте путь флагом --blender или переменной окружения BLENDER_PATH."
    )

def main():
    ap = argparse.ArgumentParser(description="Визуализатор интерьера в Blender")
    ap.add_argument("--glb", default=DEFAULT_GLB, help="Путь к room.glb")
    ap.add_argument("--json", default=DEFAULT_JSON, help="Путь к placement_result.json")
    ap.add_argument("--blender", default=None, help="Путь к бинарю Blender")
    ap.add_argument("--background", action="store_true", help="Запуск в фоне (без GUI)")
    ap.add_argument("--no-import-glb", action="store_true", help="Не импортировать room.glb, рисовать только коробку комнаты")
    ap.add_argument("--save-blend", default=None, help="Сохранить .blend в указанный путь")
    ap.add_argument("--render", default=None, help="Сохранить рендер в PNG (путь к файлу)")
    args = ap.parse_args()

    blender_bin = find_executable(([args.blender] if args.blender else []) + DEFAULT_BLENDER_CANDIDATES)

    cmd = [blender_bin]
    if args.background:
        cmd.append("-b")  # без GUI

    cmd += [
        "--python", DEFAULT_BUILDER,
        "--",
        "--glb", os.path.abspath(args.glb),
        "--json", os.path.abspath(args.json),
        "--project-root", os.path.abspath(os.path.join(THIS_DIR, "..")),
    ]
    if not args.no_import_glb:
        cmd.append("--import-glb")
    if args.save_blend:
        cmd += ["--save-blend", os.path.abspath(args.save_blend)]
    if args.render:
        cmd += ["--render", os.path.abspath(args.render)]

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()