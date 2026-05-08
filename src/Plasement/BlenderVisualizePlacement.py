#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/Plasement/BlenderVisualizePlacement.py

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_JSON = "data/output/placement_result.json"
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


def infer_reference_blend(json_path):
    try:
        json_abs = os.path.abspath(json_path)
        with open(json_abs, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    json_dir = Path(json_abs).resolve().parent
    meta = data.get("meta") or {}
    placement_meta = meta.get("placement_meta") or {}
    raw_scene_blend = str(placement_meta.get("scene_blend") or "").strip()

    candidates = []
    if raw_scene_blend:
        raw_path = Path(raw_scene_blend).expanduser()
        candidates.append(raw_path)
        candidates.append(json_dir / raw_path.name)

    candidates.append(json_dir / "infinigen_clean_scene.blend")
    candidates.append(json_dir / "scene_infinigen_clean.blend")

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return str(path.resolve())
    return None

def main():
    ap = argparse.ArgumentParser(description="Визуализатор интерьера в Blender")
    ap.add_argument("--glb", default=None, help="Compat arg, ignored")
    ap.add_argument("--json", default=DEFAULT_JSON, help="Путь к placement_result.json")
    ap.add_argument("--blender", default=None, help="Путь к бинарю Blender")
    ap.add_argument("--background", action="store_true", help="Запуск в фоне (без GUI)")
    ap.add_argument("--no-import-glb", action="store_true", help="Compat flag, ignored")
    ap.add_argument("--reference-blend", default=None, help="Открыть существующий .blend как референс-сцену")
    ap.add_argument("--overlay-bbox-only", action="store_true", help="Не импортировать меши, а наложить bbox/labels поверх reference .blend")
    ap.add_argument("--no-bbox-fallback", action="store_true", help="Отключить bbox fallback по умолчанию для объектов без mesh")
    ap.add_argument("--save-blend", default=None, help="Сохранить .blend в указанный путь")
    ap.add_argument("--render", default=None, help="Сохранить рендер в PNG (путь к файлу)")
    ap.add_argument("--build-report", default=None, help="Сохранить JSON-диагностику сборки Blender")
    ap.add_argument("--highlight-item-ids", default=None, help="Список id через запятую: для них рисуется черный bbox")
    ap.add_argument("--hide-room-shell", action="store_true", help="Скрыть walls / ceiling / exterior перед рендером")
    ap.add_argument("--render-layer", default="all", choices=["all", "kitchen", "room_shell", "surfaces", "windows", "curtains", "tables_chairs", "non_kitchen"], help="Фильтр видимости для раздельных рендеров/GIF")
    ap.add_argument("--force-tint", action="store_true", help="Принудительно заменить материалы импортированных мешей на tint-материал")
    ap.add_argument("--turntable-render-dir", default=None, help="Каталог для PNG кадров turntable")
    ap.add_argument("--turntable-frames", type=int, default=24, help="Количество кадров turntable")
    ap.add_argument("--turntable-frame-index", type=int, default=None, help="Рендерить только один кадр turntable с указанным индексом")
    ap.add_argument("--turntable-elevation-deg", type=float, default=30.0, help="Угол возвышения камеры для orbit")
    ap.add_argument("--no-pack-assets", action="store_true", help="Не паковать ассеты в .blend")
    ap.add_argument("--verbose", action="store_true", help="Печатать подробную диагностику сборки Blender-сцены")
    args = ap.parse_args()

    blender_bin = find_executable(([args.blender] if args.blender else []) + DEFAULT_BLENDER_CANDIDATES)
    reference_blend = args.reference_blend or infer_reference_blend(args.json)

    cmd = [blender_bin]
    if reference_blend:
        cmd.append(os.path.abspath(reference_blend))
    if args.background:
        cmd.append("-b")  # без GUI

    cmd += [
        "--python", DEFAULT_BUILDER,
        "--",
        "--json", os.path.abspath(args.json),
        "--project-root", os.path.abspath(os.path.join(THIS_DIR, "..")),
    ]
    if reference_blend:
        cmd += ["--reference-blend", os.path.abspath(reference_blend)]
    if args.overlay_bbox_only:
        cmd += ["--overlay-bbox-only"]
    if args.no_bbox_fallback:
        cmd += ["--no-bbox-fallback"]
    if args.highlight_item_ids:
        cmd += ["--highlight-item-ids", str(args.highlight_item_ids)]
    if args.hide_room_shell:
        cmd += ["--hide-room-shell"]
    if args.render_layer and args.render_layer != "all":
        cmd += ["--render-layer", str(args.render_layer)]
    if args.force_tint:
        cmd += ["--force-tint"]
    if args.save_blend:
        cmd += ["--save-blend", os.path.abspath(args.save_blend)]
    if args.build_report:
        cmd += ["--build-report", os.path.abspath(args.build_report)]
    if args.render:
        cmd += ["--render", os.path.abspath(args.render)]
    if args.turntable_render_dir:
        cmd += ["--turntable-render-dir", os.path.abspath(args.turntable_render_dir)]
        cmd += ["--turntable-frames", str(int(args.turntable_frames or 24))]
        if args.turntable_frame_index is not None:
            cmd += ["--turntable-frame-index", str(int(args.turntable_frame_index))]
        cmd += ["--turntable-elevation-deg", str(float(args.turntable_elevation_deg or 0.0))]
    if args.no_pack_assets:
        cmd += ["--no-pack-assets"]
    if args.verbose:
        cmd += ["--verbose"]

    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
