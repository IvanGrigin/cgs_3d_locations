#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/run_pipeline.py

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------
# Пути
# ------------------------------------------------------------
CHOOSER_SCRIPT = "src/ChooseObject/choose_obj_from_prepared.py"
CUBE_SCRIPT = "src/Plasement/CubePlacement.py"
BLENDER_VIS_SCRIPT = "src/Plasement/BlenderVisualizePlacement.py"
ML_PLACER_SCRIPT = "src/ml/infer/run_placer.py"
DIFFUSCENE_REMOTE_SCRIPT = "src/Plasement/run_diffuscene_remote.py"

DEFAULT_ROOM_GLB = "data/input/room.glb"
DEFAULT_ROOM_JSON = "data/input/room.json"

LEGACY_OBJECTS_JSON = "data/input/objects.json"
LEGACY_PLACEMENT_JSON = "data/output/placement_result.json"

PREPARED_INFO_DEFAULT = "data/sourse/3D-FRONT/prepared_model_info.json"
FUTURE_ROOT_DEFAULT = "data/sourse/3D-FRONT/3D-FUTURE-model"

MAX_ATTEMPTS = 30
TMP_ROOT = "out/tmp"

DEFAULT_MODES_BY_PLACER = {
    "cube": ["random", "relaxed"],
    "forest": ["random", "relaxed"],
    "graph_stat": ["random", "relaxed"],
    "diffusion": ["random", "relaxed"],
    "diffuscene_remote": ["diffuscene"],
}


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_room_spec_and_placements(room_json_path: str, placement_json_path: str, out_json_path: str) -> None:
    room_p = Path(room_json_path).expanduser().resolve()
    pl_p = Path(placement_json_path).expanduser().resolve()
    out_p = Path(out_json_path).expanduser().resolve()

    room = json.loads(room_p.read_text(encoding="utf-8"))
    pl = json.loads(pl_p.read_text(encoding="utf-8"))

    placements = pl.get("placements") or pl.get("items") or []
    if not isinstance(placements, list):
        placements = []

    scene = dict(room)
    scene["placements"] = placements

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_mode_run_dir(mode: str, run_dir_arg: Optional[str]) -> tuple[Path, bool]:
    """
    Если run_dir задан явно — используем его как есть.
    Если нет — создаём уникальную папку формата:
      out/tmp/<mode>_<YYYYMMDD>_<HHMMSS>_<hash>
    """
    if run_dir_arg:
        p = Path(run_dir_arg).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p, True

    run_hash = secrets.token_urlsafe(7).replace("-", "").replace("_", "").lower()
    p = Path(TMP_ROOT).resolve() / f"{mode}_{now_stamp()}_{run_hash}"
    p.mkdir(parents=True, exist_ok=True)
    return p, False


def read_prompt_from_args(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return str(args.prompt)
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().resolve().read_text(encoding="utf-8")
    if args.items:
        return "Нужно разместить следующие предметы: " + ", ".join(args.items)
    raise RuntimeError("Нужно передать либо positional items, либо --prompt, либо --prompt-file")


def sync_objects_to_legacy_input(objects_path: Path) -> None:
    dst = Path(LEGACY_OBJECTS_JSON).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(objects_path, dst)


def blender_outputs_for_mode(args: argparse.Namespace, run_dir: Path, mode: str) -> tuple[Optional[str], Optional[str]]:
    if args.save_blend:
        p = Path(args.save_blend).expanduser().resolve()
        if p.suffix.lower() == ".blend":
            blend = str(p.with_name(f"{p.stem}_{mode}.blend"))
        else:
            blend = str(p)
    else:
        blend = str((run_dir / f"scene_{mode}.blend").resolve())

    if args.render:
        p = Path(args.render).expanduser().resolve()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            render = str(p.with_name(f"{p.stem}_{mode}{p.suffix}"))
        else:
            render = str(p)
    else:
        render = str((run_dir / f"render_{mode}.png").resolve())

    return blend, render


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def parse_modes(args: argparse.Namespace) -> list[str]:
    raw = args.modes.strip() if args.modes else ""
    if raw:
        modes = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        modes = list(DEFAULT_MODES_BY_PLACER.get(args.placer, ["random", "relaxed"]))

    if not modes:
        raise RuntimeError("Список режимов пуст")

    seen = set()
    uniq = []
    for m in modes:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


# ------------------------------------------------------------
# Choose stage
# ------------------------------------------------------------
def run_choose_stage(args: argparse.Namespace, room_path: str, prompt_text: str, run_dir: Path, seed: int) -> Path:
    out_objects = run_dir / "objects.json"

    cmd = [
        sys.executable, CHOOSER_SCRIPT,
        "--room-json", os.path.abspath(room_path),
        "--prompt", prompt_text,
        "--prepared-info", os.path.abspath(args.prepared_info),
        "--future-root", os.path.abspath(args.future_root),
        "--out", str(out_objects.resolve()),
        "--run-dir", str(run_dir.resolve()),
        "--seed", str(int(seed)),
    ]

    print("▶ Выбор предметов:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_objects


# ------------------------------------------------------------
# Placer stage
# ------------------------------------------------------------
def run_cube_placer(room_path: str, objects_path: Path, mode: str, out_path: Path) -> None:
    sync_objects_to_legacy_input(objects_path)

    cube_input = f"{os.path.abspath(room_path)}\n{str(objects_path.resolve())}\n{mode}\n"
    subprocess.run([sys.executable, CUBE_SCRIPT], input=cube_input, text=True, check=True)

    legacy_out = Path(LEGACY_PLACEMENT_JSON).resolve()
    if not legacy_out.is_file():
        raise RuntimeError(f"Cube placer не создал {legacy_out}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_out, out_path)


def run_ml_placer(args: argparse.Namespace, room_path: str, objects_path: Path, mode: str, seed: int, out_path: Path) -> None:
    if not room_path.lower().endswith(".json"):
        raise RuntimeError("ML placer требует room-spec .json")

    if not args.ml_model:
        raise RuntimeError(f"--ml-model обязателен для placer={args.placer}")

    cmd = [
        sys.executable, ML_PLACER_SCRIPT,
        "--backend", args.placer,
        "--model", os.path.abspath(args.ml_model),
        "--room", os.path.abspath(room_path),
        "--objects", str(objects_path.resolve()),
        "--out", str(out_path.resolve()),
        "--device", args.ml_device,
        "--seed", str(int(seed)),
    ]

    if args.placer == "diffusion":
        cmd += ["--ddim-steps", str(int(args.diffusion_steps))]

    if mode:
        cmd += ["--mode", mode]

    print("▶ Запуск ML-расстановщика:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_diffuscene_remote_placer(
    args: argparse.Namespace,
    room_path: str,
    objects_path: Path,
    mode: str,
    seed: int,
    out_path: Path,
    run_dir: Path,
) -> None:
    del seed
    del mode

    remote_run_name = run_dir.name
    remote_artifacts_dir = Path(TMP_ROOT).resolve() / remote_run_name

    cmd = [
        sys.executable, DIFFUSCENE_REMOTE_SCRIPT,
        "--room", os.path.abspath(room_path),
        "--objects", str(objects_path.resolve()),
        "--out", str(out_path.resolve()),
        "--run-name", remote_run_name,
        "--remote-runner", os.path.abspath(args.remote_runner),
    ]

    print("▶ Запуск DiffuScene remote placer:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not out_path.is_file():
        raise RuntimeError(f"DiffuScene remote не создал итоговый placement: {out_path}")

    local_mode_artifacts = run_dir / "diffuscene_remote_artifacts"
    if remote_artifacts_dir.is_dir():
        copy_tree_contents(remote_artifacts_dir, local_mode_artifacts)
        print(f"📥 Артефакты DiffuScene -> {local_mode_artifacts}")
    else:
        print(f"⚠️ Папка remote-артефактов не найдена: {remote_artifacts_dir}")


# ------------------------------------------------------------
# Blender stage
# ------------------------------------------------------------
def run_blender_for_mode(
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    mode: str,
    placement_path: Path,
) -> None:
    is_room_json = room_path.lower().endswith(".json")

    if is_room_json:
        scene_json = run_dir / f"scene_{mode}.json"
        merge_room_spec_and_placements(room_path, str(placement_path.resolve()), str(scene_json.resolve()))
        scene_json_for_blender = scene_json
        auto_no_import_glb = True
    else:
        scene_json_for_blender = placement_path
        auto_no_import_glb = False

    blend_out, render_out = blender_outputs_for_mode(args, run_dir, mode)

    glb_for_arg = os.path.abspath(DEFAULT_ROOM_GLB)
    cmd = [
        sys.executable, BLENDER_VIS_SCRIPT,
        "--glb", glb_for_arg,
        "--json", str(scene_json_for_blender.resolve()),
    ]

    if args.blender:
        cmd += ["--blender", args.blender]
    if args.headless:
        cmd.append("--background")

    if args.no_import_glb or auto_no_import_glb:
        cmd.append("--no-import-glb")

    if blend_out:
        Path(blend_out).resolve().parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--save-blend", str(Path(blend_out).resolve())]

    if render_out:
        Path(render_out).resolve().parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--render", str(Path(render_out).resolve())]

    print("▶ Запуск Blender-визуализатора:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def visualize_existing_run(args: argparse.Namespace, room_path: str, run_dir: Path, modes: list[str]) -> bool:
    built_any = False

    for mode in modes:
        scene_path = run_dir / f"scene_{mode}.json"
        placement_path = run_dir / f"placement_{mode}.json"

        if scene_path.is_file():
            blend_out, render_out = blender_outputs_for_mode(args, run_dir, mode)
            glb_for_arg = os.path.abspath(DEFAULT_ROOM_GLB)

            cmd = [
                sys.executable, BLENDER_VIS_SCRIPT,
                "--glb", glb_for_arg,
                "--json", str(scene_path.resolve()),
            ]

            if args.blender:
                cmd += ["--blender", args.blender]
            if args.headless:
                cmd.append("--background")
            if args.no_import_glb or room_path.lower().endswith(".json"):
                cmd.append("--no-import-glb")
            if blend_out:
                cmd += ["--save-blend", str(Path(blend_out).resolve())]
            if render_out:
                cmd += ["--render", str(Path(render_out).resolve())]

            print(f"▶ Reuse Blender scene for mode={mode}:\n ", " ".join(cmd))
            subprocess.run(cmd, check=True)
            built_any = True
            continue

        if placement_path.is_file():
            run_blender_for_mode(args, room_path, run_dir, mode, placement_path)
            built_any = True

    if not built_any:
        legacy_scene = run_dir / "scene_room_and_placements.json"
        legacy_placement = run_dir / "placement_result.json"

        if legacy_scene.is_file():
            blend_out, render_out = blender_outputs_for_mode(args, run_dir, "reuse")
            glb_for_arg = os.path.abspath(DEFAULT_ROOM_GLB)
            cmd = [
                sys.executable, BLENDER_VIS_SCRIPT,
                "--glb", glb_for_arg,
                "--json", str(legacy_scene.resolve()),
            ]
            if args.blender:
                cmd += ["--blender", args.blender]
            if args.headless:
                cmd.append("--background")
            if args.no_import_glb or room_path.lower().endswith(".json"):
                cmd.append("--no-import-glb")
            if blend_out:
                cmd += ["--save-blend", str(Path(blend_out).resolve())]
            if render_out:
                cmd += ["--render", str(Path(render_out).resolve())]
            print("▶ Reuse legacy scene:\n ", " ".join(cmd))
            subprocess.run(cmd, check=True)
            built_any = True

        elif legacy_placement.is_file():
            run_blender_for_mode(args, room_path, run_dir, "reuse", legacy_placement)
            built_any = True

    return built_any

def visualize_existing_run(args: argparse.Namespace, room_path: str, run_dir: Path, modes: list[str]) -> bool:
    built_any = False

    for mode in modes:
        scene_path = run_dir / f"scene_{mode}.json"
        placement_path = run_dir / f"placement_{mode}.json"

        if scene_path.is_file():
            if args.skip_blender:
                print(f"⏭ Пропуск Blender для существующей сцены mode={mode}")
                built_any = True
                continue

            blend_out, render_out = blender_outputs_for_mode(args, run_dir, mode)
            glb_for_arg = os.path.abspath(DEFAULT_ROOM_GLB)

            cmd = [
                sys.executable, BLENDER_VIS_SCRIPT,
                "--glb", glb_for_arg,
                "--json", str(scene_path.resolve()),
            ]

            if args.blender:
                cmd += ["--blender", args.blender]
            if args.headless:
                cmd.append("--background")
            if args.no_import_glb or room_path.lower().endswith(".json"):
                cmd.append("--no-import-glb")
            if blend_out:
                cmd += ["--save-blend", str(Path(blend_out).resolve())]
            if render_out:
                cmd += ["--render", str(Path(render_out).resolve())]

            print(f"▶ Reuse Blender scene for mode={mode}:\n ", " ".join(cmd))
            subprocess.run(cmd, check=True)
            built_any = True
            continue

        if placement_path.is_file():
            if args.skip_blender:
                print(f"⏭ Пропуск Blender для существующей расстановки mode={mode}")
                built_any = True
                continue

            run_blender_for_mode(args, room_path, run_dir, mode, placement_path)
            built_any = True

    if not built_any:
        legacy_scene = run_dir / "scene_room_and_placements.json"
        legacy_placement = run_dir / "placement_result.json"

        if legacy_scene.is_file():
            if args.skip_blender:
                print("⏭ Пропуск Blender для legacy scene")
                built_any = True
            else:
                blend_out, render_out = blender_outputs_for_mode(args, run_dir, "reuse")
                glb_for_arg = os.path.abspath(DEFAULT_ROOM_GLB)
                cmd = [
                    sys.executable, BLENDER_VIS_SCRIPT,
                    "--glb", glb_for_arg,
                    "--json", str(legacy_scene.resolve()),
                ]
                if args.blender:
                    cmd += ["--blender", args.blender]
                if args.headless:
                    cmd.append("--background")
                if args.no_import_glb or room_path.lower().endswith(".json"):
                    cmd.append("--no-import-glb")
                if blend_out:
                    cmd += ["--save-blend", str(Path(blend_out).resolve())]
                if render_out:
                    cmd += ["--render", str(Path(render_out).resolve())]
                print("▶ Reuse legacy scene:\n ", " ".join(cmd))
                subprocess.run(cmd, check=True)
                built_any = True

        elif legacy_placement.is_file():
            if args.skip_blender:
                print("⏭ Пропуск Blender для legacy placement")
                built_any = True
            else:
                run_blender_for_mode(args, room_path, run_dir, "reuse", legacy_placement)
                built_any = True

    return built_any

# ------------------------------------------------------------
# Main pipeline per mode
# ------------------------------------------------------------
def run_pipeline_for_mode(
    args: argparse.Namespace,
    room_path: str,
    run_dir: Path,
    mode: str,
    prompt_text: str,
) -> None:
    print(f"\n====== РЕЖИМ {mode.upper()} ======")
    print(f"📁 mode_run_dir: {run_dir}")

    chooser_seed = int.from_bytes(secrets.token_bytes(8), "big")
    objects_path = run_choose_stage(
        args=args,
        room_path=room_path,
        prompt_text=prompt_text,
        run_dir=run_dir,
        seed=chooser_seed,
    )

    run_manifest = {
        "room": room_path,
        "prompt": prompt_text,
        "chooser_seed": chooser_seed,
        "placer": args.placer,
        "mode": mode,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    placement_out = run_dir / f"placement_{mode}.json"

    for attempt in range(1, int(args.max_attempts) + 1):
        print(f"\n---------- ПОПЫТКА {attempt} ({mode}) ----------")
        try:
            attempt_seed = int.from_bytes(secrets.token_bytes(8), "big")

            attempt_info = {
                "attempt": attempt,
                "attempt_seed": attempt_seed,
                "chooser_seed": chooser_seed,
                "mode": mode,
                "placer": args.placer,
                "objects_path": str(objects_path),
            }
            write_json(run_dir / f"attempt_{attempt:02d}.json", attempt_info)

            if args.placer == "cube":
                run_cube_placer(
                    room_path=room_path,
                    objects_path=objects_path,
                    mode=mode,
                    out_path=placement_out,
                )
            elif args.placer == "diffuscene_remote":
                run_diffuscene_remote_placer(
                    args=args,
                    room_path=room_path,
                    objects_path=objects_path,
                    mode=mode,
                    seed=attempt_seed,
                    out_path=placement_out,
                    run_dir=run_dir,
                )
            else:
                run_ml_placer(
                    args=args,
                    room_path=room_path,
                    objects_path=objects_path,
                    mode=mode,
                    seed=attempt_seed,
                    out_path=placement_out,
                )

            if args.skip_blender:
                print(f"⏭ Пропуск Blender для режима {mode}")
            else:
                run_blender_for_mode(
                    args=args,
                    room_path=room_path,
                    run_dir=run_dir,
                    mode=mode,
                    placement_path=placement_out,
                )

            print(f"\n✅ УСПЕХ! РЕЖИМ {mode}")
            return

        except subprocess.CalledProcessError:
            print(f"⚠️ Неудачная попытка ({mode}), пересборка...")
        except Exception as e:
            print(f"❌ Ошибка ({mode}): {e}")

    raise RuntimeError(f"Не удалось собрать сцену в режиме {mode}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Пайплайн: prompt -> выбор предметов -> расстановка -> Blender")

    p.add_argument("items", nargs="*", help="Опциональный список предметов. Если нет --prompt, будет собран текстовый prompt из items.")
    p.add_argument("--prompt", default=None, help="Текстовый prompt для генерации набора предметов")
    p.add_argument("--prompt-file", default=None, help="Файл с prompt")

    p.add_argument("--room", default=DEFAULT_ROOM_JSON, help="Путь комнаты (.json room-spec или .glb)")
    p.add_argument("--prepared-info", default=PREPARED_INFO_DEFAULT)
    p.add_argument("--future-root", default=FUTURE_ROOT_DEFAULT)

    p.add_argument("--run-dir", default=None, help="Папка run. Если уже содержит результаты, Blender-сцены будут построены сразу.")
    p.add_argument("--keep-tmp", dest="keep_tmp", action="store_true", default=True)
    p.add_argument("--no-keep-tmp", dest="keep_tmp", action="store_false")

    p.add_argument("--blender", default=None, help="Путь к бинарю Blender")
    p.add_argument("--headless", action="store_true", help="Запуск Blender без GUI")
    p.add_argument("--no-import-glb", action="store_true", help="Не импортировать room.glb")
    p.add_argument("--save-blend", default=None, help="Базовый путь для .blend; mode будет дописан автоматически")
    p.add_argument("--render", default=None, help="Базовый путь для render; mode будет дописан автоматически")
    p.add_argument("--skip-blender", action="store_true", help="Не запускать Blender-визуализацию")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)

    p.add_argument(
        "--placer",
        choices=["cube", "forest", "graph_stat", "diffusion", "diffuscene_remote"],
        default="cube"
    )
    p.add_argument("--ml-model", default=None)
    p.add_argument("--ml-device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--diffusion-steps", type=int, default=50)

    p.add_argument(
        "--remote-runner",
        default="src/run_room_layout_remote.sh",
        help="Локальный shell-скрипт, который отправляет room+objects на сервер DiffuScene"
    )

    p.add_argument(
        "--modes",
        default=None,
        help=(
            "Список режимов через запятую. "
            "Например: diffuscene или random,relaxed,diffuscene. "
            "Если не задан, берётся набор по умолчанию для выбранного placer."
        ),
    )

    return p
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    room_path = os.path.abspath((args.room or DEFAULT_ROOM_JSON).strip())
    modes = parse_modes(args)

    print(f"📦 modes: {', '.join(modes)}")

    explicit_run_dir = None
    if args.run_dir:
        explicit_run_dir = Path(args.run_dir).expanduser().resolve()
        explicit_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 explicit run_dir: {explicit_run_dir}")

    prompt_text = read_prompt_from_args(args)
    created_run_dirs: list[Path] = []

    try:
        if explicit_run_dir and visualize_existing_run(args, room_path, explicit_run_dir, modes):
            print("\n✅ Сцены построены из существующего run_dir")
            return

        for mode in modes:
            if explicit_run_dir:
                mode_run_dir = explicit_run_dir
                run_dir_explicit = True
            else:
                mode_run_dir, run_dir_explicit = make_mode_run_dir(mode, None)

            del run_dir_explicit
            created_run_dirs.append(mode_run_dir)

            run_pipeline_for_mode(
                args=args,
                room_path=room_path,
                run_dir=mode_run_dir,
                mode=mode,
                prompt_text=prompt_text,
            )

        print("\n✅ ВСЕ РЕЖИМЫ ОТРАБОТАЛИ УСПЕШНО")

    finally:
        if not args.keep_tmp and not explicit_run_dir:
            for p in created_run_dirs:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"🗑 Удалён run_dir: {p}")


if __name__ == "__main__":
    main()