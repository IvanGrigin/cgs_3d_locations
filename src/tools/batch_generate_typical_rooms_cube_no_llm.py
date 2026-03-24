#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/batch_generate_typical_rooms_cube_no_llm.py

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOMS_DIR = "data/input/generated_typical_rooms"
DEFAULT_OUTPUT_ROOT = "out/batch_typical_rooms"

DEFAULT_CHOOSER_SCRIPT = "src/ChooseObject/choose_obj_from_prepared.py"
DEFAULT_CUBE_SCRIPT = "src/Plasement/CubePlacement.py"
DEFAULT_NORMALIZE_SCRIPT = "src/tools/normalize_scene_format.py"

DEFAULT_PREPARED_INFO = "data/sourse/3D-FRONT/prepared_model_info.json"
DEFAULT_FUTURE_ROOT = "data/sourse/3D-FRONT/3D-FUTURE-model"

DEFAULT_REPEATS = 20
DEFAULT_MODES = ["random", "relaxed"]

# Таймауты в секундах
DEFAULT_CHOOSER_TIMEOUT = 300
DEFAULT_CUBE_TIMEOUT = 300

# Нужен только для временного кэша chooser
CHOOSER_CACHE_PREFIX = "_chooser_cache"


@dataclass(frozen=True)
class PromptTemplate:
    key: str
    room_type: str
    text: str


PROMPT_TEMPLATES: list[PromptTemplate] = [
    # bedroom
    PromptTemplate(
        key="bedroom_classic_cozy",
        room_type="bedroom",
        text="classic cozy bedroom with king-size bed, two nightstands, wardrobe and ceiling lamp",
    ),
    PromptTemplate(
        key="bedroom_classic_family",
        room_type="bedroom",
        text="classic family bedroom with double bed, two bedside tables, wardrobe, dresser and ceiling lamp",
    ),
    PromptTemplate(
        key="bedroom_classic_soft",
        room_type="bedroom",
        text="classic soft bedroom with bed, two nightstands, wardrobe, side cabinet and ceiling lamp",
    ),
    PromptTemplate(
        key="bedroom_classic_storage",
        room_type="bedroom",
        text="classic bedroom with bed, two nightstands, large wardrobe, drawer chest and ceiling lamp",
    ),

    # living
    PromptTemplate(
        key="living_classic_guest",
        room_type="living",
        text="classic living room with sofa, armchair, coffee table, TV stand, sideboard and ceiling lamp",
    ),
    PromptTemplate(
        key="living_classic_formal",
        room_type="living",
        text="classic formal living room with sofa, two armchairs, coffee table, TV stand and ceiling lamp",
    ),
    PromptTemplate(
        key="living_classic_family",
        room_type="living",
        text="classic family living room with multi-seat sofa, armchair, coffee table, TV stand and ceiling lamp",
    ),
    PromptTemplate(
        key="living_classic_storage",
        room_type="living",
        text="classic living room with sofa, coffee table, TV stand, sideboard, cabinet and ceiling lamp",
    ),

    # office
    PromptTemplate(
        key="office_classic_work",
        room_type="office",
        text="classic office with desk, office chair, bookcase, cabinet and ceiling lamp",
    ),
    PromptTemplate(
        key="office_classic_study",
        room_type="office",
        text="classic study room with desk, chair, bookshelf, storage cabinet and ceiling lamp",
    ),
    PromptTemplate(
        key="office_classic_library",
        room_type="office",
        text="classic home office with desk, office chair, bookcase, side cabinet and ceiling lamp",
    ),
    PromptTemplate(
        key="office_classic_compact",
        room_type="office",
        text="classic compact office with desk, chair, shelf, cabinet and ceiling lamp",
    ),
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def deterministic_seed(room_file: Path, prompt_key: str, repeat_index: int) -> int:
    payload = f"{room_file.resolve()}|{prompt_key}|{repeat_index}".encode("utf-8")
    h = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def load_room_type(room_json_path: Path) -> str:
    data = json.loads(room_json_path.read_text(encoding="utf-8"))
    room = data.get("room") or {}
    room_type = str(room.get("room_type") or "").strip().lower()
    if not room_type:
        raise RuntimeError(f"В {room_json_path} нет room.room_type")
    return room_type


def collect_room_files(rooms_dir: Path) -> list[Path]:
    files = sorted(p for p in rooms_dir.glob("*.json") if p.name != "summary.json")
    if not files:
        raise RuntimeError(f"В каталоге {rooms_dir} не найдено room json")
    return files


def prompts_for_room_type(room_type: str) -> list[PromptTemplate]:
    res = [x for x in PROMPT_TEMPLATES if x.room_type == room_type]
    if not res:
        raise RuntimeError(f"Нет prompt-шаблонов для типа комнаты: {room_type}")
    return res


def make_job_rel_path(room_stem: str, prompt_key: str, repeat_index: int) -> Path:
    return Path(room_stem) / prompt_key / f"run_{repeat_index:02d}"


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    fieldnames = [
        "started_at",
        "finished_at",
        "status",
        "mode",
        "room_file",
        "room_type",
        "prompt_key",
        "prompt_text",
        "repeat_index",
        "seed",
        "returncode",
        "duration_sec",
        "batch_run_dir",
        "stdout_log",
        "stderr_log",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_status_row(
    *,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    mode: str,
    room_file: Path,
    room_type: str,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    seed: int,
    returncode: int,
    duration_sec: float,
    batch_run_dir: Path,
    stdout_log: Path,
    stderr_log: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "status": status,
        "mode": mode,
        "room_file": str(room_file.resolve()),
        "room_type": room_type,
        "prompt_key": prompt_tpl.key,
        "prompt_text": prompt_tpl.text,
        "repeat_index": repeat_index,
        "seed": seed,
        "returncode": returncode,
        "duration_sec": duration_sec,
        "batch_run_dir": str(batch_run_dir.resolve()),
        "stdout_log": str(stdout_log.resolve()),
        "stderr_log": str(stderr_log.resolve()),
        "error": error,
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_subprocess(
    cmd: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: int | None = None,
    stdin_text: str | None = None,
) -> tuple[int, float]:
    t0 = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            stdout=out,
            stderr=err,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    dt = round(time.perf_counter() - t0, 3)
    return proc.returncode, dt


def append_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_chooser_once(
    *,
    chooser_script: Path,
    normalize_script: Path,
    prepared_info: Path,
    future_root: Path,
    chooser_cache_run_dir: Path,
    room_file: Path,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    seed: int,
    chooser_timeout_sec: int | None,
    chooser_extra_args: list[str],
) -> tuple[bool, str]:
    chooser_cache_run_dir.mkdir(parents=True, exist_ok=True)

    chooser_stdout = chooser_cache_run_dir / "chooser_stdout.log"
    chooser_stderr = chooser_cache_run_dir / "chooser_stderr.log"
    objects_json = chooser_cache_run_dir / "objects.json"
    objects_v1_json = chooser_cache_run_dir / "objects.v1.json"

    cmd = [
        sys.executable,
        str(chooser_script.resolve()),
        "--room-json", str(room_file.resolve()),
        "--prompt", prompt_tpl.text,
        "--prepared-info", str(prepared_info.resolve()),
        "--future-root", str(future_root.resolve()),
        "--out", str(objects_json.resolve()),
        "--run-dir", str(chooser_cache_run_dir.resolve()),
        "--seed", str(seed),
    ]
    cmd.extend(chooser_extra_args)

    rc, _ = run_subprocess(
        cmd,
        stdout_path=chooser_stdout,
        stderr_path=chooser_stderr,
        timeout_sec=chooser_timeout_sec,
    )
    if rc != 0:
        return False, f"chooser returncode={rc}"

    if not objects_json.is_file():
        return False, "chooser не создал objects.json"

    norm_stdout = chooser_cache_run_dir / "normalize_objects_stdout.log"
    norm_stderr = chooser_cache_run_dir / "normalize_objects_stderr.log"

    rc_norm, _ = run_subprocess(
        [
            sys.executable,
            str(normalize_script.resolve()),
            "--input", str(objects_json.resolve()),
            "--output", str(objects_v1_json.resolve()),
            "--target", "objects",
        ],
        stdout_path=norm_stdout,
        stderr_path=norm_stderr,
        timeout_sec=chooser_timeout_sec,
    )
    if rc_norm != 0:
        return False, f"normalize objects returncode={rc_norm}"

    if not objects_v1_json.is_file():
        return False, "normalize не создал objects.v1.json"

    meta = {
        "room_file": str(room_file.resolve()),
        "prompt_key": prompt_tpl.key,
        "prompt_text": prompt_tpl.text,
        "repeat_index": repeat_index,
        "seed": seed,
        "chooser_command": cmd,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (chooser_cache_run_dir / "chooser_job_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return True, ""


def prepare_mode_run_dir(
    *,
    mode_run_dir: Path,
    chooser_cache_run_dir: Path,
    room_file: Path,
    room_type: str,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    seed: int,
    mode: str,
) -> None:
    mode_run_dir.mkdir(parents=True, exist_ok=True)

    (mode_run_dir / "prompt.txt").write_text(prompt_tpl.text, encoding="utf-8")

    for name in [
        "objects.json",
        "objects.v1.json",
        "chooser_request.json",
        "chooser_selected_raw.json",
        "chooser_stdout.log",
        "chooser_stderr.log",
        "normalize_objects_stdout.log",
        "normalize_objects_stderr.log",
        "chooser_job_meta.json",
    ]:
        copy_if_exists(chooser_cache_run_dir / name, mode_run_dir / name)

    job_meta = {
        "room_file": str(room_file.resolve()),
        "room_type": room_type,
        "prompt_key": prompt_tpl.key,
        "prompt_text": prompt_tpl.text,
        "repeat_index": repeat_index,
        "seed": seed,
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (mode_run_dir / "job_meta.json").write_text(
        json.dumps(job_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_cube_mode(
    *,
    cube_script: Path,
    normalize_script: Path,
    mode_run_dir: Path,
    room_file: Path,
    room_type: str,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    seed: int,
    mode: str,
    cube_timeout_sec: int | None,
) -> dict[str, Any]:
    started = datetime.now()

    stdout_log = mode_run_dir / "stdout.log"
    stderr_log = mode_run_dir / "stderr.log"
    objects_json = mode_run_dir / "objects.json"
    placement_raw_json = mode_run_dir / f"placement_{mode}.json"
    placement_v1_json = mode_run_dir / "placement.v1.json"
    scene_v1_json = mode_run_dir / "scene.v1.json"

    append_run_manifest(
        mode_run_dir,
        {
            "created_at": started.isoformat(timespec="seconds"),
            "mode": mode,
            "room_file": str(room_file.resolve()),
            "room_type": room_type,
            "prompt_key": prompt_tpl.key,
            "prompt_text": prompt_tpl.text,
            "repeat_index": repeat_index,
            "seed": seed,
            "cube_script": str(cube_script.resolve()),
            "normalize_script": str(normalize_script.resolve()),
            "objects_json": str(objects_json.resolve()),
        },
    )

    if not objects_json.is_file():
        finished = datetime.now()
        return build_status_row(
            started_at=started,
            finished_at=finished,
            status="error",
            mode=mode,
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            returncode=-1,
            duration_sec=-1,
            batch_run_dir=mode_run_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            error="В mode run dir отсутствует objects.json",
        )

    output_json_global = Path("data/output/placement_result.json")
    if output_json_global.exists():
        output_json_global.unlink()

    cube_input = f"{room_file.resolve()}\n{objects_json.resolve()}\n{mode}\n"

    try:
        rc, dt = run_subprocess(
            [sys.executable, str(cube_script.resolve())],
            stdout_path=stdout_log,
            stderr_path=stderr_log,
            timeout_sec=cube_timeout_sec,
            stdin_text=cube_input,
        )
    except subprocess.TimeoutExpired:
        finished = datetime.now()
        return build_status_row(
            started_at=started,
            finished_at=finished,
            status="timeout",
            mode=mode,
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            returncode=-1,
            duration_sec=-1,
            batch_run_dir=mode_run_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            error=f"cube timeout after {cube_timeout_sec} sec",
        )

    if rc != 0:
        finished = datetime.now()
        return build_status_row(
            started_at=started,
            finished_at=finished,
            status="failed",
            mode=mode,
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            returncode=rc,
            duration_sec=dt,
            batch_run_dir=mode_run_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            error=f"CubePlacement returncode={rc}",
        )

    if not output_json_global.is_file():
        finished = datetime.now()
        return build_status_row(
            started_at=started,
            finished_at=finished,
            status="failed",
            mode=mode,
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            returncode=0,
            duration_sec=dt,
            batch_run_dir=mode_run_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            error="После CubePlacement не найден data/output/placement_result.json",
        )

    shutil.move(str(output_json_global), str(placement_raw_json))

    with stdout_log.open("a", encoding="utf-8") as out, stderr_log.open("a", encoding="utf-8") as err:
        proc_norm_placement = subprocess.run(
            [
                sys.executable,
                str(normalize_script.resolve()),
                "--input", str(placement_raw_json.resolve()),
                "--output", str(placement_v1_json.resolve()),
                "--target", "placement",
            ],
            stdout=out,
            stderr=err,
            text=True,
            check=False,
        )
        if proc_norm_placement.returncode != 0:
            finished = datetime.now()
            return build_status_row(
                started_at=started,
                finished_at=finished,
                status="failed",
                mode=mode,
                room_file=room_file,
                room_type=room_type,
                prompt_tpl=prompt_tpl,
                repeat_index=repeat_index,
                seed=seed,
                returncode=proc_norm_placement.returncode,
                duration_sec=dt,
                batch_run_dir=mode_run_dir,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                error="normalize placement failed",
            )

        proc_norm_scene = subprocess.run(
            [
                sys.executable,
                str(normalize_script.resolve()),
                "--room", str(room_file.resolve()),
                "--placement", str(placement_raw_json.resolve()),
                "--output", str(scene_v1_json.resolve()),
                "--target", "scene",
            ],
            stdout=out,
            stderr=err,
            text=True,
            check=False,
        )
        if proc_norm_scene.returncode != 0:
            finished = datetime.now()
            return build_status_row(
                started_at=started,
                finished_at=finished,
                status="failed",
                mode=mode,
                room_file=room_file,
                room_type=room_type,
                prompt_tpl=prompt_tpl,
                repeat_index=repeat_index,
                seed=seed,
                returncode=proc_norm_scene.returncode,
                duration_sec=dt,
                batch_run_dir=mode_run_dir,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                error="normalize scene failed",
            )

    finished = datetime.now()
    return build_status_row(
        started_at=started,
        finished_at=finished,
        status="ok",
        mode=mode,
        room_file=room_file,
        room_type=room_type,
        prompt_tpl=prompt_tpl,
        repeat_index=repeat_index,
        seed=seed,
        returncode=0,
        duration_sec=dt,
        batch_run_dir=mode_run_dir,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        error="",
    )


def make_failed_rows_for_all_modes(
    *,
    modes: list[str],
    mode_batch_dirs: dict[str, Path],
    room_file: Path,
    room_type: str,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    seed: int,
    error_text: str,
    chooser_cache_run_dir: Path,
) -> dict[str, dict[str, Any]]:
    res: dict[str, dict[str, Any]] = {}
    now = datetime.now()

    for mode in modes:
        mode_run_dir = mode_batch_dirs[mode] / make_job_rel_path(room_file.stem, prompt_tpl.key, repeat_index)
        mode_run_dir.mkdir(parents=True, exist_ok=True)

        # Скопируем хотя бы chooser-логи, если они есть
        for name in [
            "chooser_stdout.log",
            "chooser_stderr.log",
            "normalize_objects_stdout.log",
            "normalize_objects_stderr.log",
        ]:
            copy_if_exists(chooser_cache_run_dir / name, mode_run_dir / name)

        (mode_run_dir / "prompt.txt").write_text(prompt_tpl.text, encoding="utf-8")

        row = build_status_row(
            started_at=now,
            finished_at=now,
            status="failed",
            mode=mode,
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            returncode=-1,
            duration_sec=-1,
            batch_run_dir=mode_run_dir,
            stdout_log=mode_run_dir / "stdout.log",
            stderr_log=mode_run_dir / "stderr.log",
            error=error_text,
        )
        res[mode] = row

    return res


def update_summary(summary_json: Path, summary_obj: dict[str, Any]) -> None:
    summary_json.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-генерация типовых комнат через chooser + CubePlacement, без Blender и без LLM-флагов. "
            "Для каждого режима создаётся отдельный batch_<timestamp>_<mode>."
        )
    )
    parser.add_argument("--rooms-dir", default=DEFAULT_ROOMS_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--chooser-script", default=DEFAULT_CHOOSER_SCRIPT)
    parser.add_argument("--cube-script", default=DEFAULT_CUBE_SCRIPT)
    parser.add_argument("--normalize-script", default=DEFAULT_NORMALIZE_SCRIPT)

    parser.add_argument("--prepared-info", default=DEFAULT_PREPARED_INFO)
    parser.add_argument("--future-root", default=DEFAULT_FUTURE_ROOT)

    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=["random", "relaxed"],
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--room-limit", type=int, default=None)
    parser.add_argument("--prompt-limit-per-room", type=int, default=None)

    parser.add_argument("--chooser-timeout-sec", type=int, default=DEFAULT_CHOOSER_TIMEOUT)
    parser.add_argument("--cube-timeout-sec", type=int, default=DEFAULT_CUBE_TIMEOUT)

    # На случай если в твоей ветке chooser всё же требует явные флаги.
    parser.add_argument(
        "--chooser-extra-arg",
        action="append",
        default=[],
        help="Дополнительный аргумент, который будет добавлен в вызов choose_obj_from_prepared.py. Можно повторять.",
    )

    args = parser.parse_args()

    rooms_dir = Path(args.rooms_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    chooser_script = Path(args.chooser_script).expanduser().resolve()
    cube_script = Path(args.cube_script).expanduser().resolve()
    normalize_script = Path(args.normalize_script).expanduser().resolve()
    prepared_info = Path(args.prepared_info).expanduser().resolve()
    future_root = Path(args.future_root).expanduser().resolve()

    if not chooser_script.is_file():
        raise RuntimeError(f"Не найден chooser script: {chooser_script}")
    if not cube_script.is_file():
        raise RuntimeError(f"Не найден cube script: {cube_script}")
    if not normalize_script.is_file():
        raise RuntimeError(f"Не найден normalize script: {normalize_script}")
    if not prepared_info.is_file():
        raise RuntimeError(f"Не найден prepared_info: {prepared_info}")
    if not future_root.is_dir():
        raise RuntimeError(f"Не найден future_root: {future_root}")

    rooms = collect_room_files(rooms_dir)
    if args.room_limit is not None:
        rooms = rooms[: max(0, int(args.room_limit))]

    stamp = now_stamp()
    chooser_cache_root = output_root / f"{CHOOSER_CACHE_PREFIX}_{stamp}"

    mode_batch_dirs: dict[str, Path] = {}
    csv_logs: dict[str, Path] = {}
    jsonl_logs: dict[str, Path] = {}
    summary_jsons: dict[str, Path] = {}

    for mode in args.modes:
        batch_dir = output_root / f"batch_{stamp}_{mode}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        mode_batch_dirs[mode] = batch_dir

        csv_log = batch_dir / "batch_results.csv"
        jsonl_log = batch_dir / "batch_results.jsonl"
        summary_json = batch_dir / "summary.json"

        csv_logs[mode] = csv_log
        jsonl_logs[mode] = jsonl_log
        summary_jsons[mode] = summary_json

        ensure_csv_header(csv_log)

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "rooms_dir": str(rooms_dir),
            "output_root": str(output_root),
            "batch_dir": str(batch_dir.resolve()),
            "chooser_script": str(chooser_script.resolve()),
            "cube_script": str(cube_script.resolve()),
            "normalize_script": str(normalize_script.resolve()),
            "prepared_info": str(prepared_info.resolve()),
            "future_root": str(future_root.resolve()),
            "mode": mode,
            "repeats": args.repeats,
            "room_limit": args.room_limit,
            "prompt_limit_per_room": args.prompt_limit_per_room,
            "chooser_timeout_sec": args.chooser_timeout_sec,
            "cube_timeout_sec": args.cube_timeout_sec,
            "chooser_extra_args": list(args.chooser_extra_arg or []),
            "rooms": [str(x.resolve()) for x in rooms],
        }
        (batch_dir / "batch_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    stats_by_mode: dict[str, dict[str, int]] = {
        mode: {
            "jobs_total": 0,
            "ok_count": 0,
            "failed_count": 0,
            "timeout_count": 0,
            "error_count": 0,
        }
        for mode in args.modes
    }

    shared_job_index = 0
    chooser_extra_args = list(args.chooser_extra_arg or [])

    for room_file in rooms:
        room_type = load_room_type(room_file)
        prompts = prompts_for_room_type(room_type)
        if args.prompt_limit_per_room is not None:
            prompts = prompts[: max(0, int(args.prompt_limit_per_room))]

        for prompt_tpl in prompts:
            for repeat_index in range(1, int(args.repeats) + 1):
                shared_job_index += 1
                seed = deterministic_seed(room_file, prompt_tpl.key, repeat_index)

                print(
                    f"[{shared_job_index}] room={room_file.name} "
                    f"type={room_type} prompt={prompt_tpl.key} repeat={repeat_index} seed={seed}"
                )

                rel_path = make_job_rel_path(room_file.stem, prompt_tpl.key, repeat_index)
                chooser_cache_run_dir = chooser_cache_root / rel_path

                ok_chooser, chooser_error = run_chooser_once(
                    chooser_script=chooser_script,
                    normalize_script=normalize_script,
                    prepared_info=prepared_info,
                    future_root=future_root,
                    chooser_cache_run_dir=chooser_cache_run_dir,
                    room_file=room_file,
                    prompt_tpl=prompt_tpl,
                    repeat_index=repeat_index,
                    seed=seed,
                    chooser_timeout_sec=args.chooser_timeout_sec,
                    chooser_extra_args=chooser_extra_args,
                )

                if not ok_chooser:
                    failed_rows = make_failed_rows_for_all_modes(
                        modes=list(args.modes),
                        mode_batch_dirs=mode_batch_dirs,
                        room_file=room_file,
                        room_type=room_type,
                        prompt_tpl=prompt_tpl,
                        repeat_index=repeat_index,
                        seed=seed,
                        error_text=f"chooser failed: {chooser_error}",
                        chooser_cache_run_dir=chooser_cache_run_dir,
                    )

                    for mode, row in failed_rows.items():
                        stats_by_mode[mode]["jobs_total"] += 1
                        stats_by_mode[mode]["failed_count"] += 1
                        append_csv_row(csv_logs[mode], row)
                        append_jsonl(jsonl_logs[mode], row)

                        update_summary(
                            summary_jsons[mode],
                            {
                                **stats_by_mode[mode],
                                "last_job": row,
                            },
                        )

                        print(f"    -> {mode}: FAILED chooser")
                    continue

                for mode in args.modes:
                    mode_run_dir = mode_batch_dirs[mode] / rel_path

                    prepare_mode_run_dir(
                        mode_run_dir=mode_run_dir,
                        chooser_cache_run_dir=chooser_cache_run_dir,
                        room_file=room_file,
                        room_type=room_type,
                        prompt_tpl=prompt_tpl,
                        repeat_index=repeat_index,
                        seed=seed,
                        mode=mode,
                    )

                    row = run_cube_mode(
                        cube_script=cube_script,
                        normalize_script=normalize_script,
                        mode_run_dir=mode_run_dir,
                        room_file=room_file,
                        room_type=room_type,
                        prompt_tpl=prompt_tpl,
                        repeat_index=repeat_index,
                        seed=seed,
                        mode=mode,
                        cube_timeout_sec=args.cube_timeout_sec,
                    )

                    stats_by_mode[mode]["jobs_total"] += 1
                    if row["status"] == "ok":
                        stats_by_mode[mode]["ok_count"] += 1
                    elif row["status"] == "failed":
                        stats_by_mode[mode]["failed_count"] += 1
                    elif row["status"] == "timeout":
                        stats_by_mode[mode]["timeout_count"] += 1
                    else:
                        stats_by_mode[mode]["error_count"] += 1

                    append_csv_row(csv_logs[mode], row)
                    append_jsonl(jsonl_logs[mode], row)

                    update_summary(
                        summary_jsons[mode],
                        {
                            **stats_by_mode[mode],
                            "last_job": row,
                        },
                    )

                    print(
                        f"    -> {mode}: {row['status']} "
                        f"dt={row['duration_sec']}s "
                        f"run={mode_run_dir}"
                    )

    for mode in args.modes:
        final_summary = {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            **stats_by_mode[mode],
            "batch_dir": str(mode_batch_dirs[mode].resolve()),
            "csv_log": str(csv_logs[mode].resolve()),
            "jsonl_log": str(jsonl_logs[mode].resolve()),
        }
        update_summary(summary_jsons[mode], final_summary)

        print("\n=== MODE FINISHED ===")
        print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()