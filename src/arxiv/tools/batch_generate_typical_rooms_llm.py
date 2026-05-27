#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/batch_generate_typical_rooms_llm.py

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOMS_DIR = "data/input/generated_typical_rooms"
DEFAULT_OUTPUT_ROOT = "out/batch_typical_rooms"
DEFAULT_RUN_PIPELINE = "src/run_pipeline.py"

DEFAULT_REPEATS = 5

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_TIMEOUT = 1200
DEFAULT_OLLAMA_TEMPERATURE = 0.0
DEFAULT_OLLAMA_MAX_ATTEMPTS = 8
DEFAULT_MAX_SCENE_ATTEMPTS = 5

DEFAULT_PLAN_MODEL = None
DEFAULT_CRITIC_MODEL = None

DEFAULT_PLAN_THINK = "low"
DEFAULT_CRITIC_THINK = "low"
DEFAULT_JSON_THINK = "none"


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


def make_job_run_dir(output_root: Path, room_stem: str, prompt_key: str, repeat_index: int) -> Path:
    return output_root / room_stem / prompt_key / f"run_{repeat_index:02d}"


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    fieldnames = [
        "started_at",
        "finished_at",
        "status",
        "room_file",
        "room_type",
        "prompt_key",
        "prompt_text",
        "repeat_index",
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


def summarize_durations(csv_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        return {
            "successful_jobs": 0,
            "avg_duration_sec": None,
            "median_duration_sec": None,
            "min_duration_sec": None,
            "max_duration_sec": None,
        }

    durations: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                durations.append(float(row["duration_sec"]))
            except Exception:
                pass

    if not durations:
        return {
            "successful_jobs": 0,
            "avg_duration_sec": None,
            "median_duration_sec": None,
            "min_duration_sec": None,
            "max_duration_sec": None,
        }

    return {
        "successful_jobs": len(durations),
        "avg_duration_sec": round(sum(durations) / len(durations), 3),
        "median_duration_sec": round(st.median(durations), 3),
        "min_duration_sec": round(min(durations), 3),
        "max_duration_sec": round(max(durations), 3),
    }


def run_one_job(
    pipeline_script: Path,
    room_file: Path,
    room_type: str,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    output_root: Path,
    ollama_url: str,
    ollama_model: str,
    ollama_timeout: int,
    ollama_temperature: float,
    ollama_max_attempts: int,
    max_scene_attempts: int,
    plan_model: str | None,
    critic_model: str | None,
    plan_think: str,
    critic_think: str,
    json_think: str,
    headless: bool,
    skip_blender: bool,
) -> dict[str, Any]:
    started = datetime.now()
    run_dir = make_job_run_dir(output_root, room_file.stem, prompt_tpl.key, repeat_index)
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    meta_json = run_dir / "job_meta.json"

    cmd = [
        sys.executable,
        str(pipeline_script),
        "--placer", "ollama_llm",
        "--room", str(room_file.resolve()),
        "--prompt", prompt_tpl.text,
        "--modes", "llm",
        "--run-dir", str(run_dir.resolve()),
        "--ollama-url", ollama_url,
        "--ollama-model", ollama_model,
        "--ollama-timeout", str(int(ollama_timeout)),
        "--ollama-temperature", str(float(ollama_temperature)),
        "--ollama-max-attempts", str(int(ollama_max_attempts)),
        "--max-scene-attempts", str(int(max_scene_attempts)),
        "--plan-think", str(plan_think),
        "--critic-think", str(critic_think),
        "--llm-think", str(json_think),
    ]

    if plan_model:
        cmd += ["--plan-model", str(plan_model)]
    if critic_model:
        cmd += ["--critic-model", str(critic_model)]

    if headless:
        cmd += ["--headless"]
    if skip_blender:
        cmd += ["--skip-blender"]

    meta = {
        "room_file": str(room_file.resolve()),
        "room_type": room_type,
        "prompt_key": prompt_tpl.key,
        "prompt_text": prompt_tpl.text,
        "repeat_index": repeat_index,
        "command": cmd,
        "started_at": started.isoformat(timespec="seconds"),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, check=False)
            duration_sec = round(time.perf_counter() - t0, 3)

        finished = datetime.now()
        status = "ok" if proc.returncode == 0 else "failed"
        error = "" if proc.returncode == 0 else f"pipeline returncode={proc.returncode}"

        return {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "status": status,
            "room_file": str(room_file.resolve()),
            "room_type": room_type,
            "prompt_key": prompt_tpl.key,
            "prompt_text": prompt_tpl.text,
            "repeat_index": repeat_index,
            "returncode": proc.returncode,
            "duration_sec": duration_sec,
            "batch_run_dir": str(run_dir.resolve()),
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "error": error,
        }
    except Exception as e:
        finished = datetime.now()
        return {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "status": "error",
            "room_file": str(room_file.resolve()),
            "room_type": room_type,
            "prompt_key": prompt_tpl.key,
            "prompt_text": prompt_tpl.text,
            "repeat_index": repeat_index,
            "returncode": -1,
            "duration_sec": -1,
            "batch_run_dir": str(run_dir.resolve()),
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "error": repr(e),
        }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch LLM placement for typical rooms with 5 repeats and timing stats"
    )
    p.add_argument("--rooms-dir", default=DEFAULT_ROOMS_DIR)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--pipeline-script", default=DEFAULT_RUN_PIPELINE)
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--room-limit", type=int, default=None)
    p.add_argument("--prompt-limit-per-room", type=int, default=None)

    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--ollama-timeout", type=int, default=DEFAULT_OLLAMA_TIMEOUT)
    p.add_argument("--ollama-temperature", type=float, default=DEFAULT_OLLAMA_TEMPERATURE)
    p.add_argument("--ollama-max-attempts", type=int, default=DEFAULT_OLLAMA_MAX_ATTEMPTS)
    p.add_argument("--max-scene-attempts", type=int, default=DEFAULT_MAX_SCENE_ATTEMPTS)

    p.add_argument("--plan-model", default=DEFAULT_PLAN_MODEL)
    p.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL)
    p.add_argument("--plan-think", choices=["none", "low"], default=DEFAULT_PLAN_THINK)
    p.add_argument("--critic-think", choices=["none", "low"], default=DEFAULT_CRITIC_THINK)
    p.add_argument("--json-think", choices=["none", "low"], default=DEFAULT_JSON_THINK)

    p.add_argument("--headless", action="store_true")
    p.add_argument("--skip-blender", action="store_true", default=True)

    args = p.parse_args()

    rooms_dir = Path(args.rooms_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    pipeline_script = Path(args.pipeline_script).expanduser().resolve()

    if not pipeline_script.is_file():
        raise RuntimeError(f"Не найден pipeline script: {pipeline_script}")

    rooms = collect_room_files(rooms_dir)
    if args.room_limit is not None:
        rooms = rooms[: max(0, int(args.room_limit))]

    batch_dir = output_root / f"batch_{now_stamp()}_llm"
    batch_dir.mkdir(parents=True, exist_ok=True)

    csv_log = batch_dir / "batch_results.csv"
    jsonl_log = batch_dir / "batch_results.jsonl"
    summary_json = batch_dir / "summary.json"
    ensure_csv_header(csv_log)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "placer": "ollama_llm",
        "modes": "llm",
        "repeats": args.repeats,
        "rooms_dir": str(rooms_dir),
        "batch_dir": str(batch_dir.resolve()),
        "pipeline_script": str(pipeline_script.resolve()),
        "ollama_url": args.ollama_url,
        "ollama_model": args.ollama_model,
        "ollama_timeout": args.ollama_timeout,
        "ollama_temperature": args.ollama_temperature,
        "ollama_max_attempts": args.ollama_max_attempts,
        "max_scene_attempts": args.max_scene_attempts,
        "plan_model": args.plan_model,
        "critic_model": args.critic_model,
        "plan_think": args.plan_think,
        "critic_think": args.critic_think,
        "json_think": args.json_think,
        "skip_blender": bool(args.skip_blender),
        "rooms": [str(x) for x in rooms],
    }
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    jobs_total = 0
    ok_count = 0
    failed_count = 0
    error_count = 0

    for room_file in rooms:
        room_type = load_room_type(room_file)
        prompts = prompts_for_room_type(room_type)
        if args.prompt_limit_per_room is not None:
            prompts = prompts[: max(0, int(args.prompt_limit_per_room))]

        for prompt_tpl in prompts:
            for repeat_index in range(1, int(args.repeats) + 1):
                jobs_total += 1
                print(
                    f"[{jobs_total}] room={room_file.name} "
                    f"type={room_type} prompt={prompt_tpl.key} repeat={repeat_index}"
                )

                row = run_one_job(
                    pipeline_script=pipeline_script,
                    room_file=room_file,
                    room_type=room_type,
                    prompt_tpl=prompt_tpl,
                    repeat_index=repeat_index,
                    output_root=batch_dir,
                    ollama_url=args.ollama_url,
                    ollama_model=args.ollama_model,
                    ollama_timeout=int(args.ollama_timeout),
                    ollama_temperature=float(args.ollama_temperature),
                    ollama_max_attempts=int(args.ollama_max_attempts),
                    max_scene_attempts=int(args.max_scene_attempts),
                    plan_model=args.plan_model,
                    critic_model=args.critic_model,
                    plan_think=args.plan_think,
                    critic_think=args.critic_think,
                    json_think=args.json_think,
                    headless=bool(args.headless),
                    skip_blender=bool(args.skip_blender),
                )

                append_csv_row(csv_log, row)
                append_jsonl(jsonl_log, row)

                if row["status"] == "ok":
                    ok_count += 1
                elif row["status"] == "failed":
                    failed_count += 1
                else:
                    error_count += 1

                timing = summarize_durations(csv_log)
                print(
                    f"    -> {row['status']} dt={row['duration_sec']}s "
                    f"avg_ok={timing['avg_duration_sec']}s"
                )

                summary = {
                    "jobs_total": jobs_total,
                    "ok_count": ok_count,
                    "failed_count": failed_count,
                    "error_count": error_count,
                    "timing": timing,
                    "last_job": row,
                }
                summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    final_timing = summarize_durations(csv_log)
    final_summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "jobs_total": jobs_total,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "error_count": error_count,
        "timing": final_timing,
        "batch_dir": str(batch_dir.resolve()),
        "csv_log": str(csv_log.resolve()),
        "jsonl_log": str(jsonl_log.resolve()),
    }
    summary_json.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BATCH FINISHED ===")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()