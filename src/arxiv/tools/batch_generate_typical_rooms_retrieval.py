#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/batch_generate_typical_rooms_retrieval.py

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOMS_DIR = "data/input/generated_typical_rooms"
DEFAULT_OUTPUT_ROOT = "out/batch_typical_rooms"

DEFAULT_RETRIEVAL_SCRIPT = "src/Plasement/retrieval_knn_scene.py"

DEFAULT_DATASET_ROOT = "data/sourse/3D-FRONT/3D-FRONT-processed-mini"
DEFAULT_PREPARED_INFO = "data/sourse/3D-FRONT/prepared_model_info.json"
DEFAULT_FUTURE_ROOT = "data/sourse/3D-FRONT/3D-FUTURE-model"

DEFAULT_REPEATS = 1
DEFAULT_TOP_K = 3
DEFAULT_TIMEOUT = 600


@dataclass(frozen=True)
class PromptTemplate:
    key: str
    room_type: str
    text: str
    items: list[str]


PROMPT_TEMPLATES: list[PromptTemplate] = [
    # bedroom
    PromptTemplate(
        key="bedroom_classic_cozy",
        room_type="bedroom",
        text="classic cozy bedroom with king-size bed, two nightstands, wardrobe and ceiling lamp",
        items=["King-size Bed", "Nightstand", "Nightstand", "Wardrobe", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="bedroom_classic_family",
        room_type="bedroom",
        text="classic family bedroom with double bed, two bedside tables, wardrobe, dresser and ceiling lamp",
        items=["Double Bed", "Nightstand", "Nightstand", "Wardrobe", "Drawer Chest", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="bedroom_classic_soft",
        room_type="bedroom",
        text="classic soft bedroom with bed, two nightstands, wardrobe, side cabinet and ceiling lamp",
        items=["Double Bed", "Nightstand", "Nightstand", "Wardrobe", "Side Cabinet", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="bedroom_classic_storage",
        room_type="bedroom",
        text="classic bedroom with bed, two nightstands, large wardrobe, drawer chest and ceiling lamp",
        items=["King-size Bed", "Nightstand", "Nightstand", "Wardrobe", "Drawer Chest", "Pendant Lamp"],
    ),

    # living
    PromptTemplate(
        key="living_classic_guest",
        room_type="living",
        text="classic living room with sofa, armchair, coffee table, TV stand, sideboard and ceiling lamp",
        items=["Multi-seat Sofa", "armchair", "Coffee Table", "TV Stand", "Sideboard", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="living_classic_formal",
        room_type="living",
        text="classic formal living room with sofa, two armchairs, coffee table, TV stand and ceiling lamp",
        items=["Multi-seat Sofa", "armchair", "armchair", "Coffee Table", "TV Stand", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="living_classic_family",
        room_type="living",
        text="classic family living room with multi-seat sofa, armchair, coffee table, TV stand and ceiling lamp",
        items=["Multi-seat Sofa", "armchair", "Coffee Table", "TV Stand", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="living_classic_storage",
        room_type="living",
        text="classic living room with sofa, coffee table, TV stand, sideboard, cabinet and ceiling lamp",
        items=["Multi-seat Sofa", "Coffee Table", "TV Stand", "Sideboard", "Cabinet", "Pendant Lamp"],
    ),

    # office
    PromptTemplate(
        key="office_classic_work",
        room_type="office",
        text="classic office with desk, office chair, bookcase, cabinet and ceiling lamp",
        items=["Desk", "Office Chair", "Bookcase", "Cabinet", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="office_classic_study",
        room_type="office",
        text="classic study room with desk, chair, bookshelf, storage cabinet and ceiling lamp",
        items=["Desk", "Office Chair", "Bookcase", "Cabinet", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="office_classic_library",
        room_type="office",
        text="classic home office with desk, office chair, bookcase, side cabinet and ceiling lamp",
        items=["Desk", "Office Chair", "Bookcase", "Cabinet", "Pendant Lamp"],
    ),
    PromptTemplate(
        key="office_classic_compact",
        room_type="office",
        text="classic compact office with desk, chair, shelf, cabinet and ceiling lamp",
        items=["Desk", "Office Chair", "Shelf", "Cabinet", "Pendant Lamp"],
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
        "room_file",
        "room_type",
        "prompt_key",
        "prompt_text",
        "items_csv",
        "repeat_index",
        "seed",
        "top_k",
        "returncode",
        "duration_sec",
        "batch_run_dir",
        "stdout_log",
        "stderr_log",
        "scene_v1_json",
        "retrieval_layout_json",
        "neighbors_json",
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


def update_summary(summary_json: Path, summary_obj: dict[str, Any]) -> None:
    summary_json.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_subprocess(
    cmd: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: int | None = None,
) -> tuple[int, float]:
    t0 = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            cmd,
            stdout=out,
            stderr=err,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    dt = round(time.perf_counter() - t0, 3)
    return proc.returncode, dt


def build_status_row(
    *,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    room_file: Path,
    room_type: str,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    seed: int,
    top_k: int,
    returncode: int,
    duration_sec: float,
    batch_run_dir: Path,
    stdout_log: Path,
    stderr_log: Path,
    scene_v1_json: Path,
    retrieval_layout_json: Path,
    neighbors_json: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "status": status,
        "room_file": str(room_file.resolve()),
        "room_type": room_type,
        "prompt_key": prompt_tpl.key,
        "prompt_text": prompt_tpl.text,
        "items_csv": ",".join(prompt_tpl.items),
        "repeat_index": repeat_index,
        "seed": seed,
        "top_k": top_k,
        "returncode": returncode,
        "duration_sec": duration_sec,
        "batch_run_dir": str(batch_run_dir.resolve()),
        "stdout_log": str(stdout_log.resolve()),
        "stderr_log": str(stderr_log.resolve()),
        "scene_v1_json": str(scene_v1_json.resolve()),
        "retrieval_layout_json": str(retrieval_layout_json.resolve()),
        "neighbors_json": str(neighbors_json.resolve()),
        "error": error,
    }


def append_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_retrieval_once(
    *,
    retrieval_script: Path,
    dataset_root: Path,
    prepared_info: Path,
    future_root: Path,
    room_file: Path,
    room_type: str,
    prompt_tpl: PromptTemplate,
    repeat_index: int,
    seed: int,
    top_k: int,
    timeout_sec: int | None,
    run_dir: Path,
    extra_args: list[str],
) -> dict[str, Any]:
    started = datetime.now()

    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    prompt_txt = run_dir / "prompt.txt"
    items_txt = run_dir / "items.txt"
    scene_v1_json = run_dir / "scene.v1.json"
    retrieval_layout_json = run_dir / "retrieval_layout.json"
    neighbors_json = run_dir / "neighbors.json"
    job_meta_json = run_dir / "job_meta.json"

    prompt_txt.write_text(prompt_tpl.text, encoding="utf-8")
    items_txt.write_text("\n".join(prompt_tpl.items) + "\n", encoding="utf-8")

    job_meta = {
        "created_at": started.isoformat(timespec="seconds"),
        "room_file": str(room_file.resolve()),
        "room_type": room_type,
        "prompt_key": prompt_tpl.key,
        "prompt_text": prompt_tpl.text,
        "items": list(prompt_tpl.items),
        "repeat_index": repeat_index,
        "seed": seed,
        "top_k": top_k,
        "retrieval_script": str(retrieval_script.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "prepared_info": str(prepared_info.resolve()),
        "future_root": str(future_root.resolve()),
    }
    job_meta_json.write_text(json.dumps(job_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(retrieval_script.resolve()),
        "--dataset-root", str(dataset_root.resolve()),
        "--target-room", str(room_file.resolve()),
        "--items", ",".join(prompt_tpl.items),
        "--room-type", room_type,
        "--top-k", str(top_k),
        "--prepared-info", str(prepared_info.resolve()),
        "--future-root", str(future_root.resolve()),
        "--out", str(scene_v1_json.resolve()),
        "--dump-retrieval", str(retrieval_layout_json.resolve()),
        "--dump-neighbors", str(neighbors_json.resolve()),
    ]
    cmd.extend(extra_args)

    append_run_manifest(
        run_dir,
        {
            **job_meta,
            "command": cmd,
        },
    )

    try:
        rc, dt = run_subprocess(
            cmd,
            stdout_path=stdout_log,
            stderr_path=stderr_log,
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        finished = datetime.now()
        return build_status_row(
            started_at=started,
            finished_at=finished,
            status="timeout",
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            top_k=top_k,
            returncode=-1,
            duration_sec=-1,
            batch_run_dir=run_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            scene_v1_json=scene_v1_json,
            retrieval_layout_json=retrieval_layout_json,
            neighbors_json=neighbors_json,
            error=f"retrieval timeout after {timeout_sec} sec",
        )

    if rc != 0:
        finished = datetime.now()
        return build_status_row(
            started_at=started,
            finished_at=finished,
            status="failed",
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            top_k=top_k,
            returncode=rc,
            duration_sec=dt,
            batch_run_dir=run_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            scene_v1_json=scene_v1_json,
            retrieval_layout_json=retrieval_layout_json,
            neighbors_json=neighbors_json,
            error=f"retrieval_knn_scene returncode={rc}",
        )

    if not scene_v1_json.is_file():
        finished = datetime.now()
        return build_status_row(
            started_at=started,
            finished_at=finished,
            status="failed",
            room_file=room_file,
            room_type=room_type,
            prompt_tpl=prompt_tpl,
            repeat_index=repeat_index,
            seed=seed,
            top_k=top_k,
            returncode=0,
            duration_sec=dt,
            batch_run_dir=run_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            scene_v1_json=scene_v1_json,
            retrieval_layout_json=retrieval_layout_json,
            neighbors_json=neighbors_json,
            error="retrieval не создал scene.v1.json",
        )

    finished = datetime.now()
    return build_status_row(
        started_at=started,
        finished_at=finished,
        status="ok",
        room_file=room_file,
        room_type=room_type,
        prompt_tpl=prompt_tpl,
        repeat_index=repeat_index,
        seed=seed,
        top_k=top_k,
        returncode=0,
        duration_sec=dt,
        batch_run_dir=run_dir,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        scene_v1_json=scene_v1_json,
        retrieval_layout_json=retrieval_layout_json,
        neighbors_json=neighbors_json,
        error="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-генерация типовых комнат через retrieval_knn_scene.py. "
            "Создаёт структуру batch_<timestamp>_retrieval/<room>/<prompt>/run_xx/..."
        )
    )
    parser.add_argument("--rooms-dir", default=DEFAULT_ROOMS_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--retrieval-script", default=DEFAULT_RETRIEVAL_SCRIPT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--prepared-info", default=DEFAULT_PREPARED_INFO)
    parser.add_argument("--future-root", default=DEFAULT_FUTURE_ROOT)

    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--room-limit", type=int, default=None)
    parser.add_argument("--prompt-limit-per-room", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT)

    parser.add_argument(
        "--retrieval-extra-arg",
        action="append",
        default=[],
        help="Дополнительный аргумент, который будет добавлен в вызов retrieval_knn_scene.py. Можно повторять.",
    )

    args = parser.parse_args()

    rooms_dir = Path(args.rooms_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    retrieval_script = Path(args.retrieval_script).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    prepared_info = Path(args.prepared_info).expanduser().resolve()
    future_root = Path(args.future_root).expanduser().resolve()

    if not retrieval_script.is_file():
        raise RuntimeError(f"Не найден retrieval script: {retrieval_script}")
    if not dataset_root.is_dir():
        raise RuntimeError(f"Не найден dataset_root: {dataset_root}")
    if not prepared_info.is_file():
        raise RuntimeError(f"Не найден prepared_info: {prepared_info}")
    if not future_root.is_dir():
        raise RuntimeError(f"Не найден future_root: {future_root}")

    rooms = collect_room_files(rooms_dir)
    if args.room_limit is not None:
        rooms = rooms[: max(0, int(args.room_limit))]

    stamp = now_stamp()
    batch_dir = output_root / f"batch_{stamp}_retrieval"
    batch_dir.mkdir(parents=True, exist_ok=True)

    csv_log = batch_dir / "batch_results.csv"
    jsonl_log = batch_dir / "batch_results.jsonl"
    summary_json = batch_dir / "summary.json"

    ensure_csv_header(csv_log)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rooms_dir": str(rooms_dir),
        "output_root": str(output_root),
        "batch_dir": str(batch_dir.resolve()),
        "retrieval_script": str(retrieval_script.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "prepared_info": str(prepared_info.resolve()),
        "future_root": str(future_root.resolve()),
        "repeats": args.repeats,
        "top_k": args.top_k,
        "room_limit": args.room_limit,
        "prompt_limit_per_room": args.prompt_limit_per_room,
        "timeout_sec": args.timeout_sec,
        "retrieval_extra_args": list(args.retrieval_extra_arg or []),
        "rooms": [str(x.resolve()) for x in rooms],
        "note": "retrieval_knn_scene.py сейчас детерминированный; при repeats > 1 результаты могут совпадать.",
    }
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "jobs_total": 0,
        "ok_count": 0,
        "failed_count": 0,
        "timeout_count": 0,
        "error_count": 0,
    }

    shared_job_index = 0
    retrieval_extra_args = list(args.retrieval_extra_arg or [])

    for room_file in rooms:
        room_type = load_room_type(room_file)
        prompts = prompts_for_room_type(room_type)
        if args.prompt_limit_per_room is not None:
            prompts = prompts[: max(0, int(args.prompt_limit_per_room))]

        for prompt_tpl in prompts:
            for repeat_index in range(1, int(args.repeats) + 1):
                shared_job_index += 1
                seed = deterministic_seed(room_file, prompt_tpl.key, repeat_index)
                run_dir = batch_dir / make_job_rel_path(room_file.stem, prompt_tpl.key, repeat_index)

                print(
                    f"[{shared_job_index}] room={room_file.name} "
                    f"type={room_type} prompt={prompt_tpl.key} "
                    f"repeat={repeat_index} seed={seed}"
                )

                row = run_retrieval_once(
                    retrieval_script=retrieval_script,
                    dataset_root=dataset_root,
                    prepared_info=prepared_info,
                    future_root=future_root,
                    room_file=room_file,
                    room_type=room_type,
                    prompt_tpl=prompt_tpl,
                    repeat_index=repeat_index,
                    seed=seed,
                    top_k=int(args.top_k),
                    timeout_sec=args.timeout_sec,
                    run_dir=run_dir,
                    extra_args=retrieval_extra_args,
                )

                stats["jobs_total"] += 1
                if row["status"] == "ok":
                    stats["ok_count"] += 1
                elif row["status"] == "failed":
                    stats["failed_count"] += 1
                elif row["status"] == "timeout":
                    stats["timeout_count"] += 1
                else:
                    stats["error_count"] += 1

                append_csv_row(csv_log, row)
                append_jsonl(jsonl_log, row)

                update_summary(
                    summary_json,
                    {
                        **stats,
                        "last_job": row,
                    },
                )

                print(
                    f"    -> {row['status']} "
                    f"dt={row['duration_sec']}s "
                    f"run={run_dir}"
                )

    final_summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "retrieval",
        **stats,
        "batch_dir": str(batch_dir.resolve()),
        "csv_log": str(csv_log.resolve()),
        "jsonl_log": str(jsonl_log.resolve()),
    }
    update_summary(summary_json, final_summary)

    print("\n=== BATCH FINISHED ===")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()