#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run full pipeline + VLM review photos + per-frame VLM evaluation for example rooms."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CASES: list[dict[str, str]] = [
    {
        "key": "bedroom_medium_08",
        "room_type": "bedroom",
        "room": "data/input/example/rooms/bedroom_medium_08.json",
        "prompt": "modern cozy bedroom with bed, nightstands, wardrobe, dresser, rug, curtains, warm lighting",
    },
    {
        "key": "living_room_medium_08",
        "room_type": "living_room",
        "room": "data/input/example/rooms/living_room_medium_08.json",
        "prompt": "modern cozy living room with sofa, coffee table, TV stand, shelves, rug, curtains, warm lighting",
    },
    {
        "key": "kitchen_medium_08",
        "room_type": "kitchen",
        "room": "data/input/example/rooms/kitchen_medium_08.json",
        "prompt": "modern practical kitchen with cabinets, countertop, sink, stove, refrigerator, dining table, chairs, warm lighting",
    },
    {
        "key": "toilet_medium_08",
        "room_type": "toilet",
        "room": "data/input/example/rooms/toilet_medium_08.json",
        "prompt": "modern small toilet room with toilet, compact sink, mirror, storage, wall tiles, warm lighting",
    },
    {
        "key": "bathroom_medium_08",
        "room_type": "bathroom",
        "room": "data/input/example/rooms/bathroom_medium_08.json",
        "prompt": "modern bathroom with bathtub, sink vanity, mirror, toilet, shower area, storage, wall tiles, warm lighting",
    },
]


PROMPTS_BY_ROOM_TYPE: dict[str, str] = {
    "bedroom": "modern cozy bedroom with bed, nightstands, wardrobe, dresser, rug, curtains, warm lighting",
    "living_room": "modern cozy living room with sofa, coffee table, TV stand, shelves, rug, curtains, warm lighting",
    "kitchen": "modern practical kitchen with cabinets, countertop, sink, stove, refrigerator, dining table, chairs, warm lighting",
    "toilet": "modern small toilet room with toilet, compact sink, mirror, storage, wall tiles, warm lighting",
    "bathroom": "modern bathroom with bathtub, sink vanity, mirror, toilet, shower area, storage, wall tiles, warm lighting",
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expand_layout_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    room_types = [x.strip() for x in str(args.room_types or "").split(",") if x.strip()]
    layout_indices = [x.strip() for x in str(args.layout_indices or "").split(",") if x.strip()]
    repeats = max(1, int(args.repeats))
    if not room_types and not layout_indices and repeats == 1:
        return list(DEFAULT_CASES)
    if not room_types:
        raise SystemExit("--room-types is required when using --layout-indices or --repeats")
    if not layout_indices:
        raise SystemExit("--layout-indices is required when using --room-types or --repeats")

    cases: list[dict[str, str]] = []
    for room_type in room_types:
        if room_type not in PROMPTS_BY_ROOM_TYPE:
            raise SystemExit(f"Unsupported room type: {room_type}")
        for layout in layout_indices:
            room_rel = f"data/input/example/rooms/{room_type}_{layout}.json"
            if not (ROOT / room_rel).is_file():
                raise SystemExit(f"Room file not found: {room_rel}")
            for repeat in range(1, repeats + 1):
                cases.append(
                    {
                        "key": f"{room_type}_{layout}_r{repeat:02d}",
                        "room_type": room_type,
                        "room": room_rel,
                        "prompt": PROMPTS_BY_ROOM_TYPE[room_type],
                        "layout": layout,
                        "repeat": str(repeat),
                    }
                )
    return cases


def delete_blends(run_dir: Path) -> dict[str, Any]:
    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    for path in sorted(run_dir.rglob("*.blend")):
        try:
            path.unlink()
            deleted.append(str(path.resolve()))
        except Exception as exc:
            failed.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
    return {"deleted_count": len(deleted), "deleted": deleted, "failed": failed}


def run_logged(cmd: list[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = proc.wait()
    duration_sec = round(time.perf_counter() - t0, 3)
    return {
        "command": cmd,
        "log": str(log_path.resolve()),
        "returncode": returncode,
        "status": "ok" if returncode == 0 else "failed",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": duration_sec,
    }


def pipeline_cmd(case: dict[str, str], run_dir: Path, args: argparse.Namespace) -> list[str]:
    room_type = case["room_type"]
    cmd = [
        "python3",
        "src/run_pipeline.py",
        "--paths-config",
        "config/paths.yaml",
        "--room",
        case["room"],
        "--run-dir",
        str(run_dir),
        "--keep-tmp",
        "--placer",
        "infinigen_clean",
        "--modes",
        "infinigen_clean",
        "--max-attempts",
        str(int(args.max_attempts)),
        "--prompt",
        case["prompt"],
        "--infinigen-fast-small",
        "--infinigen-no-pose-cameras",
        "--normalize-chandeliers",
        "--repair-furniture-overlaps",
        "--supplier-catalog-json",
        args.supplier_catalog_json,
        "--supplier-selection-modes",
        args.supplier_selection_modes,
        "--supplier-top-k",
        str(int(args.supplier_top_k)),
        "--supplier-llm-provider",
        "ollama",
        "--supplier-ollama-url",
        args.ollama_url,
        "--supplier-ollama-model",
        args.supplier_model,
        "--supplier-llm-top-n",
        str(int(args.supplier_llm_top_n)),
        "--build-supplier-blend",
        "--validate-supplier-variants",
        "--blender-output",
        "render",
        "--keep-blend",
        "--headless",
        "--curtains",
        "auto",
    ]
    if bool(getattr(args, "pipeline_stop_after_placement", False)):
        cmd.append("--stop-after-placement")
    if bool(getattr(args, "pipeline_skip_existing_placement", False)):
        cmd.append("--skip-existing-placement")
    if room_type == "kitchen":
        cmd += [
            "--kitchens",
            "always",
            "--kitchen-dining",
            "auto",
            "--kitchen-accessories",
            "auto",
            "--kitchen-llm-provider",
            "ollama",
            "--kitchen-ollama-url",
            args.ollama_url,
            "--kitchen-ollama-model",
            args.supplier_model,
            "--kitchen-accessory-llm-provider",
            "ollama",
            "--kitchen-accessory-ollama-url",
            args.ollama_url,
            "--kitchen-accessory-ollama-model",
            args.supplier_model,
            "--procedural-rooms",
            "never",
        ]
    elif room_type in {"bathroom", "toilet"}:
        cmd += [
            "--kitchens",
            "never",
            "--procedural-rooms",
            "auto",
            "--procedural-replace-existing",
        ]
    else:
        cmd += [
            "--kitchens",
            "never",
            "--procedural-rooms",
            "never",
        ]
    return cmd


def render_cmd(run_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        "python3",
        "src/tools/render_blend_vlm_views.py",
        "--run-dir",
        str(run_dir),
        "--resolution-x",
        str(int(args.resolution_x)),
        "--resolution-y",
        str(int(args.resolution_y)),
        "--skip-existing",
    ]


def eval_cmd(case: dict[str, str], run_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        "python3",
        "src/tools/evaluate_vlm_review_views.py",
        "--provider",
        "ollama",
        "--ollama-url",
        args.ollama_url,
        "--model",
        args.vlm_model,
        "--run-dir",
        str(run_dir),
        "--prompt",
        case["prompt"],
        "--room-type",
        case["room_type"],
        "--style-label",
        "modern",
        "--timeout-sec",
        str(int(args.vlm_timeout_sec)),
        "--scope",
        "frames",
        "--skip-existing",
    ]


def summarize_vlm(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "vlm_review_views" / "vlm_frame_eval_summary.json"
    if not path.is_file():
        return {"available": False}
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_kind: dict[str, list[float]] = {"all": [], "topview": [], "oblique_e60": []}
    for row in rows:
        score = float(row.get("total_score") or 0.0)
        by_kind["all"].append(score)
        kind = str(row.get("view_type") or "")
        by_kind.setdefault(kind, []).append(score)
    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0
    return {
        "available": True,
        "frame_count": len(rows),
        "avg_total": avg(by_kind.get("all", [])),
        "avg_topview": avg(by_kind.get("topview", [])),
        "avg_oblique_e60": avg(by_kind.get("oblique_e60", [])),
        "summary_json": str(path.resolve()),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def attach_detail_timings(run_dir: Path, stage: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(stage, dict):
        return stage
    pipeline_detail = run_dir / "pipeline_stage_timings.json"
    remote_detail = run_dir / "infinigen_remote_timings.json"
    if pipeline_detail.is_file():
        try:
            stage["pipeline_stage_timings"] = read_json(pipeline_detail)
        except Exception as exc:
            stage["pipeline_stage_timings_error"] = f"{type(exc).__name__}: {exc}"
    if remote_detail.is_file():
        try:
            stage["infinigen_remote_timings"] = read_json(remote_detail)
        except Exception as exc:
            stage["infinigen_remote_timings_error"] = f"{type(exc).__name__}: {exc}"
    return stage


def render_manifest_ok(run_dir: Path) -> bool:
    path = run_dir / "vlm_review_views" / "manifest.json"
    if not path.is_file():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary, dict):
        return False
    return int(summary.get("failed") or 0) == 0 and int(summary.get("ok") or 0) + int(summary.get("skipped_existing") or 0) > 0


def pipeline_outputs_ready(run_dir: Path) -> bool:
    if summarize_vlm(run_dir).get("available"):
        return True
    if not any(run_dir.glob("*.blend")):
        return False
    return (run_dir / "placement_infinigen_clean.json").is_file() or (run_dir / "scene.v1.json").is_file()


def placement_stage_ready(run_dir: Path) -> bool:
    return (run_dir / "placement_infinigen_clean.json").is_file() and (run_dir / "infinigen_clean_scene.blend").is_file()


def wait_until(label: str, predicate, poll_sec: float) -> None:
    while not predicate():
        print(f"⏳ wait: {label}", flush=True)
        time.sleep(max(1.0, float(poll_sec)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full room-type pipeline + VLM photo evaluation batch.")
    p.add_argument("--out-root", default=None)
    p.add_argument("--ollama-url", default="http://127.0.0.1:11435")
    p.add_argument("--supplier-model", default="gpt-oss:20b")
    p.add_argument("--vlm-model", default="qwen2.5vl:7b")
    p.add_argument("--vlm-timeout-sec", type=int, default=180)
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--supplier-catalog-json", default="data/sourse/suppliers/supplier_catalog_canonical.json")
    p.add_argument("--supplier-selection-modes", default="optimal,best_match,cheapest")
    p.add_argument("--supplier-top-k", type=int, default=8)
    p.add_argument("--supplier-llm-top-n", type=int, default=5)
    p.add_argument("--resolution-x", type=int, default=1400)
    p.add_argument("--resolution-y", type=int, default=1050)
    p.add_argument("--only", default="", help="Comma-separated case keys to run.")
    p.add_argument("--room-types", default="", help="Comma-separated room types, e.g. bedroom,living_room,kitchen,toilet.")
    p.add_argument("--layout-indices", default="", help="Comma-separated layout suffixes, e.g. small_01,small_02,medium_06.")
    p.add_argument("--repeats", type=int, default=1, help="Number of independent runs per room/layout.")
    p.add_argument("--delete-blends-after-vlm", action="store_true", help="Delete .blend files after VLM evaluation finishes.")
    p.add_argument("--resume", action="store_true", help="Skip stages with existing successful outputs.")
    p.add_argument("--vlm-workers", type=int, default=1, help="Number of asynchronous VLM evaluation workers.")
    p.add_argument("--pipeline-stop-after-placement", action="store_true", help="Run pipeline only through Infinigen placement/download.")
    p.add_argument("--pipeline-skip-existing-placement", action="store_true", help="Pass --skip-existing-placement to run_pipeline.")
    p.add_argument("--wait-for-placement", action="store_true", help="Wait until placement_infinigen_clean.json + infinigen_clean_scene.blend exist before pipeline stage.")
    p.add_argument("--wait-for-render-views", action="store_true", help="Wait until vlm_review_views/manifest.json is complete before VLM stage.")
    p.add_argument("--poll-sec", type=float, default=30.0)
    p.add_argument("--skip-pipeline", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--skip-vlm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root or f"out/runs/full_circle_rooms_{now_stamp()}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    selected = {x.strip() for x in args.only.split(",") if x.strip()}
    cases = [case for case in expand_layout_cases(args) if not selected or case["key"] in selected]
    write_json(out_root / "batch_config.json", {"cases": cases, "args": vars(args)})

    overall_t0 = time.perf_counter()
    summary: list[dict[str, Any]] = []
    summary_lock = threading.Lock()
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []

    def append_summary(row: dict[str, Any]) -> None:
        with summary_lock:
            for idx_existing, existing in enumerate(summary):
                if existing.get("case") == row.get("case"):
                    summary[idx_existing] = row
                    break
            else:
                summary.append(row)
            write_json(out_root / "full_circle_summary.json", summary)

    def run_vlm_stage(case: dict[str, str], case_dir: Path, timings: dict[str, Any]) -> dict[str, Any]:
        if args.skip_vlm:
            timings["stages"]["vlm_eval"] = {"status": "skipped"}
        elif args.resume and summarize_vlm(case_dir).get("available"):
            timings["stages"]["vlm_eval"] = {"status": "skipped_existing"}
        else:
            if args.wait_for_render_views:
                wait_until(f"{case['key']} render views", lambda: render_manifest_ok(case_dir), float(args.poll_sec))
            timings["stages"]["vlm_eval"] = run_logged(
                eval_cmd(case, case_dir, args),
                cwd=ROOT,
                log_path=case_dir / "vlm_eval.log",
            )

        if args.delete_blends_after_vlm and timings["stages"]["vlm_eval"].get("status") in {"ok", "skipped", "skipped_existing"}:
            timings["stages"]["delete_blends"] = {
                "status": "ok",
                **delete_blends(case_dir),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }

        timings["finished_at"] = datetime.now().isoformat(timespec="seconds")
        timings["duration_sec"] = round(
            sum(float(stage.get("duration_sec") or 0.0) for stage in timings["stages"].values()),
            3,
        )
        timings["vlm_summary"] = summarize_vlm(case_dir)
        write_json(case_dir / "timings.json", timings)
        status = "ok" if all(stage.get("status") in {"ok", "skipped", "skipped_existing"} for stage in timings["stages"].values()) else "failed"
        row = {
            "case": case["key"],
            "room_type": case["room_type"],
            "status": status,
            "run_dir": str(case_dir.resolve()),
            "duration_sec": timings["duration_sec"],
            "timings": timings["stages"],
            "vlm": timings["vlm_summary"],
        }
        append_summary(row)
        return row

    vlm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.vlm_workers)))
    for idx, case in enumerate(cases, start=1):
        case_dir = out_root / case["key"]
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n====== [{idx}/{len(cases)}] {case['key']} ======", flush=True)
        timings: dict[str, Any] = {
            "case": case,
            "run_dir": str(case_dir.resolve()),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "stages": {},
        }

        if args.resume and summarize_vlm(case_dir).get("available"):
            timings["stages"]["pipeline"] = {"status": "skipped_existing"}
            timings["stages"]["render_views"] = {"status": "skipped_existing"}
            timings["stages"]["vlm_eval"] = {"status": "skipped_existing"}
            if args.delete_blends_after_vlm:
                timings["stages"]["delete_blends"] = {"status": "ok", **delete_blends(case_dir)}
            timings["finished_at"] = datetime.now().isoformat(timespec="seconds")
            timings["vlm_summary"] = summarize_vlm(case_dir)
            write_json(case_dir / "timings.json", timings)
            append_summary(
                {
                    "case": case["key"],
                    "room_type": case["room_type"],
                    "status": "skipped_existing",
                    "run_dir": str(case_dir.resolve()),
                    "timings": timings["stages"],
                    "vlm": timings["vlm_summary"],
                }
            )
            continue

        if args.skip_pipeline:
            timings["stages"]["pipeline"] = {"status": "skipped"}
        elif args.resume and pipeline_outputs_ready(case_dir) and not args.pipeline_skip_existing_placement:
            timings["stages"]["pipeline"] = {"status": "skipped_existing"}
            timings["stages"]["pipeline"] = attach_detail_timings(case_dir, timings["stages"]["pipeline"])
        else:
            if args.wait_for_placement:
                wait_until(f"{case['key']} placement", lambda: placement_stage_ready(case_dir), float(args.poll_sec))
            timings["stages"]["pipeline"] = run_logged(
                pipeline_cmd(case, case_dir, args),
                cwd=ROOT,
                log_path=case_dir / "pipeline.log",
            )
            timings["stages"]["pipeline"] = attach_detail_timings(case_dir, timings["stages"]["pipeline"])

        if timings["stages"]["pipeline"].get("status") == "failed":
            timings["finished_at"] = datetime.now().isoformat(timespec="seconds")
            timings["vlm_summary"] = summarize_vlm(case_dir)
            write_json(case_dir / "timings.json", timings)
            summary.append(
                {
                    "case": case["key"],
                    "room_type": case["room_type"],
                    "status": "pipeline_failed",
                    "run_dir": str(case_dir.resolve()),
                    "timings": timings["stages"],
                    "vlm": timings["vlm_summary"],
                }
            )
            write_json(out_root / "full_circle_summary.json", summary)
            continue

        if args.skip_render:
            timings["stages"]["render_views"] = {"status": "skipped"}
        elif args.resume and render_manifest_ok(case_dir):
            timings["stages"]["render_views"] = {"status": "skipped_existing"}
        else:
            timings["stages"]["render_views"] = run_logged(
                render_cmd(case_dir, args),
                cwd=ROOT,
                log_path=case_dir / "render_views.log",
            )

        if timings["stages"]["render_views"].get("status") == "failed":
            timings["finished_at"] = datetime.now().isoformat(timespec="seconds")
            timings["vlm_summary"] = summarize_vlm(case_dir)
            write_json(case_dir / "timings.json", timings)
            summary.append(
                {
                    "case": case["key"],
                    "room_type": case["room_type"],
                    "status": "render_failed",
                    "run_dir": str(case_dir.resolve()),
                    "timings": timings["stages"],
                    "vlm": timings["vlm_summary"],
                }
            )
            write_json(out_root / "full_circle_summary.json", summary)
            continue

        if int(args.vlm_workers) <= 1:
            run_vlm_stage(case, case_dir, timings)
        else:
            futures.append(vlm_executor.submit(run_vlm_stage, case, case_dir, timings))

    for future in concurrent.futures.as_completed(futures):
        future.result()
    vlm_executor.shutdown(wait=True)
    final = {
        "out_root": str(out_root),
        "started_at": datetime.fromtimestamp(time.time() - (time.perf_counter() - overall_t0)).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(time.perf_counter() - overall_t0, 3),
        "summary": summary,
    }
    write_json(out_root / "full_circle_summary.json", final)
    print(f"\nDONE: {out_root / 'full_circle_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
