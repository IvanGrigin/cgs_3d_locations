from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_imports() -> None:
    root = _repo_root()
    src = root / "src"
    for candidate in (root, src):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


_ensure_imports()

try:
    from src.pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage
except ModuleNotFoundError:
    from pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_scene_from_room(room_json: dict[str, Any]) -> dict[str, Any]:
    if isinstance(room_json.get("room"), dict):
        room = room_json["room"]
    else:
        room = room_json
    return {
        "schema": "scene.v1",
        "room": room,
        "placements": [],
        "meta": {
            "placer": "manual_room_input",
            "mode": "procedural_room_smoke",
        },
    }


def _batch_jobs(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [job for job in payload if isinstance(job, dict)]
    if isinstance(payload, dict):
        jobs = payload.get("jobs")
        if isinstance(jobs, list):
            return [job for job in jobs if isinstance(job, dict)]
        rooms = payload.get("rooms")
        if isinstance(rooms, list):
            return [job for job in rooms if isinstance(job, dict)]
    return []


def _job_scene_path(job: dict[str, Any], job_out_dir: Path) -> Path:
    if job.get("scene"):
        return Path(str(job["scene"]))
    if job.get("room_json"):
        room_payload = read_json(str(job["room_json"]))
        scene = build_scene_from_room(room_payload)
    else:
        room_payload = {"room": job.get("room")} if isinstance(job.get("room"), dict) else job
        scene = build_scene_from_room(room_payload)
    scene_path = job_out_dir / "input_scene_from_room.v1.json"
    write_json(scene_path, scene)
    return scene_path


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    payload = read_json(args.batch_file)
    jobs = _batch_jobs(payload)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_defaults = payload if isinstance(payload, dict) else {}
    reports: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        job_id = str(job.get("id") or job.get("tag") or f"room_{index:02d}")
        tag = str(job.get("tag") or job_id)
        job_out_dir = out_dir / tag
        job_out_dir.mkdir(parents=True, exist_ok=True)
        scene_path = _job_scene_path(job, job_out_dir)
        report = apply_procedural_room_stage(
            scene_json_path=scene_path,
            out_dir=job_out_dir,
            prompt=str(job.get("prompt", args.prompt or "")),
            policy=str(job.get("policy", batch_defaults.get("policy", args.policy))),
            density=str(job.get("density", batch_defaults.get("density", args.density))),
            replace_existing=bool(job.get("replace_existing", batch_defaults.get("replace_existing", args.replace_existing))),
            seed=int(job.get("seed", args.seed + index if args.seed is not None else index)),
            tag=tag,
        )
        report["batch_job_id"] = job_id
        reports.append(report)

    scene_bundle: list[dict[str, Any]] = []
    placement_bundle: list[dict[str, Any]] = []
    for report in reports:
        scene_path = report.get("output_scene_json")
        placement_path = report.get("output_placement_json")
        if scene_path:
            try:
                scene_bundle.append(read_json(str(scene_path)))
            except Exception:
                pass
        if placement_path:
            try:
                placement_bundle.append(read_json(str(placement_path)))
            except Exception:
                pass

    scene_bundle_json = out_dir / "scene_bundle.v1.array.json"
    placement_bundle_json = out_dir / "placement_bundle.v1.array.json"
    write_json(scene_bundle_json, scene_bundle)
    write_json(placement_bundle_json, placement_bundle)

    summary = {
        "schema": "procedural_room_batch_report/v1",
        "batch_file": str(args.batch_file),
        "out_dir": str(out_dir),
        "job_count": len(jobs),
        "scene_bundle_json": str(scene_bundle_json),
        "placement_bundle_json": str(placement_bundle_json),
        "reports": reports,
        "summary": [
            {
                "id": report.get("batch_job_id"),
                "room_type": report.get("room_type"),
                "generated_count": report.get("generated_count"),
                "accessibility_ok": (report.get("validation") or {}).get("accessibility_ok"),
                "output_scene_json": report.get("output_scene_json"),
                "output_placement_json": report.get("output_placement_json"),
                "report_json": report.get("report_json"),
            }
            for report in reports
        ],
    }
    write_json(out_dir / "procedural_room_batch_report.json", summary)
    return summary


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run procedural room stage on a scene.v1.json, room.json, or batch JSON.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="Input scene.v1.json")
    src.add_argument("--room", help="Input room.json; a temporary scene.v1 wrapper is created")
    src.add_argument("--batch-file", help="Batch JSON with jobs containing inline room objects or room_json/scene paths")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--prompt", default="", help="Prompt used for ambiguous room type normalization")
    parser.add_argument("--policy", default="always", choices=["auto", "always", "never"])
    parser.add_argument("--density", default="very_high", choices=["normal", "high", "very_high"])
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", default="standalone")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_file:
        report = run_batch(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.scene:
        scene_path = Path(args.scene)
    else:
        room_payload = read_json(args.room)
        scene = build_scene_from_room(room_payload)
        scene_path = out_dir / "input_scene_from_room.v1.json"
        write_json(scene_path, scene)

    report = apply_procedural_room_stage(
        scene_json_path=scene_path,
        out_dir=out_dir,
        prompt=args.prompt,
        policy=args.policy,
        density=args.density,
        replace_existing=args.replace_existing,
        seed=args.seed,
        tag=args.tag,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
