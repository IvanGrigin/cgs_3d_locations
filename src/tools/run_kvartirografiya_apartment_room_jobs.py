#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_ROOT = REPO_ROOT / "data/output/kvartirografiya_all_projects_with_response_windows"
DEFAULT_OUT_ROOT = REPO_ROOT / "data/output/kvartirografiya_apartment_room_jobs"
DEFAULT_PROMPTS_DIR = REPO_ROOT / "data/input/example/prompts"

ROOM_TYPE_ALIASES = {
    "bedroom": "bedroom",
    "living": "living_room",
    "living_room": "living_room",
    "studio": "living_room",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "toilet": "toilet",
    "wc": "toilet",
    "restroom": "toilet",
    "joint_bathroom": "bathroom",
    "hall": "hallway",
    "hallway": "hallway",
    "corridor": "hallway",
}

FALLBACK_PROMPTS = {
    "hallway": "Design a compact modern hallway. Keep the route from the entrance to all room doors clear. Required items: wardrobe, shoe cabinet, mirror.",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_apartment_id(value: str | int) -> str:
    text = str(value).strip()
    if text.startswith("apt_"):
        text = text.removeprefix("apt_")
    if text.isdigit():
        return f"apt_{int(text):04d}"
    return str(value).strip()


def normalize_room_type(value: str | None) -> str:
    text = str(value or "").strip().lower()
    return ROOM_TYPE_ALIASES.get(text, text or "living_room")


def shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def room_size_class(area_m2: float | None) -> str:
    if area_m2 is None:
        return "medium"
    if area_m2 <= 8.0:
        return "small"
    if area_m2 <= 18.0:
        return "medium"
    return "large"


def prompt_room_type(summary: dict[str, Any]) -> str:
    source = normalize_room_type(str(summary.get("source_room_type") or ""))
    if source in {"toilet", "wc", "restroom"}:
        return "toilet"
    return normalize_room_type(str(summary.get("room_type") or ""))


def choose_prompt(room_type: str, area_m2: float | None, prompts_dir: Path) -> tuple[str, str | None]:
    room_type = normalize_room_type(room_type)
    if room_type in FALLBACK_PROMPTS:
        return FALLBACK_PROMPTS[room_type], None
    size = room_size_class(area_m2)
    candidates = sorted(prompts_dir.glob(f"{room_type}_{size}_*.prompt.txt"))
    if not candidates:
        candidates = sorted(prompts_dir.glob(f"{room_type}_*.prompt.txt"))
    if not candidates:
        candidates = sorted(prompts_dir.glob("living_room_*.prompt.txt"))
    if not candidates:
        return "Design a modern functional room using the provided real room geometry.", None
    path = candidates[0]
    return path.read_text(encoding="utf-8").strip(), str(path.resolve())


def find_apartment_bundle(adapter_root: Path, project_id: str, apartment_id: str) -> Path:
    adapter_root = adapter_root.expanduser().resolve()
    apt_id = normalize_apartment_id(apartment_id)
    candidates = sorted(adapter_root.glob(f"{project_id}/floor_*/apartment_bundles/{apt_id}/manifest.json"))
    if not candidates:
        candidates = sorted(adapter_root.glob(f"**/{project_id}/floor_*/apartment_bundles/{apt_id}/manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"Apartment {apt_id} for project {project_id} not found under {adapter_root}")
    if len(candidates) > 1:
        # Prefer the lowest visible floor unless the caller passed a root scoped to one floor.
        candidates = sorted(candidates, key=lambda p: str(p))
    return candidates[0].parent.resolve()


def polygon_points(room: dict[str, Any]) -> list[tuple[float, float]]:
    points = []
    for point in room.get("floor_polygon") or []:
        if isinstance(point, dict) and "x" in point and "y" in point:
            points.append((float(point["x"]), float(point["y"])))
    return points


def longest_wall_m(room: dict[str, Any]) -> float:
    points = polygon_points(room)
    if len(points) < 2:
        return float(room.get("width_m") or 2.4)
    best = 0.0
    for idx, a in enumerate(points):
        b = points[(idx + 1) % len(points)]
        best = max(best, math.hypot(b[0] - a[0], b[1] - a[1]))
    return max(best, float(room.get("width_m") or 0.0), 1.2)


def room_summary(room_path: Path, source_room_path: Path | None = None) -> dict[str, Any]:
    payload = read_json(room_path)
    room = payload.get("room") if isinstance(payload, dict) else {}
    if not isinstance(room, dict):
        room = {}
    return {
        "room_json": str(room_path.resolve()),
        "source_room_json": str(source_room_path.resolve()) if source_room_path else None,
        "id": room.get("id") or room_path.stem,
        "room_type": normalize_room_type(str(room.get("room_type") or room.get("type") or "")),
        "source_room_type": room.get("source_room_type"),
        "area_m2": room.get("area_m2"),
        "width_m": room.get("width_m"),
        "depth_m": room.get("depth_m"),
        "ceiling_height_m": room.get("ceiling_height_m") or room.get("ceiling_height"),
        "coordinate_frame": (room.get("meta") or {}).get("coordinate_frame"),
        "floor_polygon": room.get("floor_polygon") or [],
        "doors": room.get("doors") or [],
        "windows": room.get("windows") or [],
        "openings": room.get("openings") or [],
        "longest_wall_m": round(longest_wall_m(room), 4),
    }


def strategy_for_mode(mode: str) -> str:
    return {"cheapest": "cheapest", "optimal": "balanced", "best_match": "style"}.get(mode, "balanced")


def build_kitchen_command(args: argparse.Namespace, summary: dict[str, Any], room_dir: Path) -> dict[str, Any]:
    mode = args.variant_mode
    run_dir = room_dir / "kitchen"
    prompt = args.kitchen_prompt or f"Functional straight kitchen for room {summary['id']}"
    width_m = max(float(summary.get("longest_wall_m") or summary.get("width_m") or 2.4), 1.2)
    cmd = [
        sys.executable,
        "-m",
        "src.suppliers.kitchen.run_kitchen_render",
        "--width-m",
        f"{width_m:.4f}",
        "--prompt",
        prompt,
        "--slug",
        str(summary["id"]),
        "--out-dir",
        str(run_dir.resolve()),
        "--mode",
        mode,
        "--budget",
        str(float(args.kitchen_budget)),
        "--kitchen-llm-provider",
        args.kitchen_llm_provider,
    ]
    if args.kitchen_no_render:
        cmd.append("--no-render")
    if args.blender:
        cmd.extend(["--blender", args.blender])
    if width_m >= args.kitchen_fridge_min_width_m:
        cmd.append("--fridge")
    if width_m >= args.kitchen_dishwasher_min_width_m:
        cmd.append("--dishwasher")
    return {
        "kind": "kitchen_render",
        "room_id": summary["id"],
        "room_type": summary["room_type"],
        "run_dir": str(run_dir.resolve()),
        "command_args": cmd,
        "command": shell_command(cmd),
        "expected_json": str((run_dir / f"{summary['id']}.json").resolve()),
        "expected_blend": str((run_dir / f"{summary['id']}.blend").resolve()),
        "expected_png": str((run_dir / f"{summary['id']}_preview.png").resolve()),
    }


def build_pipeline_command(args: argparse.Namespace, summary: dict[str, Any], room_path: Path, room_dir: Path, prompt_text: str) -> dict[str, Any]:
    mode = args.variant_mode
    run_dir = room_dir / "pipeline" / mode
    prompt_path = room_dir / "prompt.txt"
    prompt_path.write_text(prompt_text.strip() + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        "src/run_pipeline.py",
        "--room",
        str(room_path.resolve()),
        "--prompt-file",
        str(prompt_path.resolve()),
        "--run-dir",
        str(run_dir.resolve()),
        "--keep-tmp",
        "--placer",
        "infinigen_clean",
        "--supplier-selection-mode",
        mode,
        "--supplier-selection-strategy",
        strategy_for_mode(mode),
        "--supplier-top-k",
        str(args.supplier_top_k),
        "--supplier-llm-provider",
        args.supplier_llm_provider,
    ]
    if args.infinigen_fast_small:
        cmd.append("--infinigen-fast-small")
    if args.infinigen_no_pose_cameras:
        cmd.append("--infinigen-no-pose-cameras")
    if args.infinigen_solve_steps_large is not None:
        cmd.extend(["--infinigen-solve-steps-large", str(args.infinigen_solve_steps_large)])
    if args.infinigen_solve_steps_medium is not None:
        cmd.extend(["--infinigen-solve-steps-medium", str(args.infinigen_solve_steps_medium)])
    if args.infinigen_solve_steps_small is not None:
        cmd.extend(["--infinigen-solve-steps-small", str(args.infinigen_solve_steps_small)])
    if args.skip_blender:
        cmd.append("--skip-blender")
    else:
        cmd.extend(["--blender-output", args.blender_output])
        cmd.extend(["--blender-gif-frames", str(args.blender_gif_frames)])
        cmd.append("--build-supplier-blend")
        if args.keep_blend:
            cmd.append("--keep-blend")
        if args.headless:
            cmd.append("--headless")
        if args.blender:
            cmd.extend(["--blender", args.blender])
        if args.skip_supplier_gif:
            cmd.append("--skip-supplier-gif")
    if args.heuristic_llm_stages:
        cmd.extend(
            [
                "--chooser-llm-provider",
                "none",
                "--style-llm-provider",
                "none",
                "--flooring-llm-provider",
                "none",
                "--wall-llm-provider",
                "none",
            ]
        )
    if summary["room_type"] == "kitchen":
        cmd.extend(
            [
                "--kitchens",
                "always",
                "--kitchen-selection-mode",
                mode,
                "--kitchen-dining",
                "always",
                "--kitchen-accessories",
                "auto",
                "--kitchen-llm-provider",
                args.kitchen_llm_provider,
            ]
        )
    return {
        "kind": "run_pipeline_infinigen",
        "room_id": summary["id"],
        "room_type": summary["room_type"],
        "run_dir": str(run_dir.resolve()),
        "prompt_file": str(prompt_path.resolve()),
        "command_args": cmd,
        "command": shell_command(cmd),
    }


def collect_kitchen_result(run: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(run.get("expected_json") or ""))
    if not path.is_file():
        return {"status": "missing", "json": str(path)}
    data = read_json(path)
    return {
        "status": "ok",
        "json": str(path.resolve()),
        "blend": run.get("expected_blend"),
        "png": run.get("expected_png"),
        "price_estimate": data.get("price_estimate"),
        "warnings": data.get("warnings") or [],
    }


def collect_pipeline_result(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run["run_dir"]))
    summaries = []
    for path in sorted(run_dir.glob("supplier_replacements*.summary.json")):
        try:
            data = read_json(path)
        except Exception as exc:
            summaries.append({"summary_json": str(path), "error": str(exc)})
            continue
        summaries.append(
            {
                "summary_json": str(path.resolve()),
                "full_md": str(path.with_suffix("").with_suffix(".full.md").resolve()),
                "html": str(path.with_suffix("").with_suffix(".html").resolve()),
                "counts": data.get("counts"),
                "targets_count": len(data.get("targets") or []),
            }
        )
    return {
        "status": "ok" if summaries else "missing_reports",
        "run_dir": str(run_dir.resolve()),
        "supplier_reports": summaries,
        "blends": [str(p.resolve()) for p in sorted(run_dir.glob("*.blend"))],
        "renders": [str(p.resolve()) for p in sorted(run_dir.glob("*.png"))],
        "gifs": [str(p.resolve()) for p in sorted(run_dir.glob("*.gif"))],
    }


def write_report(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# Apartment Room Jobs {manifest['project_id']} / {manifest['apartment_id']}",
        "",
        f"- Bundle: `{manifest['source_bundle']}`",
        f"- Rooms: {len(manifest['rooms'])}",
        f"- Executed: {'yes' if manifest.get('executed') else 'no'}",
        "",
    ]
    for room in manifest["rooms"]:
        lines.append(f"## {room['room_id']} ({room['room_type']})")
        lines.append(f"- Room JSON: `{room['room_json']}`")
        lines.append(f"- Summary JSON: `{room['summary_json']}`")
        for run in room.get("runs") or []:
            lines.append(f"- {run['kind']}: `{run['run_dir']}`")
            result = run.get("result") or {}
            if run["kind"] == "kitchen_render" and result:
                lines.append(f"  - Kitchen JSON: `{result.get('json')}`")
                lines.append(f"  - Price: `{result.get('price_estimate')}`")
            elif result:
                lines.append(f"  - Supplier reports: {len(result.get('supplier_reports') or [])}")
                for report in (result.get("supplier_reports") or [])[:3]:
                    lines.append(f"  - Full report: `{report.get('full_md')}`")
            if not manifest.get("executed"):
                lines.append(f"  - Command: `{run['command']}`")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    apt_id = normalize_apartment_id(args.apartment)
    bundle_dir = find_apartment_bundle(Path(args.adapter_root), str(args.project_id), apt_id)
    bundle_manifest = read_json(bundle_dir / "manifest.json")
    out_dir = Path(args.out_dir).expanduser().resolve() / str(args.project_id) / apt_id
    rooms_root = out_dir / "rooms"
    rooms_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(bundle_dir / "apartment.json", out_dir / "apartment.json")
    shutil.copy2(bundle_dir / "manifest.json", out_dir / "source_bundle_manifest.json")

    manifest: dict[str, Any] = {
        "project_id": str(args.project_id),
        "apartment_id": apt_id,
        "source_bundle": str(bundle_dir.resolve()),
        "out_dir": str(out_dir),
        "rooms": [],
        "executed": False,
    }
    prompts_dir = Path(args.prompts_dir).expanduser().resolve()

    for entry in bundle_manifest.get("rooms") or []:
        source_room = Path(str(entry["room_json"])).expanduser().resolve()
        source_summary = room_summary(source_room, source_room)
        room_id = str(source_summary["id"])
        room_dir = rooms_root / room_id
        room_dir.mkdir(parents=True, exist_ok=True)
        room_path = room_dir / "room.json"
        shutil.copy2(source_room, room_path)
        summary = room_summary(room_path, source_room)
        summary_path = room_dir / "room_summary.json"
        write_json(summary_path, summary)

        prompt_type = prompt_room_type(summary)
        prompt_text, prompt_source = choose_prompt(prompt_type, float(summary.get("area_m2") or 0.0), prompts_dir)
        room_row = {
            "room_id": room_id,
            "room_type": summary["room_type"],
            "source_room_type": summary.get("source_room_type"),
            "prompt_room_type": prompt_type,
            "room_json": str(room_path.resolve()),
            "summary_json": str(summary_path.resolve()),
            "prompt_source": prompt_source,
            "runs": [],
        }
        if summary["room_type"] == "kitchen" and args.kitchen_runner == "separate":
            room_row["runs"].append(build_kitchen_command(args, summary, room_dir))
        else:
            room_row["runs"].append(build_pipeline_command(args, summary, room_path, room_dir, prompt_text))
        manifest["rooms"].append(room_row)
    return out_dir, manifest


def execute_manifest(manifest: dict[str, Any]) -> int:
    failures = 0
    for room in manifest["rooms"]:
        for run in room.get("runs") or []:
            print(f"[run] {room['room_id']} -> {run['kind']}", flush=True)
            completed = subprocess.run(run["command_args"], cwd=REPO_ROOT)
            run["returncode"] = completed.returncode
            if run["kind"] == "kitchen_render":
                run["result"] = collect_kitchen_result(run)
            else:
                run["result"] = collect_pipeline_result(run)
            if completed.returncode != 0:
                failures += 1
    manifest["executed"] = True
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and run per-room jobs for one Kvartirografiya apartment.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--apartment", required=True)
    parser.add_argument("--adapter-root", default=str(DEFAULT_ADAPTER_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--prompts-dir", default=str(DEFAULT_PROMPTS_DIR))
    parser.add_argument("--variant-mode", choices=("cheapest", "optimal", "best_match"), default="optimal")
    parser.add_argument("--execute", action="store_true")

    parser.add_argument("--skip-blender", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--blender-output", choices=("render", "gif", "both"), default="both")
    parser.add_argument("--blender-gif-frames", type=int, default=12)
    parser.add_argument("--keep-blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--skip-supplier-gif", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--supplier-top-k", type=int, default=5)
    parser.add_argument("--supplier-llm-provider", choices=("none", "ollama"), default="none")
    parser.add_argument("--heuristic-llm-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--infinigen-fast-small", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--infinigen-no-pose-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--infinigen-solve-steps-large", type=int, default=None)
    parser.add_argument("--infinigen-solve-steps-medium", type=int, default=None)
    parser.add_argument("--infinigen-solve-steps-small", type=int, default=None)

    parser.add_argument("--kitchen-prompt", default=None)
    parser.add_argument("--kitchen-runner", choices=("separate", "pipeline"), default="separate")
    parser.add_argument("--kitchen-budget", type=float, default=120000.0)
    parser.add_argument("--kitchen-no-render", action="store_true")
    parser.add_argument("--kitchen-llm-provider", choices=("none", "ollama"), default="none")
    parser.add_argument("--kitchen-fridge-min-width-m", type=float, default=2.4)
    parser.add_argument("--kitchen-dishwasher-min-width-m", type=float, default=3.0)
    args = parser.parse_args()

    out_dir, manifest = build_manifest(args)
    failures = execute_manifest(manifest) if args.execute else 0

    write_json(out_dir / "manifest.json", manifest)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}"]
    kitchen_commands = list(commands)
    pipeline_commands = list(commands)
    for room in manifest["rooms"]:
        for run in room.get("runs") or []:
            commands.append(run["command"])
            if run["kind"] == "kitchen_render":
                kitchen_commands.append(run["command"])
            else:
                pipeline_commands.append(run["command"])
    for name, content in {
        "commands.sh": commands,
        "run_kitchens.sh": kitchen_commands,
        "run_pipeline_rooms.sh": pipeline_commands,
    }.items():
        path = out_dir / name
        path.write_text("\n".join(content) + "\n", encoding="utf-8")
        path.chmod(0o755)
    write_report(out_dir, manifest)

    print(f"Apartment folder: {out_dir}")
    print(f"Rooms: {len(manifest['rooms'])}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    print(f"Commands: {out_dir / 'commands.sh'}")
    print(f"Report: {out_dir / 'report.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
