#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def resolve_blender(raw: str | None) -> str:
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(raw)
        if found:
            return found
    if Path(DEFAULT_BLENDER).is_file():
        return DEFAULT_BLENDER
    found = shutil.which("blender")
    if found:
        return found
    raise RuntimeError("Blender executable not found; pass --blender")


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_apartment_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if (root / "manifest.json").is_file() and (root / "apartment.json").is_file():
        return [root]

    direct = sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").is_file() and (p / "apartment.json").is_file())
    if direct:
        return direct

    two_level = sorted(
        p
        for parent in root.iterdir()
        if parent.is_dir()
        for p in parent.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file() and (p / "apartment.json").is_file()
    )
    if two_level:
        return two_level

    return sorted(p.parent for p in root.rglob("manifest.json") if (p.parent / "apartment.json").is_file())


def room_entries(apt_dir: Path) -> list[dict[str, Any]]:
    manifest = read_json(apt_dir / "manifest.json")
    return [entry for entry in manifest.get("rooms") or [] if isinstance(entry, dict) and entry.get("room_id")]


def scene_path_for_room(apt_dir: Path, room_id: str, mode: str) -> Path:
    return apt_dir / "rooms" / room_id / "pipeline" / mode / "scene_requirements.v1.json"


def reference_blend_for_room(apt_dir: Path, room_id: str, mode: str) -> Path | None:
    path = apt_dir / "rooms" / room_id / "pipeline" / mode / "infinigen_clean_scene.blend"
    return path if path.is_file() else None


def output_blend_for_room(apt_dir: Path, room_id: str, room_type: str, mode: str) -> Path:
    pipe = apt_dir / "rooms" / room_id / "pipeline" / mode
    if str(room_type).lower() == "kitchen":
        return pipe / "scene_kitchen_requirements.blend"
    return pipe / "scene_infinigen_clean_supplier.requirements.blend"


def blender_cmd(blender: str) -> list[str]:
    return [blender, "--factory-startup"]


def run_step(name: str, cmd: list[str], cwd: Path, steps: list[dict[str, Any]], *, allow_failure: bool = False) -> bool:
    started = time.time()
    print(f"\n[finalize] {name}")
    print("[finalize] " + " ".join(cmd))
    item = {"name": name, "cmd": cmd, "started_at": started, "status": "running"}
    steps.append(item)
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as exc:
        item["status"] = "failed"
        item["returncode"] = exc.returncode
        item["duration_sec"] = round(time.time() - started, 3)
        if allow_failure:
            return False
        raise
    item["status"] = "ok"
    item["returncode"] = 0
    item["duration_sec"] = round(time.time() - started, 3)
    return True


def rebuild_room_blends(apt_dir: Path, mode: str, blender: str, project_root: Path, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    builder = project_root / "src" / "Plasement" / "blender_scene_builder.py"
    for entry in room_entries(apt_dir):
        room_id = str(entry["room_id"])
        room_type = str(entry.get("room_type") or entry.get("prompt_room_type") or "")
        scene_json = scene_path_for_room(apt_dir, room_id, mode)
        if not scene_json.is_file():
            reports.append({"room_id": room_id, "status": "missing_scene_requirements", "scene_json": str(scene_json)})
            continue
        save_blend = output_blend_for_room(apt_dir, room_id, room_type, mode)
        build_report = save_blend.with_suffix(".build_report.json")
        reference_blend = reference_blend_for_room(apt_dir, room_id, mode)
        existed_before = save_blend.is_file()
        cmd = blender_cmd(blender)
        if reference_blend:
            cmd.append(str(reference_blend))
        cmd.extend(
            [
                "-b",
                "--python",
                str(builder),
                "--",
                "--json",
                str(scene_json),
                "--project-root",
                str(project_root / "src"),
                "--save-blend",
                str(save_blend),
                "--build-report",
                str(build_report),
                "--no-pack-assets",
            ]
        )
        if reference_blend:
            cmd.extend(["--reference-blend", str(reference_blend)])
        ok = run_step(f"rebuild room {room_id}", cmd, project_root, steps, allow_failure=True)
        if not ok and save_blend.is_file():
            reports.append(
                {
                    "room_id": room_id,
                    "room_type": room_type,
                    "status": "reused_existing_after_failed_rebuild",
                    "scene_json": str(scene_json),
                    "save_blend": str(save_blend),
                    "build_report": str(build_report),
                    "reference_blend": str(reference_blend) if reference_blend else None,
                    "existed_before": existed_before,
                }
            )
            continue
        if not ok:
            raise RuntimeError(f"Room rebuild failed and no fallback blend exists: {save_blend}")
        reports.append(
            {
                "room_id": room_id,
                "room_type": room_type,
                "status": "ok",
                "scene_json": str(scene_json),
                "save_blend": str(save_blend),
                "build_report": str(build_report),
                "reference_blend": str(reference_blend) if reference_blend else None,
            }
        )
    return reports


def write_markdown_report(apt_dir: Path, mode: str, report: dict[str, Any]) -> Path:
    out_dir = apt_dir / "apartment_pipeline" / mode
    report_path = out_dir / "report_requirements.md"
    lines = [
        "# Apartment requirements final report",
        "",
        f"- apartment: `{apt_dir}`",
        f"- mode: `{mode}`",
        f"- final blend: `{out_dir / 'scene_apartment.requirements.blend'}`",
        f"- overview render: `{out_dir / 'render_apartment.requirements.png'}`",
        f"- corner render report: `{out_dir / 'room_corner_renders.report.md'}`",
        f"- cost report: `{out_dir / 'renovation_cost_report.md'}`",
        f"- run report: `{out_dir / 'finalize_requirements.report.json'}`",
        "",
        "## Room Corner Renders",
        "",
    ]
    corner_md = out_dir / "room_corner_renders.report.md"
    if corner_md.is_file():
        lines.append(f"See `{corner_md.name}` for the full 4-corner render set.")
    else:
        lines.append("Corner render report was not found.")
    lines.extend(["", "## Steps", "", "| Step | Status | Seconds |", "|---|---|---:|"])
    for step in report.get("steps") or []:
        lines.append(f"| {step.get('name')} | {step.get('status')} | {step.get('duration_sec', '')} |")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def process_apartment(apt_dir: Path, args: argparse.Namespace, blender: str, project_root: Path) -> dict[str, Any]:
    mode = str(args.mode)
    out_dir = apt_dir / "apartment_pipeline" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    ensure_summary = out_dir / "finalize_requirements.ensure_summary.json"
    run_step(
        "requirements postprocess",
        [
            sys.executable,
            str(project_root / "src" / "tools" / "ensure_apartment_requirements.py"),
            str(apt_dir),
            "--mode",
            mode,
            "--out-summary",
            str(ensure_summary),
        ],
        project_root,
        steps,
    )

    room_reports = rebuild_room_blends(apt_dir, mode, blender, project_root, steps)

    apartment_scene = out_dir / "scene_apartment.requirements.v1.json"
    apartment_blend = out_dir / "scene_apartment.requirements.blend"
    overview_render = out_dir / "render_apartment.requirements.png"
    assemble_report = out_dir / "scene_apartment.requirements.build_report.json"
    assemble_cmd = [
        *blender_cmd(blender),
        "-b",
        "--python",
        str(project_root / "src" / "tools" / "assemble_apartment_blend.py"),
        "--",
        "--apt-dir",
        str(apt_dir),
        "--mode",
        mode,
        "--apartment-scene",
        str(apartment_scene),
        "--save-blend",
        str(apartment_blend),
        "--render",
        str(overview_render),
        "--build-report",
        str(assemble_report),
        "--width",
        str(int(args.overview_width)),
        "--height",
        str(int(args.overview_height)),
        "--samples",
        str(int(args.overview_samples)),
    ]
    run_step("assemble apartment blend", assemble_cmd, project_root, steps)

    cost_json = out_dir / "renovation_cost_report.json"
    cost_md = out_dir / "renovation_cost_report.md"
    run_step(
        "summarize renovation cost",
        [
            sys.executable,
            str(project_root / "src" / "tools" / "summarize_apartment_cost.py"),
            str(apt_dir),
            "--mode",
            mode,
            "--out-json",
            str(cost_json),
            "--out-md",
            str(cost_md),
        ],
        project_root,
        steps,
    )

    corner_dir = out_dir / "room_corner_renders"
    corner_json = out_dir / "room_corner_renders.report.json"
    corner_md = out_dir / "room_corner_renders.report.md"
    corner_cmd = [
        *blender_cmd(blender),
        str(apartment_blend),
        "-b",
        "--python",
        str(project_root / "src" / "tools" / "render_apartment_room_corner_views.py"),
        "--",
        "--apt-dir",
        str(apt_dir),
        "--mode",
        mode,
        "--apartment-scene",
        str(apartment_scene),
        "--out-dir",
        str(corner_dir),
        "--report-json",
        str(corner_json),
        "--report-md",
        str(corner_md),
        "--width",
        str(int(args.corner_width)),
        "--height",
        str(int(args.corner_height)),
        "--samples",
        str(int(args.corner_samples)),
    ]
    run_step("render room corner views", corner_cmd, project_root, steps)

    report = {
        "apartment_dir": str(apt_dir),
        "mode": mode,
        "outputs": {
            "apartment_scene": str(apartment_scene),
            "apartment_blend": str(apartment_blend),
            "overview_render": str(overview_render),
            "assemble_report": str(assemble_report),
            "cost_json": str(cost_json),
            "cost_md": str(cost_md),
            "corner_dir": str(corner_dir),
            "corner_json": str(corner_json),
            "corner_md": str(corner_md),
        },
        "room_reports": room_reports,
        "steps": steps,
    }
    report_json = out_dir / "finalize_requirements.report.json"
    report["outputs"]["run_report_json"] = str(report_json)
    report["outputs"]["run_report_md"] = str(out_dir / "report_requirements.md")
    write_json(report_json, report)
    report_md = write_markdown_report(apt_dir, mode, report)
    report["outputs"]["run_report_md"] = str(report_md)
    write_json(report_json, report)
    return report


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-command finalizer for apartment requirement postprocess, blend assembly, costs and room corner renders.")
    parser.add_argument("root", help="Apartment dir, project dir containing apt_* dirs, or output root.")
    parser.add_argument("--mode", default="optimal")
    parser.add_argument("--blender", default=None)
    parser.add_argument("--corner-width", type=int, default=960)
    parser.add_argument("--corner-height", type=int, default=720)
    parser.add_argument("--corner-samples", type=int, default=16)
    parser.add_argument("--overview-width", type=int, default=1400)
    parser.add_argument("--overview-height", type=int, default=1000)
    parser.add_argument("--overview-samples", type=int, default=16)
    parser.add_argument("--out-summary", default=None)
    return parser


def main() -> None:
    args = build_cli().parse_args()
    project_root = project_root_from_script()
    root = Path(args.root).expanduser().resolve()
    blender = resolve_blender(args.blender)
    apartments = iter_apartment_dirs(root)
    if not apartments:
        raise RuntimeError(f"No apartment folders with manifest.json and apartment.json found under {root}")
    results = [process_apartment(apt_dir, args, blender, project_root) for apt_dir in apartments]
    summary_path = Path(args.out_summary).expanduser().resolve() if args.out_summary else root / "finalize_requirements.summary.json"
    write_json(summary_path, {"root": str(root), "mode": args.mode, "count": len(results), "results": results})
    print(json.dumps({"summary": str(summary_path), "apartments": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
