#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import bpy


def argv_after_blender_separator() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def room_scene_path(apt_dir: Path, room_id: str, mode: str) -> Path:
    return apt_dir / "rooms" / room_id / "pipeline" / mode / "scene_requirements.v1.json"


def kitchen_preview_blend(apt_dir: Path, room_id: str) -> Path | None:
    path = apt_dir / "rooms" / room_id / "kitchen" / f"{room_id}.blend"
    return path if path.is_file() else None


def reference_blend_for_room(apt_dir: Path, room_id: str, mode: str) -> Path | None:
    path = apt_dir / "rooms" / room_id / "pipeline" / mode / "infinigen_clean_scene.blend"
    return path if path.is_file() else None


def room_output_blend(apt_dir: Path, room_id: str, room_type: str, mode: str) -> Path:
    pipe = apt_dir / "rooms" / room_id / "pipeline" / mode
    if str(room_type).lower() == "kitchen" or "kitchen" in room_id.lower():
        preview = kitchen_preview_blend(apt_dir, room_id)
        if preview is not None:
            return preview
        return pipe / "scene_kitchen_requirements.blend"
    return pipe / "scene_infinigen_clean_supplier.requirements.blend"


def purge_orphans() -> None:
    for _ in range(3):
        try:
            result = bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except Exception:
            return
        if "CANCELLED" in result:
            return


def run_with_argv(fn, argv: list[str]) -> None:
    old = sys.argv[:]
    sys.argv = ["Blender", "--", *argv]
    try:
        fn()
    finally:
        sys.argv = old


def run_step(name: str, steps: list[dict[str, Any]], fn) -> Any:
    started = time.time()
    item = {"name": name, "status": "running", "started_at": started}
    steps.append(item)
    print(f"[finalize_blender] {name}")
    try:
        result = fn()
    except Exception as exc:
        item["status"] = "failed"
        item["error"] = repr(exc)
        item["duration_sec"] = round(time.time() - started, 3)
        raise
    item["status"] = "ok"
    item["duration_sec"] = round(time.time() - started, 3)
    return result


def rebuild_room(builder_mod, project_root: Path, apt_dir: Path, mode: str, entry: dict[str, Any]) -> dict[str, Any]:
    room_id = str(entry.get("room_id") or "")
    room_type = str(entry.get("room_type") or entry.get("prompt_room_type") or "")
    scene_json = room_scene_path(apt_dir, room_id, mode)
    if not scene_json.is_file():
        return {"room_id": room_id, "status": "missing_scene_requirements", "scene_json": str(scene_json)}

    reference = reference_blend_for_room(apt_dir, room_id, mode)
    save_blend = room_output_blend(apt_dir, room_id, room_type, mode)
    build_report = save_blend.with_suffix(".build_report.json")

    try:
        if reference:
            bpy.ops.wm.open_mainfile(filepath=str(reference))
        else:
            bpy.ops.wm.read_factory_settings(use_empty=True)
        argv = [
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
        if reference:
            argv.extend(["--reference-blend", str(reference)])
        run_with_argv(builder_mod.main, argv)
        purge_orphans()
        return {
            "room_id": room_id,
            "room_type": room_type,
            "status": "ok",
            "scene_json": str(scene_json),
            "save_blend": str(save_blend),
            "build_report": str(build_report),
            "reference_blend": str(reference) if reference else None,
        }
    except Exception as exc:
        if save_blend.is_file():
            print(f"[finalize_blender] warning: rebuild failed for {room_id}; reusing existing {save_blend}: {exc!r}")
            return {
                "room_id": room_id,
                "room_type": room_type,
                "status": "reused_existing_after_failed_rebuild",
                "scene_json": str(scene_json),
                "save_blend": str(save_blend),
                "build_report": str(build_report),
                "reference_blend": str(reference) if reference else None,
                "error": repr(exc),
            }
        raise


def write_final_markdown(apt_dir: Path, mode: str, report: dict[str, Any]) -> Path:
    out_dir = apt_dir / "apartment_pipeline" / mode
    out_path = out_dir / "report_requirements.md"
    lines = [
        "# Apartment requirements final report",
        "",
        f"- apartment: `{apt_dir}`",
        f"- mode: `{mode}`",
        f"- final blend: `{out_dir / 'scene_apartment.requirements.blend'}`",
        f"- overview render: `{out_dir / 'render_apartment.requirements.png'}`",
        f"- room corner renders: `{out_dir / 'room_corner_renders.report.md'}`",
        f"- cost report: `{out_dir / 'renovation_cost_report.md'}`",
        f"- run report: `{out_dir / 'finalize_requirements.report.json'}`",
        "",
        "## Room Corner Renders",
        "",
        "See `room_corner_renders.report.md` for four upper-corner views per room.",
        "",
        "## Steps",
        "",
        "| Step | Status | Seconds |",
        "|---|---|---:|",
    ]
    for step in report.get("steps") or []:
        lines.append(f"| {step.get('name')} | {step.get('status')} | {step.get('duration_sec', '')} |")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def process_apartment(
    apt_dir: Path,
    mode: str,
    args: argparse.Namespace,
    modules: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    out_dir = apt_dir / "apartment_pipeline" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    def requirements_step():
        result = modules["ensure"].process_apartment(apt_dir, mode)
        write_json(out_dir / "finalize_requirements.ensure_summary.json", {"root": str(apt_dir), "count": 1, "results": [result]})
        return result

    run_step("requirements postprocess", steps, requirements_step)

    room_reports: list[dict[str, Any]] = []
    for entry in room_entries(apt_dir):
        room_id = str(entry.get("room_id") or "")
        room_type = str(entry.get("room_type") or entry.get("prompt_room_type") or "")
        existing_blend = room_output_blend(apt_dir, room_id, room_type, mode)
        if args.rebuild_rooms or not existing_blend.is_file():
            room_reports.append(run_step(f"rebuild room {room_id}", steps, lambda entry=entry: rebuild_room(modules["builder"], project_root, apt_dir, mode, entry)))
        else:
            room_reports.append(
                {
                    "room_id": room_id,
                    "room_type": room_type,
                    "status": "reused_existing_room_blend",
                    "save_blend": str(existing_blend),
                    "scene_json": str(room_scene_path(apt_dir, room_id, mode)),
                }
            )

    apartment_scene = out_dir / "scene_apartment.requirements.v1.json"
    apartment_blend = out_dir / "scene_apartment.requirements.blend"
    overview_render = out_dir / "render_apartment.requirements.png"
    assemble_report = out_dir / "scene_apartment.requirements.build_report.json"
    run_step(
        "assemble apartment blend",
        steps,
        lambda: run_with_argv(
            modules["assemble"].main,
            [
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
            ],
        ),
    )

    cost_json = out_dir / "renovation_cost_report.json"
    cost_md = out_dir / "renovation_cost_report.md"
    run_step(
        "summarize renovation cost",
        steps,
        lambda: (
            cost_json.write_text(json.dumps(modules["cost"].summarize(apt_dir, mode), ensure_ascii=False, indent=2), encoding="utf-8"),
            modules["cost"].write_markdown(modules["cost"].summarize(apt_dir, mode), cost_md),
        ),
    )

    corner_dir = out_dir / "room_corner_renders"
    corner_json = out_dir / "room_corner_renders.report.json"
    corner_md = out_dir / "room_corner_renders.report.md"
    run_step(
        "render room corner views",
        steps,
        lambda: run_with_argv(
            modules["corner"].main,
            [
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
            ],
        ),
    )

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
    report_md = write_final_markdown(apt_dir, mode, report)
    report["outputs"]["run_report_md"] = str(report_md)
    write_json(report_json, report)
    return report


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize apartment requirements inside one top-level Blender process.")
    parser.add_argument("root", help="Apartment dir, project dir containing apt_* dirs, or output root.")
    parser.add_argument("--mode", default="optimal")
    parser.add_argument("--corner-width", type=int, default=960)
    parser.add_argument("--corner-height", type=int, default=720)
    parser.add_argument("--corner-samples", type=int, default=16)
    parser.add_argument("--overview-width", type=int, default=1400)
    parser.add_argument("--overview-height", type=int, default=1000)
    parser.add_argument("--overview-samples", type=int, default=16)
    parser.add_argument("--rebuild-rooms", action="store_true", help="Rebuild per-room requirements blends before assembling the apartment. Default reuses existing room blends for stability.")
    parser.add_argument("--out-summary", default=None)
    return parser


def main() -> None:
    args = build_cli().parse_args(argv_after_blender_separator())
    project_root = project_root_from_script()
    for candidate in (project_root, project_root / "src"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    modules = {
        "ensure": import_module_from_path("cgs_ensure_apartment_requirements", project_root / "src" / "tools" / "ensure_apartment_requirements.py"),
        "builder": import_module_from_path("cgs_blender_scene_builder", project_root / "src" / "Plasement" / "blender_scene_builder.py"),
        "assemble": import_module_from_path("cgs_assemble_apartment_blend", project_root / "src" / "tools" / "assemble_apartment_blend.py"),
        "cost": import_module_from_path("cgs_summarize_apartment_cost", project_root / "src" / "tools" / "summarize_apartment_cost.py"),
        "corner": import_module_from_path("cgs_render_apartment_room_corner_views", project_root / "src" / "tools" / "render_apartment_room_corner_views.py"),
    }
    root = Path(args.root).expanduser().resolve()
    apartments = iter_apartment_dirs(root)
    if not apartments:
        raise RuntimeError(f"No apartment folders with manifest.json and apartment.json found under {root}")
    results = [process_apartment(apt_dir, str(args.mode), args, modules, project_root) for apt_dir in apartments]
    summary_path = Path(args.out_summary).expanduser().resolve() if args.out_summary else root / "finalize_requirements.summary.json"
    write_json(summary_path, {"root": str(root), "mode": args.mode, "count": len(results), "results": results})
    print(json.dumps({"summary": str(summary_path), "apartments": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
