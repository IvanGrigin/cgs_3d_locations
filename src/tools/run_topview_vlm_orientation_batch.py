#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from topview_vlm_orientation_repair import (  # noqa: E402
    collect_scene_objects,
    filter_target_objects,
    is_chair_like,
    run_topview_vlm_orientation_repair,
    write_json,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_blender(raw: str | None) -> str:
    candidates = [
        raw or "",
        os.environ.get("BLENDER_PATH", ""),
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "blender",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("Blender executable not found")


def _run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def _scene_candidates(room_dir: Path, mode: str) -> list[Path]:
    pipe = room_dir / "pipeline" / mode
    return [
        pipe / "scene_requirements.v1.json",
        pipe / "scene_supplier.optimal.v1.flooring.v1.wall_material.v1.curtains.v1.json",
        pipe / "scene_supplier.optimal.v1.flooring.v1.wall_material.v1.json",
        pipe / "scene_supplier.optimal.v1.flooring.v1.json",
        pipe / "scene_supplier.optimal.v1.json",
        pipe / "scene.v1.json",
    ]


def discover_room_jobs(apartment_dir: Path, mode: str) -> list[dict[str, Any]]:
    manifest_path = apartment_dir / "manifest.json"
    rooms_root = apartment_dir / "rooms"
    jobs: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        for row in manifest.get("rooms", []) if isinstance(manifest, dict) else []:
            if not isinstance(row, dict):
                continue
            room_id = str(row.get("room_id") or "").strip()
            if not room_id:
                continue
            room_dir = rooms_root / room_id
            scene_path = next((p for p in _scene_candidates(room_dir, mode) if p.is_file()), None)
            blend_path = _room_blend_path(room_dir, room_id, mode)
            jobs.append(
                {
                    "room_id": room_id,
                    "room_type": row.get("room_type"),
                    "room_dir": str(room_dir.resolve()),
                    "scene_json": str(scene_path.resolve()) if scene_path else None,
                    "room_blend": str(blend_path.resolve()) if blend_path else None,
                }
            )
        return jobs

    for room_dir in sorted(rooms_root.glob("*")):
        if not room_dir.is_dir():
            continue
        scene_path = next((p for p in _scene_candidates(room_dir, mode) if p.is_file()), None)
        blend_path = _room_blend_path(room_dir, room_dir.name, mode)
        jobs.append(
            {
                "room_id": room_dir.name,
                "room_type": None,
                "room_dir": str(room_dir.resolve()),
                "scene_json": str(scene_path.resolve()) if scene_path else None,
                "room_blend": str(blend_path.resolve()) if blend_path else None,
            }
        )
    return jobs


def _room_blend_path(room_dir: Path, room_id: str, mode: str) -> Path | None:
    candidates = [
        room_dir / "pipeline" / mode / "scene_infinigen_clean_supplier.requirements.blend",
        room_dir / "pipeline" / mode / "scene_kitchen_requirements.blend",
        room_dir / "pipeline" / mode / "scene_infinigen_clean_supplier.optimal.memfix.blend",
        room_dir / "pipeline" / mode / "scene_infinigen_clean_supplier.optimal.blend",
        room_dir / "kitchen" / f"{room_id}.blend",
    ]
    return next((path for path in candidates if path.is_file()), None)


def target_ids_for_scene(scene_path: Path, *, scope: str, include_armchairs: bool, max_objects: int) -> list[str]:
    scene = load_json(scene_path)
    refs = collect_scene_objects(scene, max_objects=max_objects)
    targets = filter_target_objects(refs, scope=scope, include_armchairs=include_armchairs)
    return [ref.object_id for ref in targets]


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _item_position_xy(item: dict[str, Any]) -> tuple[float | None, float | None]:
    value = item.get("position_m") or item.get("position") or item.get("center")
    if isinstance(value, list) and len(value) >= 2:
        return _as_float(value[0]), _as_float(value[1])
    if isinstance(value, dict):
        return _as_float(value.get("x")), _as_float(value.get("y"))
    return _as_float(item.get("x")), _as_float(item.get("y"))


def _item_size_xy(item: dict[str, Any]) -> tuple[float, float]:
    value = item.get("size_m") or item.get("dimensions_m") or item.get("size")
    if isinstance(value, list) and len(value) >= 2:
        sx = _as_float(value[0]) or 0.35
        sy = _as_float(value[1]) or 0.35
        return max(abs(sx), 0.08), max(abs(sy), 0.08)
    if isinstance(value, dict):
        sx = _as_float(value.get("x") or value.get("width")) or 0.35
        sy = _as_float(value.get("y") or value.get("depth")) or 0.35
        return max(abs(sx), 0.08), max(abs(sy), 0.08)
    return max(abs(_as_float(item.get("width_m") or item.get("width")) or 0.35), 0.08), max(abs(_as_float(item.get("depth_m") or item.get("depth")) or 0.35), 0.08)


def _item_yaw_deg(item: dict[str, Any]) -> float:
    value = _as_float(item.get("yaw_deg"))
    if value is None:
        value = _as_float(item.get("rotation_deg"))
    if value is None:
        rad = _as_float(item.get("yaw_rad"))
        value = rad * 180.0 / 3.141592653589793 if rad is not None else 0.0
    return float(value or 0.0) % 360.0


def _room_polygon(scene: dict[str, Any]) -> list[tuple[float, float]]:
    room = scene.get("room") if isinstance(scene.get("room"), dict) else {}
    poly = room.get("floor_polygon") or room.get("floor_polygon_xz") or []
    out: list[tuple[float, float]] = []
    if isinstance(poly, list):
        for point in poly:
            if isinstance(point, dict):
                x = _as_float(point.get("x"))
                y = _as_float(point.get("y", point.get("z")))
            elif isinstance(point, list) and len(point) >= 2:
                x = _as_float(point[0])
                y = _as_float(point[1])
            else:
                x = y = None
            if x is not None and y is not None:
                out.append((x, y))
    return out


def _semantic_color(item: dict[str, Any], target: bool) -> tuple[int, int, int, int]:
    if target:
        return (230, 70, 55, 190)
    text = " ".join(str(item.get(k) or "") for k in ("id", "category", "name", "semantic_group")).lower()
    if "table" in text or "desk" in text or "стол" in text:
        return (80, 130, 220, 155)
    if "bed" in text or "кровать" in text:
        return (120, 100, 210, 150)
    if "cabinet" in text or "wardrobe" in text or "шкаф" in text or "тумб" in text:
        return (120, 150, 120, 150)
    return (150, 150, 150, 115)


def render_schematic_topview(
    *,
    scene_path: Path,
    iteration_dir: Path,
    target_ids: list[str],
    resolution_x: int,
    resolution_y: int,
) -> dict[str, Any]:
    scene = load_json(scene_path)
    placements = scene.get("placements") if isinstance(scene.get("placements"), list) else scene.get("items")
    if not isinstance(placements, list):
        placements = []

    room_poly = _room_polygon(scene)
    points: list[tuple[float, float]] = list(room_poly)
    item_rows: list[tuple[dict[str, Any], tuple[float, float], tuple[float, float], float, list[tuple[float, float]]]] = []
    for item in placements:
        if not isinstance(item, dict):
            continue
        cx, cy = _item_position_xy(item)
        if cx is None or cy is None:
            continue
        sx, sy = _item_size_xy(item)
        yaw = _item_yaw_deg(item)
        rad = yaw * 3.141592653589793 / 180.0
        c = math.cos(rad)
        s = math.sin(rad)
        local = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
        corners = [(cx + dx * c - dy * s, cy + dx * s + dy * c) for dx, dy in local]
        item_rows.append((item, (cx, cy), (sx, sy), yaw, corners))
        points.extend(corners)
    if not points:
        points = [(-2, -2), (2, 2)]

    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    pad = 0.6
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    margin = 64
    scale = min((resolution_x - 2 * margin) / span_x, (resolution_y - 2 * margin) / span_y)

    def to_px(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return margin + (x - min_x) * scale, margin + (max_y - y) * scale

    image = Image.new("RGBA", (resolution_x, resolution_y), (250, 250, 247, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
        small_font = ImageFont.truetype("Arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    if room_poly:
        draw.polygon([to_px(p) for p in room_poly], fill=(255, 255, 255, 255), outline=(35, 35, 35, 255))

    # Draw non-targets first, target chairs last.
    ordered = sorted(item_rows, key=lambda row: 1 if str(row[0].get("id") or "") in target_ids else 0)
    for item, center, size, yaw, corners in ordered:
        item_id = str(item.get("id") or item.get("object_id") or item.get("name") or "")
        is_target = item_id in target_ids
        poly = [to_px(p) for p in corners]
        draw.polygon(poly, fill=_semantic_color(item, is_target), outline=(150, 30, 20, 255) if is_target else (100, 100, 100, 180))
        cx, cy = to_px(center)
        rad = yaw * 3.141592653589793 / 180.0
        arrow_len = max(min(size) * scale * 0.45, 14.0)
        end = (cx + math.cos(rad) * arrow_len, cy - math.sin(rad) * arrow_len)
        if is_target or is_chair_like(collect_scene_objects({"placements": [item]}, max_objects=1)[0], include_armchairs=True):
            draw.line([(cx, cy), end], fill=(0, 0, 0, 255), width=3 if is_target else 2)
            r = 4 if is_target else 3
            draw.ellipse((end[0] - r, end[1] - r, end[0] + r, end[1] + r), fill=(0, 0, 0, 255))
        if is_target:
            draw.text((cx + 5, cy + 5), item_id, fill=(0, 0, 0, 255), font=font)

    legend = [
        "Top-view schematic for VLM orientation repair",
        "Red = target chair(s); blue = tables/desks; arrow = current facing direction",
        "Return rotate_clockwise_deg: 0/90/180/270 only",
    ]
    y = 12
    for line in legend:
        draw.text((12, y), line, fill=(0, 0, 0, 255), font=small_font)
        y += 18

    out = iteration_dir / "topview.png"
    image.convert("RGB").save(out)
    log = iteration_dir / "render_topview.log"
    log.write_text(
        json.dumps(
            {
                "renderer": "schematic",
                "scene": str(scene_path),
                "topview_png": str(out.resolve()),
                "target_ids": target_ids,
                "object_count": len(item_rows),
                "bounds": [min_x, min_y, max_x, max_y],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "topview_png": str(out.resolve()),
        "build_log": None,
        "render_log": str(log.resolve()),
        "build_report": None,
        "inspection_blend": None,
        "renderer": "schematic",
    }


def render_topview(
    *,
    scene_path: Path,
    room_blend: Path | None,
    iteration_dir: Path,
    blender: str,
    target_ids: list[str],
    target_scope: str,
    include_armchairs: bool,
    resolution_x: int,
    resolution_y: int,
    elevation_deg: float,
    radius_mult: float,
    lens: float,
    keep_blends: bool,
    renderer: str,
) -> dict[str, Any]:
    if renderer == "schematic":
        return render_schematic_topview(
            scene_path=scene_path,
            iteration_dir=iteration_dir,
            target_ids=target_ids,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
        )

    if renderer == "room_blend":
        if room_blend is None or not room_blend.is_file():
            raise FileNotFoundError(f"room blend not found for real top-view render: {room_blend}")
        topview_png = iteration_dir / "topview.png"
        orientation_report = iteration_dir / "blend_orientation_apply_report.json"
        save_blend = iteration_dir / "oriented_room.blend"
        render_cmd = [
            blender,
            str(room_blend.resolve()),
            "-b",
            "--python",
            str((SRC_ROOT / "tools" / "render_saved_blend_top_view.py").resolve()),
            "--",
            "--out",
            str(topview_png.resolve()),
            "--scene-json",
            str(scene_path.resolve()),
            "--target-ids",
            ",".join(target_ids),
            "--target-scope",
            str(target_scope),
        ]
        if include_armchairs:
            render_cmd.append("--include-armchairs")
        render_cmd += [
            "--apply-scene-orientations",
            "--highlight-targets",
            "--orientation-report",
            str(orientation_report.resolve()),
            "--azimuth-deg",
            "-90.0",
            "--elevation-deg",
            str(float(elevation_deg)),
            "--radius-mult",
            str(float(radius_mult)),
            "--lens",
            str(float(lens)),
            "--resolution-x",
            str(int(resolution_x)),
            "--resolution-y",
            str(int(resolution_y)),
        ]
        if keep_blends:
            render_cmd += ["--save-blend", str(save_blend.resolve())]
        _run_logged(render_cmd, iteration_dir / "render_topview.log")
        return {
            "topview_png": str(topview_png.resolve()),
            "build_log": None,
            "render_log": str((iteration_dir / "render_topview.log").resolve()),
            "build_report": None,
            "inspection_blend": str(save_blend.resolve()) if save_blend.is_file() else None,
            "blend_orientation_report": str(orientation_report.resolve()) if orientation_report.is_file() else None,
            "renderer": "room_blend",
            "source_room_blend": str(room_blend.resolve()),
        }

    blend_path = iteration_dir / "inspection.blend"
    build_report = iteration_dir / "inspection.build_report.json"
    topview_png = iteration_dir / "topview.png"

    build_cmd = [
        sys.executable,
        str((SRC_ROOT / "Plasement" / "BlenderVisualizePlacement.py").resolve()),
        "--json",
        str(scene_path.resolve()),
        "--blender",
        blender,
        "--background",
        "--bbox-fallback",
        "--save-blend",
        str(blend_path.resolve()),
        "--build-report",
        str(build_report.resolve()),
        "--no-pack-assets",
    ]
    if target_ids:
        build_cmd += ["--highlight-item-ids", ",".join(target_ids)]
    _run_logged(build_cmd, iteration_dir / "build_blend.log")

    render_cmd = [
        blender,
        str(blend_path.resolve()),
        "-b",
        "--python",
        str((SRC_ROOT / "tools" / "render_saved_blend_top_view.py").resolve()),
        "--",
        "--out",
        str(topview_png.resolve()),
        "--azimuth-deg",
        "-90.0",
        "--elevation-deg",
        str(float(elevation_deg)),
        "--radius-mult",
        str(float(radius_mult)),
        "--lens",
        str(float(lens)),
        "--resolution-x",
        str(int(resolution_x)),
        "--resolution-y",
        str(int(resolution_y)),
    ]
    _run_logged(render_cmd, iteration_dir / "render_topview.log")

    if not keep_blends:
        with contextlib.suppress(Exception):
            blend_path.unlink(missing_ok=True)

    return {
        "topview_png": str(topview_png.resolve()),
        "build_log": str((iteration_dir / "build_blend.log").resolve()),
        "render_log": str((iteration_dir / "render_topview.log").resolve()),
        "build_report": str(build_report.resolve()) if build_report.is_file() else None,
        "inspection_blend": str(blend_path.resolve()) if blend_path.is_file() else None,
    }


def run_room(
    *,
    job: dict[str, Any],
    out_root: Path,
    args: argparse.Namespace,
    blender: str,
) -> dict[str, Any]:
    room_id = str(job["room_id"])
    room_out = out_root / room_id
    room_out.mkdir(parents=True, exist_ok=True)

    scene_raw = job.get("scene_json")
    if not scene_raw:
        return {
            "room_id": room_id,
            "room_type": job.get("room_type"),
            "status": "skipped",
            "reason": "scene_json_missing",
        }

    current_scene = Path(str(scene_raw)).expanduser().resolve()
    iterations: list[dict[str, Any]] = []
    status = "ok"
    error: str | None = None

    for idx in range(int(args.max_iterations)):
        iteration_dir = room_out / f"iter_{idx:02d}"
        if iteration_dir.exists():
            shutil.rmtree(iteration_dir)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_scene, iteration_dir / "scene.input.v1.json")

        try:
            target_ids = target_ids_for_scene(
                current_scene,
                scope=args.scope,
                include_armchairs=bool(args.include_armchairs),
                max_objects=int(args.max_objects),
            )
            render_info = render_topview(
                scene_path=current_scene,
                room_blend=Path(str(job["room_blend"])).expanduser().resolve() if job.get("room_blend") else None,
                iteration_dir=iteration_dir,
                blender=blender,
                target_ids=target_ids,
                target_scope=str(args.scope),
                include_armchairs=bool(args.include_armchairs),
                resolution_x=int(args.resolution_x),
                resolution_y=int(args.resolution_y),
                elevation_deg=float(args.elevation_deg),
                radius_mult=float(args.radius_mult),
                lens=float(args.lens),
                keep_blends=bool(args.keep_blends),
                renderer=str(args.renderer),
            )

            out_scene = iteration_dir / "scene.output.v1.json"
            out_review = iteration_dir / "vlm_review.json"
            out_report = iteration_dir / "orientation_report.json"
            out_prompt = iteration_dir / "vlm_prompt.txt"
            vlm_log = iteration_dir / "vlm_apply.log"
            with vlm_log.open("w", encoding="utf-8") as log:
                log.write(f"room_id={room_id}\niteration={idx}\nscene={current_scene}\ntopview={render_info['topview_png']}\n")
                log.flush()
                report = run_topview_vlm_orientation_repair(
                    scene_path=current_scene,
                    image_path=Path(str(render_info["topview_png"])),
                    out_scene_path=out_scene,
                    out_review_path=out_review,
                    out_report_path=out_report,
                    out_prompt_path=out_prompt,
                    provider=str(args.provider),
                    model=str(args.model or "").strip() or None,
                    max_objects=int(args.max_objects),
                    target_scope=str(args.scope),
                    include_armchairs=bool(args.include_armchairs),
                    min_confidence=float(args.min_confidence),
                    max_delta_deg=float(args.max_delta_deg),
                    snap_step_deg=float(args.snap_step_deg),
                    review_json_path=None,
                    apply=not bool(args.no_apply),
                )
                log.write(json.dumps(report.get("counts", {}), ensure_ascii=False, indent=2) + "\n")

            counts = report.get("counts") or {}
            iteration_info = {
                "iteration": idx,
                "input_scene": str(current_scene),
                "output_scene": str(out_scene.resolve()),
                "target_ids": target_ids,
                "review_json": str(out_review.resolve()),
                "report_json": str(out_report.resolve()),
                "prompt_txt": str(out_prompt.resolve()),
                "vlm_log": str(vlm_log.resolve()),
                "counts": counts,
                **render_info,
            }
            iterations.append(iteration_info)
            current_scene = out_scene.resolve()

            if int(counts.get("applied", 0) or 0) <= 0:
                break
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            (iteration_dir / "error.log").write_text(error + "\n", encoding="utf-8")
            break

    summary = {
        "room_id": room_id,
        "room_type": job.get("room_type"),
        "status": status,
        "error": error,
        "input_scene": str(Path(str(scene_raw)).expanduser().resolve()),
        "final_scene": str(current_scene),
        "iterations": iterations,
    }
    write_json(room_out / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative top-view VLM orientation repair batch for apartment rooms.")
    parser.add_argument("--apartment-dir", required=True, type=Path)
    parser.add_argument("--mode", default="optimal")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--provider", choices=["none", "openai", "openrouter", "ollama"], default="ollama")
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--scope", choices=["chairs", "all"], default="chairs")
    parser.add_argument("--include-armchairs", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=10000)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--max-delta-deg", type=float, default=180.0)
    parser.add_argument("--snap-step-deg", type=float, default=90.0)
    parser.add_argument("--resolution-x", type=int, default=1400)
    parser.add_argument("--resolution-y", type=int, default=1050)
    parser.add_argument("--elevation-deg", type=float, default=80.0)
    parser.add_argument("--radius-mult", type=float, default=0.55)
    parser.add_argument("--lens", type=float, default=32.0)
    parser.add_argument("--blender", default=None)
    parser.add_argument("--renderer", choices=["schematic", "blender", "room_blend"], default="room_blend")
    parser.add_argument("--keep-blends", action="store_true")
    parser.add_argument("--no-apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    apartment_dir = args.apartment_dir.expanduser().resolve()
    out_root = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else apartment_dir / "apartment_pipeline" / str(args.mode) / "topview_vlm_orientation_by_room"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    blender = _resolve_blender(args.blender)

    started = time.time()
    jobs = discover_room_jobs(apartment_dir, str(args.mode))
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        print(f"[topview-vlm] room {job.get('room_id')} ({job.get('room_type')})")
        summaries.append(run_room(job=job, out_root=out_root, args=args, blender=blender))

    manifest = {
        "stage": "topview_vlm_orientation_batch",
        "apartment_dir": str(apartment_dir),
        "out_dir": str(out_root),
        "provider": args.provider,
        "model": args.model,
        "scope": args.scope,
        "include_armchairs": bool(args.include_armchairs),
        "max_iterations": int(args.max_iterations),
        "started_at_unix": int(started),
        "finished_at_unix": int(time.time()),
        "rooms": summaries,
    }
    write_json(out_root / "batch_manifest.json", manifest)
    print(json.dumps({"rooms": len(summaries), "out_dir": str(out_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
