#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/blend_to_orbit_gif.py

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


BLENDER_HELPER_CODE = r'''
import bpy
import json
import math
import os
from mathutils import Vector


def parse_args():
    argv = bpy.app.driver_namespace.get("orbit_gif_args")
    if not argv:
        raise RuntimeError("orbit_gif_args not found in driver_namespace")
    return argv


def world_bbox_points(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def collect_scene_bounds():
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    if not mesh_objects:
        raise RuntimeError("В сцене нет renderable MESH-объектов")

    pts = []
    for obj in mesh_objects:
        pts.extend(world_bbox_points(obj))

    min_x = min(p.x for p in pts)
    max_x = max(p.x for p in pts)
    min_y = min(p.y for p in pts)
    max_y = max(p.y for p in pts)
    min_z = min(p.z for p in pts)
    max_z = max(p.z for p in pts)

    center = Vector((
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
        0.5 * (min_z + max_z),
    ))

    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    return {
        "center": center,
        "min_z": min_z,
        "max_z": max_z,
        "radius_xy": max(size_x, size_y) * 0.5,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
    }


def ensure_camera(scene):
    if scene.camera and scene.camera.type == "CAMERA":
        return scene.camera

    cam_data = bpy.data.cameras.new("OrbitGifCamera")
    cam_obj = bpy.data.objects.new("OrbitGifCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def ensure_track_to(camera, target):
    for c in camera.constraints:
        if c.type == "TRACK_TO" and c.target == target:
            return c

    c = camera.constraints.new(type="TRACK_TO")
    c.target = target
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'
    return c


def ensure_target_object(scene, center):
    target = bpy.data.objects.get("OrbitGifTarget")
    if target is None:
        target = bpy.data.objects.new("OrbitGifTarget", None)
        scene.collection.objects.link(target)
    target.location = center
    return target


def setup_render(scene, width, height, samples):
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    if scene.render.engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    elif scene.render.engine == 'BLENDER_EEVEE':
        # Для EEVEE samples задаются иначе; оставляем дефолт.
        pass


def compute_camera_distance(cam_obj, bounds, pitch_deg, margin):
    """
    Логика:
    - камера всегда смотрит в центр комнаты;
    - расстояние выбираем так, чтобы в кадр входил радиус комнаты по XY;
    - при больших углах сверху дополнительно учитываем высоту сцены.
    """
    cam = cam_obj.data
    angle = cam.angle if hasattr(cam, "angle") and cam.angle > 1e-6 else math.radians(50.0)
    half_fov = max(angle * 0.5, math.radians(10.0))

    radius_xy = max(bounds["radius_xy"], 0.5)
    size_z = max(bounds["size_z"], 0.5)

    dist_xy = radius_xy / math.tan(half_fov)
    dist_z = (0.5 * size_z) / math.tan(half_fov)

    pitch_rad = math.radians(pitch_deg)
    correction = 1.0 + 0.25 * abs(math.sin(pitch_rad))

    return max(dist_xy, dist_z) * margin * correction + 0.5 * max(bounds["size_x"], bounds["size_y"], bounds["size_z"])


def place_camera(camera, center, distance, yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Сферические координаты вокруг центра.
    # pitch=0 -> камера в горизонтальной плоскости
    dx = distance * math.cos(pitch) * math.cos(yaw)
    dy = distance * math.cos(pitch) * math.sin(yaw)
    dz = distance * math.sin(pitch)

    camera.location = Vector((center.x + dx, center.y + dy, center.z + dz))


def render_orbit():
    args = parse_args()
    scene = bpy.context.scene

    frames_dir = args["frames_dir"]
    width = int(args["width"])
    height = int(args["height"])
    samples = int(args["samples"])
    margin = float(args["margin"])
    yaw_step = float(args["yaw_step"])
    elevations_deg = [float(x) for x in args["elevations_deg"]]

    os.makedirs(frames_dir, exist_ok=True)

    bounds = collect_scene_bounds()
    center = bounds["center"]

    # Немного поднимаем target к геометрическому центру по высоте сцены,
    # чтобы верхние пролёты смотрели именно в комнату, а не в пол.
    target_center = Vector((center.x, center.y, center.z))

    cam_obj = ensure_camera(scene)
    target = ensure_target_object(scene, target_center)
    ensure_track_to(cam_obj, target)

    setup_render(scene, width, height, samples)

    frame_idx = 0
    for pitch_deg in elevations_deg:
        distance = compute_camera_distance(cam_obj, bounds, pitch_deg, margin)
        yaw = 0.0
        while yaw < 360.0 - 1e-9:
            place_camera(cam_obj, target_center, distance, yaw, pitch_deg)
            scene.camera = cam_obj
            out_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
            scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            print(f"[orbit_gif] rendered {out_path}")
            frame_idx += 1
            yaw += yaw_step

    print(json.dumps({
        "frames_dir": frames_dir,
        "frame_count": frame_idx,
    }, ensure_ascii=False))
    return frame_idx


render_orbit()
'''


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Открывает .blend в Blender background mode, рендерит 36 кадров облёта "
            "по трём орбитам и собирает GIF рядом с исходным .blend."
        )
    )
    p.add_argument("--blend", required=True, help="Путь к .blend")
    p.add_argument("--blender", default=None, help="Путь к бинарю Blender")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--samples", type=int, default=32, help="Cycles samples, если сцена на Cycles")
    p.add_argument("--yaw-step", type=float, default=30.0, help="Шаг по азимуту в градусах")
    p.add_argument(
        "--elevations",
        default="0,35,72",
        help="Три угла возвышения камеры в градусах, например: 0,35,72",
    )
    p.add_argument("--duration-ms", type=int, default=500, help="Длительность одного кадра в GIF")
    p.add_argument("--margin", type=float, default=1.35, help="Запас дистанции камеры")
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Не удалять промежуточные PNG-кадры",
    )
    return p


def resolve_blender_binary(blender_arg: str | None) -> str:
    candidates = []
    if blender_arg:
        candidates.append(Path(blender_arg).expanduser())

    candidates.extend([
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path(shutil.which("blender") or ""),
    ])

    for c in candidates:
        if c and str(c) and c.exists():
            return str(c.resolve())

    raise RuntimeError("Не найден Blender. Передай --blender /path/to/Blender")


def parse_elevations(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if len(vals) != 3:
        raise RuntimeError("Нужно передать ровно 3 угла возвышения, например --elevations 0,35,72")
    return vals


def run_blender_render(
    blender_bin: str,
    blend_path: Path,
    frames_dir: Path,
    width: int,
    height: int,
    samples: int,
    yaw_step: float,
    elevations_deg: list[float],
    margin: float,
) -> None:
    with tempfile.TemporaryDirectory(prefix="blend_orbit_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        helper_script = tmpdir_path / "orbit_helper.py"
        helper_script.write_text(BLENDER_HELPER_CODE, encoding="utf-8")

        args_json = tmpdir_path / "orbit_args.json"
        args_json.write_text(
            json.dumps(
                {
                    "frames_dir": str(frames_dir.resolve()),
                    "width": width,
                    "height": height,
                    "samples": samples,
                    "yaw_step": yaw_step,
                    "elevations_deg": elevations_deg,
                    "margin": margin,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        bootstrap = tmpdir_path / "bootstrap.py"
        bootstrap.write_text(
            f"""
import json
from pathlib import Path
import bpy

args = json.loads(Path(r"{args_json}").read_text(encoding="utf-8"))
bpy.app.driver_namespace["orbit_gif_args"] = args

exec(Path(r"{helper_script}").read_text(encoding="utf-8"), {{}})
""".strip(),
            encoding="utf-8",
        )

        cmd = [
            blender_bin,
            "-b",
            str(blend_path.resolve()),
            "--python",
            str(bootstrap.resolve()),
        ]

        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def build_gif_from_frames(frames_dir: Path, gif_path: Path, duration_ms: int) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Для сборки GIF нужен Pillow: pip install pillow") from e

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"Не найдено кадров в {frames_dir}")

    images = [Image.open(p).convert("RGBA") for p in frames]
    first, rest = images[0], images[1:]

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(
        gif_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        disposal=2,
    )

    for img in images:
        img.close()


def main() -> None:
    args = build_cli().parse_args()

    blend_path = Path(args.blend).expanduser().resolve()
    if not blend_path.is_file():
        raise RuntimeError(f"Не найден .blend: {blend_path}")

    blender_bin = resolve_blender_binary(args.blender)
    elevations_deg = parse_elevations(args.elevations)

    frames_expected = int(round(360.0 / float(args.yaw_step))) * len(elevations_deg)
    if frames_expected != 36:
        print(
            f"⚠️ Предупреждение: при yaw_step={args.yaw_step} и elevations={elevations_deg} "
            f"получится {frames_expected} кадров, а не 36."
        )

    base_dir = blend_path.parent
    stem = blend_path.stem
    frames_dir = base_dir / f"{stem}_gif_frames"
    gif_path = base_dir / f"{stem}.gif"

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    run_blender_render(
        blender_bin=blender_bin,
        blend_path=blend_path,
        frames_dir=frames_dir,
        width=int(args.width),
        height=int(args.height),
        samples=int(args.samples),
        yaw_step=float(args.yaw_step),
        elevations_deg=elevations_deg,
        margin=float(args.margin),
    )

    build_gif_from_frames(
        frames_dir=frames_dir,
        gif_path=gif_path,
        duration_ms=int(args.duration_ms),
    )

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"OK: GIF saved to {gif_path}")


if __name__ == "__main__":
    main()