#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
debug_asset_textures.py

Диагностика и принудительная починка текстур ассета 3D-FUTURE.

Что умеет:
1. Принимает путь к папке ассета.
2. Ищет OBJ / MTL / изображения.
3. Читает mtllib из OBJ.
4. Парсит MTL-карты:
   - map_Kd
   - map_Ka
   - map_Ks
   - map_d
   - bump / map_Bump
5. Проверяет существование файлов.
6. Выбирает лучший diffuse/basecolor.
7. По желанию создаёт:
   - fixed_model.mtl
   - fixed_<obj_name>.obj
8. Пишет texture_report.json
9. По желанию запускает Blender и создаёт .blend, где материал
   принудительно собирается через узлы:
   - map_Kd -> Base Color
   - map_d  -> Alpha
   - bump   -> Normal Map
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".webp", ".exr"}

MAP_KEYS = {
    "map_kd": "diffuse",
    "map_ka": "ambient",
    "map_ks": "specular",
    "map_d": "opacity",
    "bump": "bump",
    "map_bump": "bump",
}

OBJ_MTLLIB_RE = re.compile(r"^\s*mtllib\s+(.*)$", re.IGNORECASE)
MTL_NEWMTL_RE = re.compile(r"^\s*newmtl\s+(.*)$", re.IGNORECASE)
MTL_MAP_RE = re.compile(r"^\s*(map_Kd|map_Ka|map_Ks|map_d|bump|map_Bump)\s+(.*)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-dir", required=True, help="Путь к папке ассета")
    ap.add_argument("--obj-name", default=None, help="Какой OBJ использовать, если их несколько")
    ap.add_argument("--write-fixed", action="store_true", help="Создать fixed_model.mtl и fixed_<obj>.obj")
    ap.add_argument("--prefer", default=None, help="Явно предпочесть конкретный файл текстуры, например texture.png")

    ap.add_argument("--blender", default="/Applications/Blender.app/Contents/MacOS/Blender",
                    help="Путь к Blender")
    ap.add_argument("--test-blender", action="store_true",
                    help="Импортировать OBJ в Blender и принудительно собрать материалы через ноды")
    ap.add_argument("--save-blend", default=None,
                    help="Куда сохранить .blend. По умолчанию: <asset-dir>/debug_textured.blend")
    ap.add_argument("--render", default=None,
                    help="Куда сохранить render PNG. По умолчанию рендер не делается")
    return ap.parse_args()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def list_images(asset_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(asset_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def list_objs(asset_dir: Path) -> List[Path]:
    return sorted([p for p in asset_dir.iterdir() if p.is_file() and p.suffix.lower() == ".obj"])


def list_mtls(asset_dir: Path) -> List[Path]:
    return sorted([p for p in asset_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mtl"])


def parse_obj_mtllibs(obj_path: Path) -> List[str]:
    result: List[str] = []
    text = _read_text(obj_path)
    for line in text.splitlines():
        m = OBJ_MTLLIB_RE.match(line)
        if not m:
            continue
        tail = m.group(1).strip()
        for token in tail.split():
            token = token.strip().strip('"').strip("'")
            if token:
                result.append(token)

    uniq: List[str] = []
    seen = set()
    for x in result:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _strip_mtl_value(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw

    tokens = raw.split()
    if len(tokens) == 1:
        return tokens[0].strip('"').strip("'")

    return tokens[-1].strip('"').strip("'")


def parse_mtl(mtl_path: Path) -> Dict[str, Dict[str, str]]:
    materials: Dict[str, Dict[str, str]] = {}
    current = "__default__"
    materials[current] = {}

    text = _read_text(mtl_path)
    for line in text.splitlines():
        m_new = MTL_NEWMTL_RE.match(line)
        if m_new:
            current = m_new.group(1).strip() or "__unnamed__"
            materials.setdefault(current, {})
            continue

        m_map = MTL_MAP_RE.match(line)
        if not m_map:
            continue

        raw_key = m_map.group(1).lower()
        raw_val = _strip_mtl_value(m_map.group(2))
        logical_key = MAP_KEYS.get(raw_key, raw_key)
        materials.setdefault(current, {})
        materials[current][logical_key] = raw_val

    return materials


def resolve_texture_ref(asset_dir: Path, ref: str) -> Tuple[Optional[Path], str]:
    if not ref:
        return None, "empty"

    s = ref.strip().strip('"').strip("'")
    if not s:
        return None, "empty"

    p = Path(s)
    if p.is_absolute():
        if p.is_file():
            return p, "absolute_exists"
        return None, "absolute_missing"

    candidate = (asset_dir / p).resolve()
    if candidate.is_file():
        return candidate, "relative_exists"

    by_name = (asset_dir / p.name).resolve()
    if by_name.is_file():
        return by_name, "basename_exists"

    return None, "missing"


def choose_best_diffuse(images: List[Path], prefer: Optional[str]) -> Optional[Path]:
    if not images:
        return None

    if prefer:
        prefer_l = prefer.strip().lower()
        for p in images:
            if p.name.lower() == prefer_l:
                return p

    def score(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        s = 0

        if name == "texture.png":
            s += 1000
        if name == "texture.jpg":
            s += 950
        if "albedo" in name or "basecolor" in name or "diffuse" in name:
            s += 700
        if "color" in name:
            s += 400
        if name.startswith("image"):
            s += 200

        if p.suffix.lower() == ".png":
            s += 50

        try:
            size = p.stat().st_size
        except Exception:
            size = 0

        return (s, size, p.name.lower())

    return max(images, key=score)


def build_report(
    asset_dir: Path,
    obj_path: Optional[Path],
    mtl_paths: List[Path],
    images: List[Path],
    prefer: Optional[str],
) -> Dict:
    mtllibs: List[str] = parse_obj_mtllibs(obj_path) if obj_path else []

    parsed_mtls = {}
    for mtl in mtl_paths:
        parsed_mtls[mtl.name] = parse_mtl(mtl)

    resolved = {}
    for mtl_name, mats in parsed_mtls.items():
        resolved[mtl_name] = {}
        for mat_name, maps in mats.items():
            resolved[mtl_name][mat_name] = {}
            for key, ref in maps.items():
                rp, status = resolve_texture_ref(asset_dir, ref)
                resolved[mtl_name][mat_name][key] = {
                    "ref": ref,
                    "resolved_path": str(rp) if rp else None,
                    "status": status,
                }

    best_diffuse = choose_best_diffuse(images, prefer)

    report = {
        "asset_dir": str(asset_dir),
        "obj": str(obj_path) if obj_path else None,
        "obj_mtllibs": mtllibs,
        "mtl_files": [str(x) for x in mtl_paths],
        "image_files": [str(x) for x in images],
        "best_diffuse_candidate": str(best_diffuse) if best_diffuse else None,
        "parsed_mtls": parsed_mtls,
        "resolved_maps": resolved,
    }
    return report


def choose_obj(asset_dir: Path, obj_name: Optional[str]) -> Optional[Path]:
    objs = list_objs(asset_dir)
    if not objs:
        return None

    if obj_name:
        for p in objs:
            if p.name == obj_name:
                return p
        raise RuntimeError(f"OBJ {obj_name!r} не найден в {asset_dir}")

    for preferred in ("normalized_model.obj", "raw_model.obj"):
        for p in objs:
            if p.name == preferred:
                return p

    return objs[0]


def choose_main_mtl(asset_dir: Path, obj_path: Optional[Path]) -> Optional[Path]:
    mtls = list_mtls(asset_dir)
    if not mtls:
        return None

    if obj_path:
        refs = parse_obj_mtllibs(obj_path)
        for ref in refs:
            cand = (asset_dir / ref).resolve()
            if cand.is_file():
                return cand
            for m in mtls:
                if m.name == Path(ref).name:
                    return m

    for preferred in ("model.mtl", "normalized_model.mtl"):
        for m in mtls:
            if m.name == preferred:
                return m

    return mtls[0]


def build_fixed_mtl_text(original_text: str, best_diffuse_name: Optional[str]) -> str:
    if not best_diffuse_name:
        return original_text

    lines = original_text.splitlines()
    out: List[str] = []

    current_material_started = False
    current_material_has_map_kd = False

    def flush_material_if_needed():
        nonlocal current_material_started, current_material_has_map_kd
        if current_material_started and not current_material_has_map_kd:
            out.append(f"map_Kd {best_diffuse_name}")
        current_material_started = False
        current_material_has_map_kd = False

    for line in lines:
        m_new = MTL_NEWMTL_RE.match(line)
        if m_new:
            flush_material_if_needed()
            out.append(line)
            current_material_started = True
            continue

        m_map = MTL_MAP_RE.match(line)
        if m_map:
            key = m_map.group(1).lower()
            if key == "map_kd":
                out.append(f"map_Kd {best_diffuse_name}")
                current_material_has_map_kd = True
                continue

        out.append(line)

    flush_material_if_needed()

    if not any(MTL_NEWMTL_RE.match(line) for line in lines):
        return f"newmtl material_0\nmap_Kd {best_diffuse_name}\n" + original_text

    return "\n".join(out) + "\n"


def build_fixed_obj_text(original_text: str, fixed_mtl_name: str) -> str:
    lines = original_text.splitlines()
    out: List[str] = []
    replaced = False

    for line in lines:
        if OBJ_MTLLIB_RE.match(line):
            if not replaced:
                out.append(f"mtllib {fixed_mtl_name}")
                replaced = True
            continue
        out.append(line)

    if not replaced:
        out.insert(0, f"mtllib {fixed_mtl_name}")

    return "\n".join(out) + "\n"


def write_fixed_files(asset_dir: Path, obj_path: Path, main_mtl: Optional[Path], best_diffuse: Optional[Path]) -> Dict[str, Optional[str]]:
    result = {
        "fixed_mtl": None,
        "fixed_obj": None,
    }

    if not best_diffuse:
        return result

    fixed_mtl_name = "fixed_model.mtl"
    fixed_mtl_path = asset_dir / fixed_mtl_name

    if main_mtl and main_mtl.is_file():
        original_mtl_text = _read_text(main_mtl)
    else:
        original_mtl_text = "newmtl material_0\n"

    fixed_mtl_text = build_fixed_mtl_text(original_mtl_text, best_diffuse.name)
    _write_text(fixed_mtl_path, fixed_mtl_text)
    result["fixed_mtl"] = str(fixed_mtl_path)

    fixed_obj_name = f"fixed_{obj_path.name}"
    fixed_obj_path = asset_dir / fixed_obj_name
    original_obj_text = _read_text(obj_path)
    fixed_obj_text = build_fixed_obj_text(original_obj_text, fixed_mtl_name)
    _write_text(fixed_obj_path, fixed_obj_text)
    result["fixed_obj"] = str(fixed_obj_path)

    return result


def print_summary(report: Dict) -> None:
    print("=== TEXTURE DEBUG REPORT ===")
    print("asset_dir:", report["asset_dir"])
    print("obj:", report["obj"])
    print("obj_mtllibs:", report["obj_mtllibs"])
    print("mtl_files:", report["mtl_files"])
    print("image_files:", report["image_files"])
    print("best_diffuse_candidate:", report["best_diffuse_candidate"])
    print()

    resolved = report["resolved_maps"]
    if not resolved:
        print("MTL-карты не найдены.")
        return

    for mtl_name, mats in resolved.items():
        print(f"[MTL] {mtl_name}")
        for mat_name, maps in mats.items():
            print(f"  newmtl {mat_name}")
            if not maps:
                print("    (нет карт)")
                continue
            for key, info in maps.items():
                print(
                    f"    {key}: ref={info['ref']!r}, "
                    f"status={info['status']}, "
                    f"resolved={info['resolved_path']}"
                )


def _build_blender_debug_script(
    obj_path: Path,
    mtl_path: Optional[Path],
    save_blend: Path,
    render_path: Optional[Path],
    report_path: Path,
) -> str:
    mtl_literal = "None" if mtl_path is None else repr(str(mtl_path))
    render_literal = "None" if render_path is None else repr(str(render_path))

    return f'''# -*- coding: utf-8 -*-
import bpy
import os
import re
import traceback

OBJ_PATH = r"{str(obj_path)}"
MTL_PATH = {mtl_literal}
SAVE_BLEND = r"{str(save_blend)}"
RENDER_PATH = {render_literal}
REPORT_PATH = r"{str(report_path)}"

MAP_RE = re.compile(r"^\\s*(map_Kd|map_Ka|map_Ks|map_d|bump|map_Bump)\\s+(.*)$", re.IGNORECASE)
NEWMTL_RE = re.compile(r"^\\s*newmtl\\s+(.*)$", re.IGNORECASE)

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def parse_mtl(mtl_path):
    mats = {{}}
    current = "__default__"
    mats[current] = {{}}

    if not mtl_path or not os.path.isfile(mtl_path):
        return mats

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_new = NEWMTL_RE.match(line)
            if m_new:
                current = m_new.group(1).strip() or "__unnamed__"
                mats.setdefault(current, {{}})
                continue

            m_map = MAP_RE.match(line)
            if not m_map:
                continue

            key = m_map.group(1).lower()
            value = m_map.group(2).strip().split()[-1].strip('"').strip("'")

            if key == "map_kd":
                mats[current]["diffuse"] = value
            elif key == "map_d":
                mats[current]["opacity"] = value
            elif key in ("bump", "map_bump"):
                mats[current]["bump"] = value
            elif key == "map_ka":
                mats[current]["ambient"] = value
            elif key == "map_ks":
                mats[current]["specular"] = value

    return mats

def resolve_tex(base_dir, ref):
    if not ref:
        return None
    p = os.path.join(base_dir, ref)
    p = os.path.normpath(p)
    if os.path.isfile(p):
        return p
    p2 = os.path.join(base_dir, os.path.basename(ref))
    p2 = os.path.normpath(p2)
    if os.path.isfile(p2):
        return p2
    return None

def import_obj(path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    elif hasattr(bpy.ops.import_scene, "obj"):
        bpy.ops.import_scene.obj(filepath=path)
    else:
        raise RuntimeError("OBJ importer not available")
    after = [o for o in bpy.data.objects if o not in before]
    return after

def ensure_camera_and_light():
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (2.5, -4.0, 2.0)
    cam.rotation_euler = (1.15, 0.0, 0.75)
    scene.camera = cam

    light_data = bpy.data.lights.new(name="Light", type="AREA")
    light_data.energy = 4000
    light_data.shape = 'RECTANGLE'
    light_data.size = 3.0
    light_data.size_y = 3.0
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    scene.collection.objects.link(light)
    light.location = (0.0, 0.0, 3.0)
    light.rotation_euler = (0.0, 0.0, 0.0)

def build_material(mat_name, info, base_dir):
    mat = bpy.data.materials.new(mat_name + "_DEBUG")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (420, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-400, 0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-200, 0)
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])

    diffuse_path = resolve_tex(base_dir, info.get("diffuse"))
    opacity_path = resolve_tex(base_dir, info.get("opacity"))
    bump_path = resolve_tex(base_dir, info.get("bump"))

    report = {{
        "material_name": mat_name,
        "diffuse_path": diffuse_path,
        "opacity_path": opacity_path,
        "bump_path": bump_path,
    }}

    if diffuse_path:
        tex = nodes.new("ShaderNodeTexImage")
        tex.location = (0, 120)
        tex.image = bpy.data.images.load(diffuse_path, check_existing=True)
        try:
            tex.image.colorspace_settings.name = "sRGB"
        except Exception:
            pass
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if opacity_path and "Alpha" in bsdf.inputs:
        tex_a = nodes.new("ShaderNodeTexImage")
        tex_a.location = (0, -80)
        tex_a.image = bpy.data.images.load(opacity_path, check_existing=True)
        try:
            tex_a.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        links.new(mapping.outputs["Vector"], tex_a.inputs["Vector"])
        links.new(tex_a.outputs["Color"], bsdf.inputs["Alpha"])
        try:
            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
        except Exception:
            pass

    if bump_path:
        tex_b = nodes.new("ShaderNodeTexImage")
        tex_b.location = (0, -280)
        tex_b.image = bpy.data.images.load(bump_path, check_existing=True)
        try:
            tex_b.image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        links.new(mapping.outputs["Vector"], tex_b.inputs["Vector"])

        bump = nodes.new("ShaderNodeBump")
        bump.location = (220, -280)
        links.new(tex_b.outputs["Color"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat, report

def main():
    reset_scene()
    imported = import_obj(OBJ_PATH)
    ensure_camera_and_light()

    base_dir = os.path.dirname(OBJ_PATH)
    parsed = parse_mtl(MTL_PATH)

    debug_report = {{
        "obj_path": OBJ_PATH,
        "mtl_path": MTL_PATH,
        "materials_in_mtl": parsed,
        "materials_applied": []
    }}

    for obj in imported:
        if obj.type != "MESH":
            continue

        if not obj.data.materials:
            continue

        for i, old_mat in enumerate(obj.data.materials):
            if old_mat is None:
                continue

            name = old_mat.name
            info = parsed.get(name) or parsed.get(name.strip()) or {{}}

            if not info and len(parsed) == 2 and "__default__" in parsed:
                # бывает ровно один реальный material + __default__
                real_names = [k for k in parsed.keys() if k != "__default__"]
                if len(real_names) == 1:
                    info = parsed[real_names[0]]

            if not info and "__default__" in parsed:
                info = parsed["__default__"]

            new_mat, mat_report = build_material(name, info, base_dir)
            obj.data.materials[i] = new_mat
            debug_report["materials_applied"].append(mat_report)

    if RENDER_PATH:
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = RENDER_PATH
        bpy.ops.render.render(write_still=True)

    bpy.ops.wm.save_as_mainfile(filepath=SAVE_BLEND)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(debug_report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
'''


def run_blender_texture_test(
    blender_path: Path,
    obj_path: Path,
    mtl_path: Optional[Path],
    save_blend: Path,
    render_path: Optional[Path],
    asset_dir: Path,
) -> Dict[str, Optional[str]]:
    if not blender_path.is_file():
        raise RuntimeError(f"Blender не найден: {blender_path}")

    script_path = Path(tempfile.gettempdir()) / "oai_debug_asset_texture_blender.py"
    report_path = asset_dir / "blender_texture_debug_report.json"

    script_text = _build_blender_debug_script(
        obj_path=obj_path,
        mtl_path=mtl_path,
        save_blend=save_blend,
        render_path=render_path,
        report_path=report_path,
    )
    _write_text(script_path, script_text)

    cmd = [
        str(blender_path),
        "--background",
        "--python",
        str(script_path),
    ]

    subprocess.run(cmd, check=True)

    return {
        "blend": str(save_blend),
        "render": str(render_path) if render_path else None,
        "blender_report": str(report_path),
        "script": str(script_path),
    }


def main() -> None:
    args = parse_args()

    asset_dir = Path(args.asset_dir).expanduser().resolve()
    if not asset_dir.exists() or not asset_dir.is_dir():
        raise RuntimeError(f"Папка не существует: {asset_dir}")

    obj_path = choose_obj(asset_dir, args.obj_name)
    mtl_paths = list_mtls(asset_dir)
    images = list_images(asset_dir)

    report = build_report(
        asset_dir=asset_dir,
        obj_path=obj_path,
        mtl_paths=mtl_paths,
        images=images,
        prefer=args.prefer,
    )

    print_summary(report)

    fixed_info = None
    fixed_obj_for_blender = obj_path
    fixed_mtl_for_blender = choose_main_mtl(asset_dir, obj_path)

    if args.write_fixed:
        if obj_path is None:
            raise RuntimeError("Не найден OBJ для создания fixed-файлов")

        main_mtl = choose_main_mtl(asset_dir, obj_path)
        best_diffuse = Path(report["best_diffuse_candidate"]) if report["best_diffuse_candidate"] else None
        fixed_info = write_fixed_files(
            asset_dir=asset_dir,
            obj_path=obj_path,
            main_mtl=main_mtl,
            best_diffuse=best_diffuse,
        )
        print()
        print("fixed files:", fixed_info)

        if fixed_info.get("fixed_obj"):
            fixed_obj_for_blender = Path(fixed_info["fixed_obj"])
        if fixed_info.get("fixed_mtl"):
            fixed_mtl_for_blender = Path(fixed_info["fixed_mtl"])

    blender_info = None
    if args.test_blender:
        if fixed_obj_for_blender is None:
            raise RuntimeError("Не найден OBJ для Blender-проверки")

        save_blend = Path(args.save_blend).expanduser().resolve() if args.save_blend else (asset_dir / "debug_textured.blend")
        render_path = Path(args.render).expanduser().resolve() if args.render else None

        blender_info = run_blender_texture_test(
            blender_path=Path(args.blender).expanduser().resolve(),
            obj_path=fixed_obj_for_blender,
            mtl_path=fixed_mtl_for_blender,
            save_blend=save_blend,
            render_path=render_path,
            asset_dir=asset_dir,
        )
        print()
        print("blender output:", blender_info)

    report_path = asset_dir / "texture_report.json"
    final_report = dict(report)
    final_report["fixed_files"] = fixed_info
    final_report["blender_output"] = blender_info
    _write_text(report_path, json.dumps(final_report, indent=2, ensure_ascii=False))
    print()
    print("report saved:", report_path)


if __name__ == "__main__":
    main()