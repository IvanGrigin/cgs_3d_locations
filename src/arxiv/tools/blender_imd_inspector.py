#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/tools/imd_asset_labeler.py

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


AXES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

YES_WORDS = {"y", "yes", "да", "д", "игрек", "true", "1"}
NO_WORDS = {"n", "no", "нет", "н", "false", "0"}


# --------------------------
# Defaults storage
# --------------------------

@dataclass
class Defaults:
    up: str = "+Z"
    front: str = "-Y"
    right: str = "-X"

    textures_ok: bool = True
    textures_dir: str = "."
    is_single: bool = True

    def to_json(self) -> Dict:
        return {
            "up": self.up,
            "front": self.front,
            "right": self.right,
            "textures_ok": self.textures_ok,
            "textures_dir": self.textures_dir,
            "is_single": self.is_single,
        }

    @staticmethod
    def from_json(d: Dict) -> "Defaults":
        out = Defaults()
        out.up = str(d.get("up", out.up))
        out.front = str(d.get("front", out.front))
        out.right = str(d.get("right", out.right))
        out.textures_ok = bool(d.get("textures_ok", out.textures_ok))
        out.textures_dir = str(d.get("textures_dir", out.textures_dir))
        out.is_single = bool(d.get("is_single", out.is_single))
        # sanity
        if out.up not in AXES: out.up = "+Z"
        if out.front not in AXES: out.front = "-Y"
        if out.right not in AXES: out.right = "-X"
        return out


def load_defaults(path: Path) -> Defaults:
    if path.is_file():
        try:
            return Defaults.from_json(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    # если файла нет/битый — создаём дефолт с нужной ориентацией
    d = Defaults()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return d


def save_defaults(path: Path, d: Defaults) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------
# CLI helpers
# --------------------------

def norm_answer(s: str) -> str:
    return s.strip().lower()

def is_yes(s: str) -> bool:
    t = norm_answer(s)
    return (t == "") or (t in YES_WORDS)

def is_no(s: str) -> bool:
    t = norm_answer(s)
    return t in NO_WORDS

def prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    default_str = "Y" if default_yes else "N"
    while True:
        ans = input(f"{question} [Y/N] (по умолчанию {default_str}): ").strip()
        if ans == "":
            return default_yes
        if is_yes(ans):
            return True
        if is_no(ans):
            return False
        print("  ! Введи Y/Yes/Enter или N/No.")

def prompt_axis(name: str, default_axis: str) -> str:
    while True:
        ans = input(f"{name} [{default_axis}] (варианты: {', '.join(AXES)}): ").strip()
        if ans == "":
            return default_axis
        ans = ans.upper()
        if ans in AXES:
            return ans
        print("  ! Неверная ось. Пример: +Z, -Y, +X ...")

def prompt_text(question: str, default: str = "") -> str:
    ans = input(f"{question} [{default}]: ").strip()
    return ans if ans != "" else default


# --------------------------
# Axis -> semantic mapping for Blender inspector
# --------------------------

def _opposite_axis(a: str) -> str:
    return ("-" if a[0] == "+" else "+") + a[1:]

def _mapping_world_axis_to_semantic(up: str, front: str, right: str) -> Dict[str, str]:
    """
    Returns mapping for inspector:
      "+X": semantic_side, "-X": ..., "+Y": ..., ...
    Based on: UP axis -> top, FRONT axis -> front, RIGHT axis -> right.
    """
    m = {k: "front" for k in ["+X","-X","+Y","-Y","+Z","-Z"]}

    m[up] = "top"
    m[_opposite_axis(up)] = "bottom"

    m[front] = "front"
    m[_opposite_axis(front)] = "back"

    m[right] = "right"
    m[_opposite_axis(right)] = "left"
    return m


# --------------------------
# OBJ/MTL texture parsing
# --------------------------

_MTL_TEX_KEYS = {
    "map_kd", "map_ks", "map_ka", "map_bump", "bump", "map_d", "map_ns",
    "disp", "decal", "refl"
}

def parse_obj_mtllib(obj_path: Path) -> Optional[str]:
    try:
        for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("mtllib "):
                return line.split(None, 1)[1].strip()
    except Exception:
        pass
    return None

def parse_mtl_texture_files(mtl_path: Path) -> List[str]:
    out: List[str] = []
    try:
        for raw in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            key = parts[0].lower()
            if key not in _MTL_TEX_KEYS:
                continue
            # MTL может содержать флаги (-o, -s, -bm, ...)
            # берём "последний токен", который выглядит как путь/файл
            # (это практично для IMD)
            cand = parts[-1].strip()
            if cand:
                out.append(cand)
    except Exception:
        pass
    # уникализируем, сохраняя порядок
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def resolve_texture_paths(asset_dir: Path, obj_dir: Path, tex_rel_list: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      (tex_files_declared, tex_paths_found, tex_files_missing)
    We try resolve relative to:
      - obj_dir
      - asset_dir
      - their subdirs (common ones)
    """
    declared = tex_rel_list[:]
    found: List[str] = []
    missing: List[str] = []

    search_roots = [
        obj_dir,
        asset_dir,
        obj_dir / "textures",
        obj_dir / "Textures",
        asset_dir / "textures",
        asset_dir / "Textures",
    ]

    for rel in tex_rel_list:
        rel_clean = rel.replace("\\", "/")
        # если в mtl путь с подпапкой, проверим напрямую
        candidates = []
        p = Path(rel_clean)
        if p.is_absolute():
            candidates.append(p)
        else:
            for root in search_roots:
                candidates.append(root / p)

        ok_path = None
        for c in candidates:
            if c.is_file():
                ok_path = c
                break

        if ok_path is not None:
            found.append(str(ok_path))
        else:
            missing.append(rel)

    return declared, found, missing


# --------------------------
# Asset scanning
# --------------------------

def find_obj_in_dir(d: Path) -> Optional[Path]:
    objs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".obj"])
    if not objs:
        return None
    # эвристика: избегаем слишком общих названий
    preferred = [p for p in objs if p.name.lower() not in {"scene.obj", "model.obj"}]
    return preferred[0] if preferred else objs[0]

def collect_asset_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # фильтры мусора
        dn = os.path.basename(dirpath)
        if dn.startswith(".") or dn == "__MACOSX":
            continue
        has_obj = any(fn.lower().endswith(".obj") for fn in filenames)
        if has_obj:
            out.append(Path(dirpath))
    out = sorted(out)
    return out


# --------------------------
# Meta writing
# --------------------------

def write_asset_meta(meta_path: Path, payload: Dict) -> None:
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------
# Blender runner
# --------------------------

def run_blender(blender_path: str, inspector_py: Path, obj_path: Path, mapping: Dict[str, str]) -> int:
    cmd = [
        blender_path,
        "--factory-startup",
        "--python",
        str(inspector_py),
        "--",
        "--obj",
        str(obj_path),
        "--default-plus-x",  mapping["+X"],
        "--default-minus-x", mapping["-X"],
        "--default-plus-y",  mapping["+Y"],
        "--default-minus-y", mapping["-Y"],
        "--default-plus-z",  mapping["+Z"],
        "--default-minus-z", mapping["-Z"],
    ]
    return subprocess.call(cmd)


# --------------------------
# Main
# --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory with IMD assets")
    ap.add_argument("--blender", required=True, help="Path to Blender executable")
    ap.add_argument("--inspector", default="src/tools/blender_imd_inspector.py", help="Blender inspector script")
    ap.add_argument("--defaults", default="src/tools/imd_defaults.json", help="Defaults json")
    ap.add_argument("--start", type=int, default=1, help="1-based start index")
    ap.add_argument("--max", type=int, default=0, help="Max items (0 = all)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    blender = str(args.blender)
    inspector = Path(args.inspector).expanduser().resolve()
    defaults_path = Path(args.defaults).expanduser().resolve()

    if not root.is_dir():
        print("[error] root not found:", root)
        sys.exit(2)
    if not inspector.is_file():
        print("[error] inspector not found:", inspector)
        sys.exit(2)

    defaults = load_defaults(defaults_path)

    asset_dirs = collect_asset_dirs(root)
    print(f"[scan] Found {len(asset_dirs)} asset dirs under: {root}")
    print(f"[blender] {blender}")
    print(f"[defaults] {defaults_path}")

    start_idx = max(1, int(args.start))
    max_items = int(args.max)
    end_idx = len(asset_dirs) if max_items <= 0 else min(len(asset_dirs), start_idx - 1 + max_items)

    for i in range(start_idx - 1, end_idx):
        asset_dir = asset_dirs[i]
        obj_path = find_obj_in_dir(asset_dir)
        if obj_path is None:
            continue

        # MTL discovery
        obj_dir = obj_path.parent
        mtllib = parse_obj_mtllib(obj_path)
        mtl_path = None
        if mtllib:
            candidate = (obj_dir / mtllib)
            if candidate.is_file():
                mtl_path = candidate
        if mtl_path is None:
            # fallback: any .mtl in same dir
            mtls = sorted([p for p in obj_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mtl"])
            mtl_path = mtls[0] if mtls else None

        meta_path = asset_dir / "asset_meta.json"

        print("\n" + "=" * 80)
        print(f"[{i+1}/{len(asset_dirs)}] {asset_dir}")
        print(f"  OBJ: {obj_path.name}")
        print(f"  MTL: {mtl_path.name if mtl_path else '(none)'}")
        print(f"  META: {meta_path.name}")
        print("  -> Открываю Blender. Посмотри ориентацию и текстуры. Закрой Blender, чтобы продолжить.")

        # Compute mapping for inspector cube labels from current defaults axes
        mapping = _mapping_world_axis_to_semantic(defaults.up, defaults.front, defaults.right)

        rc = run_blender(blender, inspector, obj_path, mapping)
        if rc != 0:
            print(f"[warn] Blender returned code {rc}")

        # Always print texture resolution details (what / where)
        tex_declared: List[str] = []
        tex_found: List[str] = []
        tex_missing: List[str] = []
        if mtl_path is not None and mtl_path.is_file():
            tex_declared = parse_mtl_texture_files(mtl_path)
            tex_declared, tex_found, tex_missing = resolve_texture_paths(asset_dir, obj_dir, tex_declared)

        print("\n[texture-report]")
        print(f"  asset_dir: {asset_dir}")
        print(f"  obj_dir:   {obj_dir}")
        if mtl_path:
            print(f"  mtl:       {mtl_path}")
        else:
            print("  mtl:       (none)")
        if tex_declared:
            print("  declared in MTL:")
            for t in tex_declared:
                print(f"    - {t}")
        else:
            print("  declared in MTL: (none)")

        if tex_found:
            print("  found on disk:")
            for p in tex_found:
                print(f"    + {p}")
        else:
            print("  found on disk: (none)")

        if tex_missing:
            print("  missing:")
            for t in tex_missing:
                print(f"    ! {t}")
        else:
            print("  missing: (none)")

        # 1) Global gate: "everything ok?"
        ok_all = prompt_yes_no("Всё корректно? (ориентация + текстуры + одиночность)", default_yes=True)

        if ok_all:
            # AUTO-FILL PATH (Enter/Yes): no more questions
            up = defaults.up
            front = defaults.front
            right = defaults.right

            textures_ok = True
            textures_dir = defaults.textures_dir if defaults.textures_dir else "."
            is_single = True

            note = ""

        else:
            # Detailed path
            print("\nВведи семантические оси предмета относительно МИРОВЫХ осей:")
            up = prompt_axis("UP (вверх предмета)", defaults.up)
            front = prompt_axis("FRONT (вперёд предмета)", defaults.front)
            right = prompt_axis("RIGHT (право предмета)", defaults.right)

            textures_ok = prompt_yes_no("Текстуры подтянулись корректно?", default_yes=defaults.textures_ok)
            if textures_ok:
                textures_dir = defaults.textures_dir if defaults.textures_dir else "."
            else:
                textures_dir = prompt_text("Где лежат текстуры (папка относительно ассета)", defaults.textures_dir if defaults.textures_dir else ".")
                # если текстуры не ок — полезно явно зафиксировать, какие файлы должны быть
                # (можно оставить пусто, тогда пишем то, что нашли в MTL)
                extra = prompt_text("Если знаешь, перечисли нужные файлы текстур через запятую (можно пусто)", "")
                if extra.strip():
                    # перезапишем declared списком пользователя
                    tex_declared = [x.strip() for x in extra.split(",") if x.strip()]
                    tex_declared, tex_found, tex_missing = resolve_texture_paths(asset_dir, obj_dir, tex_declared)

            # single/multi
            is_single = prompt_yes_no("Это одиночный предмет (не набор/не сцена)?", default_yes=defaults.is_single)

            note = prompt_text("Комментарий/заметка (можно пусто)", "")

            # optional: update defaults for next assets
            if prompt_yes_no("Сделать эти значения дефолтами для следующих ассетов?", default_yes=False):
                defaults.up = up
                defaults.front = front
                defaults.right = right
                defaults.textures_ok = textures_ok
                defaults.textures_dir = textures_dir
                defaults.is_single = is_single
                save_defaults(defaults_path, defaults)
                print(f"  ✅ defaults saved: {defaults_path}")

        # Save asset meta
        payload = {
            "up": up,
            "front": front,
            "right": right,

            "textures_ok": bool(textures_ok),
            "textures_dir": str(textures_dir),
            "textures_files_declared": tex_declared,
            "textures_files_found": tex_found,
            "textures_files_missing": tex_missing,
            "mtl_file": str(mtl_path.name) if mtl_path else None,

            "is_single": bool(is_single),
            "note": str(note),

            "obj_file": str(obj_path.name),
        }

        write_asset_meta(meta_path, payload)
        print(f"  ✅ saved: {meta_path}")


if __name__ == "__main__":
    main()