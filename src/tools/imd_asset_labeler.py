#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

AXES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

DEFAULT_META_NAME = "asset_meta.json"
DEFAULT_DEFAULTS_NAME = "imd_defaults.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_blender(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    candidates = [
        os.environ.get("BLENDER_PATH"),
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
        "blender",  # PATH
    ]
    for c in candidates:
        if not c:
            continue
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        if c == "blender":
            return c
    return "blender"


def is_asset_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    objs = list(p.glob("*.obj")) + list(p.glob("*.OBJ"))
    return len(objs) > 0


def choose_obj_file(asset_dir: Path) -> Optional[Path]:
    objs = list(asset_dir.glob("*.obj")) + list(asset_dir.glob("*.OBJ"))
    if not objs:
        return None
    objs.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
    return objs[0]


def find_mtl_for_obj(obj_path: Path) -> Optional[Path]:
    cand = obj_path.with_suffix(".mtl")
    if cand.exists():
        return cand
    cand = obj_path.with_suffix(".MTL")
    if cand.exists():
        return cand

    try:
        txt = obj_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^\s*mtllib\s+(.+?)\s*$", txt, flags=re.MULTILINE)
        if m:
            name = m.group(1).strip()
            p = (obj_path.parent / name).resolve()
            if p.exists():
                return p
    except Exception:
        pass

    mtls = list(obj_path.parent.glob("*.mtl")) + list(obj_path.parent.glob("*.MTL"))
    if len(mtls) == 1:
        return mtls[0]
    return None


def parse_mtl_textures(mtl_path: Path) -> List[str]:
    """
    Вытаскиваем пути из map_* / bump / map_Bump / disp и т.п.
    Поддерживаем опции вида: map_Kd -s 1 1 1 textures/diffuse.png
    Берём "самый правый" токен (как правило это файл).
    """
    if not mtl_path or not mtl_path.exists():
        return []

    tex: List[str] = []
    keys = ("map_kd", "map_ks", "map_ke", "map_bump", "bump", "disp", "map_d", "map_ns")
    try:
        for line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if not parts:
                continue
            k = parts[0].lower()
            if k not in keys:
                continue
            candidate = parts[-1]
            tex.append(candidate)
    except Exception:
        return []

    seen = set()
    out: List[str] = []
    for t in tex:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def resolve_texture_paths(asset_dir: Path, mtl_rel_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Возвращаем:
      existing_relative:  как в MTL
      missing_relative:   как в MTL
      existing_abs:       абсолютные пути существующих
    """
    existing: List[str] = []
    missing: List[str] = []
    existing_abs: List[str] = []
    for rel in mtl_rel_paths:
        p = (asset_dir / rel).resolve()
        if p.exists():
            existing.append(rel)
            existing_abs.append(str(p))
        else:
            missing.append(rel)
    return existing, missing, existing_abs


def infer_texture_root_from_existing(existing_rel: List[str]) -> str:
    if not existing_rel:
        return "."
    dirs = []
    for r in existing_rel:
        pr = Path(r)
        d = str(pr.parent).replace("\\", "/")
        if d == ".":
            d = "."
        dirs.append(d)
    if len(set(dirs)) == 1:
        return dirs[0]

    split_dirs = [d.split("/") if d != "." else [] for d in dirs]
    if not split_dirs:
        return "."
    common: List[str] = []
    for parts in zip(*split_dirs):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    return "/".join(common) if common else "."


def prompt_axis(name: str, default: str) -> str:
    while True:
        ans = input(f"{name} [{default}] (варианты: {', '.join(AXES)}): ").strip()
        if not ans:
            return default
        ans = ans.upper().replace(" ", "")
        if ans in AXES:
            return ans
        print("  Неверное значение. Пример: +Y")


def _is_yes_token(s: str) -> bool:
    s = s.strip().lower()
    return s in ("y", "yes", "да", "д", "игрек", "у")


def _is_no_token(s: str) -> bool:
    s = s.strip().lower()
    return s in ("n", "no", "нет", "н")


def prompt_yes_no(msg: str, default: bool) -> bool:
    d = "Y" if default else "N"
    while True:
        ans = input(f"{msg} [Y/N] (по умолчанию {d}): ").strip().lower()
        if not ans:
            return default
        if _is_yes_token(ans):
            return True
        if _is_no_token(ans):
            return False
        print("  Введите Y или N.")


def prompt_text(msg: str, default: str) -> str:
    ans = input(f"{msg} [{default}]: ").strip()
    return default if not ans else ans


def prompt_ok_all_quick() -> bool:
    """
    Первый вопрос "всё корректно?".
    Enter / Y / YES / ДА / ИГРЕК => True
    N / Н / НЕТ => False
    """
    while True:
        ans = input("Всё корректно? (ориентация, текстуры, одиночность) [Y/N] (по умолчанию Y): ").strip()
        if not ans:
            return True
        if _is_yes_token(ans):
            return True
        if _is_no_token(ans):
            return False
        print("  Введите Y или N (или просто Enter).")


def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def start_blender_preview_async(
    blender_exe: str,
    inspector_script: Path,
    obj_path: Path,
    mtl_path: Optional[Path],
) -> Optional[subprocess.Popen]:
    """
    Запускает Blender параллельно (не блокируя терминал).
    Blender остаётся открыт, пока пользователь не закроет окно,
    либо пока мы не завершим и не вызовем stop_blender(proc).
    """
    if not inspector_script.exists():
        print(f"[warn] Inspector script not found: {inspector_script}")
        return None

    # Никаких "жёстких" проверок содержимого инспектора (они часто дают ложные срабатывания).
    # Но добавим мягкое предупреждение, если скрипт вообще не похож на bpy-скрипт.
    txt = _read_text_safe(inspector_script)
    if "import bpy" not in txt:
        print(
            "[warn] Inspector script не содержит 'import bpy'. "
            "Если Blender ругается на аргументы --root/--blender — вероятно, указан не тот скрипт."
        )

    cmd = [
        blender_exe,
        "--factory-startup",
        "--python",
        str(inspector_script),
        "--",
        "--obj",
        str(obj_path),
    ]
    if mtl_path:
        cmd += ["--mtl", str(mtl_path)]

    try:
        return subprocess.Popen(cmd)
    except Exception as e:
        print(f"[warn] Failed to start Blender: {e}")
        return None


def stop_blender(proc: subprocess.Popen, timeout_sec: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout_sec)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


@dataclass
class Defaults:
    # Требуемые дефолты:
    #   UP    = +Z
    #   FRONT = -Y
    #   RIGHT = -X
    up: str = "+Z"
    front: str = "-Y"
    right: str = "-X"

    textures_ok_default: bool = True
    texture_root_default: str = "."
    is_single_default: bool = True

    @staticmethod
    def from_dict(d: dict) -> "Defaults":
        axes = (d or {}).get("axes") or {}
        return Defaults(
            up=str(axes.get("up", "+Z")).upper(),
            front=str(axes.get("front", "-Y")).upper(),
            right=str(axes.get("right", "-X")).upper(),
            textures_ok_default=bool((d or {}).get("textures_ok_default", True)),
            texture_root_default=str((d or {}).get("texture_root_default", ".")),
            is_single_default=bool((d or {}).get("is_single_default", True)),
        )

    def to_dict(self) -> dict:
        return {
            "axes": {"up": self.up, "front": self.front, "right": self.right},
            "textures_ok_default": self.textures_ok_default,
            "texture_root_default": self.texture_root_default,
            "is_single_default": self.is_single_default,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Корневая папка IMD (рекурсивный обход)")
    ap.add_argument("--blender", default=None, help="Путь к Blender (или BLENDER_PATH)")
    ap.add_argument("--meta-name", default=DEFAULT_META_NAME, help="Имя json в папке ассета")
    ap.add_argument("--defaults", default=DEFAULT_DEFAULTS_NAME, help="Файл дефолтов (рядом со скриптом или абсолютный)")
    ap.add_argument("--force", action="store_true", help="Перезаписывать существующий meta json без вопросов")
    ap.add_argument("--skip-existing", action="store_true", help="Пропускать папки, где meta json уже есть")
    ap.add_argument("--no-blender", action="store_true", help="Не запускать Blender (только вопросы в консоли)")
    ap.add_argument("--keep-blender-open", action="store_true", help="Не закрывать Blender автоматически после ввода")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    blender_exe = find_blender(args.blender)

    defaults_path = Path(args.defaults)
    if not defaults_path.is_absolute():
        defaults_path = (Path(__file__).resolve().parent / defaults_path).resolve()

    d = read_json(defaults_path) or {}
    defaults = Defaults.from_dict(d)

    inspector_script = (Path(__file__).resolve().parent / "blender_imd_inspector.py").resolve()
    if not args.no_blender and not inspector_script.exists():
        raise SystemExit(f"Inspector script not found: {inspector_script}")

    asset_dirs: List[Path] = []
    for dirpath, _, _ in os.walk(root):
        p = Path(dirpath)
        if is_asset_dir(p):
            asset_dirs.append(p)

    asset_dirs.sort()

    print(f"[scan] Found {len(asset_dirs)} asset dirs under: {root}")
    print(f"[blender] {blender_exe}")
    print(f"[defaults] {defaults_path}")

    for idx, asset_dir in enumerate(asset_dirs, 1):
        meta_path = asset_dir / args.meta_name
        if meta_path.exists() and args.skip_existing:
            continue

        obj_path = choose_obj_file(asset_dir)
        if not obj_path:
            continue
        mtl_path = find_mtl_for_obj(obj_path)

        existing_meta = read_json(meta_path)

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(asset_dirs)}] {asset_dir}")
        print(f"  OBJ: {obj_path.name}")
        print(f"  MTL: {mtl_path.name if mtl_path else '(none)'}")
        print(f"  META: {meta_path.name} {'(exists)' if meta_path.exists() else ''}")

        if existing_meta and not args.force:
            if not prompt_yes_no("Обновить существующий meta json?", default=False):
                continue

        # Текстуры из MTL (для логов и дефолтов)
        mtl_tex = parse_mtl_textures(mtl_path) if mtl_path else []
        exist_tex, miss_tex, exist_tex_abs = resolve_texture_paths(asset_dir, mtl_tex)
        inferred_tex_root = infer_texture_root_from_existing(exist_tex)

        if mtl_tex:
            print("  [mtl] texture references:")
            for t in mtl_tex:
                print("   -", t)
        if exist_tex:
            print("  [mtl] resolved existing files:")
            for a in exist_tex_abs:
                print("   -", a)
        if miss_tex:
            print("  [mtl] missing referenced files:")
            for t in miss_tex:
                print("   -", t)
        if mtl_tex:
            print(f"  [mtl] inferred texture_root: {inferred_tex_root}")

        # Blender preview (async, stays open during questioning)
        blender_proc: Optional[subprocess.Popen] = None
        if not args.no_blender:
            print("  -> Открываю Blender (параллельно). Пока Blender открыт — введи ответы в терминале.")
            blender_proc = start_blender_preview_async(blender_exe, inspector_script, obj_path, mtl_path)

        try:
            ok_all = prompt_ok_all_quick()

            if ok_all:
                # авто-заполнение
                up = defaults.up
                front = defaults.front
                right = defaults.right

                textures_ok = True
                tex_root_default = inferred_tex_root if inferred_tex_root != "." else defaults.texture_root_default
                tex_root = tex_root_default

                is_single = defaults.is_single_default
                notes = ""
            else:
                print("\nВведи семантические оси предмета относительно МИРОВЫХ осей:")
                up = prompt_axis("UP (вверх предмета)", defaults.up)
                front = prompt_axis("FRONT (вперёд предмета)", defaults.front)
                right = prompt_axis("RIGHT (право предмета)", defaults.right)

                def axis_abs(a: str) -> str:
                    return a.replace("+", "").replace("-", "")

                if len({axis_abs(up), axis_abs(front), axis_abs(right)}) != 3:
                    print("  ⚠️ Предупреждение: UP/FRONT/RIGHT используют не 3 разные оси. Это часто ошибка.")

                tex_ok_default = defaults.textures_ok_default and (len(miss_tex) == 0)
                textures_ok = prompt_yes_no("Текстуры подтянулись корректно?", default=tex_ok_default)

                tex_root_suggest = inferred_tex_root if inferred_tex_root != "." else defaults.texture_root_default
                tex_root = prompt_text("Где лежат текстуры (папка относительно ассета)", tex_root_suggest)

                is_single = prompt_yes_no("Предмет один (single item)?", default=defaults.is_single_default)
                notes = prompt_text("Комментарий/заметка (можно пусто)", "")

                if prompt_yes_no("Сделать эти значения дефолтами для следующих ассетов?", default=False):
                    defaults.up = up
                    defaults.front = front
                    defaults.right = right
                    defaults.textures_ok_default = textures_ok
                    defaults.texture_root_default = tex_root
                    defaults.is_single_default = is_single
                    write_json(defaults_path, defaults.to_dict())
                    print(f"  [defaults] updated: {defaults_path}")

            meta = {
                "version": 2,
                "obj_file": obj_path.name,
                "mtl_file": (mtl_path.name if mtl_path else None),
                "is_single": bool(is_single),
                "axes": {"up": up, "front": front, "right": right},
                "textures": {
                    "source": "mtl" if mtl_path else "none",
                    "mtl_texture_refs": mtl_tex,
                    "resolved_existing": exist_tex,
                    "resolved_missing": miss_tex,
                    "resolved_existing_abs": exist_tex_abs,
                    "inferred_texture_root": inferred_tex_root,
                    "user_confirmed_ok": bool(textures_ok),
                    "texture_root": tex_root,
                },
                "notes": notes,
                "updated_utc": utc_now_iso(),
            }

            write_json(meta_path, meta)
            print(f"  ✅ saved: {meta_path}")

        finally:
            if blender_proc is not None and not args.keep_blender_open:
                stop_blender(blender_proc)

    print("\nDone.")


if __name__ == "__main__":
    main()