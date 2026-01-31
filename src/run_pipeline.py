# src/run_pipeline.py
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import subprocess
import sys
import time
import secrets
from typing import Any
from pathlib import Path


# ------------------------------------------------------------
# Пути
# ------------------------------------------------------------
CUBE_SCRIPT = "src/Plasement/CubePlacement.py"
BLENDER_VIS_SCRIPT = "src/Plasement/BlenderVisualizePlacement.py"

DEFAULT_ROOM_GLB = "data/input/room.glb"
DEFAULT_ROOM_JSON = "data/input/room.json"   # <-- ДОБАВИЛИ: room-spec

FURNITURE_DB = "data/input/furniture_types.json"
OBJECTS_JSON = "data/input/objects.json"

PLACEMENT_JSON = "data/output/placement_result.json"
SCENE_JSON = "data/output/scene_room_and_placements.json"  # <-- ДОБАВИЛИ: склеенный JSON для Blender

FIND_OBJ_SCRIPT = "src/tools/find_obj_from_db.py"
EXTRACT_ROOT = "data/sourse/imodern"

MAX_ATTEMPTS = 30


# ------------------------------------------------------------
# Нормализация и метрики похожести
# ------------------------------------------------------------
_word_re = re.compile(r"[А-Яа-яA-Za-z0-9]+")

def norm(s: str) -> str:
    return " ".join(_word_re.findall(s.lower()))

def token_set(s: str) -> set[str]:
    return set(norm(s).split())

def char_multiset(s: str) -> dict[str, int]:
    d: dict[str, int] = {}
    for ch in norm(s).replace(" ", ""):
        d[ch] = d.get(ch, 0) + 1
    return d

def jaccard_tokens(a: str, b: str) -> float:
    A, B = token_set(a), token_set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def overlap_chars(a: str, b: str) -> float:
    A, B = char_multiset(a), char_multiset(b)
    inter = sum(min(A.get(k, 0), B.get(k, 0)) for k in set(A) | set(B))
    denom = max(sum(A.values()), sum(B.values()), 1)
    return inter / denom

def fuzzy_score(q: str, name: str) -> float:
    return 0.7 * jaccard_tokens(q, name) + 0.3 * overlap_chars(q, name)


# ------------------------------------------------------------
# Поиск OBJ
# ------------------------------------------------------------
def _fallback_any_obj(root=EXTRACT_ROOT) -> str | None:
    best, best_size = None, -1
    for r, _d, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".obj"):
                p = os.path.join(r, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > best_size:
                    best_size, best = sz, p
    return os.path.abspath(best) if best else None

def resolve_mesh_path(name: str) -> str | None:
    try:
        proc = subprocess.run(
            [sys.executable, FIND_OBJ_SCRIPT, name],
            capture_output=True, text=True, check=True
        )
        line = (proc.stdout or "").strip().splitlines()
        line = line[-1].strip() if line else ""
        if (not line) or ("no .obj" in line.lower()):
            raise RuntimeError("finder returned no obj")
        if " " in line:
            abs_path, _fname = line.rsplit(" ", 1)
        else:
            abs_path = line
        abs_path = abs_path.strip().strip('"').strip("'")
        if os.path.isfile(abs_path):
            return os.path.abspath(abs_path)
    except Exception:
        pass
    return _fallback_any_obj()


# ------------------------------------------------------------
# Загрузка furniture_types.json
# ------------------------------------------------------------
def load_furniture_db() -> list[dict[str, Any]]:
    try:
        with open(FURNITURE_DB, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("items", []))
    except Exception:
        return []

def find_best_spec_from_db(query: str, db: list[dict[str, Any]]) -> dict[str, Any] | None:
    best, best_s = None, 0.0
    for it in db:
        s = fuzzy_score(query, it.get("name", ""))
        if s > best_s:
            best_s, best = s, it
    return best if best_s >= 0.45 else None


# ------------------------------------------------------------
# Пресеты fallback (мм)
# ------------------------------------------------------------
PRESETS = {
    "кровать":        (([1900, 1400, 450], [2200, 1800, 900]), {"touch_wall": {"sides": ["back", "left", "right"]}}),
    "двуспальная":    (([1900, 1600, 450], [2200, 2000, 900]), {"touch_wall": {"sides": ["back", "left", "right"]}}),
    "односпальная":   (([1800,  900, 450], [2000, 1200, 900]), {"touch_wall": {"sides": ["back", "left", "right"]}}),
    "тумбочка":       (([350,   350, 400], [600,  500, 700]),  {"mount_type": "floor"}),
    "комод":          (([800,   400, 700], [1800, 600, 1200]), {"mount_type": "floor"}),
    "диван":          (([1700,  800, 700], [2600, 1000, 950]), {"touch_wall": {"sides": ["back"]}}),
    "кресло":         (([700,   700, 700], [1100, 900, 1100]), {"mount_type": "floor"}),
    "стол":           (([800,   500, 720], [2000, 900, 780]),  {"mount_type": "floor"}),
    "стул":           (([350,   350, 450], [500,  500, 1000]), {"mount_type": "floor"}),
    "шкаф":           (([800,   500, 2000],[2400, 800, 2700]), {"touch_wall": {"sides": ["back"]}}),
    "полка":          (([400,    200, 30], [1600, 400, 60]),   {"mount_type": "wall", "mount_height_m": 1.5, "mount_anchor": "center"}),
}

def guess_spec_fallback(query: str) -> dict[str, Any]:
    qn = norm(query)
    for key, (rng, cons) in PRESETS.items():
        if key in qn.split() or key in qn:
            (min_mm, max_mm), constraints = rng, cons
            return {
                "name": query,
                "min_size_mm": min_mm,
                "max_size_mm": max_mm,
                "color": [0.8, 0.8, 0.8],
                "constraints": constraints,
            }
    return {
        "name": query,
        "min_size_mm": [600, 400, 600],
        "max_size_mm": [1200, 800, 1000],
        "color": [0.8, 0.8, 0.8],
        "constraints": {"mount_type": "floor"},
    }

def _texture_dirs_for_mesh(mesh_path: str | None) -> list[str]:
    if not mesh_path:
        return []
    p = Path(mesh_path).resolve()
    dirs = [
        str(p.parent),
        str(p.parent.parent),
    ]
    out = []
    seen = set()
    for d in dirs:
        if d not in seen:
            out.append(d); seen.add(d)
    return out


# ------------------------------------------------------------
# Генерация objects.json (с mesh_path)
# ------------------------------------------------------------
def generate_objects_json(requested_names: list[str], seed: int | None = None):
    db = load_furniture_db()

    items: list[dict[str, Any]] = []
    for raw_name in requested_names:
        picked = find_best_spec_from_db(raw_name, db)
        if picked is None:
            picked = guess_spec_fallback(raw_name)

        mesh_path = resolve_mesh_path(raw_name)
        if mesh_path:
            print(f"[mesh] {raw_name} → {mesh_path}")
        else:
            print(f"[mesh] {raw_name} → OBJ не найден (будет только AABB)")

        items.append({
            "name": picked["name"],
            "min_size_mm": picked["min_size_mm"],
            "max_size_mm": picked["max_size_mm"],
            "color": picked.get("color", [0.7, 0.7, 0.7]),
            "constraints": picked.get("constraints", {}),
            "mesh_path": mesh_path,
            "mesh_fit_mode": "uniform",
            "mesh_texture_dirs": _texture_dirs_for_mesh(mesh_path),
        })

    data = {"seed": int(seed) if seed is not None else None, "items": items}
    os.makedirs(os.path.dirname(OBJECTS_JSON), exist_ok=True)
    with open(OBJECTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ objects.json сгенерирован: {len(items)} предметов; seed={data['seed']}")


# ------------------------------------------------------------
# Склейка room.json + placements -> scene JSON для Blender
# ------------------------------------------------------------
def merge_room_spec_and_placements(room_json_path: str, placement_json_path: str, out_json_path: str) -> None:
    room_p = Path(room_json_path).expanduser().resolve()
    pl_p = Path(placement_json_path).expanduser().resolve()
    out_p = Path(out_json_path).expanduser().resolve()

    room = json.loads(room_p.read_text(encoding="utf-8"))
    pl = json.loads(pl_p.read_text(encoding="utf-8"))

    # placements могут лежать по-разному
    placements = pl.get("placements") or pl.get("items") or []
    if not isinstance(placements, list):
        placements = []

    # Делаем именно room-spec JSON + placements (НЕ кладём "items", чтобы builder зашёл в ветку room-spec)
    scene = dict(room)
    scene["placements"] = placements

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# Один прогон (placement → blender)
# ------------------------------------------------------------
def run_pipeline_for_mode(room_path: str, mode: str, vis_opts: argparse.Namespace, requested_items: list[str]):
    """
    room_path может быть:
      - .json (room-spec)  -> CubePlacement должен уметь читать JSON
      - .glb  (старый)     -> CubePlacement читает GLB
    """
    print(f"\n====== РЕЖИМ {mode.upper()} ======")

    room_path = os.path.abspath(room_path)
    is_room_json = room_path.lower().endswith(".json")

    for attempt in range(1, vis_opts.max_attempts + 1):
        print(f"\n---------- ПОПЫТКА {attempt} ({mode}) ----------")
        try:
            if vis_opts.regen_per_attempt:
                seed = int.from_bytes(secrets.token_bytes(8), "big")
                generate_objects_json(requested_items, seed=seed)

            # 1) Расстановка
            cube_input = f"{room_path}\n{OBJECTS_JSON}\n{mode}\n"
            subprocess.run([sys.executable, CUBE_SCRIPT], input=cube_input, text=True, check=True)

            # 2) Для Blender: если room-spec -> склеиваем room.json + placements
            scene_json_for_blender = os.path.abspath(PLACEMENT_JSON)
            auto_no_import_glb = False

            if is_room_json:
                merge_room_spec_and_placements(room_path, PLACEMENT_JSON, SCENE_JSON)
                scene_json_for_blender = os.path.abspath(SCENE_JSON)
                auto_no_import_glb = True

            # 3) Blender
            # BlenderVisualizePlacement требует --glb, поэтому:
            # - в room-spec режиме даём любой валидный путь (default room.glb), но добавляем --no-import-glb
            glb_for_arg = os.path.abspath(DEFAULT_ROOM_GLB)

            cmd = [
                sys.executable, BLENDER_VIS_SCRIPT,
                "--glb", glb_for_arg,
                "--json", scene_json_for_blender,
            ]

            if vis_opts.blender:
                cmd += ["--blender", vis_opts.blender]
            if vis_opts.headless:
                cmd.append("--background")

            # пользовательский флаг
            if vis_opts.no_import_glb:
                cmd.append("--no-import-glb")
            # автозапрет импорта GLB если room-spec
            if auto_no_import_glb and ("--no-import-glb" not in cmd):
                cmd.append("--no-import-glb")

            if vis_opts.save_blend:
                os.makedirs(os.path.dirname(vis_opts.save_blend), exist_ok=True)
                cmd += ["--save-blend", os.path.abspath(vis_opts.save_blend)]
            if vis_opts.render:
                os.makedirs(os.path.dirname(vis_opts.render), exist_ok=True)
                cmd += ["--render", os.path.abspath(vis_opts.render)]

            print("▶ Запуск Blender-визуализатора:\n ", " ".join(cmd))
            subprocess.run(cmd, check=True)

            print(f"\n✅ УСПЕХ! РЕЖИМ {mode} — сцена собрана и визуализирована")
            return
        except subprocess.CalledProcessError:
            print(f"⚠️ Неудачная попытка ({mode}), пересборка...")
            time.sleep(0.2)

    print(f"\n❌ Не удалось собрать сцену в режиме {mode} за {vis_opts.max_attempts} попыток")
    sys.exit(1)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_cli():
    p = argparse.ArgumentParser(description="Пайплайн: свободные названия → OBJ → расстановка → визуализация в Blender")
    p.add_argument("items", nargs="+", help="Любые русские названия предметов (например: кровать тумбочка диван)")

    # ВАЖНО: room-path теперь может быть glb или json
    p.add_argument("--room", default=None, help=f"Путь комнаты (.json room-spec или .glb). По умолчанию: {DEFAULT_ROOM_JSON}")

    p.add_argument("--blender", default=None, help="Путь к бинарю Blender")
    p.add_argument("--headless", action="store_true", help="Запуск Blender без GUI")
    p.add_argument("--no-import-glb", action="store_true", help="Не импортировать room.glb (только геометрия из JSON)")
    p.add_argument("--save-blend", default=None, help="Сохранить .blend")
    p.add_argument("--render", default=None, help="Сохранить PNG-рендер")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--regen-per-attempt", action="store_true", help="Перед каждой попыткой генерировать objects.json с новым seed")
    return p


def main():
    parser = build_cli()
    args = parser.parse_args()

    requested_items = args.items
    print("📦 Запрошенные предметы:")
    for it in requested_items:
        print(" -", it)

    # room path (json по умолчанию)
    room_path = args.room or DEFAULT_ROOM_JSON
    room_path = room_path.strip()

    # база габаритов (мягкая)
    global FURNITURE_DB
    furn_default = FURNITURE_DB
    furn_path = input(f"Файл базы мебели (.json) [{furn_default}]: ").strip() or furn_default
    FURNITURE_DB = furn_path

    # objects.json (с mesh_path)
    if not args.regen_per_attempt:
        seed0 = int.from_bytes(secrets.token_bytes(8), "big")
        generate_objects_json(requested_items, seed=seed0)

    # два режима
    run_pipeline_for_mode(room_path, mode="random",  vis_opts=args, requested_items=requested_items)
    run_pipeline_for_mode(room_path, mode="relaxed", vis_opts=args, requested_items=requested_items)

    print("\n✅ ОБА РЕЖИМА ОТРАБОТАЛИ УСПЕШНО")


if __name__ == "__main__":
    main()
