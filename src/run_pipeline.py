#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/run_pipeline.py

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import secrets
from typing import Any, Optional
from pathlib import Path


# ------------------------------------------------------------
# Пути
# ------------------------------------------------------------
CUBE_SCRIPT = "src/Plasement/CubePlacement.py"
BLENDER_VIS_SCRIPT = "src/Plasement/BlenderVisualizePlacement.py"

DEFAULT_ROOM_GLB = "data/input/room.glb"
DEFAULT_ROOM_JSON = "data/input/room.json"  # room-spec (JSON)

FURNITURE_DB = "data/input/furniture_types.json"
OBJECTS_JSON = "data/input/objects.json"

PLACEMENT_JSON = "data/output/placement_result.json"
SCENE_JSON = "data/output/scene_room_and_placements.json"  # room-spec + placements для Blender

IMODERN_ROOT_DEFAULT = "data/sourse/imodern"
FUTURE_ROOT_DEFAULT = "data/sourse/3D-FRONT/3D-FUTURE-model"
FUTURE_INFO_DEFAULT = "data/sourse/3D-FRONT/3D-FUTURE-model/model_info.json"

MAX_ATTEMPTS = 30


# ------------------------------------------------------------
# Нормализация и метрики похожести (fallback)
# ------------------------------------------------------------
_word_re = re.compile(r"[А-Яа-яA-Za-z0-9]+")


def norm(s: str) -> str:
    return " ".join(_word_re.findall((s or "").lower().replace("ё", "е")))


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
# Типы (категории) по тексту
# ------------------------------------------------------------
_TYPE_KEYS = {
    "bed":      {"кровать", "кроват", "двуспаль", "односпаль", "полутор", "трехспаль", "трёхспаль"},
    "sofa":     {"диван", "реклайнер", "софа"},
    "tv_stand": {"тумба", "тумбочка", "комод", "тв", "tv"},
    "chair":    {"стул", "табурет"},
    "armchair": {"кресло", "кресл"},
    "table":    {"стол", "письменный", "журнальный", "обеденный", "раскладной", "консоль"},
    "wardrobe": {"шкаф", "гардероб", "стеллаж", "витрина", "буфет", "полка", "тумба"},
    "lighting": {"лампа", "свет", "люстра", "бра", "торшер"},
}


def _guess_type_from_text(text: str) -> str | None:
    ts = token_set(text)
    for typ, keys in _TYPE_KEYS.items():
        if ts & keys:
            return typ
    return None


def _split_by_underscore(name: str) -> list[str]:
    s = (name or "").replace("ё", "е")
    s = s.replace(" ", "_").replace("-", "_")
    parts = [p.strip().lower() for p in s.split("_")]
    return [p for p in parts if p]


# ------------------------------------------------------------
# IMODERN: каталог mesh по ФС
# ------------------------------------------------------------
_IMODERN_CATALOG: list[dict[str, Any]] | None = None


def _scan_imodern_catalog(root: str) -> list[dict[str, Any]]:
    """
    Сканируем root рекурсивно.
    Для каждой найденной .obj берём asset_name = имя папки 1-го уровня внутри root:
      root/<asset_name>/<something>/model.obj
      root/<asset_name>/model.obj
    """
    out: list[dict[str, Any]] = []
    root_abs = Path(root).resolve()
    if not root_abs.is_dir():
        return out

    for dp, _d, files in os.walk(root_abs):
        mesh = None
        for f in files:
            if f.lower().endswith(".obj"):
                mesh = str((Path(dp) / f).resolve())
                break
        if not mesh:
            continue

        dp_path = Path(dp).resolve()
        try:
            rel = dp_path.relative_to(root_abs)
            asset_name = rel.parts[0] if rel.parts else dp_path.name
        except Exception:
            asset_name = dp_path.name

        out.append({
            "name": asset_name,
            "name_tokens": _split_by_underscore(asset_name),
            "mesh_path": mesh,
        })

    return out


def _candidate_has_any_key(candidate_tokens: list[str], keys: set[str]) -> bool:
    if not candidate_tokens:
        return False
    for t in candidate_tokens:
        for k in keys:
            if t == k or t.startswith(k):
                return True
    return False


def resolve_mesh_path_imodern(query_name: str, rng: random.Random, imodern_root: str) -> str | None:
    global _IMODERN_CATALOG
    if _IMODERN_CATALOG is None:
        _IMODERN_CATALOG = _scan_imodern_catalog(imodern_root)

    if not _IMODERN_CATALOG:
        return None

    q_type = _guess_type_from_text(query_name)

    if q_type is not None and q_type in _TYPE_KEYS:
        keys = _TYPE_KEYS[q_type]
        cands = [it for it in _IMODERN_CATALOG if _candidate_has_any_key(it["name_tokens"], keys)]
        if cands:
            chosen = rng.choice(cands)
            return chosen["mesh_path"]

    best, best_s = None, -1.0
    for it in _IMODERN_CATALOG:
        s = fuzzy_score(query_name, it["name"])
        if s > best_s:
            best_s, best = s, it
    return best["mesh_path"] if best else None


# ------------------------------------------------------------
# 3D-FUTURE: загрузка model_info.json и выбор по категориям
# ------------------------------------------------------------
_3DFUTURE_ASSETS: list[dict[str, Any]] | None = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_3dfuture_index(
    future_root: str,
    future_info: str,
    style: Optional[str],
    material: Optional[str],
    theme: Optional[str],
) -> list[dict[str, Any]]:
    """
    Индекс строится по model_info.json. В индекс попадает только то, что реально существует на диске:
      <root>/<model_id>/normalized_model.obj (pref) или raw_model.obj (fallback)
      model.mtl (optional)
      texture.png (optional)
    """
    root = Path(future_root).resolve()
    info_path = Path(future_info).resolve()
    if not root.is_dir():
        return []
    if not info_path.is_file():
        return []

    info = _read_json(info_path)
    out: list[dict[str, Any]] = []

    for rec in info:
        mid = rec.get("model_id")
        if not mid:
            continue

        if style is not None and rec.get("style") != style:
            continue
        if material is not None and rec.get("material") != material:
            continue
        if theme is not None and rec.get("theme") != theme:
            continue

        d = root / mid
        if not d.is_dir():
            continue

        obj_norm = d / "normalized_model.obj"
        obj_raw = d / "raw_model.obj"
        obj_path = obj_norm if obj_norm.is_file() else obj_raw
        if not obj_path.is_file():
            continue

        mtl_path = d / "model.mtl"
        tex_path = d / "texture.png"

        out.append({
            "model_id": mid,
            "dir": str(d),
            "obj": str(obj_path),
            "mtl": str(mtl_path) if mtl_path.is_file() else None,
            "texture": str(tex_path) if tex_path.is_file() else None,
            "super_category": rec.get("super-category"),
            "category": rec.get("category"),
            "style": rec.get("style"),
            "theme": rec.get("theme"),
            "material": rec.get("material"),
        })

    return out


def _supercats_for_type(q_type: str | None) -> list[str]:
    """
    Логика соответствия: внутренний тип -> super-category из 3D-FUTURE.
    """
    if q_type == "chair":
        return ["Chair"]
    if q_type == "armchair":
        return ["Sofa"]  # armchair внутри super-category "Sofa"
    if q_type == "sofa":
        return ["Sofa"]
    if q_type == "table":
        return ["Table", "Cabinet/Shelf/Desk"]  # coffee/side table иногда в Cabinet/Shelf/Desk
    if q_type == "bed":
        return ["Bed"]
    if q_type in ("wardrobe", "tv_stand"):
        return ["Cabinet/Shelf/Desk"]
    if q_type == "lighting":
        return ["Lighting"]
    return ["Chair", "Table", "Sofa", "Cabinet/Shelf/Desk", "Bed", "Lighting"]


def _prefer_category_for_type(q_type: str | None, a: dict[str, Any]) -> bool:
    """
    Дополнительное уточнение внутри super-category, чтобы armchair != sofa, chair != stool и т.д.
    Здесь только грубый фильтр, без жёстких требований.
    """
    cat = (a.get("category") or "").lower()

    if q_type == "armchair":
        return "armchair" in cat or "chair" in cat
    if q_type == "chair":
        return "chair" in cat or "barstool" in cat or "stool" in cat
    if q_type == "tv_stand":
        return "tv" in cat or "stand" in cat or "console" in cat or "sideboard" in cat
    if q_type == "wardrobe":
        return "wardrobe" in cat or "bookcase" in cat or "shelf" in cat or "cabinet" in cat or "drawer" in cat or "sideboard" in cat
    if q_type == "table":
        return "table" in cat or "desk" in cat or "bar" in cat
    if q_type == "bed":
        return "bed" in cat
    if q_type == "lighting":
        return "lamp" in cat
    return True


def resolve_mesh_path_3dfuture(
    query_name: str,
    rng: random.Random,
    future_root: str,
    future_info: str,
    style: Optional[str],
    material: Optional[str],
    theme: Optional[str],
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Возвращает:
      mesh_path, meta_record
    """
    global _3DFUTURE_ASSETS
    if _3DFUTURE_ASSETS is None:
        _3DFUTURE_ASSETS = _build_3dfuture_index(
            future_root=future_root,
            future_info=future_info,
            style=style,
            material=material,
            theme=theme,
        )

    if not _3DFUTURE_ASSETS:
        return None, None

    q_type = _guess_type_from_text(query_name)
    supercats = set(_supercats_for_type(q_type))

    # 1) кандидаты по super-category
    cands = [a for a in _3DFUTURE_ASSETS if a.get("super_category") in supercats]
    if not cands:
        cands = list(_3DFUTURE_ASSETS)

    # 2) мягкое уточнение по category
    refined = [a for a in cands if _prefer_category_for_type(q_type, a)]
    if refined:
        cands = refined

    # 3) если всё ещё очень много, слегка приоритезируем по “похожести” на текст запроса
    # (это не поиск точного соответствия, только подсказка)
    scored = []
    for a in cands:
        name = f'{a.get("super_category","")} {a.get("category","")} {a.get("style","")}'
        scored.append((fuzzy_score(query_name, name), a))
    scored.sort(key=lambda x: x[0], reverse=True)

    # берем верхнюю “корзину” и семплируем из неё, чтобы не получать всегда один и тот же
    top_k = min(200, len(scored))
    pool = [a for _s, a in scored[:top_k]] if top_k > 0 else cands

    chosen = rng.choice(pool)
    return chosen["obj"], chosen


def _texture_dirs_for_mesh(mesh_path: str | None) -> list[str]:
    if not mesh_path:
        return []
    p = Path(mesh_path).resolve()
    dirs = [str(p.parent), str(p.parent.parent)]
    out: list[str] = []
    seen: set[str] = set()
    for d in dirs:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def _texture_dirs_for_3dfuture(meta: dict[str, Any] | None) -> list[str]:
    if not meta:
        return []
    d = meta.get("dir")
    if not d:
        return []
    return [str(Path(d).resolve())]


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
    "люстра":         (([300,   300, 200], [1200, 1200, 800]), {"mount_type": "ceiling"}),
    "лампа":          (([200,   200, 200], [800,  800, 1800]), {"mount_type": "floor"}),
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


# ------------------------------------------------------------
# objects.json: генерация
# ------------------------------------------------------------
def generate_objects_json(
    requested_names: list[str],
    seed: int | None,
    asset_source: str,
    imodern_root: str,
    future_root: str,
    future_info: str,
    future_style: Optional[str],
    future_material: Optional[str],
    future_theme: Optional[str],
) -> None:
    """
    Логика:
      - габариты берём из furniture_types.json (если нашли) иначе fallback пресеты
      - mesh_path выбираем из источника:
          imodern: скан ФС + ключи по типам
          3dfuture: model_info + super-category/category фильтры
      - seed кладём в objects.json (CubePlacement использует его как RNG seed расстановки)
    """
    db = load_furniture_db()
    rng = random.Random(int(seed)) if seed is not None else random.Random()

    items: list[dict[str, Any]] = []
    for raw_name in requested_names:
        picked = find_best_spec_from_db(raw_name, db)
        if picked is None:
            picked = guess_spec_fallback(raw_name)

        mesh_path: Optional[str] = None
        src_meta: Optional[dict[str, Any]] = None

        if asset_source == "imodern":
            mesh_path = resolve_mesh_path_imodern(raw_name, rng=rng, imodern_root=imodern_root)
            if mesh_path:
                print(f"[mesh][imodern] {raw_name} → {mesh_path}")
            else:
                print(f"[mesh][imodern] {raw_name} → OBJ не найден (будет только AABB)")

        elif asset_source == "3dfuture":
            mesh_path, src_meta = resolve_mesh_path_3dfuture(
                query_name=raw_name,
                rng=rng,
                future_root=future_root,
                future_info=future_info,
                style=future_style,
                material=future_material,
                theme=future_theme,
            )
            if mesh_path:
                sc = src_meta.get("super_category") if src_meta else None
                cat = src_meta.get("category") if src_meta else None
                mid = src_meta.get("model_id") if src_meta else None
                print(f"[mesh][3dfuture] {raw_name} → {mesh_path} | {sc}/{cat} | {mid}")
            else:
                print(f"[mesh][3dfuture] {raw_name} → OBJ не найден (будет только AABB)")

        else:
            raise ValueError(f"Unknown asset_source: {asset_source}")

        mesh_texture_dirs = (
            _texture_dirs_for_3dfuture(src_meta) if asset_source == "3dfuture" else _texture_dirs_for_mesh(mesh_path)
        )

        item = {
            "name": picked["name"],
            "min_size_mm": picked["min_size_mm"],
            "max_size_mm": picked["max_size_mm"],
            "color": picked.get("color", [0.7, 0.7, 0.7]),
            "constraints": picked.get("constraints", {}),
            "mesh_path": mesh_path,
            "mesh_fit_mode": "uniform",
            "mesh_texture_dirs": mesh_texture_dirs,
            "asset_source": asset_source,
        }

        if asset_source == "3dfuture" and src_meta is not None:
            item["asset_meta"] = {
                "model_id": src_meta.get("model_id"),
                "super_category": src_meta.get("super_category"),
                "category": src_meta.get("category"),
                "style": src_meta.get("style"),
                "theme": src_meta.get("theme"),
                "material": src_meta.get("material"),
                "dir": src_meta.get("dir"),
                "mtl": src_meta.get("mtl"),
                "texture": src_meta.get("texture"),
            }

        items.append(item)

    data = {"seed": int(seed) if seed is not None else None, "items": items}
    Path(OBJECTS_JSON).resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(OBJECTS_JSON).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ objects.json сгенерирован: {len(items)} предметов; seed={data['seed']}; source={asset_source}")


def rewrite_objects_seed(seed: int) -> None:
    p = Path(OBJECTS_JSON).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"objects.json не найден: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["seed"] = int(seed)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# Склейка room.json + placements -> scene JSON для Blender
# ------------------------------------------------------------
def merge_room_spec_and_placements(room_json_path: str, placement_json_path: str, out_json_path: str) -> None:
    room_p = Path(room_json_path).expanduser().resolve()
    pl_p = Path(placement_json_path).expanduser().resolve()
    out_p = Path(out_json_path).expanduser().resolve()

    room = json.loads(room_p.read_text(encoding="utf-8"))
    pl = json.loads(pl_p.read_text(encoding="utf-8"))

    placements = pl.get("placements") or pl.get("items") or []
    if not isinstance(placements, list):
        placements = []

    scene = dict(room)
    scene["placements"] = placements

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------
# Один прогон (placement → blender)
# ------------------------------------------------------------
def run_pipeline_for_mode(room_path: str, mode: str, vis_opts: argparse.Namespace, requested_items: list[str]) -> None:
    print(f"\n====== РЕЖИМ {mode.upper()} ======")

    room_path = os.path.abspath(room_path)
    is_room_json = room_path.lower().endswith(".json")

    for attempt in range(1, vis_opts.max_attempts + 1):
        print(f"\n---------- ПОПЫТКА {attempt} ({mode}) ----------")
        try:
            attempt_seed = int.from_bytes(secrets.token_bytes(8), "big")

            if vis_opts.regen_per_attempt:
                generate_objects_json(
                    requested_names=requested_items,
                    seed=attempt_seed,
                    asset_source=vis_opts.asset_source,
                    imodern_root=vis_opts.imodern_root,
                    future_root=vis_opts.future_root,
                    future_info=vis_opts.future_info,
                    future_style=vis_opts.future_style,
                    future_material=vis_opts.future_material,
                    future_theme=vis_opts.future_theme,
                )
            else:
                if mode == "random":
                    rewrite_objects_seed(attempt_seed)

            cube_input = f"{room_path}\n{OBJECTS_JSON}\n{mode}\n"
            subprocess.run([sys.executable, CUBE_SCRIPT], input=cube_input, text=True, check=True)

            scene_json_for_blender = os.path.abspath(PLACEMENT_JSON)
            auto_no_import_glb = False

            if is_room_json:
                merge_room_spec_and_placements(room_path, PLACEMENT_JSON, SCENE_JSON)
                scene_json_for_blender = os.path.abspath(SCENE_JSON)
                auto_no_import_glb = True

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

            if vis_opts.no_import_glb:
                cmd.append("--no-import-glb")
            if auto_no_import_glb and ("--no-import-glb" not in cmd):
                cmd.append("--no-import-glb")

            if vis_opts.save_blend:
                Path(vis_opts.save_blend).resolve().parent.mkdir(parents=True, exist_ok=True)
                cmd += ["--save-blend", os.path.abspath(vis_opts.save_blend)]
            if vis_opts.render:
                Path(vis_opts.render).resolve().parent.mkdir(parents=True, exist_ok=True)
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
    p = argparse.ArgumentParser(description="Пайплайн: названия → OBJ → расстановка → визуализация в Blender")
    p.add_argument("items", nargs="+", help="Русские названия предметов (например: кровать тумбочка диван)")

    p.add_argument("--room", default=None, help=f"Путь комнаты (.json room-spec или .glb). По умолчанию: {DEFAULT_ROOM_JSON}")

    p.add_argument("--blender", default=None, help="Путь к бинарю Blender")
    p.add_argument("--headless", action="store_true", help="Запуск Blender без GUI")
    p.add_argument("--no-import-glb", action="store_true", help="Не импортировать room.glb (только геометрия из JSON)")
    p.add_argument("--save-blend", default=None, help="Сохранить .blend")
    p.add_argument("--render", default=None, help="Сохранить PNG-рендер")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--regen-per-attempt", action="store_true", help="На каждой попытке пересобирать objects.json (и mesh, и seed)")

    p.add_argument("--asset-source", choices=["imodern", "3dfuture"], default="3dfuture")

    p.add_argument("--imodern-root", default=IMODERN_ROOT_DEFAULT)

    p.add_argument("--future-root", default=FUTURE_ROOT_DEFAULT)
    p.add_argument("--future-info", default=FUTURE_INFO_DEFAULT)

    p.add_argument("--future-style", default=None)     # например "Modern"
    p.add_argument("--future-material", default=None)  # например "Wood"
    p.add_argument("--future-theme", default=None)     # например "Lines"
    return p


def main():
    parser = build_cli()
    args = parser.parse_args()

    requested_items = args.items
    print("📦 Запрошенные предметы:")
    for it in requested_items:
        print(" -", it)

    room_path = (args.room or DEFAULT_ROOM_JSON).strip()

    global FURNITURE_DB
    furn_default = FURNITURE_DB
    furn_path = input(f"Файл базы мебели (.json) [{furn_default}]: ").strip() or furn_default
    FURNITURE_DB = furn_path

    if not args.regen_per_attempt:
        seed0 = int.from_bytes(secrets.token_bytes(8), "big")
        generate_objects_json(
            requested_names=requested_items,
            seed=seed0,
            asset_source=args.asset_source,
            imodern_root=args.imodern_root,
            future_root=args.future_root,
            future_info=args.future_info,
            future_style=args.future_style,
            future_material=args.future_material,
            future_theme=args.future_theme,
        )

    run_pipeline_for_mode(room_path, mode="random",  vis_opts=args, requested_items=requested_items)
    run_pipeline_for_mode(room_path, mode="relaxed", vis_opts=args, requested_items=requested_items)

    print("\n✅ ОБА РЕЖИМА ОТРАБОТАЛИ УСПЕШНО")


if __name__ == "__main__":
    main()