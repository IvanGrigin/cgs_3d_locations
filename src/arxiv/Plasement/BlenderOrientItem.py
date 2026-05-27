# BlenderOrientItem.py (Blender 5.x safe) + DB write + wizard
# Usage:
#   blender --factory-startup --python BlenderOrientItem.py -- \
#     --db <path.sqlite> --item-id <int> --bx-id <bx_...> --obj <path.obj>

import json
import sys
import threading
import queue
import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy

# --- make local imports work when run from absolute path ---
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import blender_build_from_obj as build

DEFAULT_FRONT_CLEARANCE_M = build.DEFAULT_FRONT_CLEARANCE_M

# ----------------------------- Small helpers -----------------------------

def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def _is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

def _normalize_probs(p: Dict[str, float], fallback: Dict[str, float]) -> Dict[str, float]:
    return build._normalize_probs(p, fallback)

# ----------------------------- DB write -----------------------------

def _db_connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path), timeout=5.0)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;")
    return con

def _db_update_item(db_path: Path, item_id: int, **fields) -> None:
    if not fields:
        return
    con = _db_connect(db_path)
    try:
        keys = list(fields.keys())
        sql = "UPDATE imodern_item SET " + ", ".join([f"{k}=?" for k in keys]) + " WHERE id=?"
        con.execute(sql, [fields[k] for k in keys] + [item_id])
        con.commit()
    finally:
        con.close()

# ----------------------------- UI / Metadata -----------------------------

@dataclass
class OrientationMeta:
    item_source: str
    item_name: str
    scale_power10_k: int
    scale_factor: float
    bbox_dims_before_raw: Tuple[float, float, float]
    bbox_dims_after_m: Tuple[float, float, float]
    unit_guess: str

    wall_contact_required: bool
    wall_contact_prob: Dict[str, float]
    floor_contact_required: bool
    ceiling_contact_required: bool
    front_clearance_m: float

    world_matrix_4x4: Optional[List[List[float]]] = None

    has_base_textures: bool = False
    texture_files: Optional[List[str]] = None
    texture_apply_error: Optional[str] = None

def _matrix_to_list(m) -> List[List[float]]:
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]

# ----------------------------- Console wizard -----------------------------

_WIZ_Q: "queue.Queue[dict]" = queue.Queue()
_WIZ_THREAD: Optional[threading.Thread] = None

_SIDE_ALIASES = {
    "top": "top", "up": "top", "ceiling": "top",
    "bottom": "bottom", "bot": "bottom", "down": "bottom", "floor": "bottom",
    "left": "left", "l": "left",
    "right": "right", "r": "right",
    "front": "front", "f": "front",
    "back": "back", "rear": "back",
    "верх": "top", "верхний": "top",
    "низ": "bottom", "нижний": "bottom",
    "лево": "left", "левый": "left",
    "право": "right", "правый": "right",
    "перед": "front", "передний": "front",
    "зад": "back", "задний": "back",
}

def _ask_bool(prompt: str, default: bool) -> bool:
    d = "y" if default else "n"
    s = input(f"{prompt} [y/n, default={d}]: ").strip().lower()
    if not s:
        return default
    if s in ("y", "yes", "1", "да", "д"):
        return True
    if s in ("n", "no", "0", "нет", "н"):
        return False
    return default

def _ask_side(prompt: str, default_side: str) -> str:
    s = input(f"{prompt} [top/bottom/left/right/front/back, default={default_side}]: ").strip().lower()
    if not s:
        return default_side
    s = s.replace("+", "").replace("-", "").strip()
    return _SIDE_ALIASES.get(s, default_side)

def _ask_probs(default: Dict[str, float]) -> Dict[str, float]:
    print("Введите вероятности касания со стеной по сторонам в порядке:")
    print("  back left right front")
    print(f"По умолчанию: {default['back']} {default['left']} {default['right']} {default['front']}")
    s = input("Ввод (пусто = default): ").strip()
    if not s:
        return default.copy()
    parts = s.replace(",", ".").split()
    if len(parts) != 4:
        print("Ожидалось 4 числа -> использую default.")
        return default.copy()
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        print("Некорректные числа -> использую default.")
        return default.copy()
    probs = dict(zip(["back", "left", "right", "front"], vals))
    return _normalize_probs(probs, fallback=default)

def _wizard_thread_body():
    if not _is_tty():
        _WIZ_Q.put({"mode": "error", "msg": "stdin is not a TTY. Запускайте Blender из терминала. При необходимости добавьте '</dev/tty'."})
        return

    print("\n[CGS] ===== Console wizard =====")
    print("[CGS] В Blender: вращайте ITEM как нужно (мышь/Rotate).")
    input("[CGS] Когда закончите ориентацию, нажмите Enter в консоли, чтобы продолжить... ")

    ok = _ask_bool("[CGS] Подтверждаете, что ориентация корректна?", default=True)
    if not ok:
        print("\n[CGS] Ок. Тогда задайте, КАК ДОЛЖНО БЫТЬ (для следующего запуска).")
        red_to   = _ask_side("RED (+X) -> куда?", "right")
        green_to = _ask_side("GREEN (+Y) -> куда?", "front")
        blue_to  = _ask_side("BLUE (+Z) -> куда?", "top")
        _WIZ_Q.put({"mode": "restart", "want": {"red": red_to, "green": green_to, "blue": blue_to}})
        return

    floor_required = _ask_bool("[CGS] Должен ли касаться пола (нижней частью)?", default=True)
    ceiling_required = _ask_bool("[CGS] Должен ли касаться потолка (верхней частью)?", default=False)
    wall_required = _ask_bool("[CGS] Должен ли касаться стены?", default=True)

    default_probs = {"back": 0.8, "left": 0.1, "right": 0.1, "front": 0.0}
    wall_prob = default_probs.copy()
    if wall_required:
        wall_prob = _ask_probs(default_probs)

    _WIZ_Q.put({
        "mode": "save",
        "floor_required": bool(floor_required),
        "ceiling_required": bool(ceiling_required),
        "wall_required": bool(wall_required),
        "wall_prob": wall_prob,
    })

def _wizard_poll_timer():
    try:
        msg = _WIZ_Q.get_nowait()
    except queue.Empty:
        return 0.2

    mode = msg.get("mode")

    if mode == "error":
        print(f"[CGS] ❌ {msg.get('msg')}")
        try:
            init = bpy.context.scene.get("cgs_orient_init_meta", None)
            if init:
                _db_update_item(
                    Path(init["db_path"]), int(init["item_id"]),
                    orient_status="error",
                    orient_error=str(msg.get("msg")),
                    orient_updated_at=_utc_now_iso(),
                )
        except Exception:
            pass
        return None

    if mode == "restart":
        want = msg["want"]
        print("\n[CGS] ===== ORIENTATION NOT CONFIRMED =====")
        print(f"  RED   (+X) -> {want['red']}")
        print(f"  GREEN (+Y) -> {want['green']}")
        print(f"  BLUE  (+Z) -> {want['blue']}")
        print("[CGS] Закрываю Blender...\n")
        try:
            init = bpy.context.scene.get("cgs_orient_init_meta", None)
            if init:
                _db_update_item(
                    Path(init["db_path"]), int(init["item_id"]),
                    orient_status="not_confirmed",
                    orient_error="user not confirmed orientation",
                    orient_updated_at=_utc_now_iso(),
                )
        except Exception:
            pass
        try:
            bpy.ops.wm.quit_blender()
        except Exception:
            pass
        return None

    if mode == "save":
        s = bpy.context.scene.cgs_orient_settings
        item = bpy.data.objects.get("ITEM")
        init = bpy.context.scene.get("cgs_orient_init_meta", None)

        if item is None or not init:
            print("[CGS] ❌ Не найден ITEM или init_meta. Закрываю Blender.")
            try:
                if init:
                    _db_update_item(
                        Path(init["db_path"]), int(init["item_id"]),
                        orient_status="error",
                        orient_error="ITEM/init_meta missing",
                        orient_updated_at=_utc_now_iso(),
                    )
            except Exception:
                pass
            try:
                bpy.ops.wm.quit_blender()
            except Exception:
                pass
            return None

        s.floor_required = msg["floor_required"]
        s.ceiling_required = msg["ceiling_required"]
        s.wall_required = msg["wall_required"]

        probs = _normalize_probs(
            msg["wall_prob"],
            fallback={"back": 0.8, "left": 0.1, "right": 0.1, "front": 0.0},
        )

        meta = OrientationMeta(
            item_source=init["item_source"],
            item_name=init["item_name"],
            scale_power10_k=int(init["scale_power10_k"]),
            scale_factor=float(init["scale_factor"]),
            bbox_dims_before_raw=tuple(init["bbox_before_raw"]),
            bbox_dims_after_m=tuple(init["bbox_after_m"]),
            unit_guess=str(init.get("unit_guess", "unknown")),
            wall_contact_required=bool(s.wall_required),
            wall_contact_prob=probs,
            floor_contact_required=bool(s.floor_required),
            ceiling_contact_required=bool(s.ceiling_required),
            front_clearance_m=float(getattr(s, "front_clearance", DEFAULT_FRONT_CLEARANCE_M)),
            world_matrix_4x4=_matrix_to_list(item.matrix_world),
            has_base_textures=bool(init.get("has_base_textures", False)),
            texture_files=list(init.get("texture_files", [])) or [],
            texture_apply_error=init.get("texture_apply_error", None),
        )

        db_path = Path(init["db_path"])
        item_id = int(init["item_id"])

        try:
            _db_update_item(
                db_path, item_id,
                orient_status="ok",
                orient_error=None,
                orient_updated_at=_utc_now_iso(),

                has_base_textures=1 if meta.has_base_textures else 0,
                texture_files=json.dumps(meta.texture_files or [], ensure_ascii=False),
                texture_apply_error=meta.texture_apply_error,

                scale_power10_k=int(meta.scale_power10_k),
                scale_factor=float(meta.scale_factor),
                unit_guess=str(meta.unit_guess),

                bbox_raw_x=float(meta.bbox_dims_before_raw[0]),
                bbox_raw_y=float(meta.bbox_dims_before_raw[1]),
                bbox_raw_z=float(meta.bbox_dims_before_raw[2]),

                bbox_m_x=float(meta.bbox_dims_after_m[0]),
                bbox_m_y=float(meta.bbox_dims_after_m[1]),
                bbox_m_z=float(meta.bbox_dims_after_m[2]),

                wall_contact_required=1 if meta.wall_contact_required else 0,
                floor_contact_required=1 if meta.floor_contact_required else 0,
                ceiling_contact_required=1 if meta.ceiling_contact_required else 0,
                front_clearance_m=float(meta.front_clearance_m),

                wall_prob_back=float(meta.wall_contact_prob["back"]),
                wall_prob_left=float(meta.wall_contact_prob["left"]),
                wall_prob_right=float(meta.wall_contact_prob["right"]),
                wall_prob_front=float(meta.wall_contact_prob["front"]),

                world_matrix_4x4=json.dumps(meta.world_matrix_4x4, ensure_ascii=False),
            )
            print(f"\n[CGS] ✅ Saved to DB: {db_path} (imodern_item.id={item_id})")
        except Exception as e:
            print(f"\n[CGS] ❌ DB save failed: {e!r}")
            try:
                _db_update_item(
                    db_path, item_id,
                    orient_status="db_save_failed",
                    orient_error=repr(e),
                    orient_updated_at=_utc_now_iso(),
                )
            except Exception:
                pass

        print("[CGS] Закрываю Blender.\n")
        try:
            bpy.ops.wm.quit_blender()
        except Exception:
            pass
        return None

    return None

def _start_console_wizard_async():
    global _WIZ_THREAD
    if _WIZ_THREAD is not None and _WIZ_THREAD.is_alive():
        return None

    _WIZ_THREAD = threading.Thread(target=_wizard_thread_body, daemon=True)
    _WIZ_THREAD.start()
    bpy.app.timers.register(_wizard_poll_timer, first_interval=0.2)
    return None

# ----------------------------- Main -----------------------------

def _parse_args(argv: List[str]) -> Tuple[Path, Path, int, str]:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    db = None
    item_id = None
    bx_id = None
    obj = None

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--db" and i + 1 < len(argv):
            db = argv[i + 1]; i += 2
        elif a == "--item-id" and i + 1 < len(argv):
            item_id = argv[i + 1]; i += 2
        elif a == "--bx-id" and i + 1 < len(argv):
            bx_id = argv[i + 1]; i += 2
        elif a == "--obj" and i + 1 < len(argv):
            obj = argv[i + 1]; i += 2
        else:
            i += 1

    if not db or not item_id or not bx_id or not obj:
        raise RuntimeError("Required args: --db --item-id --bx-id --obj")

    db_path = Path(db).expanduser().resolve()
    obj_path = Path(obj).expanduser().resolve()
    return obj_path, db_path, int(item_id), str(bx_id)

_DEFER_OBJ: Optional[Path] = None
_DEFER_DB: Optional[Path] = None
_DEFER_ITEM_ID: Optional[int] = None
_DEFER_BX: Optional[str] = None
_ATTEMPTS = 0
_MAX_ATTEMPTS = 200

def _ui_ready() -> bool:
    wm = bpy.context.window_manager
    if wm is None or not wm.windows:
        return False
    win = wm.windows[0]
    if win.screen is None:
        return False
    for area in win.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    return True
    return False

def _deferred_start():
    global _ATTEMPTS
    _ATTEMPTS += 1
    if not _ui_ready():
        if _ATTEMPTS < _MAX_ATTEMPTS:
            return 0.1
        print("[CGS] UI not ready after many attempts; abort.")
        return None

    try:
        # mark started
        try:
            _db_update_item(
                _DEFER_DB, int(_DEFER_ITEM_ID),
                orient_status="in_blender",
                orient_error=None,
                orient_updated_at=_utc_now_iso(),
            )
        except Exception:
            pass

        init_meta_scene = build.build_scene_from_obj(_DEFER_OBJ)

        bpy.context.scene["cgs_orient_init_meta"] = {
            "db_path": str(_DEFER_DB),
            "item_id": int(_DEFER_ITEM_ID),
            "bx_id": str(_DEFER_BX),
            **init_meta_scene,
        }

        # early texture info
        try:
            _db_update_item(
                _DEFER_DB, int(_DEFER_ITEM_ID),
                has_base_textures=1 if init_meta_scene.get("has_base_textures") else 0,
                texture_files=json.dumps(init_meta_scene.get("texture_files") or [], ensure_ascii=False),
                texture_apply_error=init_meta_scene.get("texture_apply_error"),
                orient_updated_at=_utc_now_iso(),
            )
        except Exception:
            pass

        print(f"[CGS] DB target: {_DEFER_DB} imodern_item.id={_DEFER_ITEM_ID} bx_id={_DEFER_BX}\n")

        bpy.app.timers.register(_start_console_wizard_async, first_interval=0.3)

    except Exception as e:
        print("[CGS] Unhandled exception:", repr(e))
        try:
            if _DEFER_DB and _DEFER_ITEM_ID is not None:
                _db_update_item(
                    _DEFER_DB, int(_DEFER_ITEM_ID),
                    orient_status="exception",
                    orient_error=repr(e),
                    orient_updated_at=_utc_now_iso(),
                )
        except Exception:
            pass

    return None

def main():
    global _DEFER_OBJ, _DEFER_DB, _DEFER_ITEM_ID, _DEFER_BX
    obj_path, db_path, item_id, bx_id = _parse_args(sys.argv)
    _DEFER_OBJ = obj_path
    _DEFER_DB = db_path
    _DEFER_ITEM_ID = item_id
    _DEFER_BX = bx_id
    bpy.app.timers.register(_deferred_start, first_interval=0.1)

if __name__ == "__main__":
    main()
