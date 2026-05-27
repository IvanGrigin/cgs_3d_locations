#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ensure_pillow():
    try:
        from PIL import Image, ImageTk, ImageDraw  # type: ignore
        return Image, ImageTk, ImageDraw
    except Exception:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        answer = messagebox.askyesno(
            "Требуется Pillow / Pillow required",
            "Для открытия PNG/JPG/JPEG нужен пакет Pillow.\n\n"
            "Установить его автоматически сейчас?\n\n"
            "The Pillow package is required for PNG/JPG/JPEG support.\n\n"
            "Install it automatically now?",
        )
        root.destroy()
        if not answer:
            raise SystemExit(
                "Установите Pillow командой: python -m pip install Pillow\n"
                "Install Pillow with: python -m pip install Pillow"
            )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageTk, ImageDraw  # type: ignore
        return Image, ImageTk, ImageDraw


Image, ImageTk, ImageDraw = ensure_pillow()

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

APP_TITLE = "Floorplan Annotator"
APP_VERSION = "2.9"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ANNOT_SUFFIX = ".floorplan_annot.json"
STATE_SUFFIX = ".floorplan_state.json"
HISTORY_SUFFIX = ".floorplan_history.jsonl"
SESSION_FILE = "_floorplan_annotator_session.json"
GLOBAL_LOG_FILE = "_floorplan_annotator.log"
ACTIVITY_LOG_FILE = "_floorplan_activity.jsonl"
SNAPSHOT_EVERY_SECONDS = 30
IDLE_GAP_SECONDS = 120

ROOM_FILL_ALPHA = 64
ROOM_FILL_ALPHA_ACTIVE = 82
ROOM_FILL_ALPHA_SELECTED = 96
ITEM_BORDER_WIDTH = 2
ROOM_BORDER_WIDTH = 3
SELECTED_BORDER_WIDTH = 4
ACTIVE_BORDER_WIDTH = 4
ITEM_SELECTED_OUTLINE = "#5A1E24"
ITEM_LABEL_COLOR = "#7A3440"
UNDO_LIMIT = 200

ROOM_CATEGORIES = [
    ("living_room", "Гостиная", "Living room", "#D6A430"),
    ("kitchen", "Кухня", "Kitchen", "#5AA06A"),
    ("bedroom", "Спальня", "Bedroom", "#8E6AD8"),
    ("bathroom", "Ванная", "Bathroom", "#5A8ED8"),
    ("toilet", "Туалет", "Toilet", "#B08E23"),
    ("corridor", "Коридор", "Corridor", "#C66E3B"),
    ("office", "Кабинет", "Office", "#4B9B8A"),
    ("other", "Другое", "Other", "#777777"),
]

ITEM_CATEGORIES_RAW = [
    ("air_conditioner", "Кондиционер", "Air conditioner", "#6FA8DC"),
    ("armchair", "Кресло", "Armchair", "#3F9A85"),
    ("bath", "Ванна", "Bath", "#4AA2D0"),
    ("bed", "Кровать", "Bed", "#A16AE8"),
    ("bench", "Банкетка", "Bench", "#8F7A63"),
    ("bidet", "Биде", "Bidet", "#7BA7C5"),
    ("boiler", "Бойлер", "Boiler", "#A36C5C"),
    ("bookcase", "Стеллаж", "Bookcase", "#4A70C2"),
    ("cabinet", "Тумба", "Cabinet", "#3E7FC1"),
    ("chair", "Стул", "Chair", "#4E9B9B"),
    ("chandelier", "Люстра", "Chandelier", "#B89E58"),
    ("coffee_table", "Журнальный столик", "Coffee table", "#C89C4B"),
    ("console", "Консоль", "Console", "#92745E"),
    ("desk", "Рабочий стол", "Desk", "#A48A3A"),
    ("dining_table", "Обеденный стол", "Dining table", "#D7A43A"),
    ("dishwasher", "Посудомоечная машина", "Dishwasher", "#6E92A8"),
    ("dresser", "Комод", "Dresser", "#5674C9"),
    ("dryer", "Сушильная машина", "Dryer", "#8E72D5"),
    ("floor_lamp", "Торшер", "Floor lamp", "#8B7D4F"),
    ("fridge", "Холодильник", "Fridge", "#7A7A7A"),
    ("hanger", "Вешалка", "Hanger", "#7E6A58"),
    ("hood", "Вытяжка", "Hood", "#808890"),
    ("kitchen_island", "Кухонный остров", "Kitchen island", "#9C8A62"),
    ("microwave", "Микроволновка", "Microwave", "#8C8C8C"),
    ("mirror", "Зеркало", "Mirror", "#8AB2D1"),
    ("nightstand", "Прикроватная тумба", "Nightstand", "#8C6BE0"),
    ("ottoman", "Пуф", "Ottoman", "#8B6E86"),
    ("oven", "Духовка", "Oven", "#B85B5B"),
    ("radiator", "Радиатор", "Radiator", "#999999"),
    ("shelf", "Полка", "Shelf", "#5A93C7"),
    ("shoe_rack", "Обувница", "Shoe rack", "#8F7A68"),
    ("shower", "Душ", "Shower", "#3C9AD0"),
    ("sideboard", "Буфет", "Sideboard", "#8B6B54"),
    ("sink", "Кухонная мойка", "Kitchen sink", "#5FA9E5"),
    ("sink_bathroom", "Раковина", "Sink", "#6DB9D2"),
    ("sofa", "Диван", "Sofa", "#53A653"),
    ("stove", "Плита", "Stove", "#D46A6A"),
    ("table", "Стол", "Table", "#E0B64A"),
    ("toilet", "Унитаз", "Toilet", "#9B7B35"),
    ("towel_radiator", "Полотенцесушитель", "Towel radiator", "#A5A5A5"),
    ("tv", "Телевизор", "TV", "#444444"),
    ("tv_stand", "ТВ-тумба", "TV stand", "#516FAF"),
    ("vanity_table", "Туалетный столик", "Vanity table", "#B17D8E"),
    ("wardrobe", "Шкаф", "Wardrobe", "#4C78FF"),
    ("washing_machine", "Стиральная машина", "Washing machine", "#7B5FC8"),
    ("water_filter", "Фильтр воды", "Water filter", "#88AFCB"),
    ("other", "Другое", "Other", "#777777"),
]

ITEM_CATEGORIES = sorted(
    [row for row in ITEM_CATEGORIES_RAW if row[0] != "other"],
    key=lambda row: row[1].lower(),
) + [row for row in ITEM_CATEGORIES_RAW if row[0] == "other"]

ROOM_CATEGORY_MAP = {k: {"ru": ru, "en": en, "color": color} for k, ru, en, color in ROOM_CATEGORIES}
ITEM_CATEGORY_MAP = {k: {"ru": ru, "en": en, "color": color} for k, ru, en, color in ITEM_CATEGORIES}

INSTRUCTIONS_TEXT = """РУССКИЙ
1. Откройте папку с планами квартир.
2. Для каждого изображения сначала обведите комнаты полигоном.
3. После завершения контура выберите тип комнаты. Комната сразу подсветится полупрозрачным цветом.
4. При выборе категории комнаты программа автоматически переключается в режим комнат.
5. При выборе категории предмета программа автоматически переключается в режим предметов.
6. Справа категории предметов показаны как длинный прокручиваемый список по алфавиту. Пункт «Другое» находится в самом конце.
7. Поле ввода своей категории доступно только когда выбрано «Другое».
8. Если выбрана обычная категория, будет использована именно она, даже если в поле «Другое» уже есть текст.
9. Предмет можно рисовать полигоном или прямоугольником. В файле он всё равно сохраняется как полигон.
10. Для добавления предмета не нужно заранее выбирать активную комнату. Предмет автоматически привязывается к той комнате, внутри которой находится центр его полигона.
11. Если центр полигона предмета не попадает ни в одну комнату, это ошибка, и предмет не сохраняется.
12. Все промежуточные состояния, история и журнал активности сохраняются автоматически рядом с исходным изображением.
13. Кнопка «Сохранить и далее» сохраняет итоговую разметку, блокирует её от редактирования и открывает следующий файл.
14. Enter завершает текущий полигон. В режиме прямоугольника Enter завершает текущий прямоугольник.
15. Backspace/Delete удаляет выбранную сущность. Esc отменяет текущий черновик. Ctrl+Z и Cmd+Z отменяют последнее действие.
16. Названия комнат показываются в плане. Для предметов подпись выводится только у выделенного предмета, чтобы не загромождать изображение.
17. Колесо мыши — масштаб. Средняя кнопка мыши или Space+drag — панорама.

ENGLISH
1. Open the folder that contains the apartment plans.
2. For each image, annotate rooms first using polygons.
3. After closing the contour, choose the room type. The room is highlighted with a semi-transparent color.
4. Choosing a room category automatically switches the app to room mode.
5. Choosing an item category automatically switches the app to item mode.
6. The item categories on the right are shown as one long alphabetical scrollable list. “Other” stays at the end.
7. The custom text field becomes editable only when “Other” is selected.
8. If a regular category is selected, that category is used even if the “Other” field already contains text.
9. An item can be drawn as a polygon or as a rectangle. It is still saved as a polygon in the file.
10. You do not need to preselect an active room to add an item. The item is automatically attached to the room that contains the center of its polygon.
11. If the item polygon center does not fall inside any room, it is an error and the item is rejected.
12. All intermediate states, history and activity logs are autosaved next to the original image.
13. “Save and Next” saves the final annotation, locks it against editing and opens the next file.
14. Enter closes the current polygon. In rectangle mode Enter completes the current rectangle.
15. Backspace/Delete removes the selected entity. Esc cancels the current draft. Ctrl+Z and Cmd+Z undo the last action.
16. Room labels stay visible on the plan. Item labels are shown only for the selected item to avoid clutter.
17. Mouse wheel zooms. Middle mouse button or Space+drag pans the view.
"""


class Geometry:
    @staticmethod
    def polygon_area(points: List[List[float]]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    @staticmethod
    def polygon_centroid(points: List[List[float]]) -> List[float]:
        if not points:
            return [0.0, 0.0]
        if len(points) < 3:
            sx = sum(p[0] for p in points)
            sy = sum(p[1] for p in points)
            return [sx / len(points), sy / len(points)]
        area = Geometry.polygon_area(points)
        if abs(area) < 1e-9:
            sx = sum(p[0] for p in points)
            sy = sum(p[1] for p in points)
            return [sx / len(points), sy / len(points)]
        cx = 0.0
        cy = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            cross = x1 * y2 - x2 * y1
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        return [cx, cy]

    @staticmethod
    def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
        if len(polygon) < 3:
            return False
        x, y = point
        inside = False
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            cond = ((y1 > y) != (y2 > y))
            if cond:
                denom = (y2 - y1) if abs(y2 - y1) > 1e-12 else 1e-12
                x_intersect = (x2 - x1) * (y - y1) / denom + x1
                if x < x_intersect:
                    inside = not inside
        return inside

    @staticmethod
    def distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * abx + (py - ay) * aby) / ab2
        t = max(0.0, min(1.0, t))
        cx = ax + t * abx
        cy = ay + t * aby
        return math.hypot(px - cx, py - cy)

    @staticmethod
    def point_near_polygon(point: List[float], polygon: List[List[float]], tolerance: float) -> bool:
        if Geometry.point_in_polygon(point, polygon):
            return True
        if len(polygon) < 2:
            return False
        px, py = point
        for i in range(len(polygon)):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % len(polygon)]
            if Geometry.distance_point_to_segment(px, py, ax, ay, bx, by) <= tolerance:
                return True
        return False

    @staticmethod
    def rect_to_polygon(a: List[float], b: List[float]) -> List[List[float]]:
        x1, y1 = a
        x2, y2 = b
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    @staticmethod
    def bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def safe_read_json(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return deepcopy(default)


class Storage:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.session_path = root_dir / SESSION_FILE
        self.global_log_path = root_dir / GLOBAL_LOG_FILE
        self.activity_log_path = root_dir / ACTIVITY_LOG_FILE

    @staticmethod
    def ann_path(image_path: Path) -> Path:
        return image_path.with_name(image_path.stem + ANNOT_SUFFIX)

    @staticmethod
    def state_path(image_path: Path) -> Path:
        return image_path.with_name(image_path.stem + STATE_SUFFIX)

    @staticmethod
    def history_path(image_path: Path) -> Path:
        return image_path.with_name(image_path.stem + HISTORY_SUFFIX)

    @staticmethod
    def set_readonly(path: Path, readonly: bool) -> None:
        if not path.exists():
            return
        try:
            mode = path.stat().st_mode
            if readonly:
                path.chmod(mode & ~0o222)
            else:
                path.chmod(mode | 0o200)
        except Exception:
            pass

    def load_annotation(self, image_path: Path, image_size: Tuple[int, int]) -> dict:
        default = {
            "version": 2,
            "image_file": image_path.name,
            "image_path": str(image_path),
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "created_at": now_ts(),
            "updated_at": now_ts(),
            "complete": False,
            "locked": False,
            "locked_at": None,
            "finalized_at": None,
            "rooms": [],
            "items": [],
            "stats": {"room_count": 0, "item_count": 0},
        }
        data = safe_read_json(self.ann_path(image_path), default)
        data.setdefault("rooms", [])
        data.setdefault("items", [])
        data.setdefault("complete", False)
        data.setdefault("locked", False)
        data.setdefault("locked_at", None)
        data.setdefault("finalized_at", None)
        data.setdefault("stats", {})
        data["image_file"] = image_path.name
        data["image_path"] = str(image_path)
        data["image_size"] = {"width": image_size[0], "height": image_size[1]}
        return data

    def load_state(self, image_path: Path) -> dict:
        default = {
            "active_room_id": None,
            "selected_entity": None,
            "mode": "rooms",
            "item_draw_mode": "polygon",
            "draft_points": [],
            "draft_rect_start": None,
            "zoom": 1.0,
            "pan_x": 0.0,
            "pan_y": 0.0,
            "show_overlay": True,
            "current_room_category": None,
            "current_room_custom_label": "",
            "current_item_category": None,
            "current_item_custom_label": "",
            "last_saved_reason": "",
        }
        return safe_read_json(self.state_path(image_path), default)

    def save_annotation(self, image_path: Path, annotation: dict) -> None:
        annotation = deepcopy(annotation)
        annotation["updated_at"] = now_ts()
        annotation["stats"] = {
            "room_count": len(annotation.get("rooms", [])),
            "item_count": len(annotation.get("items", [])),
        }
        ann_path = self.ann_path(image_path)
        self.set_readonly(ann_path, False)
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

    def save_state(self, image_path: Path, state: dict) -> None:
        state_path = self.state_path(image_path)
        self.set_readonly(state_path, False)
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def append_history(self, image_path: Path, payload: dict) -> None:
        with self.history_path(image_path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_session(self, payload: dict) -> None:
        with self.session_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_session(self) -> dict:
        return safe_read_json(self.session_path, {"current_folder": "", "current_index": 0, "updated_at": ""})

    def log(self, text: str) -> None:
        line = f"[{now_ts()}] {text}\n"
        with self.global_log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def log_activity(self, payload: dict) -> None:
        event = deepcopy(payload)
        event.setdefault("timestamp", now_ts())
        with self.activity_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class CategoryDialog(simpledialog.Dialog):
    def __init__(self, parent, title: str, categories: List[Tuple[str, str, str, str]], mode_name: str, initial_key=None):
        self.categories = categories
        self.mode_name = mode_name
        self.result_value = None
        self.initial_key = initial_key
        super().__init__(parent, title)

    def body(self, master):
        ttk.Label(master, text=f"Выберите категорию / Choose category ({self.mode_name})").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.var = tk.StringVar(value=self.initial_key or self.categories[0][0])
        self.custom_var = tk.StringVar(value="")
        wrap = ttk.Frame(master)
        wrap.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        for i, (key, ru, en, _color) in enumerate(self.categories):
            ttk.Radiobutton(wrap, text=f"{ru} / {en}", variable=self.var, value=key).grid(row=i, column=0, sticky="w", pady=1)
        ttk.Label(master, text="Пользовательская метка / Custom label:").grid(row=2, column=0, sticky="w", padx=6, pady=(10, 4))
        ttk.Entry(master, textvariable=self.custom_var, width=40).grid(row=3, column=0, sticky="ew", padx=6)
        return master

    def apply(self):
        custom = self.custom_var.get().strip()
        self.result_value = {"key": self.var.get(), "user_label": custom}


class AnnotatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} {APP_VERSION}")
        self.geometry("1600x980")
        self.minsize(1180, 760)

        self.folder: Optional[Path] = None
        self.storage: Optional[Storage] = None
        self.images: List[Path] = []
        self.current_index = 0
        self.current_image_path: Optional[Path] = None
        self.original_image = None
        self.current_photo = None
        self.overlay_fill_photo = None
        self.render_signature = None
        self.image_size = (1, 1)
        self.annotation = {}
        self.image_state = {}
        self.last_snapshot_time = 0.0
        self.current_locked = False
        self.last_user_action_mono = 0.0
        self.activity_segment_started_mono: Optional[float] = None
        self.activity_segment_started_ts: Optional[str] = None
        self.activity_segment_image: Optional[str] = None

        self.mode_var = tk.StringVar(value="rooms")
        self.item_draw_mode_var = tk.StringVar(value="polygon")
        self.room_category_var = tk.StringVar(value="")
        self.room_custom_var = tk.StringVar(value="")
        self.item_category_var = tk.StringVar(value="")
        self.item_custom_var = tk.StringVar(value="")
        self.show_overlay_var = tk.BooleanVar(value=True)

        self.space_pressed = False
        self.panning = False
        self.pan_start_screen = None
        self.pan_origin = (0.0, 0.0)
        self.zoom = 1.0
        self.base_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.draft_points: List[List[float]] = []
        self.draft_rect_start: Optional[List[float]] = None
        self.preview_point: Optional[List[float]] = None
        self.active_room_id: Optional[str] = None
        self.selected_entity: Optional[Tuple[str, str]] = None
        self.undo_stack: List[dict] = []
        self.room_custom_entry: Optional[ttk.Entry] = None
        self.item_custom_entry: Optional[ttk.Entry] = None

        self.status_var = tk.StringVar(value="Выберите папку с планами / Choose a folder with plans")
        self.file_var = tk.StringVar(value="Файл: —")
        self.counter_var = tk.StringVar(value="0 / 0")
        self.active_room_var = tk.StringVar(value="Активная комната: — / Active room: —")
        self.current_target_var = tk.StringVar(value="Будет добавлено: — / Will add: —")
        self.selected_var = tk.StringVar(value="Выделено: — / Selected: —")

        self._build_ui()
        self._bind_events()
        self.after(100, self.choose_folder_on_start)

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Button(top, text="Открыть папку / Open folder", command=self.choose_folder).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(top, textvariable=self.file_var).grid(row=0, column=1, sticky="w")
        ttk.Label(top, textvariable=self.counter_var, font=("Segoe UI", 11, "bold")).grid(row=0, column=2, padx=10)
        ttk.Button(top, text="◀", width=4, command=self.prev_image).grid(row=0, column=3, padx=2)
        ttk.Button(top, text="▶", width=4, command=self.next_image).grid(row=0, column=4, padx=2)
        ttk.Button(top, text="Сохранить / Save", command=self.save_now).grid(row=0, column=5, padx=(12, 4))
        ttk.Button(top, text="Сохранить и далее / Save and Next", command=self.save_and_next).grid(row=0, column=6)

        main = ttk.Frame(self, padding=(8, 4, 8, 8))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, width=330)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        mode_box = ttk.LabelFrame(left, text="Режим / Mode", padding=8)
        mode_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(mode_box, text="Комнаты / Rooms", variable=self.mode_var, value="rooms", command=self.on_mode_change).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_box, text="Предметы / Items", variable=self.mode_var, value="items", command=self.on_mode_change).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Checkbutton(mode_box, text="Показывать разметку / Show overlay", variable=self.show_overlay_var, command=self.redraw_overlay).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        room_box = ttk.LabelFrame(left, text="Категории комнат / Room categories", padding=8)
        room_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._build_category_buttons(room_box, ROOM_CATEGORIES, self.room_category_var, self.room_custom_var, is_room=True, columns=2)

        instr_box = ttk.LabelFrame(left, text="Инструкция / Instructions", padding=8)
        instr_box.grid(row=2, column=0, sticky="nsew")
        left.rowconfigure(2, weight=1)
        self.instructions = tk.Text(instr_box, wrap="word", height=20)
        self.instructions.pack(fill="both", expand=True)
        self.instructions.insert("1.0", INSTRUCTIONS_TEXT)
        self.instructions.configure(state="disabled")

        center = ttk.Frame(main)
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        header_box = ttk.LabelFrame(center, text="Текущая разметка / Current annotation", padding=8)
        header_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header_box.columnconfigure(0, weight=1)
        ttk.Label(header_box, textvariable=self.active_room_var, font=("Segoe UI", 10, "bold"), wraplength=860, justify="left").grid(row=0, column=0, sticky="w")
        ttk.Label(header_box, textvariable=self.current_target_var, font=("Segoe UI", 10), wraplength=860, justify="left").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header_box, textvariable=self.selected_var, font=("Segoe UI", 9), wraplength=860, justify="left").grid(row=2, column=0, sticky="w", pady=(4, 0))

        self.canvas = tk.Canvas(center, bg="#2B2B2B", highlightthickness=1, highlightbackground="#444444")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        bottom_tools = ttk.Frame(center, padding=(0, 6, 0, 6))
        bottom_tools.grid(row=2, column=0, sticky="ew")
        ttk.Button(bottom_tools, text="Сбросить вид / Reset view", command=self.reset_view).pack(side="left")
        ttk.Button(bottom_tools, text="Отменить черновик / Cancel draft", command=self.cancel_draft).pack(side="left", padx=4)
        ttk.Button(bottom_tools, text="Удалить выбранное / Delete selected", command=self.delete_selected).pack(side="left", padx=4)

        lists_area = ttk.Frame(center)
        lists_area.grid(row=3, column=0, sticky="ew")
        lists_area.columnconfigure(0, weight=1)
        lists_area.columnconfigure(1, weight=2)

        rooms_list_box = ttk.LabelFrame(lists_area, text="Комнаты / Rooms", padding=8)
        rooms_list_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        rooms_list_box.rowconfigure(0, weight=1)
        rooms_list_box.columnconfigure(0, weight=1)
        rooms_wrap = ttk.Frame(rooms_list_box)
        rooms_wrap.grid(row=0, column=0, sticky="nsew")
        rooms_wrap.rowconfigure(0, weight=1)
        rooms_wrap.columnconfigure(0, weight=1)
        self.rooms_list = tk.Listbox(rooms_wrap, height=9, exportselection=False)
        self.rooms_list.grid(row=0, column=0, sticky="nsew")
        self.rooms_list_scroll = ttk.Scrollbar(rooms_wrap, orient="vertical", command=self.rooms_list.yview)
        self.rooms_list_scroll.grid(row=0, column=1, sticky="ns")
        self.rooms_list.configure(yscrollcommand=self.rooms_list_scroll.set)
        ttk.Button(rooms_list_box, text="Сделать активной / Set active", command=self.activate_room_from_list).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        items_list_box = ttk.LabelFrame(lists_area, text="Предметы / Items", padding=8)
        items_list_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        items_list_box.rowconfigure(0, weight=1)
        items_list_box.columnconfigure(0, weight=1)
        items_wrap = ttk.Frame(items_list_box)
        items_wrap.grid(row=0, column=0, sticky="nsew")
        items_wrap.rowconfigure(0, weight=1)
        items_wrap.columnconfigure(0, weight=1)
        self.items_list = tk.Listbox(items_wrap, height=9, exportselection=False)
        self.items_list.grid(row=0, column=0, sticky="nsew")
        self.items_list_scroll = ttk.Scrollbar(items_wrap, orient="vertical", command=self.items_list.yview)
        self.items_list_scroll.grid(row=0, column=1, sticky="ns")
        self.items_list.configure(yscrollcommand=self.items_list_scroll.set)
        ttk.Button(items_list_box, text="Удалить выбранное / Delete selected", command=self.delete_selected).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        right = ttk.Frame(main, width=420)
        right.grid(row=0, column=2, sticky="nse", padx=(8, 0))
        right.grid_propagate(False)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        item_box = ttk.LabelFrame(right, text="Разметка предметов / Item annotation", padding=8)
        item_box.grid(row=0, column=0, sticky="nsew")
        ttk.Radiobutton(item_box, text="Полигон / Polygon", variable=self.item_draw_mode_var, value="polygon", command=self.on_item_draw_mode_change).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(item_box, text="Прямоугольник / Rectangle", variable=self.item_draw_mode_var, value="rectangle", command=self.on_item_draw_mode_change).grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._build_category_buttons(
            item_box,
            ITEM_CATEGORIES,
            self.item_category_var,
            self.item_custom_var,
            is_room=False,
            start_row=1,
            scrollable_height=760,
            columns=1,
        )

        status = ttk.Frame(self, padding=(8, 0, 8, 8))
        status.grid(row=2, column=0, sticky="ew")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

    def _bind_scroll_canvas_mousewheel(self, canvas: tk.Canvas, *widgets):
        def _on_mousewheel(event):
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta:
                canvas.yview_scroll(delta, "units")
                return "break"
            return None

        all_widgets = (canvas,) + widgets
        for widget in all_widgets:
            widget.bind("<MouseWheel>", _on_mousewheel, add="+")
            widget.bind("<Button-4>", _on_mousewheel, add="+")
            widget.bind("<Button-5>", _on_mousewheel, add="+")

    def _update_custom_entry_state(self, is_room: bool):
        entry = self.room_custom_entry if is_room else self.item_custom_entry
        var = self.room_category_var if is_room else self.item_category_var
        if entry is None:
            return
        state = "normal" if var.get().strip() == "other" else "disabled"
        try:
            entry.configure(state=state)
        except Exception:
            return
        if state == "disabled":
            entry.selection_clear()
        else:
            entry.focus_set()

    def _build_category_buttons(self, parent, categories, var, custom_var, is_room: bool, start_row: int = 0, scrollable_height: Optional[int] = None, columns: int = 2):
        if scrollable_height is not None:
            scroll_wrap = ttk.Frame(parent)
            scroll_wrap.grid(row=start_row, column=0, columnspan=3, sticky="nsew")
            scroll_wrap.columnconfigure(0, weight=1)
            scroll_wrap.rowconfigure(0, weight=1)

            scroll_canvas = tk.Canvas(scroll_wrap, height=scrollable_height, highlightthickness=0, bd=0)
            scroll_canvas.grid(row=0, column=0, sticky="nsew")
            scroll_bar = ttk.Scrollbar(scroll_wrap, orient="vertical", command=scroll_canvas.yview)
            scroll_bar.grid(row=0, column=1, sticky="ns")
            scroll_canvas.configure(yscrollcommand=scroll_bar.set)

            buttons_frame = ttk.Frame(scroll_canvas)
            window_id = scroll_canvas.create_window((0, 0), window=buttons_frame, anchor="nw")
            buttons_frame.bind("<Configure>", lambda e, c=scroll_canvas: c.configure(scrollregion=c.bbox("all")))
            scroll_canvas.bind("<Configure>", lambda e, c=scroll_canvas, wid=window_id: c.itemconfigure(wid, width=e.width))
            self._bind_scroll_canvas_mousewheel(scroll_canvas, buttons_frame)
            bottom_row = start_row + 1
        else:
            buttons_frame = ttk.Frame(parent)
            buttons_frame.grid(row=start_row, column=0, columnspan=3, sticky="ew")
            bottom_row = start_row + math.ceil(len(categories) / max(columns, 1))

        columns = max(1, int(columns))
        for i, (key, ru, en, _color) in enumerate(categories):
            text = f"{ru} / {en}" if columns == 1 else f"{ru}\n{en}"
            if is_room:
                cmd = (lambda k=key: self.on_room_category_clicked(k))
            else:
                cmd = (lambda k=key: self.on_item_category_clicked(k))
            rb = ttk.Radiobutton(buttons_frame, text=text, value=key, variable=var, command=cmd)
            row = i if columns == 1 else i // columns
            col = 0 if columns == 1 else i % columns
            rb.grid(row=row, column=col, sticky="ew", padx=2, pady=2)

        for col in range(columns):
            buttons_frame.columnconfigure(col, weight=1)

        label_text = "Своя комната / Custom room" if is_room else "Своя категория / Custom category"
        ttk.Label(parent, text=label_text).grid(row=bottom_row, column=0, columnspan=3, sticky="w", pady=(6, 2))

        entry = ttk.Entry(parent, textvariable=custom_var)
        entry.grid(row=bottom_row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 2))
        entry.configure(state="disabled")

        if is_room:
            self.room_custom_entry = entry
        else:
            self.item_custom_entry = entry

        def _on_var_write(*_args):
            self._update_custom_entry_state(is_room)

        var.trace_add("write", _on_var_write)
        self._update_custom_entry_state(is_room)

    def choose_folder_on_start(self):
        self.choose_folder()

    def choose_folder(self):
        path = filedialog.askdirectory(title="Выберите папку с планами / Choose folder with plans")
        if not path:
            if not self.images:
                self.status("Папка не выбрана / No folder selected")
            return
        folder = Path(path)
        images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        if not images:
            messagebox.showerror(APP_TITLE, "В папке нет PNG/JPG/JPEG файлов.\nNo PNG/JPG/JPEG files found in the folder.")
            return
        self.folder = folder
        self.storage = Storage(folder)
        self.images = images
        self.log_user_activity("open_folder", folder=str(folder), image_count=len(images))
        session = self.storage.load_session()
        self.current_index = 0
        if session.get("current_folder") == str(folder):
            idx = int(session.get("current_index", 0))
            if 0 <= idx < len(images):
                self.current_index = idx
        self.load_current_image()

    def load_current_image(self):
        if not self.images:
            return
        if self.activity_segment_started_mono is not None:
            self.end_activity_segment("image_switch")
        self.current_image_path = self.images[self.current_index]
        self.counter_var.set(f"{self.current_index + 1} / {len(self.images)}")
        self.status(f"Загрузка: {self.current_image_path.name} / Loading: {self.current_image_path.name}")
        self.original_image = Image.open(self.current_image_path).convert("RGB")
        self.image_size = self.original_image.size
        assert self.storage is not None
        self.annotation = self.storage.load_annotation(self.current_image_path, self.image_size)
        self.image_state = self.storage.load_state(self.current_image_path)
        self.current_locked = bool(self.annotation.get("locked"))
        lock_suffix = " [LOCKED]" if self.current_locked else ""
        self.file_var.set(f"Файл / File: {self.current_image_path.name}{lock_suffix}")
        self.undo_stack = []
        self.restore_image_state()
        self.update_lists()
        self.reset_view(keep_user_values=True)
        self.render_base_image(force=True)
        self.redraw_overlay()
        self.write_session("load_image")
        self.storage.log_activity({
            "event": "load_image",
            "image": self.current_image_path.name,
            "index": self.current_index,
            "locked": self.current_locked,
            "complete": self.annotation.get("complete", False),
        })
        if self.current_locked:
            self.status(
                f"Готово: {self.current_image_path.name}. Финальная разметка заблокирована / "
                f"Ready: {self.current_image_path.name}. Final annotation is locked"
            )
        else:
            self.status(f"Готово: {self.current_image_path.name} / Ready: {self.current_image_path.name}")

    def restore_image_state(self):
        self.mode_var.set(self.image_state.get("mode", "rooms"))
        self.item_draw_mode_var.set(self.image_state.get("item_draw_mode", "polygon"))
        self.room_category_var.set(self.image_state.get("current_room_category") or "")
        self.room_custom_var.set(self.image_state.get("current_room_custom_label") or "")
        self.item_category_var.set(self.image_state.get("current_item_category") or "")
        self.item_custom_var.set(self.image_state.get("current_item_custom_label") or "")
        self.show_overlay_var.set(bool(self.image_state.get("show_overlay", True)))
        self.draft_points = deepcopy(self.image_state.get("draft_points", []))
        self.draft_rect_start = deepcopy(self.image_state.get("draft_rect_start"))
        self.preview_point = deepcopy(self.image_state.get("preview_point"))
        self.active_room_id = self.image_state.get("active_room_id")
        sel = self.image_state.get("selected_entity")
        self.selected_entity = tuple(sel) if isinstance(sel, list) and len(sel) == 2 else None
        self.zoom = float(self.image_state.get("zoom", 1.0))
        self.pan_x = float(self.image_state.get("pan_x", 0.0))
        self.pan_y = float(self.image_state.get("pan_y", 0.0))
        if self.current_locked:
            self.draft_points = []
            self.draft_rect_start = None
            self.preview_point = None
        self.sync_active_room_label()
        self.sync_current_target_label()
        self.sync_selected_label()

    # ---------- Save ----------
    def current_state_payload(self) -> dict:
        return {
            "active_room_id": self.active_room_id,
            "selected_entity": list(self.selected_entity) if self.selected_entity else None,
            "mode": self.mode_var.get(),
            "item_draw_mode": self.item_draw_mode_var.get(),
            "draft_points": deepcopy(self.draft_points),
            "draft_rect_start": deepcopy(self.draft_rect_start),
            "preview_point": deepcopy(self.preview_point),
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "show_overlay": bool(self.show_overlay_var.get()),
            "current_room_category": self.room_category_var.get() or None,
            "current_room_custom_label": self.room_custom_var.get().strip(),
            "current_item_category": self.item_category_var.get() or None,
            "current_item_custom_label": self.item_custom_var.get().strip(),
            "last_saved_reason": "",
        }

    def save_all(self, reason: str, snapshot: bool = True, complete_override: Optional[bool] = None, lock_after_save: bool = False):
        if not self.current_image_path or not self.storage:
            return
        if self.is_current_locked():
            self.write_session(reason + "_locked_skip")
            self.storage.log(f"{self.current_image_path.name}: {reason} skipped because annotation is locked")
            self.storage.log_activity({
                "event": "save_skipped_locked",
                "reason": reason,
                "image": self.current_image_path.name,
                "index": self.current_index,
            })
            return

        state = self.current_state_payload()
        state["last_saved_reason"] = reason
        if complete_override is not None:
            self.annotation["complete"] = bool(complete_override)
        if lock_after_save:
            self.annotation["locked"] = True
            self.annotation["locked_at"] = now_ts()
            self.annotation["finalized_at"] = self.annotation["locked_at"]
        self.annotation["updated_at"] = now_ts()
        self.storage.save_annotation(self.current_image_path, self.annotation)
        self.storage.save_state(self.current_image_path, state)
        if lock_after_save:
            self.storage.set_readonly(self.storage.ann_path(self.current_image_path), True)
            self.storage.set_readonly(self.storage.state_path(self.current_image_path), True)
            self.current_locked = True

        event = {
            "timestamp": now_ts(),
            "reason": reason,
            "image": self.current_image_path.name,
            "complete": self.annotation.get("complete", False),
            "locked": self.annotation.get("locked", False),
            "rooms": len(self.annotation.get("rooms", [])),
            "items": len(self.annotation.get("items", [])),
            "active_room_id": self.active_room_id,
            "draft_points_count": len(self.draft_points),
            "mode": self.mode_var.get(),
        }
        self.storage.append_history(self.current_image_path, event)
        self.storage.log(
            f"{self.current_image_path.name}: {reason}; complete={self.annotation.get('complete', False)}; "
            f"locked={self.annotation.get('locked', False)}; rooms={len(self.annotation.get('rooms', []))}; "
            f"items={len(self.annotation.get('items', []))}"
        )
        self.storage.log_activity({
            "event": "file_saved",
            "reason": reason,
            "image": self.current_image_path.name,
            "index": self.current_index,
            "complete": self.annotation.get("complete", False),
            "locked": self.annotation.get("locked", False),
            "rooms": len(self.annotation.get("rooms", [])),
            "items": len(self.annotation.get("items", [])),
            "annotation_file": self.storage.ann_path(self.current_image_path).name,
            "state_file": self.storage.state_path(self.current_image_path).name,
        })
        self.write_session(reason)
        if snapshot and time.time() - self.last_snapshot_time >= SNAPSHOT_EVERY_SECONDS:
            snap_path = self.current_image_path.with_name(self.current_image_path.stem + f".snapshot_{int(time.time())}.json")
            with snap_path.open("w", encoding="utf-8") as f:
                json.dump({"annotation": self.annotation, "state": state}, f, ensure_ascii=False, indent=2)
            self.last_snapshot_time = time.time()

    def write_session(self, reason: str):
        if not self.storage or not self.folder:
            return
        self.storage.save_session({
            "current_folder": str(self.folder),
            "current_index": self.current_index,
            "current_image": self.current_image_path.name if self.current_image_path else "",
            "updated_at": now_ts(),
            "reason": reason,
        })

    def is_current_locked(self) -> bool:
        return bool(self.current_locked or self.annotation.get("locked"))

    def ensure_editable(self, action_name: str) -> bool:
        if not self.is_current_locked():
            return True
        self.status(
            "Финальная разметка заблокирована: редактирование запрещено / "
            "Final annotation is locked: editing is disabled"
        )
        if self.storage and self.current_image_path:
            self.storage.log_activity({
                "event": "edit_blocked_locked",
                "action": action_name,
                "image": self.current_image_path.name,
                "index": self.current_index,
                "mode": self.mode_var.get(),
            })
        return False

    def end_activity_segment(self, reason: str):
        if not self.storage or self.activity_segment_started_mono is None:
            return
        duration = max(0.0, time.time() - self.activity_segment_started_mono)
        self.storage.log_activity({
            "event": "activity_segment_end",
            "reason": reason,
            "image": self.activity_segment_image,
            "started_at": self.activity_segment_started_ts,
            "ended_at": now_ts(),
            "duration_seconds": round(duration, 3),
        })
        self.activity_segment_started_mono = None
        self.activity_segment_started_ts = None
        self.activity_segment_image = None

    def log_user_activity(self, action: str, **extra):
        if not self.storage:
            return
        now_mono = time.time()
        image_name = self.current_image_path.name if self.current_image_path else ""
        need_new_segment = (
            self.activity_segment_started_mono is None
            or self.activity_segment_image != image_name
            or (self.last_user_action_mono and now_mono - self.last_user_action_mono > IDLE_GAP_SECONDS)
        )
        if need_new_segment:
            if self.activity_segment_started_mono is not None:
                self.end_activity_segment("idle_or_switch")
            self.activity_segment_started_mono = now_mono
            self.activity_segment_started_ts = now_ts()
            self.activity_segment_image = image_name
            self.storage.log_activity({
                "event": "activity_segment_start",
                "trigger": action,
                "image": image_name,
                "index": self.current_index,
                "mode": self.mode_var.get(),
            })
        self.last_user_action_mono = now_mono
        payload = {
            "event": action,
            "image": image_name,
            "index": self.current_index,
            "mode": self.mode_var.get(),
            "locked": self.is_current_locked(),
            "rooms": len(self.annotation.get("rooms", [])),
            "items": len(self.annotation.get("items", [])),
            "active_room_id": self.active_room_id,
            "selected_entity": list(self.selected_entity) if self.selected_entity else None,
        }
        payload.update(extra)
        self.storage.log_activity(payload)

    def save_now(self):
        self.log_user_activity("manual_save_click")
        if self.is_current_locked():
            self.save_all("manual_save", snapshot=True)
            self.status("Файл заблокирован: финальная разметка не изменяется / File is locked: final annotation is not modified")
            return
        self.save_all("manual_save", snapshot=True)
        self.status("Сохранено / Saved")

    def save_and_next(self):
        self.log_user_activity("save_and_next_click")
        if self.is_current_locked():
            self.status("Файл уже зафиксирован и заблокирован / File is already finalized and locked")
        else:
            self.save_all("save_and_next", snapshot=True, complete_override=True, lock_after_save=True)
            self.status("Сохранено, зафиксировано и заблокировано. Переход к следующему файлу / Saved, finalized, locked. Moving to next file")
        if self.current_index < len(self.images) - 1:
            self.end_activity_segment("save_and_next")
            self.current_index += 1
            self.load_current_image()
        else:
            messagebox.showinfo(APP_TITLE, "Это последний файл.\nThis is the last file.")

    def autosave(self, reason: str):
        try:
            self.save_all(reason, snapshot=False)
        except Exception as exc:
            self.status(f"Ошибка автосохранения / Autosave error: {exc}")

    def _text_input_has_focus(self) -> bool:
        widget = self.focus_get()
        if widget is None:
            return False
        try:
            cls = widget.winfo_class()
        except Exception:
            return False
        return cls in {"Entry", "TEntry", "Text", "Spinbox", "TCombobox"}

    def push_undo_state(self, reason: str):
        snapshot = {
            "annotation": deepcopy(self.annotation),
            "state": self.current_state_payload(),
            "reason": reason,
        }
        self.undo_stack.append(snapshot)
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack = self.undo_stack[-UNDO_LIMIT:]

    def restore_snapshot(self, snapshot: dict):
        self.annotation = deepcopy(snapshot.get("annotation", {}))
        self.image_state = deepcopy(snapshot.get("state", {}))
        self.restore_image_state()
        self.update_lists()
        self.render_base_image(force=True)
        self.redraw_overlay()

    def undo_last_action(self):
        if not self.ensure_editable("undo"):
            return
        if not self.undo_stack:
            self.status("Нечего отменять / Nothing to undo")
            return
        snapshot = self.undo_stack.pop()
        self.restore_snapshot(snapshot)
        self.log_user_activity("undo")
        self.autosave("undo")
        self.status("Отменено последнее действие / Undid last action")

    def on_shortcut_return(self, _event=None):
        if self._text_input_has_focus():
            return None
        self.finish_current_polygon()
        return "break"

    def on_shortcut_backspace(self, _event=None):
        if self._text_input_has_focus():
            return None
        if self.draft_points or self.draft_rect_start is not None:
            self.remove_last_draft_point()
        elif self.selected_entity:
            self.delete_selected()
        return "break"

    def on_shortcut_escape(self, _event=None):
        if self._text_input_has_focus():
            return None
        self.cancel_draft()
        return "break"

    def on_shortcut_delete(self, _event=None):
        if self._text_input_has_focus():
            return None
        self.delete_selected()
        return "break"

    def on_shortcut_undo(self, _event=None):
        if self._text_input_has_focus():
            return None
        self.undo_last_action()
        return "break"

    # ---------- Events ----------
    def _bind_events(self):
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Control-Button-1>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel_linux(1, e))
        self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel_linux(-1, e))
        self.canvas.bind("<Button-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)

        self.bind_all("<Return>", self.on_shortcut_return, add="+")
        self.bind_all("<KP_Enter>", self.on_shortcut_return, add="+")
        self.bind_all("<BackSpace>", self.on_shortcut_backspace, add="+")
        self.bind_all("<Escape>", self.on_shortcut_escape, add="+")
        self.bind_all("<Delete>", self.on_shortcut_delete, add="+")
        self.bind_all("<KP_Delete>", self.on_shortcut_delete, add="+")
        self.bind_all("<Control-z>", self.on_shortcut_undo, add="+")
        self.bind_all("<Control-Z>", self.on_shortcut_undo, add="+")
        self.bind_all("<Command-z>", self.on_shortcut_undo, add="+")
        self.bind_all("<Command-Z>", self.on_shortcut_undo, add="+")
        self.bind_all("<Control-s>", lambda e: self.save_now(), add="+")
        self.bind_all("<Control-Return>", lambda e: self.save_and_next(), add="+")
        self.bind("<KeyPress-space>", self.on_space_press)
        self.bind("<KeyRelease-space>", self.on_space_release)

        self.rooms_list.bind("<<ListboxSelect>>", lambda e: self.on_room_list_select())
        self.items_list.bind("<<ListboxSelect>>", lambda e: self.on_item_list_select())
        self.rooms_list.bind("<Delete>", self.on_list_delete, add="+")
        self.rooms_list.bind("<BackSpace>", self.on_list_delete, add="+")
        self.items_list.bind("<Delete>", self.on_list_delete, add="+")
        self.items_list.bind("<BackSpace>", self.on_list_delete, add="+")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_mode_change(self):
        self.cancel_draft(redraw=False, autosave=False)
        self.sync_active_room_label()
        self.sync_current_target_label()
        self.redraw_overlay()
        if not self.is_current_locked():
            self.autosave("mode_change")
        self.log_user_activity("mode_change", new_mode=self.mode_var.get())
        if self.mode_var.get() == "items" and not self.annotation.get("rooms"):
            self.status("Сначала нужно разметить комнату, затем предметы / Annotate at least one room before items")

    def on_item_draw_mode_change(self):
        self.cancel_draft(redraw=True, autosave=False)
        self.sync_current_target_label()
        if not self.is_current_locked():
            self.autosave("item_draw_mode_change")
        self.log_user_activity("item_draw_mode_change", draw_mode=self.item_draw_mode_var.get())

    def on_canvas_resize(self, _event):
        self.reset_view(keep_user_values=True)
        self.render_base_image(force=True)
        self.redraw_overlay()

    def on_space_press(self, _event):
        self.space_pressed = True

    def on_space_release(self, _event):
        self.space_pressed = False
        self.panning = False

    def on_middle_down(self, event):
        self.start_pan(event)

    def on_middle_drag(self, event):
        self.move_pan(event)

    def on_middle_up(self, _event):
        self.panning = False

    def start_pan(self, event):
        self.panning = True
        self.pan_start_screen = (event.x, event.y)
        self.pan_origin = (self.pan_x, self.pan_y)

    def move_pan(self, event):
        if not self.panning or not self.pan_start_screen:
            return
        dx = event.x - self.pan_start_screen[0]
        dy = event.y - self.pan_start_screen[1]
        self.pan_x = self.pan_origin[0] + dx
        self.pan_y = self.pan_origin[1] + dy
        self.render_base_image(force=True)
        self.redraw_overlay()

    def on_left_click(self, event):
        if self.space_pressed:
            self.start_pan(event)
            return
        if not self.current_image_path:
            return
        if not self.ensure_editable("left_click_draw"):
            return
        point = self.screen_to_world(event.x, event.y)
        if point is None:
            return

        if self.mode_var.get() == "rooms":
            self.log_user_activity("room_point_added")
            self.push_undo_state("room_draft_add_point")
            self.draft_rect_start = None
            self.draft_points.append(point)
            self.preview_point = point
            self.redraw_overlay()
            self.autosave("room_draft_add_point")
            return

        # items mode
        if not self.annotation.get("rooms"):
            messagebox.showwarning(APP_TITLE, "Сначала нужно разметить хотя бы одну комнату.\nYou must annotate at least one room first.")
            return

        if self.item_draw_mode_var.get() == "polygon":
            self.log_user_activity("item_point_added")
            self.push_undo_state("item_polygon_add_point")
            self.draft_points.append(point)
            self.preview_point = point
            self.redraw_overlay()
            self.autosave("item_polygon_add_point")
        else:
            if self.draft_rect_start is None:
                self.log_user_activity("item_rect_start")
                self.push_undo_state("item_rect_start")
                self.draft_rect_start = point
                self.preview_point = point
                self.redraw_overlay()
                self.autosave("item_rect_start")
            else:
                polygon = Geometry.rect_to_polygon(self.draft_rect_start, point)
                self.finish_item_polygon(polygon, source="rectangle")

    def on_double_click(self, _event):
        self.finish_current_polygon()

    def on_right_click(self, event):
        point = self.screen_to_world(event.x, event.y)
        if point is None:
            return "break"

        has_polygon_draft = bool(self.draft_points)
        has_rect_draft = self.mode_var.get() == "items" and self.item_draw_mode_var.get() == "rectangle" and self.draft_rect_start is not None

        if has_polygon_draft or has_rect_draft:
            self.cancel_draft(redraw=True, autosave=True)
            self.status("Черновик сброшен / Draft cleared")
            return "break"

        hit = self.find_entity_by_point(point)
        if not hit:
            return "break"
        if self.selected_entity == hit:
            self.delete_selected()
            return "break"
        self.select_entity(hit, autosave_reason="select_entity_by_right_click")
        return "break"

    def on_mouse_move(self, event):
        if self.panning and self.space_pressed:
            self.move_pan(event)
            return
        point = self.screen_to_world(event.x, event.y)
        if point is None:
            return
        self.preview_point = point
        self.redraw_overlay()

    def on_mouse_wheel(self, event):
        delta = 1 if event.delta > 0 else -1
        self._apply_zoom(delta, event.x, event.y)

    def on_mouse_wheel_linux(self, delta, event):
        self._apply_zoom(delta, event.x, event.y)

    def _apply_zoom(self, delta: int, x: int, y: int):
        if not self.current_image_path:
            return
        old_scale = self.total_scale()
        factor = 1.1 if delta > 0 else 1.0 / 1.1
        new_zoom = min(max(self.zoom * factor, 0.1), 15.0)
        if abs(new_zoom - self.zoom) < 1e-12:
            return
        wx, wy = self.screen_to_world(x, y)
        self.zoom = new_zoom
        new_scale = self.total_scale()
        self.pan_x = x - wx * new_scale - self.centering_offset()[0]
        self.pan_y = y - wy * new_scale - self.centering_offset()[1]
        self.render_base_image(force=True)
        self.redraw_overlay()
        self.autosave("zoom_change")

    def on_room_list_select(self):
        idxs = self.rooms_list.curselection()
        if not idxs:
            return
        room = self.annotation.get("rooms", [])[idxs[0]]
        self.select_entity(("room", room["id"]), autosave_reason="room_list_select")
        self.log_user_activity("room_list_select", room_id=room["id"], room_label=room["label"])

    def on_item_list_select(self):
        idxs = self.items_list.curselection()
        if not idxs:
            return
        item = self.annotation.get("items", [])[idxs[0]]
        self.select_entity(("item", item["id"]), autosave_reason="item_list_select")
        self.log_user_activity("item_list_select", item_id=item["id"], item_label=item["label"])

    def on_list_delete(self, _event=None):
        self.delete_selected()
        return "break"

    # ---------- Geometry / transforms ----------
    def centering_offset(self) -> Tuple[float, float]:
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        img_w, img_h = self.image_size
        scale = self.total_scale()
        ox = (canvas_w - img_w * scale) / 2.0
        oy = (canvas_h - img_h * scale) / 2.0
        return ox, oy

    def total_scale(self) -> float:
        return self.base_scale * self.zoom

    def reset_view(self, keep_user_values: bool = False):
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        img_w, img_h = self.image_size
        if img_w <= 0 or img_h <= 0:
            return
        fit = min(canvas_w / img_w, canvas_h / img_h)
        if fit <= 0:
            fit = 1.0
        self.base_scale = fit
        if not keep_user_values:
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0

    def world_to_screen(self, point: List[float]) -> Tuple[float, float]:
        scale = self.total_scale()
        ox, oy = self.centering_offset()
        x = point[0] * scale + ox + self.pan_x
        y = point[1] * scale + oy + self.pan_y
        return x, y

    def screen_to_world(self, sx: float, sy: float) -> Optional[List[float]]:
        if not self.current_image_path:
            return None
        scale = self.total_scale()
        ox, oy = self.centering_offset()
        x = (sx - ox - self.pan_x) / scale
        y = (sy - oy - self.pan_y) / scale
        x = min(max(x, 0.0), max(self.image_size[0] - 1, 0))
        y = min(max(y, 0.0), max(self.image_size[1] - 1, 0))
        return [float(x), float(y)]

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        raw = (hex_color or "").lstrip("#")
        if len(raw) != 6:
            return (120, 120, 120)
        try:
            return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
        except ValueError:
            return (120, 120, 120)

    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        r, g, b = [max(0, min(255, int(v))) for v in rgb]
        return f"#{r:02X}{g:02X}{b:02X}"

    def _mix_color(self, base_color: str, target_color: str, target_weight: float) -> str:
        br, bg, bb = self._hex_to_rgb(base_color)
        tr, tg, tb = self._hex_to_rgb(target_color)
        w = max(0.0, min(1.0, float(target_weight)))
        rgb = (
            round(br * (1.0 - w) + tr * w),
            round(bg * (1.0 - w) + tg * w),
            round(bb * (1.0 - w) + tb * w),
        )
        return self._rgb_to_hex(rgb)

    def _rgba_color(self, hex_color: str, alpha: int) -> Tuple[int, int, int, int]:
        r, g, b = self._hex_to_rgb(hex_color)
        return (r, g, b, max(0, min(255, int(alpha))))

    def _contrast_text_color(self, hex_color: str) -> str:
        r, g, b = self._hex_to_rgb(hex_color)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#111111" if luminance >= 160 else "#F8F8F8"

    def _outline_color(self, base_color: str, active: bool = False, selected: bool = False, room: bool = True) -> str:
        safe_base = self._mix_color(base_color, "#FFFFFF", 0.40) if self._contrast_text_color(base_color) == "#F8F8F8" else base_color
        if not room and selected:
            return ITEM_SELECTED_OUTLINE
        if selected:
            return self._mix_color(safe_base, "#FFFFFF", 0.25)
        if active:
            return self._mix_color(safe_base, "#FFFFFF", 0.12)
        if not room:
            return self._mix_color(safe_base, "#111111", 0.22)
        return safe_base

    def _label_box_color(self, room: bool, base_color: str) -> str:
        if room:
            return ""
        return "#161616"

    def _label_text(self, raw_text: str, room: bool) -> str:
        text = (raw_text or "").strip()
        if not text:
            return "—"
        if room:
            return text
        if len(text) <= 20:
            return text
        parts = text.split()
        if len(parts) >= 2:
            return "\n".join(parts[:2])
        return text[:20] + "…"

    def _current_category_label(self, is_room: bool) -> str:
        if is_room:
            key = self.room_category_var.get().strip()
            custom = self.room_custom_var.get().strip()
            mapping = ROOM_CATEGORY_MAP
        else:
            key = self.item_category_var.get().strip()
            custom = self.item_custom_var.get().strip()
            mapping = ITEM_CATEGORY_MAP
        if not key:
            return "—"
        info = mapping.get(key, mapping.get("other", {"ru": "Другое"}))
        if key == "other" and custom:
            return custom
        return info.get("ru", key)

    def sync_current_target_label(self):
        if self.mode_var.get() == "rooms":
            room_name = self._current_category_label(True)
            self.current_target_var.set(f"Будет добавлена комната: {room_name} / Room to add: {room_name}")
            return
        item_name = self._current_category_label(False)
        draw_mode_ru = "полигон" if self.item_draw_mode_var.get() == "polygon" else "прямоугольник"
        draw_mode_en = "polygon" if self.item_draw_mode_var.get() == "polygon" else "rectangle"
        self.current_target_var.set(
            f"Будет добавлен предмет: {item_name} ({draw_mode_ru}), комната определится автоматически / "
            f"Item to add: {item_name} ({draw_mode_en}), room will be detected automatically"
        )

    # ---------- Drawing ----------
    def render_base_image(self, force: bool = False):
        if not self.original_image:
            return
        scale = self.total_scale()
        ox, oy = self.centering_offset()
        sig = (round(scale, 5), round(ox + self.pan_x, 2), round(oy + self.pan_y, 2), self.canvas.winfo_width(), self.canvas.winfo_height())
        if not force and sig == self.render_signature:
            return
        self.render_signature = sig
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        view = Image.new("RGB", (canvas_w, canvas_h), (43, 43, 43))
        scaled_w = max(1, int(round(self.image_size[0] * scale)))
        scaled_h = max(1, int(round(self.image_size[1] * scale)))
        resized = self.original_image.resize((scaled_w, scaled_h), Image.LANCZOS)
        px = int(round(ox + self.pan_x))
        py = int(round(oy + self.pan_y))
        view.paste(resized, (px, py))
        self.current_photo = ImageTk.PhotoImage(view)
        self.canvas.delete("base_image")
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_photo, tags=("base_image",))

    def _draw_fill_overlay(self):
        self.canvas.delete("overlay_fill")
        self.overlay_fill_photo = None
        if not self.show_overlay_var.get() or not self.current_image_path:
            return
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        for room in self.annotation.get("rooms", []):
            coords = []
            for p in room["polygon"]:
                sx, sy = self.world_to_screen(p)
                coords.append((int(round(sx)), int(round(sy))))
            if len(coords) < 3:
                continue
            is_active = room["id"] == self.active_room_id
            is_selected = self.selected_entity == ("room", room["id"])
            alpha = ROOM_FILL_ALPHA_SELECTED if is_selected else (ROOM_FILL_ALPHA_ACTIVE if is_active else ROOM_FILL_ALPHA)
            draw.polygon(coords, fill=self._rgba_color(room.get("color", "#777777"), alpha))
        self.overlay_fill_photo = ImageTk.PhotoImage(overlay)
        self.canvas.create_image(0, 0, anchor="nw", image=self.overlay_fill_photo, tags=("overlay_fill",))

    def redraw_overlay(self):
        self.canvas.delete("overlay")
        self.canvas.delete("overlay_fill")
        if not self.show_overlay_var.get():
            return

        self._draw_fill_overlay()

        for room in self.annotation.get("rooms", []):
            is_active = room["id"] == self.active_room_id
            is_selected = self.selected_entity == ("room", room["id"])
            self.draw_polygon(room["polygon"], room.get("color", "#777777"), room=True, active=is_active, selected=is_selected)
            centroid = room.get("centroid") or Geometry.polygon_centroid(room["polygon"])
            self.draw_center_label(
                centroid,
                room.get("label", room.get("category_key", "room")),
                font=("Segoe UI", 14, "bold"),
                polygon=room["polygon"],
                room=True,
                accent_color=room.get("color", "#777777"),
                selected=is_selected,
            )

        for item in self.annotation.get("items", []):
            is_selected = self.selected_entity == ("item", item["id"])
            self.draw_polygon(item["polygon"], item.get("color", "#777777"), room=False, active=False, selected=is_selected)
            if is_selected:
                centroid = item.get("centroid") or Geometry.polygon_centroid(item["polygon"])
                self.draw_center_label(
                    centroid,
                    item.get("label", item.get("category_key", "item")),
                    font=("Segoe UI", 9, "bold"),
                    polygon=item["polygon"],
                    room=False,
                    accent_color=item.get("color", "#777777"),
                    selected=True,
                )

        if self.draft_points:
            draft_coords = []
            for p in self.draft_points:
                sx, sy = self.world_to_screen(p)
                draft_coords.extend([sx, sy])
                self.canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#FFFFFF", outline="#202020", width=1, tags=("overlay",))
            if self.preview_point and (not self.draft_points or self.preview_point != self.draft_points[-1]):
                sx, sy = self.world_to_screen(self.preview_point)
                draft_coords.extend([sx, sy])
            if len(draft_coords) >= 4:
                self.canvas.create_line(*draft_coords, fill="#FFFFFF", width=3, dash=(8, 4), capstyle=tk.ROUND, joinstyle=tk.ROUND, tags=("overlay",))

        if self.draft_rect_start and self.preview_point and self.mode_var.get() == "items" and self.item_draw_mode_var.get() == "rectangle":
            rect_poly = Geometry.rect_to_polygon(self.draft_rect_start, self.preview_point)
            coords = []
            for p in rect_poly:
                sx, sy = self.world_to_screen(p)
                coords.extend([sx, sy])
            self.canvas.create_polygon(*coords, outline="#FFFFFF", fill="", width=3, dash=(8, 4), joinstyle=tk.ROUND, tags=("overlay",))

    def draw_center_label(self, centroid: List[float], text: str, font, polygon: List[List[float]], room: bool, accent_color: str, selected: bool = False):
        sx, sy = self.world_to_screen(centroid)
        x1, y1, x2, y2 = Geometry.bbox(polygon)
        bx1, by1 = self.world_to_screen([x1, y1])
        bx2, by2 = self.world_to_screen([x2, y2])
        available_width = max(80, abs(bx2 - bx1) * 0.72)
        shown_text = self._label_text(text, room=room)

        if room:
            base_text = self._contrast_text_color(accent_color)
            text_color = self._mix_color(base_text, accent_color, 0.28)
            shadow_color = self._mix_color(base_text, "#000000" if base_text == "#F8F8F8" else "#FFFFFF", 0.35)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self.canvas.create_text(
                    sx + dx,
                    sy + dy,
                    text=shown_text,
                    fill=shadow_color,
                    font=font,
                    width=available_width,
                    justify="center",
                    anchor="center",
                    tags=("overlay",),
                )
            self.canvas.create_text(
                sx,
                sy,
                text=shown_text,
                fill=text_color,
                font=font,
                width=available_width,
                justify="center",
                anchor="center",
                tags=("overlay",),
            )
            return

        if not selected:
            return

        text_color = self._mix_color(ITEM_LABEL_COLOR, "#FFFFFF", 0.18)
        shadow_color = self._mix_color(ITEM_SELECTED_OUTLINE, "#FFFFFF", 0.55)
        self.canvas.create_text(
            sx + 1,
            sy + 1,
            text=shown_text,
            fill=shadow_color,
            font=font,
            width=available_width,
            justify="center",
            anchor="center",
            tags=("overlay",),
        )
        self.canvas.create_text(
            sx,
            sy,
            text=shown_text,
            fill=text_color,
            font=font,
            width=available_width,
            justify="center",
            anchor="center",
            tags=("overlay",),
        )

    def draw_polygon(self, polygon: List[List[float]], color: str, room: bool, active: bool, selected: bool):
        coords = []
        for p in polygon:
            sx, sy = self.world_to_screen(p)
            coords.extend([sx, sy])

        outline = self._outline_color(color, active=active, selected=selected, room=room)
        width = SELECTED_BORDER_WIDTH if selected else (ACTIVE_BORDER_WIDTH if active else (ROOM_BORDER_WIDTH if room else ITEM_BORDER_WIDTH))

        if room:
            self.canvas.create_polygon(
                *coords,
                fill="",
                outline=outline,
                width=width,
                dash=(9, 5),
                joinstyle=tk.ROUND,
                tags=("overlay",),
            )
        else:
            self.canvas.create_polygon(
                *coords,
                fill="",
                outline=outline,
                width=width,
                joinstyle=tk.ROUND,
                tags=("overlay",),
            )

    # ---------- Finish polygons ----------
    def finish_current_polygon(self):
        if self.mode_var.get() == "rooms":
            if len(self.draft_points) < 3:
                return
            self.finish_room_polygon(deepcopy(self.draft_points))
            return

        if self.item_draw_mode_var.get() == "polygon":
            if len(self.draft_points) < 3:
                return
            self.finish_item_polygon(deepcopy(self.draft_points), source="polygon")
            return

        if self.item_draw_mode_var.get() == "rectangle" and self.draft_rect_start is not None and self.preview_point is not None:
            polygon = Geometry.rect_to_polygon(self.draft_rect_start, self.preview_point)
            self.finish_item_polygon(polygon, source="rectangle")

    def finish_room_polygon(self, polygon: List[List[float]]):
        if not self.ensure_editable("finish_room_polygon"):
            return
        choice = self.get_current_category(is_room=True, allow_dialog=True)
        if not choice:
            self.status("Комната не сохранена: категория не выбрана / Room not saved: category not selected")
            return
        info = ROOM_CATEGORY_MAP.get(choice["key"], ROOM_CATEGORY_MAP["other"])
        label = choice["user_label"].strip() or info["ru"]
        self.push_undo_state("room_saved")
        room = {
            "id": str(uuid.uuid4()),
            "category_key": choice["key"],
            "label": label,
            "user_label": choice["user_label"].strip(),
            "color": info["color"],
            "polygon": polygon,
            "centroid": Geometry.polygon_centroid(polygon),
            "created_at": now_ts(),
            "updated_at": now_ts(),
        }
        self.annotation.setdefault("rooms", []).append(room)
        self.active_room_id = room["id"]
        self.selected_entity = ("room", room["id"])
        self.cancel_draft(redraw=False, autosave=False)
        self.update_lists()
        self.sync_active_room_label()
        self.sync_selected_label()
        self.redraw_overlay()
        self.log_user_activity("room_saved", room_id=room["id"], room_label=room["label"], category_key=room["category_key"])
        self.autosave("room_saved")
        self.status("Комната сохранена / Room saved")

    def find_room_for_point(self, point: List[float]) -> Optional[dict]:
        matches = []
        for room in self.annotation.get("rooms", []):
            if Geometry.point_in_polygon(point, room["polygon"]):
                matches.append(room)
        if not matches:
            return None
        active_room = self.find_room(self.active_room_id) if self.active_room_id else None
        if active_room is not None and active_room in matches:
            return active_room
        return min(matches, key=lambda room: abs(Geometry.polygon_area(room["polygon"])))

    def finish_item_polygon(self, polygon: List[List[float]], source: str):
        if not self.ensure_editable("finish_item_polygon"):
            return
        if not self.annotation.get("rooms"):
            messagebox.showwarning(APP_TITLE, "Сначала нужно разметить хотя бы одну комнату.\nYou must annotate at least one room first.")
            return
        centroid = Geometry.polygon_centroid(polygon)
        room = self.find_room_for_point(centroid)
        if room is None:
            messagebox.showerror(
                APP_TITLE,
                "Центр полигона предмета должен находиться внутри какой-нибудь размеченной комнаты.\n"
                "Это ошибка, предмет не сохранён.\n\n"
                "The item polygon center must be inside at least one annotated room.\n"
                "This is an error, the item was not saved.",
            )
            self.cancel_draft(redraw=True, autosave=True)
            return
        choice = self.get_current_category(is_room=False, allow_dialog=True)
        if not choice:
            self.status("Предмет не сохранён: категория не выбрана / Item not saved: category not selected")
            return
        info = ITEM_CATEGORY_MAP.get(choice["key"], ITEM_CATEGORY_MAP["other"])
        label = choice["user_label"].strip() or info["ru"]
        self.push_undo_state("item_saved")
        item = {
            "id": str(uuid.uuid4()),
            "room_id": room["id"],
            "room_label": room["label"],
            "category_key": choice["key"],
            "label": label,
            "user_label": choice["user_label"].strip(),
            "color": info["color"],
            "polygon": polygon,
            "centroid": centroid,
            "shape_source": source,
            "created_at": now_ts(),
            "updated_at": now_ts(),
        }
        self.annotation.setdefault("items", []).append(item)
        self.active_room_id = room["id"]
        self.selected_entity = ("item", item["id"])
        self.cancel_draft(redraw=False, autosave=False)
        self.update_lists()
        self.sync_active_room_label()
        self.sync_current_target_label()
        self.sync_selected_label()
        self.redraw_overlay()
        self.log_user_activity(
            "item_saved",
            item_id=item["id"],
            item_label=item["label"],
            category_key=item["category_key"],
            room_id=room["id"],
            source=source,
        )
        self.autosave("item_saved")
        self.status("Предмет сохранён / Item saved")

    # ---------- Category helpers ----------
    def set_annotation_mode(self, mode: str, autosave_reason: Optional[str] = None):
        if mode not in {"rooms", "items"}:
            return
        if self.mode_var.get() != mode:
            self.mode_var.set(mode)
            self.cancel_draft(redraw=False, autosave=False)
            self.sync_active_room_label()
            self.redraw_overlay()
        self.sync_current_target_label()
        if autosave_reason and not self.is_current_locked():
            self.autosave(autosave_reason)

    def on_room_category_clicked(self, key: str):
        self.room_category_var.set(key)
        self.selected_entity = None
        self.sync_current_target_label()
        self.sync_selected_label()
        self.redraw_overlay()
        self.set_annotation_mode("rooms", autosave_reason="room_category_select")
        self.log_user_activity("room_category_select", category_key=key)

    def on_item_category_clicked(self, key: str):
        self.item_category_var.set(key)
        self.selected_entity = None
        self.sync_current_target_label()
        self.sync_selected_label()
        self.redraw_overlay()
        self.set_annotation_mode("items", autosave_reason="item_category_select")
        self.log_user_activity("item_category_select", category_key=key)

    def ask_custom_label(self, is_room: bool):
        title = "Своя комната" if is_room else "Своя категория предмета"
        text = simpledialog.askstring(
            title,
            "Введите пользовательскую метку\nEnter custom label",
            parent=self,
        )
        if text is None:
            return
        text = text.strip()
        if is_room:
            self.room_category_var.set("other")
            self.room_custom_var.set(text)
            self.sync_current_target_label()
            self.set_annotation_mode("rooms", autosave_reason="room_custom_label")
            self.log_user_activity("room_custom_label", value=text)
        else:
            self.item_category_var.set("other")
            self.item_custom_var.set(text)
            self.sync_current_target_label()
            self.set_annotation_mode("items", autosave_reason="item_custom_label")
            self.log_user_activity("item_custom_label", value=text)

    def get_current_category(self, is_room: bool, allow_dialog: bool = False) -> Optional[Dict[str, str]]:
        if is_room:
            key = self.room_category_var.get().strip()
            raw_user_label = self.room_custom_var.get().strip()
            categories = ROOM_CATEGORIES
        else:
            key = self.item_category_var.get().strip()
            raw_user_label = self.item_custom_var.get().strip()
            categories = ITEM_CATEGORIES

        user_label = raw_user_label if key == "other" else ""

        if key:
            return {"key": key, "user_label": user_label}
        if not allow_dialog:
            return None
        dialog = CategoryDialog(self, "Категория / Category", categories, "room" if is_room else "item")
        if not dialog.result_value:
            return None
        if is_room:
            self.room_category_var.set(dialog.result_value["key"])
            self.room_custom_var.set(dialog.result_value["user_label"])
            self.sync_current_target_label()
            self.set_annotation_mode("rooms")
        else:
            self.item_category_var.set(dialog.result_value["key"])
            self.item_custom_var.set(dialog.result_value["user_label"])
            self.sync_current_target_label()
            self.set_annotation_mode("items")
        if dialog.result_value["key"] != "other":
            dialog.result_value["user_label"] = ""
        return dialog.result_value

    # ---------- Selection / deletion ----------
    def find_entity_by_point(self, point: List[float]) -> Optional[Tuple[str, str]]:
        tolerance = max(5.0 / self.total_scale(), 2.0)
        for item in reversed(self.annotation.get("items", [])):
            if Geometry.point_near_polygon(point, item["polygon"], tolerance):
                return ("item", item["id"])
        for room in reversed(self.annotation.get("rooms", [])):
            if Geometry.point_near_polygon(point, room["polygon"], tolerance):
                return ("room", room["id"])
        return None

    def select_entity(self, entity: Tuple[str, str], autosave_reason: Optional[str] = None):
        kind, entity_id = entity
        self.selected_entity = entity
        if kind == "item":
            item = self.find_item(entity_id)
            if item is None:
                self.selected_entity = None
                return
            self.active_room_id = item.get("room_id") or self.active_room_id
            self.set_annotation_mode("items")
            self.rooms_list.selection_clear(0, tk.END)
            self.update_lists(select_item_id=entity_id)
        else:
            room = self.find_room(entity_id)
            if room is None:
                self.selected_entity = None
                return
            self.active_room_id = room["id"]
            self.set_annotation_mode("rooms")
            self.items_list.selection_clear(0, tk.END)
            self.update_lists(select_room_id=entity_id)
        self.sync_active_room_label()
        self.sync_current_target_label()
        self.sync_selected_label()
        self.redraw_overlay()
        self.log_user_activity("select_entity", entity_kind=kind, entity_id=entity_id)
        if autosave_reason and not self.is_current_locked():
            self.autosave(autosave_reason)

    def select_entity_by_point(self, point: List[float]):
        hit = self.find_entity_by_point(point)
        if hit is None:
            return None
        kind, _ = hit
        reason = "select_item_by_point" if kind == "item" else "select_room_by_point"
        self.select_entity(hit, autosave_reason=reason)
        return hit

    def activate_room_from_list(self):
        idxs = self.rooms_list.curselection()
        if not idxs:
            return
        room = self.annotation.get("rooms", [])[idxs[0]]
        self.select_entity(("room", room["id"]), autosave_reason="activate_room_from_list")
        self.log_user_activity("activate_room_from_list", room_id=room["id"], room_label=room["label"])

    def delete_selected(self):
        if not self.selected_entity:
            room_idxs = self.rooms_list.curselection()
            item_idxs = self.items_list.curselection()
            if room_idxs:
                room = self.annotation.get("rooms", [])[room_idxs[0]]
                self.selected_entity = ("room", room["id"])
            elif item_idxs:
                item = self.annotation.get("items", [])[item_idxs[0]]
                self.selected_entity = ("item", item["id"])
            else:
                return
        if not self.ensure_editable("delete_selected"):
            return
        kind, entity_id = self.selected_entity
        if kind == "room":
            room = self.find_room(entity_id)
            if room is None:
                return
            self.push_undo_state("delete_room")
            self.annotation["rooms"] = [r for r in self.annotation.get("rooms", []) if r["id"] != entity_id]
            self.annotation["items"] = [it for it in self.annotation.get("items", []) if it.get("room_id") != entity_id]
            if self.active_room_id == entity_id:
                self.active_room_id = None
            self.log_user_activity("delete_room", room_id=entity_id, room_label=room["label"])
        else:
            item = self.find_item(entity_id)
            if item is None:
                return
            self.push_undo_state("delete_item")
            self.annotation["items"] = [it for it in self.annotation.get("items", []) if it["id"] != entity_id]
            self.log_user_activity("delete_item", item_id=entity_id, item_label=item["label"])

        self.selected_entity = None
        self.sync_active_room_label()
        self.sync_current_target_label()
        self.sync_selected_label()
        self.update_lists()
        self.redraw_overlay()
        self.autosave("delete_selected")
        self.status("Удалено / Deleted")

    def cancel_draft(self, redraw: bool = True, autosave: bool = True):
        if self.is_current_locked() and (self.draft_points or self.draft_rect_start is not None):
            self.draft_points = []
            self.draft_rect_start = None
            self.preview_point = None
            if redraw:
                self.redraw_overlay()
            return
        had_draft = bool(self.draft_points or self.draft_rect_start is not None)
        if had_draft:
            self.push_undo_state("cancel_draft")
        self.draft_points = []
        self.draft_rect_start = None
        self.preview_point = None
        if redraw:
            self.redraw_overlay()
        if autosave and not self.is_current_locked():
            if had_draft:
                self.log_user_activity("cancel_draft")
            self.autosave("cancel_draft")

    def remove_last_draft_point(self):
        if not self.ensure_editable("remove_last_draft_point"):
            return
        if self.item_draw_mode_var.get() == "rectangle" and self.draft_rect_start is not None:
            self.push_undo_state("remove_rect_start")
            self.draft_rect_start = None
            self.preview_point = None
            self.redraw_overlay()
            self.log_user_activity("remove_rect_start")
            self.autosave("remove_rect_start")
            return
        if not self.draft_points:
            return
        self.push_undo_state("remove_last_draft_point")
        self.draft_points.pop()
        self.redraw_overlay()
        self.log_user_activity("remove_last_draft_point")
        self.autosave("remove_last_draft_point")

    # ---------- Lists / labels ----------
    def update_lists(self, select_room_id: Optional[str] = None, select_item_id: Optional[str] = None):
        self.rooms_list.delete(0, tk.END)
        for idx, room in enumerate(self.annotation.get("rooms", [])):
            marker = "* " if room["id"] == self.active_room_id else "  "
            self.rooms_list.insert(tk.END, f"{marker}{idx + 1}. {room['label']}")
            if select_room_id and room["id"] == select_room_id:
                self.rooms_list.selection_set(idx)

        self.items_list.delete(0, tk.END)
        for idx, item in enumerate(self.annotation.get("items", [])):
            self.items_list.insert(tk.END, f"{idx + 1}. [{item.get('room_label', '')}] {item['label']}")
            if select_item_id and item["id"] == select_item_id:
                self.items_list.selection_set(idx)

    def sync_active_room_label(self):
        room = self.find_room(self.active_room_id) if self.active_room_id else None
        if room:
            self.active_room_var.set(f"Активная комната: {room['label']} / Active room: {room['label']}")
        else:
            self.active_room_var.set("Активная комната: — / Active room: —")

    def sync_selected_label(self):
        if not self.selected_entity:
            self.selected_var.set("Выделено: — / Selected: —")
            return
        kind, entity_id = self.selected_entity
        if kind == "room":
            room = self.find_room(entity_id)
            if room:
                self.selected_var.set(
                    f"Выделено: комната '{room['label']}' ({len(room['polygon'])} точек) / "
                    f"Selected: room '{room['label']}' ({len(room['polygon'])} points)"
                )
                return
        if kind == "item":
            item = self.find_item(entity_id)
            if item:
                self.selected_var.set(
                    f"Выделено: предмет '{item['label']}' в комнате '{item.get('room_label', '')}' / "
                    f"Selected: item '{item['label']}' in room '{item.get('room_label', '')}'"
                )
                return
        self.selected_var.set("Выделено: — / Selected: —")

    # ---------- Navigation ----------
    def prev_image(self):
        if not self.images:
            return
        self.log_user_activity("prev_image_click")
        self.save_all("prev_image_autosave", snapshot=False)
        if self.current_index > 0:
            self.end_activity_segment("prev_image")
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if not self.images:
            return
        self.log_user_activity("next_image_click")
        self.save_all("next_image_autosave", snapshot=False)
        if self.current_index < len(self.images) - 1:
            self.end_activity_segment("next_image")
            self.current_index += 1
            self.load_current_image()
        else:
            messagebox.showinfo(APP_TITLE, "Это последний файл.\nThis is the last file.")

    # ---------- Utilities ----------
    def find_room(self, room_id: Optional[str]) -> Optional[dict]:
        if not room_id:
            return None
        for room in self.annotation.get("rooms", []):
            if room["id"] == room_id:
                return room
        return None

    def find_item(self, item_id: Optional[str]) -> Optional[dict]:
        if not item_id:
            return None
        for item in self.annotation.get("items", []):
            if item["id"] == item_id:
                return item
        return None

    def status(self, text: str):
        self.status_var.set(text)

    def on_close(self):
        try:
            self.log_user_activity("app_close")
            self.save_all("app_close", snapshot=True)
            self.end_activity_segment("app_close")
        except Exception:
            pass
        self.destroy()


def main():
    app = AnnotatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
