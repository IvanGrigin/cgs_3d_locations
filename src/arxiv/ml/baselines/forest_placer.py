# src/ml/baselines/forest_placer.py
# -*- coding: utf-8 -*-
"""
ForestPlacer: baseline «лес деревьев» для предсказания расстановки предметов по геометрии комнаты и атрибутам объектов.

Вход (inference): mini-json комнаты:
- room.floor_polygon_xz: список вершин полигона пола в плоскости XZ
- room.objects: список объектов; для предсказания достаточно имени + категориальных id + размеров (если есть)
  Если bbox_world_xy отсутствует (обычно при инференсе), можно подать dims в объекте (см. _get_object_dims).

Выход:
- для каждого объекта: pos{x,y,z} (y=0) и yaw_deg (поворот по вертикали)

Модель:
- RandomForestRegressor предсказывает (x,z)
- RandomForestClassifier предсказывает дискретизированный yaw (yaw_bins корзин)

Ограничения/постобработка:
- предсказанная точка проецируется внутрь полигона (через выбор ближайшей допустимой точки на сетке/семплах)
- детерминированность через random_state

Зависимости:
- numpy
- scikit-learn

CLI:
1) Обучение + сохранение:
   python -m src.ml.baselines.forest_placer \
     --train "data/sourse/3D-FRONT/3D-FRONT-processed-mini/**/*.mini.json" \
     --out_model runs/forest/model.pkl

2) Предикт (без переобучения, модель грузится из файла):
   python -m src.ml.baselines.forest_placer \
     --model runs/forest/model.pkl \
     --predict_room_json "data/sourse/3D-FRONT/3D-FRONT-processed-mini/<file>.mini.json" \
     --out_pred runs/forest/pred.json

Примечание по логике:
Это не автогрессивная расстановка (не учитывает столкновения/доступность).
Это быстрый статистический prior по данным (3D-FRONT), который затем следует прогонять через feasibility-checker
(например, src/ml/Plasement/placement_checks.py) и/или greedy/ILP/A* доработку.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except Exception as e:
    raise ImportError("ForestPlacer требует scikit-learn. Установите: pip install scikit-learn") from e


# -----------------------------
# Геометрия: полигон XZ
# -----------------------------

def _poly_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    z = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


def _poly_perimeter(poly: np.ndarray) -> float:
    d = np.diff(np.vstack([poly, poly[0]]), axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))


def _poly_centroid(poly: np.ndarray) -> Tuple[float, float]:
    x = poly[:, 0]
    z = poly[:, 1]
    x2 = np.roll(x, -1)
    z2 = np.roll(z, -1)
    cross = x * z2 - x2 * z
    a = float(np.sum(cross)) * 0.5
    if abs(a) < 1e-12:
        return float(np.mean(x)), float(np.mean(z))
    cx = float(np.sum((x + x2) * cross) / (6.0 * a))
    cz = float(np.sum((z + z2) * cross) / (6.0 * a))
    return cx, cz


def _point_in_poly(point: Tuple[float, float], poly: np.ndarray) -> bool:
    # ray casting (2D)
    x, z = point
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x1, z1 = poly[i]
        x2, z2 = poly[(i + 1) % n]
        cond = ((z1 > z) != (z2 > z))
        if cond:
            x_int = x1 + (x2 - x1) * (z - z1) / (z2 - z1 + 1e-18)
            if x_int > x:
                inside = not inside
    return inside


def _closest_feasible_point(
    xz: Tuple[float, float],
    poly: np.ndarray,
    rng: np.random.RandomState,
    n_grid: int = 25,
    n_rand: int = 300,
) -> Tuple[float, float]:
    if _point_in_poly(xz, poly):
        return xz

    xmin, zmin = np.min(poly, axis=0)
    xmax, zmax = np.max(poly, axis=0)

    candidates: List[Tuple[float, float]] = []

    xs = np.linspace(xmin, xmax, n_grid)
    zs = np.linspace(zmin, zmax, n_grid)
    for xx in xs:
        for zz in zs:
            if _point_in_poly((float(xx), float(zz)), poly):
                candidates.append((float(xx), float(zz)))

    for _ in range(n_rand):
        xx = float(rng.uniform(xmin, xmax))
        zz = float(rng.uniform(zmin, zmax))
        if _point_in_poly((xx, zz), poly):
            candidates.append((xx, zz))

    if not candidates:
        return (float(0.5 * (xmin + xmax)), float(0.5 * (zmin + zmax)))

    px, pz = xz
    cand = np.array(candidates, dtype=np.float32)
    d2 = (cand[:, 0] - px) ** 2 + (cand[:, 1] - pz) ** 2
    best = cand[int(np.argmin(d2))]
    return float(best[0]), float(best[1])


# -----------------------------
# Признаки: комната + объект
# -----------------------------

def _hash_bucket(text: str, n_buckets: int) -> int:
    # детерминированный bucket (FNV-1a)
    h = 2166136261
    for ch in text.encode("utf-8", errors="ignore"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % n_buckets)


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _get_object_dims(obj: Dict[str, Any]) -> Tuple[float, float]:
    """
    Возвращает (width, depth) в плоскости XZ.
    - bbox_world_xy: [xmin, xmax, zmin, zmax]
    - dims: {"width":..., "depth":...} или [w, d]
    """
    bb = obj.get("bbox_world_xy")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        xmin, xmax, zmin, zmax = map(float, bb)
        return max(0.0, xmax - xmin), max(0.0, zmax - zmin)

    dims = obj.get("dims")
    if isinstance(dims, dict):
        w = float(dims.get("width", 0.0))
        d = float(dims.get("depth", 0.0))
        return max(0.0, w), max(0.0, d)
    if isinstance(dims, (list, tuple)) and len(dims) >= 2:
        w = float(dims[0])
        d = float(dims[1])
        return max(0.0, w), max(0.0, d)

    return 0.0, 0.0


def _room_features(poly: np.ndarray) -> Dict[str, float]:
    area = _poly_area(poly)
    per = _poly_perimeter(poly)
    cx, cz = _poly_centroid(poly)
    xmin, zmin = np.min(poly, axis=0)
    xmax, zmax = np.max(poly, axis=0)
    w = float(xmax - xmin)
    d = float(zmax - zmin)
    aspect = float(w / (d + 1e-9))
    return {
        "room_area": float(area),
        "room_perim": float(per),
        "room_cx": float(cx),
        "room_cz": float(cz),
        "room_bbox_w": float(w),
        "room_bbox_d": float(d),
        "room_aspect": float(aspect),
    }


def _object_features(
    obj: Dict[str, Any],
    room_feat: Dict[str, float],
    n_name_buckets: int,
) -> Dict[str, float]:
    name = str(obj.get("name", ""))
    label = str(obj.get("label", ""))

    super_id = _safe_int(obj.get("super_id", obj.get("super-id", -1)), -1)
    cat_id = _safe_int(obj.get("cat_id", obj.get("cat-id", -1)), -1)
    style_id = _safe_int(obj.get("style_id", obj.get("style-id", -1)), -1)
    material_id = _safe_int(obj.get("material_id", obj.get("material-id", -1)), -1)
    theme_id = _safe_int(obj.get("theme_id", obj.get("theme-id", -1)), -1)

    w, d = _get_object_dims(obj)

    rw = float(w / (room_feat["room_bbox_w"] + 1e-9))
    rd = float(d / (room_feat["room_bbox_d"] + 1e-9))

    bucket_name = _hash_bucket(name, n_name_buckets)
    bucket_label = _hash_bucket(label, n_name_buckets)

    return {
        "obj_super_id": float(super_id),
        "obj_cat_id": float(cat_id),
        "obj_style_id": float(style_id),
        "obj_material_id": float(material_id),
        "obj_theme_id": float(theme_id),
        "obj_w": float(w),
        "obj_d": float(d),
        "obj_rw": float(rw),
        "obj_rd": float(rd),
        "name_bucket": float(bucket_name),
        "label_bucket": float(bucket_label),
    }


def _yaw_to_bin(yaw_deg: float, n_bins: int) -> int:
    y = float(yaw_deg) % 360.0
    step = 360.0 / float(n_bins)
    return int(math.floor(y / step)) % n_bins


def _bin_to_yaw(bin_id: int, n_bins: int) -> float:
    step = 360.0 / float(n_bins)
    return float((int(bin_id) + 0.5) * step)


# -----------------------------
# Датасет для обучения
# -----------------------------

@dataclass
class TrainSample:
    x_feat: np.ndarray
    y_xz: np.ndarray
    y_yaw_bin: int


def _iter_rooms_from_mini_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rooms = data.get("rooms", [])
    if not isinstance(rooms, list):
        return []
    return rooms


def build_training_samples(
    mini_json_paths: List[str],
    n_name_buckets: int = 512,
    yaw_bins: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    samples: List[TrainSample] = []
    feature_names: Optional[List[str]] = None

    total_obj = 0
    dropped_bad_target = 0

    for p in mini_json_paths:
        for room in _iter_rooms_from_mini_json(p):
            poly_raw = room.get("floor_polygon_xz", [])
            if not isinstance(poly_raw, list) or len(poly_raw) < 3:
                continue
            poly = np.array([(float(v["x"]), float(v["z"])) for v in poly_raw], dtype=np.float32)
            rfeat = _room_features(poly)

            objects = room.get("objects", [])
            if not isinstance(objects, list):
                continue

            for obj in objects:
                total_obj += 1
                pos = obj.get("pos")
                yaw = obj.get("yaw_deg")
                if not isinstance(pos, dict) or yaw is None:
                    continue

                x = float(pos.get("x", float("nan")))
                z = float(pos.get("z", float("nan")))
                yaw_deg = float(yaw)

                if not (math.isfinite(x) and math.isfinite(z) and math.isfinite(yaw_deg)):
                    dropped_bad_target += 1
                    continue

                ofeat = _object_features(obj, rfeat, n_name_buckets=n_name_buckets)
                feat_dict = {**rfeat, **ofeat}
                if feature_names is None:
                    feature_names = sorted(feat_dict.keys())

                x_feat = np.array([float(feat_dict.get(k, 0.0)) for k in feature_names], dtype=np.float32)
                y_xz = np.array([x, z], dtype=np.float32)
                y_yaw_bin = _yaw_to_bin(yaw_deg, yaw_bins)

                samples.append(TrainSample(x_feat=x_feat, y_xz=y_xz, y_yaw_bin=y_yaw_bin))

    if feature_names is None or not samples:
        raise ValueError("Не удалось собрать обучающие примеры: пустой список samples.")

    X = np.stack([s.x_feat for s in samples], axis=0)
    Y_xz = np.stack([s.y_xz for s in samples], axis=0)
    Y_yaw = np.array([s.y_yaw_bin for s in samples], dtype=np.int64)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y_xz).all(axis=1)
    X = X[mask]
    Y_xz = Y_xz[mask]
    Y_yaw = Y_yaw[mask]

    if X.shape[0] == 0:
        raise ValueError("После фильтрации NaN/Inf обучающих примеров не осталось.")

    print(
        f"[build_training_samples] total_obj={total_obj}, "
        f"dropped_bad_target={dropped_bad_target}, "
        f"kept={int(X.shape[0])}"
    )

    return X, Y_xz, Y_yaw, feature_names


# -----------------------------
# ForestPlacer
# -----------------------------

class ForestPlacer:
    def __init__(
        self,
        n_name_buckets: int = 512,
        yaw_bins: int = 8,
        rf_regressor_params: Optional[Dict[str, Any]] = None,
        rf_classifier_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        self.n_name_buckets = int(n_name_buckets)
        self.yaw_bins = int(yaw_bins)
        self.random_state = int(random_state)
        self.rng = np.random.RandomState(self.random_state)

        reg_params = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=self.random_state,
        )
        if rf_regressor_params:
            reg_params.update(rf_regressor_params)

        clf_params = dict(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=self.random_state,
        )
        if rf_classifier_params:
            clf_params.update(rf_classifier_params)

        self.reg = RandomForestRegressor(**reg_params)
        self.clf = RandomForestClassifier(**clf_params)

        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def fit(self, mini_json_paths: List[str]) -> "ForestPlacer":
        X, Y_xz, Y_yaw, feat_names = build_training_samples(
            mini_json_paths=mini_json_paths,
            n_name_buckets=self.n_name_buckets,
            yaw_bins=self.yaw_bins,
        )
        self.feature_names = feat_names

        self.reg.fit(X, Y_xz)
        self.clf.fit(X, Y_yaw)

        self.is_fitted = True
        return self

    def _build_feature_vector(self, room: Dict[str, Any], obj: Dict[str, Any]) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("ForestPlacer: feature_names=None. Сначала fit() или load().")

        poly_raw = room.get("floor_polygon_xz", [])
        if not isinstance(poly_raw, list) or len(poly_raw) < 3:
            raise ValueError("room.floor_polygon_xz должен быть полигоном (>=3 вершины).")

        poly = np.array([(float(v["x"]), float(v["z"])) for v in poly_raw], dtype=np.float32)
        rfeat = _room_features(poly)
        ofeat = _object_features(obj, rfeat, n_name_buckets=self.n_name_buckets)
        feat_dict = {**rfeat, **ofeat}

        return np.array([float(feat_dict.get(k, 0.0)) for k in self.feature_names], dtype=np.float32)

    def predict_room(self, room: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("ForestPlacer: модель не обучена. Вызовите fit() или load().")

        poly_raw = room.get("floor_polygon_xz", [])
        if not isinstance(poly_raw, list) or len(poly_raw) < 3:
            raise ValueError("room.floor_polygon_xz должен быть полигоном (>=3 вершины).")
        poly = np.array([(float(v["x"]), float(v["z"])) for v in poly_raw], dtype=np.float32)

        objs = room.get("objects", [])
        if not isinstance(objs, list):
            raise ValueError("room.objects должен быть списком объектов.")

        out: List[Dict[str, Any]] = []
        for obj in objs:
            x_feat = self._build_feature_vector(room, obj).reshape(1, -1)
            pred_xz = self.reg.predict(x_feat)[0]
            pred_yaw_bin = int(self.clf.predict(x_feat)[0])

            px = float(pred_xz[0])
            pz = float(pred_xz[1])
            px, pz = _closest_feasible_point((px, pz), poly, rng=self.rng)

            yaw_deg = _bin_to_yaw(pred_yaw_bin, self.yaw_bins)

            out.append(
                {
                    "instanceid": obj.get("instanceid"),
                    "name": obj.get("name"),
                    "label": obj.get("label"),
                    "pred": {
                        "pos": {"x": px, "y": 0.0, "z": pz},
                        "yaw_deg": yaw_deg,
                        "yaw_bin": pred_yaw_bin,
                    },
                }
            )
        return out

    def save(self, path: str) -> None:
        if not self.is_fitted or self.feature_names is None:
            raise RuntimeError("ForestPlacer: нечего сохранять — модель не обучена.")

        payload = {
            "feature_names": self.feature_names,
            "n_name_buckets": self.n_name_buckets,
            "yaw_bins": self.yaw_bins,
            "random_state": self.random_state,
            "reg": self.reg,
            "clf": self.clf,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @staticmethod
    def load(path: str) -> "ForestPlacer":
        with open(path, "rb") as f:
            payload = pickle.loads(f.read())

        placer = ForestPlacer(
            n_name_buckets=int(payload["n_name_buckets"]),
            yaw_bins=int(payload["yaw_bins"]),
            random_state=int(payload["random_state"]),
        )
        placer.feature_names = list(payload["feature_names"])
        placer.reg = payload["reg"]
        placer.clf = payload["clf"]
        placer.is_fitted = True
        return placer


# -----------------------------
# CLI
# -----------------------------

def _glob_mini_json(pattern_or_path: str) -> List[str]:
    s = str(pattern_or_path).strip()
    if not s:
        return []

    # glob-паттерн
    if any(ch in s for ch in ["*", "?", "[", "]"]):
        paths = glob.glob(s, recursive=True)
        paths = [p for p in paths if p.endswith(".mini.json")]
        paths.sort()
        return paths

    # директория
    p = Path(s)
    if p.exists() and p.is_dir():
        paths = [str(x) for x in p.rglob("*.mini.json")]
        paths.sort()
        return paths

    # единичный файл
    if p.exists() and p.is_file() and str(p).endswith(".mini.json"):
        return [str(p)]

    return []


def _load_first_room(mini_json_path: str) -> Dict[str, Any]:
    with open(mini_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rooms = data.get("rooms", [])
    if not isinstance(rooms, list) or not rooms:
        raise SystemExit("В predict_room_json нет rooms.")
    return rooms[0]


def main() -> None:
    ap = argparse.ArgumentParser()

    # режим train
    ap.add_argument("--train", type=str, default="", help="Glob или директория с *.mini.json для обучения.")
    ap.add_argument("--out_model", type=str, default="", help="Куда сохранить модель (pickle).")

    # режим predict
    ap.add_argument("--model", type=str, default="", help="Путь к сохраненной модели (pickle).")
    ap.add_argument("--predict_room_json", type=str, default="", help="Путь к одному *.mini.json для предсказания.")
    ap.add_argument("--out_pred", type=str, default="", help="Куда сохранить pred JSON (если пусто — печатает в stdout).")

    # параметры модели
    ap.add_argument("--yaw_bins", type=int, default=8)
    ap.add_argument("--name_buckets", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    # параметры RF (для контроля размера/скорости)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=0, help="0 = None")

    args = ap.parse_args()

    # -------- predict mode --------
    if args.model and args.predict_room_json:
        placer = ForestPlacer.load(args.model)
        room = _load_first_room(args.predict_room_json)
        preds = placer.predict_room(room)
        text = json.dumps(preds, ensure_ascii=False, indent=2)
        if args.out_pred:
            Path(args.out_pred).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_pred).write_text(text, encoding="utf-8")
            print(f"[forest_placer] Saved predictions: {args.out_pred}")
        else:
            print(text)
        return

    # -------- train mode --------
    if args.train and args.out_model:
        paths = _glob_mini_json(args.train)
        if not paths:
            raise SystemExit(f"Не найдено *.mini.json по: {args.train}")

        max_depth = None if int(args.max_depth) == 0 else int(args.max_depth)

        rf_regressor_params = dict(
            n_estimators=int(args.n_estimators),
            min_samples_leaf=int(args.min_samples_leaf),
            max_depth=max_depth,
        )
        rf_classifier_params = dict(
            n_estimators=int(args.n_estimators),
            min_samples_leaf=int(args.min_samples_leaf),
            max_depth=max_depth,
        )

        placer = ForestPlacer(
            n_name_buckets=int(args.name_buckets),
            yaw_bins=int(args.yaw_bins),
            rf_regressor_params=rf_regressor_params,
            rf_classifier_params=rf_classifier_params,
            random_state=int(args.seed),
        ).fit(paths)

        placer.save(args.out_model)
        print(f"[forest_placer] Saved model: {args.out_model}")
        return

    raise SystemExit(
        "Неверный режим.\n"
        "Train:\n"
        "  python -m src.ml.baselines.forest_placer --train '<glob>' --out_model runs/forest/model.pkl\n"
        "Predict:\n"
        "  python -m src.ml.baselines.forest_placer --model runs/forest/model.pkl --predict_room_json '<file>.mini.json' [--out_pred out.json]\n"
    )


if __name__ == "__main__":
    main()
