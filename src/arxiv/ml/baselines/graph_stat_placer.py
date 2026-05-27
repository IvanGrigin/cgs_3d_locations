#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/ml/graph_stat_angle/train_models.py

Обучение нескольких моделей по CSV пар (cat1, cat2, ...).

Главное изменение относительно прежней версии:
- теперь можно выбирать, по какому столбцу учить угол и расстояние:
    --angle_col angle_q30 | angle_q60 | angle_q90 | angle_deg
    --dist_col  dist_norm_q40 | dist_norm | dist_q40 | dist
- число классов по углу определяется автоматически:
    - для angle_q30: 12 (0..330 шаг 30)
    - для angle_q60: 6
    - для angle_q90: 4
    - для angle_deg: используется --bins (по умолчанию 36)

Валидация:
- по строкам (как было).
  Если захотите: можно сделать split по room_id (GroupSplit), но это отдельная правка.

Скоринг (меньше лучше):
  score = angle_logloss + 0.3 * dist_mae
  если logloss недоступен: score = (1-accuracy) + 0.3 * dist_mae

Сохранение:
  --out_dir/<model>.pkl
  --out_dir/best.pkl
  --out_dir/metrics.json

Примеры запуска:

1) УЧИТЬ ПО УГЛУ 30° И ДИСТАНЦИИ 40 СМ:
  python -m src.ml.graph_stat_angle.train_models \
    --csv data/input/graph_stat/pairs.csv \
    --out_dir runs/graph_stat_angle_q30 \
    --angle_col angle_q30 \
    --dist_col dist_norm_q40 \
    --test_size 0.2 \
    --seed 42

2) По 60°:
  ... --angle_col angle_q60 --dist_col dist_norm_q40

3) По 90°:
  ... --angle_col angle_q90 --dist_col dist_norm_q40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
except Exception as e:
    raise ImportError("scikit-learn is required. Install: pip install scikit-learn") from e


# -----------------------------
# Data
# -----------------------------

@dataclass
class Dataset:
    X_dict: List[Dict[str, Any]]
    y_cls: np.ndarray
    y_dist: np.ndarray
    n_classes: int


def _parse_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _infer_classes_for_quantized_angle(angle_col: str) -> Optional[int]:
    # angle_q30: 0..330 (12 values), angle_q60: 6, angle_q90: 4
    if angle_col == "angle_q30":
        return 12
    if angle_col == "angle_q60":
        return 6
    if angle_col == "angle_q90":
        return 4
    return None


def _angle_to_bin_deg(angle_deg: float, bins: int) -> int:
    a = angle_deg % 360.0
    w = 360.0 / float(bins)
    b = int(a // w)
    if b < 0:
        b = 0
    if b >= bins:
        b = bins - 1
    return b


def _quantized_angle_to_class(angle_q: float, step_deg: float) -> int:
    """
    angle_q уже кратен step_deg и лежит в [0,360).
    Переводим в класс: 0..(360/step_deg - 1)
    """
    a = angle_q % 360.0
    k = int(round(a / step_deg))
    n = int(round(360.0 / step_deg))
    k %= n
    return k


def load_pairs_csv(
    csv_path: Path,
    angle_col: str,
    dist_col: str,
    bins_for_angle_deg: int,
    max_rows: int = 0,
) -> Dataset:
    """
    Требуемые столбцы: cat1, cat2, angle_col, dist_col.
    Features: только {"cat1":..., "cat2":...}.
    """
    Xd: List[Dict[str, Any]] = []
    y_cls: List[int] = []
    y_dist: List[float] = []

    n_classes_q = _infer_classes_for_quantized_angle(angle_col)
    if angle_col == "angle_deg":
        n_classes = int(bins_for_angle_deg)
    elif n_classes_q is not None:
        n_classes = int(n_classes_q)
    else:
        # если пользователь дал нестандартный столбец, оценим по уникальным значениям
        n_classes = -1

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit("[train_models] CSV has no header.")

        need = {"cat1", "cat2", angle_col, dist_col}
        if not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"[train_models] Bad CSV header. Need columns: {sorted(need)}")

        # если n_classes неизвестно, собираем уникальные значения угла
        uniq_angles: set[float] = set() if n_classes == -1 else set()

        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break

            c1 = (row.get("cat1") or "").strip()
            c2 = (row.get("cat2") or "").strip()
            if not c1 or not c2:
                continue

            ang_raw = _parse_float(row.get(angle_col, ""))
            d_raw = _parse_float(row.get(dist_col, ""))
            if ang_raw is None or d_raw is None:
                continue
            if d_raw < 0.0:
                continue

            a, b = (c1, c2) if c1 <= c2 else (c2, c1)
            Xd.append({"cat1": a, "cat2": b})
            y_dist.append(float(d_raw))

            if angle_col == "angle_deg":
                y_cls.append(_angle_to_bin_deg(ang_raw, bins=int(bins_for_angle_deg)))
            elif angle_col == "angle_q30":
                y_cls.append(_quantized_angle_to_class(ang_raw, 30.0))
            elif angle_col == "angle_q60":
                y_cls.append(_quantized_angle_to_class(ang_raw, 60.0))
            elif angle_col == "angle_q90":
                y_cls.append(_quantized_angle_to_class(ang_raw, 90.0))
            else:
                uniq_angles.add(float(ang_raw))

        if n_classes == -1:
            # сделаем маппинг по уникальным значениям
            vals = sorted(uniq_angles)
            if not vals:
                raise SystemExit("[train_models] Cannot infer classes: no angle values.")
            mapping = {v: i for i, v in enumerate(vals)}
            # пересчитать второй проход без хранения всех строк невозможно,
            # поэтому запрещаем нестандартный angle_col без фиксированных правил.
            raise SystemExit(
                f"[train_models] Unsupported angle_col='{angle_col}'. "
                f"Use angle_deg/angle_q30/angle_q60/angle_q90."
            )

    if not Xd:
        raise SystemExit("[train_models] Empty dataset after filtering.")

    return Dataset(
        X_dict=Xd,
        y_cls=np.array(y_cls, dtype=np.int32),
        y_dist=np.array(y_dist, dtype=np.float32),
        n_classes=int(n_classes),
    )


# -----------------------------
# Models
# -----------------------------

@dataclass
class DualModel:
    name: str
    vec: DictVectorizer
    angle_model: Any
    dist_model: Any
    n_classes: int

    def fit(self, X_dict: List[Dict[str, Any]], y_cls: np.ndarray, y_dist: np.ndarray) -> None:
        X = self.vec.fit_transform(X_dict)
        self.angle_model.fit(X, y_cls)
        self.dist_model.fit(X, y_dist)

    def predict_angle(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        X = self.vec.transform(X_dict)
        return self.angle_model.predict(X)

    def predict_angle_proba(self, X_dict: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        X = self.vec.transform(X_dict)
        if not hasattr(self.angle_model, "predict_proba"):
            return None
        p = self.angle_model.predict_proba(X)

        if p.shape[1] == self.n_classes:
            return p

        if hasattr(self.angle_model, "classes_"):
            classes = getattr(self.angle_model, "classes_")
            out = np.zeros((p.shape[0], self.n_classes), dtype=np.float64)
            for j, c in enumerate(classes):
                ci = int(c)
                if 0 <= ci < self.n_classes:
                    out[:, ci] = p[:, j]
            s = out.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return out / s

        return None

    def predict_dist(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        X = self.vec.transform(X_dict)
        return self.dist_model.predict(X)


class BaselineStat:
    """
    Статистика по паре категорий:
      - угол: гистограмма по классам + Laplace smoothing
      - расстояние: медиана dist_col
    """
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.pair_hist: Dict[Tuple[str, str], np.ndarray] = {}
        self.pair_med: Dict[Tuple[str, str], float] = {}
        self.global_hist = np.ones((n_classes,), dtype=np.float64)
        self.global_med = 0.0

    def fit(self, X_dict: List[Dict[str, Any]], y_cls: np.ndarray, y_dist: np.ndarray) -> None:
        self.global_med = float(np.median(np.array(y_dist, dtype=np.float64)))

        buckets: Dict[Tuple[str, str], List[float]] = {}

        for x, c, d in zip(X_dict, y_cls, y_dist):
            key = (x["cat1"], x["cat2"])
            if key not in self.pair_hist:
                self.pair_hist[key] = np.ones((self.n_classes,), dtype=np.float64)  # Laplace
            ci = int(c)
            if 0 <= ci < self.n_classes:
                self.pair_hist[key][ci] += 1.0
            buckets.setdefault(key, []).append(float(d))

        for key, arr in buckets.items():
            self.pair_med[key] = float(np.median(np.array(arr, dtype=np.float64)))

    def predict_angle(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(X_dict),), dtype=np.int32)
        for i, x in enumerate(X_dict):
            key = (x["cat1"], x["cat2"])
            hist = self.pair_hist.get(key, self.global_hist)
            out[i] = int(np.argmax(hist))
        return out

    def predict_angle_proba(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(X_dict), self.n_classes), dtype=np.float64)
        for i, x in enumerate(X_dict):
            key = (x["cat1"], x["cat2"])
            hist = self.pair_hist.get(key, self.global_hist)
            p = hist / float(hist.sum())
            out[i] = p
        return out

    def predict_dist(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(X_dict),), dtype=np.float32)
        for i, x in enumerate(X_dict):
            key = (x["cat1"], x["cat2"])
            out[i] = float(self.pair_med.get(key, self.global_med))
        return out


# -----------------------------
# Eval
# -----------------------------

def evaluate_model(model: Any, X_te: List[Dict[str, Any]], y_cls_te: np.ndarray, y_dist_te: np.ndarray, n_classes: int) -> Dict[str, float]:
    y_pred = model.predict_angle(X_te)
    acc = float(accuracy_score(y_cls_te, y_pred))

    ll: Optional[float] = None
    proba = model.predict_angle_proba(X_te) if hasattr(model, "predict_angle_proba") else None
    if proba is not None:
        try:
            ll = float(log_loss(y_cls_te, proba, labels=list(range(n_classes))))
        except Exception:
            ll = None

    d_pred = model.predict_dist(X_te)
    mae = float(mean_absolute_error(y_dist_te, d_pred))
    rmse = float(math.sqrt(mean_squared_error(y_dist_te, d_pred)))

    score = float((1.0 - acc) + 0.3 * mae) if ll is None else float(ll + 0.3 * mae)

    out: Dict[str, float] = {
        "angle_accuracy": acc,
        "dist_mae": mae,
        "dist_rmse": rmse,
        "score": score,
    }
    if ll is not None:
        out["angle_logloss"] = ll
    return out


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--angle_col", default="angle_deg", choices=["angle_deg", "angle_q30", "angle_q60", "angle_q90"])
    ap.add_argument(
        "--dist_col",
        default="dist_norm",
        choices=["dist_norm", "dist_norm_q40", "dist", "dist_q40"],
    )

    ap.add_argument("--bins", type=int, default=36, help="Используется только если angle_col=angle_deg")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_pairs_csv(
        csv_path=csv_path,
        angle_col=str(args.angle_col),
        dist_col=str(args.dist_col),
        bins_for_angle_deg=int(args.bins),
        max_rows=int(args.max_rows),
    )

    Xd, y_cls, y_dist, n_classes = ds.X_dict, ds.y_cls, ds.y_dist, ds.n_classes

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        Xd, y_cls, y_dist,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y_cls,
    )

    models: Dict[str, Any] = {}

    baseline = BaselineStat(n_classes=n_classes)
    baseline.fit(X_tr, y_tr, d_tr)
    models["baseline"] = baseline

    models["forest"] = DualModel(
        name="forest",
        vec=DictVectorizer(sparse=True),
        angle_model=RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=int(args.seed),
            n_jobs=-1,
        ),
        dist_model=RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=int(args.seed),
            n_jobs=-1,
        ),
        n_classes=n_classes,
    )

    models["gbdt"] = DualModel(
        name="gbdt",
        vec=DictVectorizer(sparse=True),
        angle_model=GradientBoostingClassifier(random_state=int(args.seed)),
        dist_model=GradientBoostingRegressor(random_state=int(args.seed)),
        n_classes=n_classes,
    )

    # multi_class убран (warning исчезнет), multinomial будет по умолчанию
    models["linear"] = DualModel(
        name="linear",
        vec=DictVectorizer(sparse=True),
        angle_model=LogisticRegression(
            max_iter=400,
            solver="lbfgs",
        ),
        dist_model=Ridge(alpha=1.0, random_state=int(args.seed)),
        n_classes=n_classes,
    )

    # fit
    for name, m in models.items():
        if isinstance(m, DualModel):
            m.fit(X_tr, y_tr, d_tr)

    # eval
    metrics: Dict[str, Dict[str, float]] = {}
    for name, m in models.items():
        metrics[name] = evaluate_model(m, X_te, y_te, d_te, n_classes=n_classes)

    best_name = min(metrics.keys(), key=lambda k: metrics[k]["score"])
    best_model = models[best_name]

    # save
    for name, m in models.items():
        save_pickle(m, out_dir / f"{name}.pkl")
    save_pickle(best_model, out_dir / "best.pkl")

    meta = {
        "csv": str(csv_path),
        "angle_col": str(args.angle_col),
        "dist_col": str(args.dist_col),
        "bins_arg": int(args.bins),
        "n_classes": int(n_classes),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "max_rows": int(args.max_rows),
        "best_name": best_name,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[train_models] saved models to: {out_dir}")
    print(f"[train_models] best: {best_name} score={metrics[best_name]['score']:.6f}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
