#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/ml/graph_stat_angle/train_models.py

Цель:
- Обучить несколько моделей по данным pairs.csv:
  1) "forest"  : RandomForestClassifier для бина угла + RandomForestRegressor для dist_norm
  2) "gbdt"    : GradientBoostingClassifier + GradientBoostingRegressor
  3) "linear"  : LogisticRegression (multinomial) + Ridge
  4) "baseline": чистая статистика (гистограмма углов + медиана dist_norm) как эталон

- Сделать единый протокол:
  * Вход: cat1, cat2, angle_deg, dist_norm (+ room_id/source_file как meta, но в модели не используем)
  * Фичи: (cat1, cat2) -> one-hot через DictVectorizer
  * Цели:
      y_angle_bin: бин угла (0..bins-1)
      y_dist: dist_norm
  * Валидация: stratified holdout по y_angle_bin
  * Метрики:
      - angle_accuracy (по бинам)
      - angle_logloss (если доступно predict_proba)
      - dist_mae
      - dist_rmse
  * Итоговый скоринг для выбора лучшей модели:
      score = angle_logloss + 0.3 * dist_mae   (меньше = лучше)
    (если logloss недоступен, используем 1-accuracy вместо logloss)

Сохранение:
- Папка: src/ml/graph_stat_angle/
- Артефакты: runs/graph_stat_angle/<model_name>.pkl, runs/graph_stat_angle/metrics.json
- Также сохраняется "best.pkl" и best_name в metrics.json

Запуск:
  python -m src.ml.graph_stat_angle.train_models \
    --csv data/input/graph_stat/pairs.csv \
    --out_dir runs/graph_stat_angle \
    --bins 36 \
    --test_size 0.2 \
    --seed 42 \
    --max_rows 0
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
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
except Exception as e:
    raise ImportError(
        "scikit-learn is required for this module. Install: pip install scikit-learn"
    ) from e


# -----------------------------
# Data loading
# -----------------------------

@dataclass
class Dataset:
    X_dict: List[Dict[str, Any]]
    y_bin: np.ndarray
    y_dist: np.ndarray


def _parse_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _angle_to_bin(angle_deg: float, bins: int) -> int:
    a = angle_deg % 360.0
    w = 360.0 / float(bins)
    b = int(a // w)
    if b >= bins:
        b = bins - 1
    return b


def load_pairs_csv(csv_path: Path, bins: int, max_rows: int = 0) -> Dataset:
    """
    CSV columns required:
      cat1, cat2, angle_deg, dist_norm
    Features:
      {"cat1":..., "cat2":...}  (DictVectorizer -> one-hot)
    """
    Xd: List[Dict[str, Any]] = []
    yb: List[int] = []
    yd: List[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        need = {"cat1", "cat2", "angle_deg", "dist_norm"}
        if r.fieldnames is None or not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"[train_models] Bad CSV header. Need columns: {sorted(need)}")

        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break

            c1 = (row.get("cat1") or "").strip()
            c2 = (row.get("cat2") or "").strip()
            if not c1 or not c2:
                continue

            ang = _parse_float(row.get("angle_deg", ""))
            dn = _parse_float(row.get("dist_norm", ""))
            if ang is None or dn is None or dn < 0.0:
                continue

            # В CSV уже cat1<=cat2; но фиксируем на всякий случай
            a, b = (c1, c2) if c1 <= c2 else (c2, c1)
            Xd.append({"cat1": a, "cat2": b})
            yb.append(_angle_to_bin(ang, bins=bins))
            yd.append(float(dn))

    if not Xd:
        raise SystemExit("[train_models] Empty dataset after filtering.")

    return Dataset(
        X_dict=Xd,
        y_bin=np.array(yb, dtype=np.int32),
        y_dist=np.array(yd, dtype=np.float32),
    )


# -----------------------------
# Models
# -----------------------------

@dataclass
class DualModel:
    """
    Единая обёртка: классификация угла + регрессия расстояния.
    """
    name: str
    vec: DictVectorizer
    angle_model: Any
    dist_model: Any
    bins: int

    def fit(self, X_dict: List[Dict[str, Any]], y_bin: np.ndarray, y_dist: np.ndarray) -> None:
        X = self.vec.fit_transform(X_dict)
        self.angle_model.fit(X, y_bin)
        self.dist_model.fit(X, y_dist)

    def predict_angle_bin(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        X = self.vec.transform(X_dict)
        return self.angle_model.predict(X)

    def predict_angle_proba(self, X_dict: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        X = self.vec.transform(X_dict)
        if hasattr(self.angle_model, "predict_proba"):
            p = self.angle_model.predict_proba(X)
            # гарантируем shape [n, bins] (некоторые модели могут не иметь всех классов)
            if p.shape[1] != self.bins and hasattr(self.angle_model, "classes_"):
                classes = getattr(self.angle_model, "classes_")
                out = np.full((p.shape[0], self.bins), 0.0, dtype=np.float64)
                for j, c in enumerate(classes):
                    out[:, int(c)] = p[:, j]
                # нормировка
                s = out.sum(axis=1, keepdims=True)
                s[s == 0] = 1.0
                out = out / s
                return out
            return p
        return None

    def predict_dist(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        X = self.vec.transform(X_dict)
        return self.dist_model.predict(X)


class BaselineStat:
    """
    Baseline: по каждой паре категорий:
    - угол: распределение по бинам + Laplace smoothing
    - расстояние: медиана dist_norm
    """
    def __init__(self, bins: int) -> None:
        self.bins = bins
        self.pair_hist: Dict[Tuple[str, str], np.ndarray] = {}
        self.pair_med: Dict[Tuple[str, str], float] = {}
        self.global_hist = np.ones((bins,), dtype=np.float64)
        self.global_med = 0.0

    def fit(self, X_dict: List[Dict[str, Any]], y_bin: np.ndarray, y_dist: np.ndarray) -> None:
        dists = []
        for x, b, d in zip(X_dict, y_bin, y_dist):
            a = x["cat1"]; c = x["cat2"]
            key = (a, c)
            if key not in self.pair_hist:
                self.pair_hist[key] = np.ones((self.bins,), dtype=np.float64)  # Laplace
            self.pair_hist[key][int(b)] += 1.0
            dists.append(float(d))

        self.global_med = float(np.median(np.array(dists, dtype=np.float64)))

        # медианы по парам
        buckets: Dict[Tuple[str, str], List[float]] = {}
        for x, d in zip(X_dict, y_dist):
            key = (x["cat1"], x["cat2"])
            buckets.setdefault(key, []).append(float(d))
        for key, arr in buckets.items():
            self.pair_med[key] = float(np.median(np.array(arr, dtype=np.float64)))

    def predict_angle_bin(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(X_dict),), dtype=np.int32)
        for i, x in enumerate(X_dict):
            key = (x["cat1"], x["cat2"])
            hist = self.pair_hist.get(key)
            if hist is None:
                hist = self.global_hist
            out[i] = int(np.argmax(hist))
        return out

    def predict_angle_proba(self, X_dict: List[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(X_dict), self.bins), dtype=np.float64)
        for i, x in enumerate(X_dict):
            key = (x["cat1"], x["cat2"])
            hist = self.pair_hist.get(key)
            if hist is None:
                hist = self.global_hist
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
# Evaluation & selection
# -----------------------------

def evaluate_model(name: str, model: Any, X_te: List[Dict[str, Any]], yb_te: np.ndarray, yd_te: np.ndarray) -> Dict[str, float]:
    yb_pred = model.predict_angle_bin(X_te)
    acc = float(accuracy_score(yb_te, yb_pred))

    proba = None
    if hasattr(model, "predict_angle_proba"):
        proba = model.predict_angle_proba(X_te)

    ll = None
    if proba is not None:
        try:
            ll = float(log_loss(yb_te, proba, labels=list(range(int(np.max(yb_te)) + 1))))
        except Exception:
            ll = None

    yd_pred = model.predict_dist(X_te)
    mae = float(mean_absolute_error(yd_te, yd_pred))
    rmse = float(math.sqrt(mean_squared_error(yd_te, yd_pred)))

    # unified score (lower is better)
    if ll is None:
        score = float((1.0 - acc) + 0.3 * mae)
    else:
        score = float(ll + 0.3 * mae)

    out = {
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bins", type=int, default=36)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0, help="0=all, иначе ограничить строки для отладки")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_pairs_csv(csv_path, bins=int(args.bins), max_rows=int(args.max_rows))
    Xd, yb, yd = ds.X_dict, ds.y_bin, ds.y_dist

    # Стратификация по бинам угла
    X_tr, X_te, yb_tr, yb_te, yd_tr, yd_te = train_test_split(
        Xd, yb, yd,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=yb,
    )

    models: Dict[str, Any] = {}

    # baseline
    baseline = BaselineStat(bins=int(args.bins))
    baseline.fit(X_tr, yb_tr, yd_tr)
    models["baseline"] = baseline

    # forest
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
        bins=int(args.bins),
    )

    # gbdt
    models["gbdt"] = DualModel(
        name="gbdt",
        vec=DictVectorizer(sparse=True),
        angle_model=GradientBoostingClassifier(
            random_state=int(args.seed),
        ),
        dist_model=GradientBoostingRegressor(
            random_state=int(args.seed),
        ),
        bins=int(args.bins),
    )

    # linear
    models["linear"] = DualModel(
        name="linear",
        vec=DictVectorizer(sparse=True),
        angle_model=LogisticRegression(
            multi_class="multinomial",
            max_iter=500,
            n_jobs=-1,
        ),
        dist_model=Ridge(alpha=1.0, random_state=int(args.seed)),
        bins=int(args.bins),
    )

    # Fit DualModels
    for k, m in list(models.items()):
        if isinstance(m, DualModel):
            m.fit(X_tr, yb_tr, yd_tr)

    # Evaluate
    metrics: Dict[str, Dict[str, float]] = {}
    for name, m in models.items():
        metrics[name] = evaluate_model(name, m, X_te, yb_te, yd_te)

    # Select best (min score)
    best_name = min(metrics.keys(), key=lambda k: metrics[k]["score"])
    best_model = models[best_name]

    # Save all models
    for name, m in models.items():
        save_pickle(m, out_dir / f"{name}.pkl")

    # Save best
    save_pickle(best_model, out_dir / "best.pkl")

    meta = {
        "csv": str(csv_path),
        "bins": int(args.bins),
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
