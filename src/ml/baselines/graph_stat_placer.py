# src/ml/baselines/graph_stat_placer.py
# -*- coding: utf-8 -*-
"""
GraphStatPlacer (graph_stat):
- Учится предсказывать попарные смещения (dx,dz) между объектами по их атрибутам и НАЗВАНИЯМ (name/label).
- На инференсе строит граф связей между объектами и восстанавливает совместные позиции через итеративное согласование
  + collision repair (для снижения CollisionPairRate).

Входные данные: mini-json (как в примере пользователя).

Зависимости:
- numpy
- scikit-learn
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception as e:
    raise ImportError("graph_stat требует scikit-learn. Установите: pip install scikit-learn") from e


# -----------------------------
# Геометрия полигона (XZ)
# -----------------------------

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
# Размеры объекта (для AABB-коллизий)
# -----------------------------

def _get_object_dims(obj: Dict[str, Any]) -> Tuple[float, float]:
    if "bbox_world_xy" in obj and obj["bbox_world_xy"] is not None:
        bb = obj["bbox_world_xy"]
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


def _rect_from_center_dims(x: float, z: float, w: float, d: float) -> Tuple[float, float, float, float]:
    hw = 0.5 * float(w)
    hd = 0.5 * float(d)
    return (x - hw, x + hw, z - hd, z + hd)


def _rect_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ax2, az1, az2 = a
    bx1, bx2, bz1, bz2 = b
    if ax2 <= bx1 or bx2 <= ax1:
        return False
    if az2 <= bz1 or bz2 <= az1:
        return False
    return True


# -----------------------------
# Текстовые признаки (name/label) — hashing trick
# -----------------------------

_TOKEN_SPLIT_RE = re.compile(r"[\/,_\-\s]+")


def _iter_tokens(text: str) -> List[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    parts = _TOKEN_SPLIT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 32:
            p = p[:32]
        out.append(p)
    return out


def _iter_char_ngrams(text: str, n: int = 3) -> List[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    if len(text) < n:
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _fnv1a_32(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _signed_hash_to_index_and_sign(s: str, dim: int) -> Tuple[int, float]:
    h = _fnv1a_32(s.encode("utf-8", errors="ignore"))
    idx = int(h % dim)
    sign = -1.0 if (h & 1) else 1.0
    return idx, sign


def _hash_text_vector(
    name: str,
    label: str,
    tok_dim: int,
    chr_dim: int,
    char_n: int = 3,
) -> np.ndarray:
    v_tok = np.zeros((tok_dim,), dtype=np.float32)
    v_chr = np.zeros((chr_dim,), dtype=np.float32)

    toks = _iter_tokens(name or "") + _iter_tokens(label or "")
    for t in toks:
        idx, sgn = _signed_hash_to_index_and_sign("tok:" + t, tok_dim)
        v_tok[idx] += sgn

    chrs = _iter_char_ngrams(name or "", n=char_n) + _iter_char_ngrams(label or "", n=char_n)
    if len(chrs) > 512:
        chrs = chrs[:512]
    for g in chrs:
        idx, sgn = _signed_hash_to_index_and_sign("chr:" + g, chr_dim)
        v_chr[idx] += sgn

    def _l2norm(x: np.ndarray) -> np.ndarray:
        nrm = float(np.linalg.norm(x))
        if nrm < 1e-6:
            return x
        return x / nrm

    v_tok = _l2norm(v_tok)
    v_chr = _l2norm(v_chr)
    return np.concatenate([v_tok, v_chr], axis=0)


# -----------------------------
# Узловые/парные признаки
# -----------------------------

def _safe_int(v: Any, default: int = -1) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _room_bbox(poly: np.ndarray) -> Tuple[float, float, float, float]:
    xmin, zmin = np.min(poly, axis=0)
    xmax, zmax = np.max(poly, axis=0)
    return float(xmin), float(xmax), float(zmin), float(zmax)


def _node_features(
    obj: Dict[str, Any],
    room_poly: np.ndarray,
    tok_dim: int,
    chr_dim: int,
) -> np.ndarray:
    super_id = float(_safe_int(obj.get("super_id", obj.get("super-id", -1)), -1))
    cat_id = float(_safe_int(obj.get("cat_id", obj.get("cat-id", -1)), -1))
    style_id = float(_safe_int(obj.get("style_id", obj.get("style-id", -1)), -1))
    material_id = float(_safe_int(obj.get("material_id", obj.get("material-id", -1)), -1))
    theme_id = float(_safe_int(obj.get("theme_id", obj.get("theme-id", -1)), -1))

    w, d = _get_object_dims(obj)

    xmin, xmax, zmin, zmax = _room_bbox(room_poly)
    room_w = max(1e-6, xmax - xmin)
    room_d = max(1e-6, zmax - zmin)

    rw = float(w / room_w)
    rd = float(d / room_d)

    name = str(obj.get("name", ""))
    label = str(obj.get("label", ""))
    tv = _hash_text_vector(name, label, tok_dim=tok_dim, chr_dim=chr_dim, char_n=3)

    # Важно: НЕ используем GT-позицию в признаках (без утечек).
    num = np.array(
        [
            super_id,
            cat_id,
            style_id,
            material_id,
            theme_id,
            float(w),
            float(d),
            rw,
            rd,
        ],
        dtype=np.float32,
    )
    return np.concatenate([num, tv], axis=0)


def _pair_features(hi: np.ndarray, hj: np.ndarray) -> np.ndarray:
    diff = (hj - hi).astype(np.float32)
    prod = (hi * hj).astype(np.float32)
    adiff = np.abs(diff).astype(np.float32)
    return np.concatenate([hi, hj, diff, prod, adiff], axis=0)


# -----------------------------
# Датасет для обучения попарных смещений
# -----------------------------

@dataclass
class PairSample:
    x: np.ndarray
    y: np.ndarray  # [dx, dz]


def _iter_rooms_from_mini_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rooms = data.get("rooms", [])
    if not isinstance(rooms, list):
        return []
    return rooms


def _poly_from_room(room: Dict[str, Any]) -> np.ndarray:
    poly_raw = room.get("floor_polygon_xz", [])
    if not isinstance(poly_raw, list) or len(poly_raw) < 3:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(float(v["x"]), float(v["z"])) for v in poly_raw], dtype=np.float32)


def _get_gt_xz(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pos = obj.get("pos")
    if not isinstance(pos, dict):
        return None
    x = float(pos.get("x", float("nan")))
    z = float(pos.get("z", float("nan")))
    if not (math.isfinite(x) and math.isfinite(z)):
        return None
    return x, z


def _knn_edges_by_gt(xz: np.ndarray, k: int) -> List[Tuple[int, int]]:
    n = xz.shape[0]
    if n <= 1:
        return []
    edges = []
    for i in range(n):
        d2 = np.sum((xz - xz[i:i + 1]) ** 2, axis=1)
        order = np.argsort(d2)
        cnt = 0
        for j in order:
            if j == i:
                continue
            edges.append((i, int(j)))
            cnt += 1
            if cnt >= k:
                break
    return edges


def build_pair_dataset(
    mini_json_paths: List[str],
    tok_dim: int,
    chr_dim: int,
    k_edges: int,
    max_rooms: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    samples: List[PairSample] = []
    rooms_seen = 0
    pairs_total = 0
    dropped = 0

    for p in mini_json_paths:
        for room in _iter_rooms_from_mini_json(p):
            if max_rooms and rooms_seen >= max_rooms:
                break

            poly = _poly_from_room(room)
            if poly.shape[0] < 3:
                continue

            objs = room.get("objects", [])
            if not isinstance(objs, list) or len(objs) < 2:
                continue

            idx_map: List[int] = []
            gt: List[Tuple[float, float]] = []
            for i, obj in enumerate(objs):
                g = _get_gt_xz(obj)
                if g is None:
                    continue
                idx_map.append(i)
                gt.append(g)
            if len(idx_map) < 2:
                continue

            gt_xz = np.array(gt, dtype=np.float32)
            m = gt_xz.shape[0]

            H = []
            for ii in idx_map:
                H.append(_node_features(objs[ii], poly, tok_dim=tok_dim, chr_dim=chr_dim))
            H = np.stack(H, axis=0).astype(np.float32)

            edges = _knn_edges_by_gt(gt_xz, k=k_edges)
            if not edges:
                continue

            for (a, b) in edges:
                pairs_total += 1
                dx = float(gt_xz[b, 0] - gt_xz[a, 0])
                dz = float(gt_xz[b, 1] - gt_xz[a, 1])
                if not (math.isfinite(dx) and math.isfinite(dz)):
                    dropped += 1
                    continue

                x_feat = _pair_features(H[a], H[b])
                y = np.array([dx, dz], dtype=np.float32)

                if (not np.isfinite(x_feat).all()) or (not np.isfinite(y).all()):
                    dropped += 1
                    continue
                samples.append(PairSample(x=x_feat, y=y))

            rooms_seen += 1

        if max_rooms and rooms_seen >= max_rooms:
            break

    if not samples:
        raise ValueError("build_pair_dataset: пусто (нет обучающих пар).")

    X = np.stack([s.x for s in samples], axis=0).astype(np.float32)
    Y = np.stack([s.y for s in samples], axis=0).astype(np.float32)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]
    Y = Y[mask]

    print(f"[graph_stat] rooms={rooms_seen} pairs_total={pairs_total} kept={int(X.shape[0])} dropped={dropped}")
    return X, Y


# -----------------------------
# GraphStatPlacer
# -----------------------------

class GraphStatPlacer:
    def __init__(
        self,
        tok_dim: int = 128,
        chr_dim: int = 128,
        k_train_edges: int = 8,
        k_infer_edges: int = 10,
        random_state: int = 42,
        iters: int = 80,
        relax: float = 0.35,
        center_pull: float = 0.02,
        coll_steps: int = 40,
        coll_push: float = 0.15,
        coll_margin: float = 0.05,
    ) -> None:
        self.tok_dim = int(tok_dim)
        self.chr_dim = int(chr_dim)
        self.k_train_edges = int(k_train_edges)
        self.k_infer_edges = int(k_infer_edges)
        self.random_state = int(random_state)
        self.rng = np.random.RandomState(self.random_state)

        self.iters = int(iters)
        self.relax = float(relax)
        self.center_pull = float(center_pull)

        self.coll_steps = int(coll_steps)
        self.coll_push = float(coll_push)
        self.coll_margin = float(coll_margin)

        base = dict(
            max_depth=None,
            max_iter=300,
            learning_rate=0.05,
            max_bins=255,
            l2_regularization=1e-3,
            random_state=self.random_state,
        )
        self.reg_dx = HistGradientBoostingRegressor(**base)
        self.reg_dz = HistGradientBoostingRegressor(**base)

        self.is_fitted = False
        self.feat_dim: Optional[int] = None

    def fit(self, mini_json_paths: List[str], max_rooms: int = 0) -> "GraphStatPlacer":
        X, Y = build_pair_dataset(
            mini_json_paths=mini_json_paths,
            tok_dim=self.tok_dim,
            chr_dim=self.chr_dim,
            k_edges=self.k_train_edges,
            max_rooms=max_rooms,
        )
        self.feat_dim = int(X.shape[1])
        self.reg_dx.fit(X, Y[:, 0])
        self.reg_dz.fit(X, Y[:, 1])
        self.is_fitted = True
        return self

    def save(self, path: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("GraphStatPlacer: нечего сохранять — модель не обучена.")
        payload = {
            "tok_dim": self.tok_dim,
            "chr_dim": self.chr_dim,
            "k_train_edges": self.k_train_edges,
            "k_infer_edges": self.k_infer_edges,
            "random_state": self.random_state,
            "iters": self.iters,
            "relax": self.relax,
            "center_pull": self.center_pull,
            "coll_steps": self.coll_steps,
            "coll_push": self.coll_push,
            "coll_margin": self.coll_margin,
            "feat_dim": self.feat_dim,
            "reg_dx": self.reg_dx,
            "reg_dz": self.reg_dz,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(pickle.dumps(payload))
        print(f"[graph_stat] Saved model: {path}")

    @staticmethod
    def load(path: str) -> "GraphStatPlacer":
        with open(path, "rb") as f:
            payload = pickle.loads(f.read())
        m = GraphStatPlacer(
            tok_dim=int(payload["tok_dim"]),
            chr_dim=int(payload["chr_dim"]),
            k_train_edges=int(payload["k_train_edges"]),
            k_infer_edges=int(payload["k_infer_edges"]),
            random_state=int(payload["random_state"]),
            iters=int(payload["iters"]),
            relax=float(payload["relax"]),
            center_pull=float(payload["center_pull"]),
            coll_steps=int(payload["coll_steps"]),
            coll_push=float(payload["coll_push"]),
            coll_margin=float(payload["coll_margin"]),
        )
        m.feat_dim = int(payload.get("feat_dim") or 0)
        m.reg_dx = payload["reg_dx"]
        m.reg_dz = payload["reg_dz"]
        m.is_fitted = True
        return m

    def _infer_graph_edges(self, H: np.ndarray) -> List[Tuple[int, int]]:
        n, _ = H.shape
        if n <= 1:
            return []

        num_dim = 9
        txt = H[:, num_dim:]
        txt_norm = np.linalg.norm(txt, axis=1, keepdims=True)
        txt_norm = np.maximum(txt_norm, 1e-6)
        txt_u = txt / txt_norm

        edges: List[Tuple[int, int]] = []
        for i in range(n):
            sims = np.dot(txt_u, txt_u[i:i + 1].T).reshape(-1)
            wi = H[i, 5]
            di = H[i, 6]
            size = np.abs(H[:, 5] - wi) + np.abs(H[:, 6] - di)
            sims = sims - 0.05 * size

            order = np.argsort(-sims)
            cnt = 0
            for j in order:
                if j == i:
                    continue
                edges.append((i, int(j)))
                cnt += 1
                if cnt >= self.k_infer_edges:
                    break
        return edges

    def _predict_delta(self, hi: np.ndarray, hj: np.ndarray) -> Tuple[float, float]:
        x = _pair_features(hi, hj).reshape(1, -1)
        dx = float(self.reg_dx.predict(x)[0])
        dz = float(self.reg_dz.predict(x)[0])
        return dx, dz

    def _collision_repair(self, poly: np.ndarray, objs: List[Dict[str, Any]], P: np.ndarray) -> np.ndarray:
        n = P.shape[0]
        dims = [_get_object_dims(o) for o in objs]

        for _ in range(self.coll_steps):
            moved = False
            rects: List[Optional[Tuple[float, float, float, float]]] = []
            for i in range(n):
                w, d = dims[i]
                if w <= 0 or d <= 0:
                    rects.append(None)
                else:
                    rects.append(_rect_from_center_dims(P[i, 0], P[i, 1], w + self.coll_margin, d + self.coll_margin))

            for i in range(n):
                if rects[i] is None:
                    continue
                for j in range(i + 1, n):
                    if rects[j] is None:
                        continue
                    if not _rect_intersect(rects[i], rects[j]):
                        continue

                    v = P[i] - P[j]
                    norm = float(np.linalg.norm(v))
                    if norm < 1e-6:
                        ang = float(self.rng.uniform(0, 2 * math.pi))
                        v = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
                        norm = 1.0
                    v = v / norm

                    step = self.coll_push
                    P[i] = P[i] + step * v
                    P[j] = P[j] - step * v
                    moved = True

                    P[i, 0], P[i, 1] = _closest_feasible_point((float(P[i, 0]), float(P[i, 1])), poly, self.rng)
                    P[j, 0], P[j, 1] = _closest_feasible_point((float(P[j, 0]), float(P[j, 1])), poly, self.rng)

            if not moved:
                break
        return P

    def predict_room(self, room: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("GraphStatPlacer: модель не обучена. Вызовите fit() или load().")

        poly = _poly_from_room(room)
        if poly.shape[0] < 3:
            raise ValueError("room.floor_polygon_xz должен быть полигоном (>=3 вершины).")

        objs = room.get("objects", [])
        if not isinstance(objs, list) or not objs:
            return []

        n = len(objs)
        H = np.stack([_node_features(objs[i], poly, self.tok_dim, self.chr_dim) for i in range(n)], axis=0).astype(
            np.float32
        )

        edges = self._infer_graph_edges(H)
        if not edges:
            cx, cz = _poly_centroid(poly)
            out = []
            for obj in objs:
                out.append(
                    {
                        "instanceid": obj.get("instanceid"),
                        "name": obj.get("name"),
                        "label": obj.get("label"),
                        "pred": {"pos": {"x": float(cx), "y": 0.0, "z": float(cz)}, "yaw_deg": 0.0},
                    }
                )
            return out

        cx, cz = _poly_centroid(poly)
        P = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            dx = float(self.rng.normal(0.0, 0.3))
            dz = float(self.rng.normal(0.0, 0.3))
            px, pz = _closest_feasible_point((cx + dx, cz + dz), poly, self.rng)
            P[i, 0] = px
            P[i, 1] = pz

        for _ in range(self.iters):
            P += self.center_pull * (np.array([cx, cz], dtype=np.float32) - P)

            for (i, j) in edges:
                dx, dz = self._predict_delta(H[i], H[j])

                target_j = P[i] + np.array([dx, dz], dtype=np.float32)
                target_i = P[j] - np.array([dx, dz], dtype=np.float32)

                P[j] = (1.0 - self.relax) * P[j] + self.relax * target_j
                P[i] = (1.0 - self.relax) * P[i] + self.relax * target_i

                P[i, 0], P[i, 1] = _closest_feasible_point((float(P[i, 0]), float(P[i, 1])), poly, self.rng)
                P[j, 0], P[j, 1] = _closest_feasible_point((float(P[j, 0]), float(P[j, 1])), poly, self.rng)

        P = self._collision_repair(poly, objs, P)

        out: List[Dict[str, Any]] = []
        for i, obj in enumerate(objs):
            out.append(
                {
                    "instanceid": obj.get("instanceid"),
                    "name": obj.get("name"),
                    "label": obj.get("label"),
                    "pred": {
                        "pos": {"x": float(P[i, 0]), "y": 0.0, "z": float(P[i, 1])},
                        "yaw_deg": 0.0,
                    },
                }
            )
        return out


# -----------------------------
# CLI
# -----------------------------

def _glob_mini_json(pattern_or_dir: str) -> List[str]:
    s = str(pattern_or_dir)
    if any(ch in s for ch in ["*", "?", "[", "]"]):
        files = glob.glob(s, recursive=True)
        files = [f for f in files if f.endswith(".mini.json")]
        files.sort()
        return files

    p = Path(s)
    if p.exists() and p.is_file():
        if str(p).endswith(".mini.json"):
            return [str(p)]
        p = p.parent

    out: List[str] = []
    if p.exists() and p.is_dir():
        for dirpath, _, filenames in os.walk(str(p)):
            for fn in filenames:
                if fn.endswith(".mini.json"):
                    out.append(os.path.join(dirpath, fn))
        out.sort()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", type=str, default="", help="Glob/dir for *.mini.json to train")
    ap.add_argument("--out_model", type=str, default="", help="Where to save model.pkl after training")
    ap.add_argument("--max_rooms", type=int, default=0, help="0=all rooms, иначе ограничение по комнатам")

    ap.add_argument("--model", type=str, default="", help="Path to saved model.pkl for inference")
    ap.add_argument("--predict_room_json", type=str, default="", help="Path to one *.mini.json to predict (first room)")
    ap.add_argument("--out_pred", type=str, default="", help="Where to write predictions json")

    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--relax", type=float, default=0.35)
    ap.add_argument("--center_pull", type=float, default=0.02)
    ap.add_argument("--coll_steps", type=int, default=40)
    ap.add_argument("--coll_push", type=float, default=0.15)
    ap.add_argument("--coll_margin", type=float, default=0.05)

    args = ap.parse_args()

    if args.train:
        paths = _glob_mini_json(args.train)
        if not paths:
            raise SystemExit(f"[graph_stat] No *.mini.json found by: {args.train}")
        if not args.out_model:
            raise SystemExit("[graph_stat] --out_model is required when --train is set")

        m = GraphStatPlacer(
            random_state=42,
            iters=int(args.iters),
            relax=float(args.relax),
            center_pull=float(args.center_pull),
            coll_steps=int(args.coll_steps),
            coll_push=float(args.coll_push),
            coll_margin=float(args.coll_margin),
        ).fit(paths, max_rooms=int(args.max_rooms))
        m.save(args.out_model)
        return

    if args.model and args.predict_room_json:
        m = GraphStatPlacer.load(args.model)
        with open(args.predict_room_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        rooms = data.get("rooms", [])
        if not rooms:
            raise SystemExit("[graph_stat] predict_room_json: no rooms")

        preds = m.predict_room(rooms[0])

        if args.out_pred:
            os.makedirs(os.path.dirname(args.out_pred) or ".", exist_ok=True)
            Path(args.out_pred).write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[graph_stat] Saved predictions: {args.out_pred}")
        else:
            print(json.dumps(preds, ensure_ascii=False, indent=2))
        return

    raise SystemExit(
        "[graph_stat] Usage:\n"
        "  Train:   python -m src.ml.baselines.graph_stat_placer --train '<glob>' --out_model runs/graph_stat/model.pkl\n"
        "  Predict: python -m src.ml.baselines.graph_stat_placer --model runs/graph_stat/model.pkl --predict_room_json <file> --out_pred runs/graph_stat/pred.json"
    )


if __name__ == "__main__":
    main()
