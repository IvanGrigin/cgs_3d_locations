from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.multioutput import MultiOutputRegressor
except Exception as e:
    raise ImportError(
        "scikit-learn is required for tree_regressor baseline. "
        "Install: pip install scikit-learn"
    ) from e


@dataclass
class TreeLayoutModel:
    num_cats: int
    model: object  # MultiOutputRegressor(ExtraTreesRegressor)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # returns [M,2] in canonical coords
        y = self.model.predict(X).astype(np.float32)
        if y.ndim == 1:
            y = y[:, None]
        return y


def _one_hot_cat(cat_id: np.ndarray, num_cats: int) -> np.ndarray:
    # cat_id: [K] integer in [0..num_cats-1]
    out = np.zeros((cat_id.shape[0], num_cats), dtype=np.float32)
    valid = (cat_id >= 0) & (cat_id < num_cats)
    idx = np.where(valid)[0]
    out[idx, cat_id[idx]] = 1.0
    return out


def build_room_cat_counts(cat_id_room: np.ndarray, mask_room: np.ndarray, num_cats: int) -> np.ndarray:
    # cat_id_room: [N], mask_room: [N]
    counts = np.zeros((num_cats,), dtype=np.float32)
    idx = np.where(mask_room > 0.5)[0]
    if idx.size == 0:
        return counts
    for j in idx:
        c = int(cat_id_room[j])
        if 0 <= c < num_cats:
            counts[c] += 1.0
    # нормируем на число объектов (чтобы был сопоставимый масштаб)
    counts /= float(idx.size)
    return counts


def make_features_for_room(
    room_h_world: np.ndarray,          # [3] meters
    cat_id_room: np.ndarray,           # [N]
    size_room: np.ndarray,             # [N,3] normalized
    mask_room: np.ndarray,             # [N]
    num_cats: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает:
      X: [K, D] фичи для K объектов (mask==1)
      obj_idx: [K] их индексы в комнате (0..N-1)
    """
    N = cat_id_room.shape[0]
    obj_idx = np.where(mask_room > 0.5)[0]
    K = obj_idx.size
    if K == 0:
        return np.zeros((0, 1), dtype=np.float32), obj_idx

    # room-level
    hx_w, hy_w, hz_w = float(room_h_world[0]), float(room_h_world[1]), float(room_h_world[2])
    n_obj = float(K)
    cat_counts = build_room_cat_counts(cat_id_room, mask_room, num_cats)  # [C]

    # object-level
    cats = cat_id_room[obj_idx].astype(np.int64)
    oh = _one_hot_cat(cats, num_cats)  # [K,C]

    sz = size_room[obj_idx].astype(np.float32)  # [K,3]
    # индекс объекта в последовательности (как слабый сигнал структуры)
    j_norm = (obj_idx.astype(np.float32) / max(1.0, float(N - 1))).reshape(-1, 1)

    # дублируем room-cat histogram для каждого объекта
    cat_counts_rep = np.repeat(cat_counts.reshape(1, -1), K, axis=0)

    # основные фичи
    room_feats = np.tile(np.array([hx_w, hz_w, n_obj], dtype=np.float32).reshape(1, 3), (K, 1))

    X = np.concatenate(
        [
            room_feats,          # 3
            sz,                 # 3
            j_norm,             # 1
            oh,                 # C
            cat_counts_rep,     # C
        ],
        axis=1,
    ).astype(np.float32)

    return X, obj_idx


def clamp_to_room(pred_xz: np.ndarray, size_room: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    pred_xz: [N,2]
    size_room: [N,3] normalized sizes
    """
    out = pred_xz.copy().astype(np.float32)
    idx = np.where(mask > 0.5)[0]
    if idx.size == 0:
        return out

    # half-sizes in canonical coords (since room bounds are [-1,1])
    hx = 0.5 * size_room[idx, 0]
    hz = 0.5 * size_room[idx, 2]

    # если объект шире комнаты — просто clamp в центр (это редкость, но возможно)
    # (в твоём v2 таких unplaceable нет, но оставим страховку)
    low_x = -1.0 + hx
    high_x = 1.0 - hx
    low_z = -1.0 + hz
    high_z = 1.0 - hz

    x = out[idx, 0]
    z = out[idx, 1]

    # где high<low -> ставим в 0
    bad_x = (high_x < low_x)
    bad_z = (high_z < low_z)

    x = np.where(bad_x, 0.0, np.clip(x, low_x, high_x))
    z = np.where(bad_z, 0.0, np.clip(z, low_z, high_z))

    out[idx, 0] = x
    out[idx, 1] = z
    return out


def fit_tree_model(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_trees: int = 400,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 2,
    seed: int = 42,
) -> object:
    base = ExtraTreesRegressor(
        n_estimators=num_trees,
        random_state=seed,
        n_jobs=-1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X, Y)
    return model
