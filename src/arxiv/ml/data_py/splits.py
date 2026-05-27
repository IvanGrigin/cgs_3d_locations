# src/ml/data/splits.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def make_house_id_splits(
    room_house_ids: Sequence[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[int]]:
    """
    room_house_ids: список house_id по комнатам в порядке массива датасета.
    Возвращает индексы комнат для train/val/test.
    """
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1

    uniq_houses = sorted(set(room_house_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq_houses)

    n = len(uniq_houses)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_h = set(uniq_houses[:n_train])
    val_h = set(uniq_houses[n_train:n_train+n_val])
    test_h = set(uniq_houses[n_train+n_val:])

    splits = {"train": [], "val": [], "test": []}
    for i, hid in enumerate(room_house_ids):
        if hid in train_h:
            splits["train"].append(i)
        elif hid in val_h:
            splits["val"].append(i)
        else:
            splits["test"].append(i)
    return splits


def save_splits(splits: Dict[str, List[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)


def load_splits(path: Path) -> Dict[str, List[int]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
