# src/ml/data/schemas.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass(frozen=True)
class RoomKey:
    house_id: str
    room_name: str
    scene_glb: str


@dataclass(frozen=True)
class RoomOBB:
    # World-space
    C: Tuple[float, float, float]      # center
    R: Tuple[float, float, float]      # right axis (unit)
    U: Tuple[float, float, float]      # up axis (unit)
    F: Tuple[float, float, float]      # forward axis (unit)
    hx: float
    hy: float
    hz: float


@dataclass(frozen=True)
class ObjOBB:
    uuid: str
    category: str
    object_id: str

    C: Tuple[float, float, float]      # center in world
    R: Tuple[float, float, float]      # right axis (unit)
    U: Tuple[float, float, float]      # up axis (unit)
    F: Tuple[float, float, float]      # forward axis (unit)
    hx: float
    hy: float
    hz: float
