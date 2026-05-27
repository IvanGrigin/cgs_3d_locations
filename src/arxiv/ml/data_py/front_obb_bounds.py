from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RoomKey:
    house_id: str
    room_name: str
    scene_glb: str


@dataclass(frozen=True)
class RoomOBB:
    # World-space room OBB
    c: Tuple[float, float, float]        # center
    h: Tuple[float, float, float]        # half sizes (hx, hy, hz) in world units
    R: Tuple[float, float, float]        # right axis (world)
    U: Tuple[float, float, float]        # up axis (world)
    F: Tuple[float, float, float]        # forward axis (world)


def _sf(row: dict, k: str, default: float = float("nan")) -> float:
    s = row.get(k, "")
    if s is None or s == "":
        return default
    try:
        v = float(s)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _norm(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12 or not math.isfinite(n):
        return (0.0, 0.0, 0.0)
    return (x / n, y / n, z / n)


def _dot(a, b) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _mul(a, s: float):
    return (a[0]*s, a[1]*s, a[2]*s)


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _orthonormalize(R, U, F) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    # Логика: приводим оси к ортонормальному базису, чтобы B^{-1} = B^T работало устойчиво.
    Rn = _norm(R)
    # Gram-Schmidt для U относительно R
    Uproj = _mul(Rn, _dot(U, Rn))
    Un = _norm(_sub(U, Uproj))
    # F получаем крестом, чтобы гарантировать ортогональность
    Fn = _norm(_cross(Rn, Un))
    # Если исходный F смотрел "в другую сторону", переворачиваем Fn
    if _dot(Fn, F) < 0:
        Fn = _mul(Fn, -1.0)
    return Rn, Un, Fn


def read_room_obbs(obb_csv: str | Path) -> Dict[RoomKey, RoomOBB]:
    obb_csv = Path(obb_csv)
    if not obb_csv.exists():
        raise FileNotFoundError(f"OBB CSV not found: {obb_csv}")

    out: Dict[RoomKey, RoomOBB] = {}

    with obb_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = RoomKey(row["house_id"], row["room_name"], row["scene_glb"])

            # Логика: для комнаты в твоём OBB CSV есть специальная строка None.obj/None.obj.
            obj_id = (row.get("object_id") or "").strip()
            cat = (row.get("category") or "").strip()
            if not ((obj_id == "None.obj") or (cat == "None.obj")):
                continue

            cx, cy, cz = _sf(row, "obb_cx"), _sf(row, "obb_cy"), _sf(row, "obb_cz")
            hx, hy, hz = _sf(row, "obb_hx"), _sf(row, "obb_hy"), _sf(row, "obb_hz")

            Rx = (_sf(row, "right_x"), _sf(row, "right_y"), _sf(row, "right_z"))
            Ux = (_sf(row, "up_x"), _sf(row, "up_y"), _sf(row, "up_z"))
            Fx = (_sf(row, "fwd_x"), _sf(row, "fwd_y"), _sf(row, "fwd_z"))

            if not all(math.isfinite(v) for v in [cx, cy, cz, hx, hy, hz, *Rx, *Ux, *Fx]):
                continue
            if hx <= 1e-8 or hy <= 1e-8 or hz <= 1e-8:
                continue

            Rn, Un, Fn = _orthonormalize(Rx, Ux, Fx)
            out[k] = RoomOBB(c=(cx, cy, cz), h=(hx, hy, hz), R=Rn, U=Un, F=Fn)

    return out
