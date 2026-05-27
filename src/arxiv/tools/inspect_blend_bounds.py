#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import mathutils


def object_bounds(obj: bpy.types.Object):
    if obj.type != "MESH" or obj.hide_render:
        return None
    pts = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    if not pts:
        return None
    bmin = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    bmax = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    size = bmax - bmin
    center = (bmin + bmax) * 0.5
    return bmin, bmax, size, center


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=80)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    args = p.parse_args(argv)

    rows = []
    for obj in bpy.data.objects:
        bounds = object_bounds(obj)
        if bounds is None:
            continue
        bmin, bmax, size, center = bounds
        max_abs = max(abs(float(v)) for v in [bmin.x, bmax.x, bmin.y, bmax.y, bmin.z, bmax.z])
        max_size = max(float(size.x), float(size.y), float(size.z))
        rows.append(
            {
                "name": obj.name,
                "type": obj.type,
                "source_room_id": str(obj.get("source_room_id") or ""),
                "max_abs": round(max_abs, 4),
                "max_size": round(max_size, 4),
                "center": [round(float(center.x), 4), round(float(center.y), 4), round(float(center.z), 4)],
                "size": [round(float(size.x), 4), round(float(size.y), 4), round(float(size.z), 4)],
                "min": [round(float(bmin.x), 4), round(float(bmin.y), 4), round(float(bmin.z), 4)],
                "max": [round(float(bmax.x), 4), round(float(bmax.y), 4), round(float(bmax.z), 4)],
            }
        )
    rows.sort(key=lambda row: (row["max_abs"], row["max_size"]), reverse=True)
    payload = {"blend": bpy.data.filepath, "count": len(rows), "top": rows[: max(1, int(args.limit))]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
