from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_imports() -> None:
    root = _repo_root()
    src = root / "src"
    for candidate in (root, src):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


_ensure_imports()

try:
    from src.pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage
except ModuleNotFoundError:
    from pipeline.procedural_rooms.procedural_room_stage import apply_procedural_room_stage


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_scene_from_room(room_json: dict[str, Any]) -> dict[str, Any]:
    if isinstance(room_json.get("room"), dict):
        room = room_json["room"]
    else:
        room = room_json
    return {
        "schema": "scene.v1",
        "room": room,
        "placements": [],
        "meta": {
            "placer": "manual_room_input",
            "mode": "procedural_room_smoke",
        },
    }


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run procedural room stage on a scene.v1.json or room.json.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="Input scene.v1.json")
    src.add_argument("--room", help="Input room.json; a temporary scene.v1 wrapper is created")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--prompt", default="", help="Prompt used for ambiguous room type normalization")
    parser.add_argument("--policy", default="always", choices=["auto", "always", "never"])
    parser.add_argument("--density", default="very_high", choices=["normal", "high", "very_high"])
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", default="standalone")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scene:
        scene_path = Path(args.scene)
    else:
        room_payload = read_json(args.room)
        scene = build_scene_from_room(room_payload)
        scene_path = out_dir / "input_scene_from_room.v1.json"
        write_json(scene_path, scene)

    report = apply_procedural_room_stage(
        scene_json_path=scene_path,
        out_dir=out_dir,
        prompt=args.prompt,
        policy=args.policy,
        density=args.density,
        replace_existing=args.replace_existing,
        seed=args.seed,
        tag=args.tag,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
