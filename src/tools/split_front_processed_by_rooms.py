# src/tools/split_front_processed_by_rooms.py
# Разбивает каждый JSON из data/sourse/3D-FRONT/3D-FRONT-processed на отдельные файлы по комнатам.
# Старый файл удаляется, вместо него создаются новые: <uid>__<room_id>.json
# Запуск:
#   python3 src/tools/split_front_processed_by_rooms.py \
#     --input_dir data/sourse/3D-FRONT/3D-FRONT-processed \
#     --delete_original 1

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def safe_name(s: str) -> str:
    # Безопасное имя файла: оставляем буквы/цифры/._- , остальное заменяем на _
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_")


def is_json_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".json"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split_one_scene(scene: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Возвращает список (room_id, room_scene_json).
    Каждый room_scene_json имеет формат как исходный, но rooms = [одна комната].
    """
    uid = scene.get("uid")
    north_vector = scene.get("north_vector")
    rooms = scene.get("rooms") or []

    out: List[Tuple[str, Dict[str, Any]]] = []
    for r in rooms:
        room_id = str(r.get("id", "Room"))
        one = {
            "uid": uid,
            "north_vector": north_vector,
            "rooms": [r],
        }
        out.append((room_id, one))
    return out


def process_file(path: Path, delete_original: bool) -> int:
    """
    Обрабатывает один JSON-файл, создаёт room-файлы в той же папке.
    Возвращает количество созданных файлов.
    """
    scene = load_json(path)

    uid = str(scene.get("uid") or path.stem)
    splits = split_one_scene(scene)

    # Если нет комнат — ничего не делаем (оставляем файл как есть).
    if not splits:
        return 0

    out_dir = path.parent
    created = 0

    for room_id, room_json in splits:
        out_name = f"{safe_name(uid)}__{safe_name(room_id)}.json"
        out_path = out_dir / out_name
        write_json(out_path, room_json)
        created += 1

    if delete_original:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return created


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Папка с processed JSON (например data/sourse/3D-FRONT/3D-FRONT-processed)")
    ap.add_argument("--delete_original", type=int, default=1, help="1=удалять исходный файл после разбиения, 0=оставлять (по умолчанию 1)")
    ap.add_argument("--dry_run", type=int, default=0, help="1=ничего не писать/не удалять, только логировать")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir не существует или не папка: {input_dir}")

    delete_original = bool(args.delete_original)
    dry_run = bool(args.dry_run)

    files = sorted([p for p in input_dir.iterdir() if is_json_file(p)])

    total_in = 0
    total_out = 0
    total_skipped = 0

    for f in files:
        total_in += 1
        try:
            scene = load_json(f)
        except Exception as e:
            print(f"[SKIP] {f.name}: не удалось прочитать JSON: {e}")
            total_skipped += 1
            continue

        rooms = scene.get("rooms") or []
        if not rooms:
            print(f"[SKIP] {f.name}: rooms пустой")
            total_skipped += 1
            continue

        uid = str(scene.get("uid") or f.stem)
        out_names = [f"{safe_name(uid)}__{safe_name(str(r.get('id','Room')))}.json" for r in rooms]

        if dry_run:
            print(f"[DRY] {f.name} -> {len(out_names)} файлов; delete_original={delete_original}")
            for n in out_names:
                print(f"      + {n}")
            if delete_original:
                print(f"      - {f.name}")
            total_out += len(out_names)
            continue

        created = process_file(f, delete_original=delete_original)
        total_out += created
        print(f"[OK] {f.name} -> {created} файлов; delete_original={delete_original}")

    print("----")
    print(f"Входных файлов: {total_in}")
    print(f"Пропущено:      {total_skipped}")
    print(f"Создано:        {total_out}")


if __name__ == "__main__":
    main()
