#!/usr/bin/env bash
set -euo pipefail

# Путь к Blender (можно переопределить: BLENDER_BIN=... ./src/tools/view_many_objs.sh)
BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"

# Blender python-скрипт, который умеет показывать OBJ из папки
SCRIPT="${SCRIPT:-src/Plasement/BlenderViewOBJFolder.py}"

# Корневая папка с предметами (каждый предмет — отдельная директория)
ROOT_DIR="${ROOT_DIR:-src/data/sourse/imodern}"

# Сколько предметов открыть
N="${1:-12}"

# 1 = случайный порядок, 0 = по алфавиту
SHUFFLE="${SHUFFLE:-0}"

# --- выбрать самый большой OBJ рекурсивно внутри папки предмета ---
pick_largest_obj() {
  /usr/bin/python3 - "$1" <<'PY'
import os, sys
root = sys.argv[1]
best = None  # (size, path)
for dirpath, _, files in os.walk(root):
    for fn in files:
        if fn.lower().endswith(".obj"):
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if best is None or sz > best[0]:
                best = (sz, os.path.abspath(p))
if best is None:
    sys.exit(2)
print(best[1])
PY
}

# --- список верхнеуровневых папок-предметов ---
list_items() {
  find "$ROOT_DIR" -mindepth 1 -maxdepth 1 -type d -print | sort
}

shuffle_items_py() {
  /usr/bin/python3 - <<'PY'
import sys, random
items = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
random.shuffle(items)
for x in items:
    print(x)
PY
}

# --- проверки ---
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "[VIEW] ERROR: ROOT_DIR is not a directory: $ROOT_DIR" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "[VIEW] ERROR: SCRIPT not found: $SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$BLENDER_BIN" ]]; then
  echo "[VIEW] ERROR: Blender not executable: $BLENDER_BIN" >&2
  exit 1
fi

# --- получить список предметов ---
if [[ "$SHUFFLE" == "1" ]]; then
  ITEMS_STREAM="$(list_items | shuffle_items_py)"
else
  ITEMS_STREAM="$(list_items)"
fi

if [[ -z "${ITEMS_STREAM// }" ]]; then
  echo "[VIEW] No item folders in: $ROOT_DIR" >&2
  exit 1
fi

count=0

# ВАЖНО: без mapfile. Читаем поток построчно.
echo "$ITEMS_STREAM" | while IFS= read -r item_dir; do
  [[ -z "$item_dir" ]] && continue

  if [[ "$count" -ge "$N" ]]; then
    exit 0
  fi

  # пропускаем служебные каталоги
  base="$(basename "$item_dir")"
  if [[ "$base" == "_archives" ]]; then
    continue
  fi

  echo
  echo "=============================="
  echo "[VIEW] $(($count + 1)) / $N : $base"
  echo "[VIEW] Folder: $item_dir"

  # Находим конкретный OBJ, чтобы показать (самый большой)
  OBJ_PATH=""
  if OBJ_PATH="$(pick_largest_obj "$item_dir" 2>/dev/null)"; then
    :
  else
    echo "[VIEW] Skip: no OBJ inside: $item_dir"
    continue
  fi

  echo "[VIEW] OBJ: $OBJ_PATH"
  echo "[VIEW] Close Blender window to continue to next item."

  # Запуск Blender в GUI-режиме. Он блокирует скрипт до закрытия окна.
  "$BLENDER_BIN" --python "$SCRIPT" -- \
    --dir "$item_dir" \
    --fit 4.0

  count=$((count + 1))
done