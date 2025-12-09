import json
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================

CUBE_SCRIPT = "src/Plasement/CubePlacement.py"
VIS_SCRIPT = "src/Plasement/VisualizePlacement.py"
FURNITURE_DB = "src/data/input/furniture_types.json"
OBJECTS_JSON = "src/data/input/objects.json"

MAX_ATTEMPTS = 30  # сколько раз пытаться пересобрать сцену


# ============================================================
# ЗАГРУЗКА БАЗЫ МЕБЕЛИ
# ============================================================

def load_furniture_db():
    with open(FURNITURE_DB, "r", encoding="utf-8") as f:
        data = json.load(f)

    db = {item["name"]: item for item in data["items"]}
    return db


# ============================================================
# ГЕНЕРАЦИЯ objects.json ИЗ ВВОДА
# ============================================================

def generate_objects_json(requested_names):
    db = load_furniture_db()

    items = []

    for name in requested_names:
        if name not in db:
            raise RuntimeError(f"❌ В базе нет предмета: {name}")

        src = db[name]

        items.append({
            "name": src["name"],
            "min_size_mm": src["min_size_mm"],
            "max_size_mm": src["max_size_mm"],
            "color": [0.7, 0.7, 0.7],
            "constraints": src.get("constraints", {})
        })

    data = {"items": items}

    with open(OBJECTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ objects.json сгенерирован: {len(items)} предметов")


# ============================================================
# ЗАПУСК СБОРКИ + ВИЗУАЛИЗАЦИИ
# ============================================================

def run_pipeline():
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n========== ПОПЫТКА {attempt} ==========")

        try:
            # ✅ ВСЕГДА ИСПОЛЬЗУЕМ ТОТ ЖЕ PYTHON, ЧТО ЗАПУСТИЛ ЭТОТ ФАЙЛ
            subprocess.run(
                [sys.executable, CUBE_SCRIPT],
                check=True
            )

            subprocess.run(
                [sys.executable, VIS_SCRIPT],
                check=True
            )

            print("\n✅ УСПЕХ! СЦЕНА СОБРАНА И ПРОХОДЫ КОРРЕКТНЫ")
            return

        except subprocess.CalledProcessError:
            print("⚠️ Неудачная попытка, пересборка...")
            time.sleep(0.2)

    print("\n❌ НЕ УДАЛОСЬ СОБРАТЬ КОРРЕКТНУЮ СЦЕНУ")
    sys.exit(1)


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("❌ Нужно передать список мебели!")
        print("Пример:")
        print("python src/run_pipeline.py bed sofa wardrobe table lamp")
        sys.exit(1)

    requested_items = sys.argv[1:]

    print("📦 Запрошенные предметы:")
    for it in requested_items:
        print(" -", it)

    generate_objects_json(requested_items)
    run_pipeline()


if __name__ == "__main__":
    main()