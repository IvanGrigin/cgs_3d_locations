import sqlite3
import json
import random
import os

DB_NAME = "furniture.db"

# ==============================
#  Категории мебели
# ==============================

CATEGORIES = {
    "kitchen": "appliance",
    "bedroom": "bed",
    "living": "seating",
    "storage": "storage",
    "bathroom": "appliance",
    "hall": "storage",
    "kids": "bed",
    "lighting": "other",
    "seating": "seating"     # ← ЭТОГО НЕ ХВАТАЛО
}

# ==============================
#  Генераторы размеров
# ==============================

def rand_range(base_min, base_max, spread=200):
    """Создаёт минимальный и максимальный размеры с небольшим разбросом."""
    min_v = random.randint(base_min, base_max)
    max_v = min_v + random.randint(50, spread)
    return min_v, max_v


def generate_item(category_prefix, name_base, cat, count, base_dims):
    """Автоматически создаёт множество вариаций предметов."""
    items = []
    for i in range(count):
        dx_min, dx_max = rand_range(base_dims[0][0], base_dims[0][1])
        dy_min, dy_max = rand_range(base_dims[1][0], base_dims[1][1])
        dz_min, dz_max = rand_range(base_dims[2][0], base_dims[2][1])

        item = {
            "name": f"{category_prefix}_{name_base}_{i+1}",
            "category": cat,
            "min_size_mm": [dx_min, dy_min, dz_min],
            "max_size_mm": [dx_max, dy_max, dz_max],
            "constraints": {
                "on_floor": True if cat != "lighting" else False,
                "touch_wall": random.choice([True, False]),
                "human_approach": random.choice([True, False]),
                "free_side": {"distance": random.randint(300, 900)} if random.random() < 0.3 else None,
                "free_side_named": {"side": "front", "distance": random.randint(400, 800)} if random.random() < 0.2 else None,
                "in_corner": random.choice([False, False, True]),
                "under_ceiling": True if cat == "other" else False
            }
        }

        items.append(item)

    return items


# ==============================
#  Автоматическое создание 100+ предметов
# ==============================

GENERATED_ITEMS = []

# Формат base_dims: ((minX, maxX), (minY, maxY), (minZ, maxZ))
GENERATED_ITEMS += generate_item("bedroom", "bed", CATEGORIES["bedroom"], 12,
                                 ((1800, 2100), (700, 1800), (400, 550)))

GENERATED_ITEMS += generate_item("bedroom", "wardrobe", CATEGORIES["storage"], 10,
                                 ((800, 2000), (400, 700), (1800, 2500)))

GENERATED_ITEMS += generate_item("kitchen", "cabinet", CATEGORIES["storage"], 15,
                                 ((400, 1200), (300, 700), (700, 1000)))

GENERATED_ITEMS += generate_item("kitchen", "fridge", CATEGORIES["kitchen"], 5,
                                 ((600, 900), (600, 800), (1700, 2200)))

GENERATED_ITEMS += generate_item("living", "sofa", CATEGORIES["living"], 10,
                                 ((1400, 2600), (700, 1200), (800, 1000)))

GENERATED_ITEMS += generate_item("living", "chair", CATEGORIES["seating"], 8,
                                 ((400, 700), (400, 700), (700, 1000)))

GENERATED_ITEMS += generate_item("storage", "shelf", CATEGORIES["storage"], 10,
                                 ((600, 1600), (300, 600), (1000, 2200)))

GENERATED_ITEMS += generate_item("bathroom", "sink", CATEGORIES["bathroom"], 5,
                                 ((400, 700), (300, 600), (800, 1100)))

GENERATED_ITEMS += generate_item("bathroom", "bathtub", CATEGORIES["bathroom"], 4,
                                 ((1400, 1800), (600, 900), (400, 700)))

GENERATED_ITEMS += generate_item("hall", "shoe_rack", CATEGORIES["hall"], 8,
                                 ((500, 900), (250, 400), (400, 700)))

GENERATED_ITEMS += generate_item("kids", "toybox", CATEGORIES["kids"], 6,
                                 ((400, 700), (300, 600), (400, 600)))

GENERATED_ITEMS += generate_item("lighting", "lamp", CATEGORIES["lighting"], 12,
                                 ((150, 300), (150, 300), (150, 300)))


# Итого: >100 предметов
assert len(GENERATED_ITEMS) >= 100

# ==============================
#  Создание БД
# ==============================

def create_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS furniture (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            category TEXT,
            min_x INTEGER, min_y INTEGER, min_z INTEGER,
            max_x INTEGER, max_y INTEGER, max_z INTEGER,
            constraints_json TEXT
        )
    """)

    conn.commit()
    conn.close()


# ==============================
#  Заполнение БД
# ==============================

def insert_items():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    for it in GENERATED_ITEMS:
        min_x, min_y, min_z = it["min_size_mm"]
        max_x, max_y, max_z = it["max_size_mm"]

        cur.execute("""
            INSERT INTO furniture
            (name, category, min_x, min_y, min_z, max_x, max_y, max_z, constraints_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            it["name"],
            it["category"],
            min_x, min_y, min_z,
            max_x, max_y, max_z,
            json.dumps(it["constraints"], ensure_ascii=False)
        ))

    conn.commit()
    conn.close()


# ==============================
#  Экспорт в JSON
# ==============================

def export_json():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT name, category, min_x, min_y, min_z, max_x, max_y, max_z, constraints_json FROM furniture")

    items = []
    for row in cur.fetchall():
        name, category, min_x, min_y, min_z, max_x, max_y, max_z, constraints = row
        items.append({
            "name": name,
            "category": category,
            "min_size_mm": [min_x, min_y, min_z],
            "max_size_mm": [max_x, max_y, max_z],
            "constraints": json.loads(constraints)
        })

    with open("exported_furniture.json", "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, indent=2, ensure_ascii=False)

    conn.close()


if __name__ == "__main__":
    create_db()
    insert_items()
    export_json()
    print(f"Готово! Создано {len(GENERATED_ITEMS)} предметов.")
