# src/FromTextToJSON.py

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import requests  # pip install requests

# Пути относительно корня проекта
FURNITURE_DB = Path("src/data/input/furniture_types.json")
OBJECTS_JSON = Path("src/data/input/objects.json")

# Настройки LLM (можно переопределить через переменные окружения)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # пример для Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")              # или любой другой
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))


# ============================================================
# ЗАГРУЗКА БАЗЫ МЕБЕЛИ
# ============================================================

def load_furniture_db() -> Dict[str, Dict[str, Any]]:
    """
    furniture_types.json:
    {
      "items": [
        {
          "name": "sofa",
          "category": "...",
          "min_size_mm": [...],
          "max_size_mm": [...],
          "constraints": {...}
        },
        ...
      ]
    }
    """
    with FURNITURE_DB.open("r", encoding="utf-8") as f:
        data = json.load(f)

    db = {item["name"]: item for item in data["items"]}
    return db


def build_system_prompt(available_names: List[str]) -> str:
    """
    Жёстко описываем LLM, какой JSON нам нужен.
    """
    names_str = ", ".join(sorted(available_names))

    return f"""
Ты — парсер запроса на дизайн комнаты в JSON.

Тебе дают ТЕХНИЧЕСКОЕ ЗАДАНИЕ на русском языке, описывающее комнату и мебель.

Твоя задача — вывести ТОЛЬКО один JSON без комментариев и текста вокруг,
строгого вида:

{{
  "items": [
    {{
      "name": "<одно из допустимых имён>",
      "count": <целое число, сколько таких предметов нужно>
    }},
    ...
  ]
}}

Допустимые значения поля "name" (типовая мебель, соответствующая базе данных):

{names_str}

Правила:

1. НЕ выдумывай новые имена, используй только этот список.
2. Если пользователь пишет "двуспальная кровать" — выбери подходящий тип кровати
   из списка (например, bed_double или bed).
3. Если пользователь говорит "две прикроватные тумбочки" —
   это один тип (nightstand) с "count": 2.
4. Если пользователь явно говорит, что предмет не нужен, не добавляй его.
5. Если непонятно, какую именно модель выбрать, выбери наиболее типовую
   (например, "кровать" → bed или bed_double, "шкаф" → wardrobe).
6. Не добавляй в JSON другие поля, кроме "name" и "count".
7. Никогда не добавляй комментарии, пояснения, текст до или после JSON.

Всегда возвращай корректный JSON.
    """.strip()


# ============================================================
# ВЫЗОВ LLM
# ============================================================

def call_llm(user_text: str, available_names: List[str]) -> Dict[str, Any]:
    """
    Вызов лёгкой LLM через OpenAI-совместимое API (например, Ollama / OpenRouter).
    Ожидаем, что она вернёт JSON согласно нашему system-prompt’у.
    """
    system_prompt = build_system_prompt(available_names)

    url = f"{LLM_BASE_URL}/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Вот текст ТЗ пользователя. "
                    "Прочитай его и верни JSON согласно инструкции.\n\n"
                    + user_text
                ),
            },
        ],
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Для Ollama / OpenAI-стиля ожидаем:
    # { choices: [ { message: { content: "...." } } ] }
    content = data["choices"][0]["message"]["content"]

    # На всякий случай вырезаем "лишнее" вокруг JSON
    content_stripped = content.strip()
    first = content_stripped.find("{")
    last = content_stripped.rfind("}")
    if first == -1 or last == -1:
        raise ValueError(f"LLM вернула не-JSON: {content_stripped}")

    json_str = content_stripped[first:last + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Не удалось распарсить JSON из LLM: {e}\nТекст: {json_str}")

    if "items" not in parsed or not isinstance(parsed["items"], list):
        raise ValueError(f"Некорректная структура JSON из LLM: {parsed}")

    return parsed


# ============================================================
# ПРЕОБРАЗОВАНИЕ В objects.json ДЛЯ CubePlacement
# ============================================================

def build_objects_json(
    llm_items: List[Dict[str, Any]],
    furniture_db: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    На вход: [{"name": "...", "count": N}, ...] из LLM.
    На выход: objects.json в формате, который ест CubePlacement:

    {
      "items": [
        {
          "name": "sofa",
          "min_size_mm": [...],
          "max_size_mm": [...],
          "color": [0.7, 0.7, 0.7],
          "constraints": {...}
        },
        ...
      ]
    }
    """

    result_items: List[Dict[str, Any]] = []

    for req in llm_items:
        name = req.get("name")
        count = int(req.get("count", 1))

        if name not in furniture_db:
            # Игнорируем неизвестные имена, но можно и падать с ошибкой
            print(f"⚠️ Предмет {name!r} не найден в furniture_types.json — пропускаю")
            continue

        proto = furniture_db[name]

        for _ in range(max(count, 0)):
            result_items.append(
                {
                    "name": proto["name"],
                    "min_size_mm": proto["min_size_mm"],
                    "max_size_mm": proto["max_size_mm"],
                    "color": proto.get("color", [0.7, 0.7, 0.7]),
                    "constraints": proto.get("constraints", {}),
                }
            )

    return {"items": result_items}


def save_objects_json(data: Dict[str, Any]) -> None:
    OBJECTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OBJECTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# ENTRYPOINT
# ============================================================

def read_user_text() -> str:
    """
    Если передан путь к файлу: читаем его.
    Иначе читаем всё из stdin (можно просто вставить текст и нажать Ctrl+D).
    """
    if len(sys.argv) >= 2:
        # первый аргумент — путь к файлу с ТЗ
        path = Path(sys.argv[1])
        return path.read_text(encoding="utf-8")
    else:
        print("Вставьте текст ТЗ (Ctrl+D для конца ввода):")
        return sys.stdin.read()


def main():
    furniture_db = load_furniture_db()
    available_names = list(furniture_db.keys())

    user_text = read_user_text().strip()
    if not user_text:
        print("❌ Пустой текст. Нечего разбирать.")
        sys.exit(1)

    print("📖 ТЗ прочитано, вызываю LLM для разметки мебели...")

    parsed = call_llm(user_text, available_names)
    llm_items = parsed["items"]

    print("📦 LLM вернула список:")
    for it in llm_items:
        print(f" - {it.get('name')} × {it.get('count', 1)}")

    objects_data = build_objects_json(llm_items, furniture_db)
    save_objects_json(objects_data)

    print(f"\n✅ Файл {OBJECTS_JSON} сгенерирован, предметов: {len(objects_data['items'])}")
    print("Теперь можно запускать:")
    print("  python src/run_pipeline.py sofa wardrobe ...  — или")
    print("  напрямую CubePlacement, если он читает src/data/input/objects.json")


if __name__ == "__main__":
    main()