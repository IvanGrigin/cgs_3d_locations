# src/tools/find_obj_from_db.py
# -*- coding: utf-8 -*-
import os
import re
import sys
import glob
import sqlite3
from difflib import SequenceMatcher

DB_PATH = os.path.join("src", "data", "sourse", "imodern.db")
EXTRACT_ROOT = os.path.join("src", "data", "sourse", "imodern")

# очень лёгкий словарь синонимов/нормализаций
SYNONYMS = {
    "тумбочка": "тумба",
    "тумба тв": "тумба",
    "тв": "тв",
    "tv": "тв",
    "журнальный столик": "столик",
    "кроватка": "кровать",
    "диван-кровать": "кровать диван",
}

_word_re = re.compile(r"[a-zа-яё0-9]+")

def normalize_text(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    # подставим синонимы целиком
    for k, v in SYNONYMS.items():
        s = s.replace(k, v)
    return s

def tokenize(s: str) -> set[str]:
    s = normalize_text(s)
    return set(_word_re.findall(s))

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def char_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def find_item_folder_by_bx(bx_id: str) -> str | None:
    # Ищем распакованную папку с суффиксом "__bx_<id>"
    pattern = os.path.join(EXTRACT_ROOT, f"*__{bx_id}")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None

def find_largest_obj(root_dir: str) -> str | None:
    best_path, best_size = None, -1
    for r, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".obj"):
                p = os.path.join(r, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > best_size:
                    best_size = sz
                    best_path = p
    return best_path

def iter_items_with_obj(conn):
    # Берём все записи; фильтровать будем по наличию obj на лету
    q = "SELECT bx_id, name FROM imodern_item"
    for bx_id, name in conn.execute(q):
        bx_id = str(bx_id)
        folder = find_item_folder_by_bx(bx_id)
        if not folder:
            continue
        obj_path = find_largest_obj(folder)
        if not obj_path:
            continue
        yield {"bx_id": bx_id, "name": name, "obj": obj_path}

def fallback_any_obj() -> str | None:
    return find_largest_obj(EXTRACT_ROOT)

def choose_best_item(term: str, candidates: list[dict]) -> dict | None:
    term_tokens = tokenize(term)
    best, best_score = None, -1.0
    for it in candidates:
        name = it["name"] or ""
        name_tokens = tokenize(name)
        s_tok = jaccard(term_tokens, name_tokens)
        s_chr = char_ratio(term, name)
        # весами повышаем приоритет совпадения по словам
        score = 2.0 * s_tok + 1.0 * s_chr
        # небольшие бонусы за прямые ключевые слова
        if "кровать" in term_tokens and ("кровать" in name_tokens or "диван" in name_tokens):
            score += 0.05
        if "тумба" in term_tokens and ("тумба" in name_tokens or "комод" in name_tokens):
            score += 0.05
        if score > best_score:
            best_score = score
            best = it
    return best

def main():
    term = sys.argv[1] if len(sys.argv) > 1 else "кровать"

    if not os.path.isfile(DB_PATH):
        # если БД отсутствует — сразу общий обход
        any_obj = fallback_any_obj()
        if any_obj:
            full = os.path.abspath(any_obj)
            print(f"{full} {os.path.basename(full)}")
            return 0
        # совсем ничего — тихий выход (по ТЗ печатаем только объект)
        return 2

    with sqlite3.connect(DB_PATH) as conn:
        conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
        candidates = list(iter_items_with_obj(conn))

    if candidates:
        best = choose_best_item(term, candidates)
        if best and best.get("obj"):
            full = os.path.abspath(best["obj"])
            print(f"{full} {os.path.basename(full)}")
            return 0

    # fallback: берём любой .obj из дерева
    any_obj = fallback_any_obj()
    if any_obj:
        full = os.path.abspath(any_obj)
        print(f"{full} {os.path.basename(full)}")
        return 0

    # Ничего не нашли — ничего не печатаем (строгий stdout)
    return 2

if __name__ == "__main__":
    sys.exit(main())