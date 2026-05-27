#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_source_category_id_to_label.py

Строит глобальный mapping:
    sourceCategoryId -> semantic label

по полным JSON-файлам 3D-FRONT.

Зачем:
- в 3D-FRONT у части объектов пустые category/title;
- при этом sourceCategoryId часто заполнен;
- по множеству сцен можно восстановить наиболее вероятную метку для каждого sourceCategoryId;
- это полезно для подготовки данных под FID / KID / SCA / CKL,
  особенно для semantic top-down renderings и Category KL.

Логика:
1. Проходим по всем JSON в папке full 3D-FRONT.
2. Для каждого furniture-объекта смотрим:
   - sourceCategoryId
   - category
   - title
3. Нормализуем текстовые метки.
4. Для каждого sourceCategoryId собираем частоты меток.
5. В качестве итоговой метки выбираем самую частую.
6. Сохраняем:
   - source_category_id_to_label.json
   - source_category_id_to_label_detailed.json
   - source_category_id_to_label.csv

Использование:
python src/tools/build_source_category_id_to_label.py \
  --front-root data/sourse/3D-FRONT/3D-FRONT \
  --out-dir data/sourse/3D-FRONT/source_category_label_index

Быстрый запуск на части данных:
python src/tools/build_source_category_id_to_label.py \
  --front-root data/sourse/3D-FRONT/3D-FRONT \
  --out-dir out/source_category_label_index_test \
  --limit-files 500
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# IO
# ============================================================

def load_json(path: str | Path) -> Any:
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# Базовые утилиты
# ============================================================

def as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def iter_json_files(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Папка не найдена: {root}")
    files = sorted(x for x in root.rglob("*.json") if x.is_file())
    if not files:
        raise RuntimeError(f"В папке нет *.json: {root}")
    return files


# ============================================================
# Нормализация меток
# ============================================================

_SPLIT_SLASH_RE = re.compile(r"\s*/\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _cleanup_label(s: str) -> str:
    s = s.strip()
    s = s.replace("_", " ")
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s.strip()


def _looks_meaningful_label(s: str) -> bool:
    if not s:
        return False

    bad_exact = {
        "",
        "null",
        "none",
        "nan",
        "unknown",
        "others",
        "other",
        "object",
        "furniture",
        "standard",
    }
    if s.lower() in bad_exact:
        return False

    return True


def normalize_label(raw: str) -> Optional[str]:
    """
    Нормализация category/title.
    Примеры:
      "lighting/pendant light" -> "Pendant Light"
      "table/night table" -> "Night Table"
      "storage unit/armoire" -> "Armoire"
      "door/entry/single swing door" -> "Single Swing Door"
    """
    s = _cleanup_label(raw)
    if not s:
        return None

    # Если строка вида a/b/c, чаще всего полезнее взять последний сегмент.
    parts = [p.strip() for p in _SPLIT_SLASH_RE.split(s) if p.strip()]
    if parts:
        s = parts[-1]

    s = _cleanup_label(s)
    if not _looks_meaningful_label(s):
        return None

    # Title Case, но без агрессивной порчи уже нормального текста
    words = s.split(" ")
    s = " ".join(w if any(ch.isupper() for ch in w[1:]) else w.capitalize() for w in words)

    # Небольшие правки частых артефактов
    replacements = {
        "Tv": "TV",
        "Tv Stand": "TV Stand",
        "Night Table": "Night Table",
        "Pendant Light": "Pendant Light",
        "Single Swing Door": "Single Swing Door",
        "Double Swing Door - Asymmetrical": "Double Swing Door - Asymmetrical",
        "Floor-Based Window": "Floor-Based Window",
        "Floor-Based Media Unit": "Floor-Based Media Unit",
        "Side Cabinet": "Side Cabinet",
    }
    s = replacements.get(s, s)

    return s if _looks_meaningful_label(s) else None


def candidate_labels_from_furniture_row(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Возвращает список кандидатов вида:
      (source_name, normalized_label)
    Приоритет потом будет задаваться весами, а не порядком в списке.
    """
    out: List[Tuple[str, str]] = []

    raw_category = normalize_label(as_str(row.get("category")).strip())
    if raw_category:
        out.append(("category", raw_category))

    raw_title = normalize_label(as_str(row.get("title")).strip())
    if raw_title:
        out.append(("title", raw_title))

    return out


# ============================================================
# Основная агрегация
# ============================================================

def build_source_category_index(
    *,
    front_root: Path,
    limit_files: Optional[int] = None,
) -> Dict[str, Any]:
    files = iter_json_files(front_root)
    if limit_files is not None and limit_files > 0:
        files = files[:limit_files]

    stats: Dict[str, Any] = {
        "processed_files": 0,
        "failed_files": 0,
        "total_furniture_rows": 0,
        "rows_with_source_category_id": 0,
        "rows_with_any_label": 0,
        "unique_source_category_ids": 0,
    }

    errors: List[Dict[str, Any]] = []

    # sourceCategoryId -> Counter(label -> weighted count)
    label_counter_by_source_id: Dict[str, Counter] = defaultdict(Counter)

    # sourceCategoryId -> raw meta
    meta_by_source_id: Dict[str, Dict[str, Any]] = {}

    for file_path in files:
        try:
            root = load_json(file_path)
            stats["processed_files"] += 1
        except Exception as exc:
            stats["failed_files"] += 1
            errors.append({
                "file": str(file_path),
                "error": str(exc),
            })
            continue

        furniture = root.get("furniture")
        if not isinstance(furniture, list):
            continue

        source_uid = as_str(root.get("uid")).strip()

        for row in furniture:
            if not isinstance(row, dict):
                continue

            stats["total_furniture_rows"] += 1

            source_category_id = as_str(row.get("sourceCategoryId")).strip()
            if not source_category_id:
                continue

            stats["rows_with_source_category_id"] += 1

            # Вес category выше title
            candidates = candidate_labels_from_furniture_row(row)
            if candidates:
                stats["rows_with_any_label"] += 1

            meta = meta_by_source_id.setdefault(source_category_id, {
                "examples": [],
                "seen_count": 0,
                "valid_true_count": 0,
                "valid_false_count": 0,
                "category_counter_raw": Counter(),
                "title_counter_raw": Counter(),
                "jid_counter": Counter(),
                "uids_examples": [],
            })

            meta["seen_count"] += 1
            if row.get("valid") is True:
                meta["valid_true_count"] += 1
            elif row.get("valid") is False:
                meta["valid_false_count"] += 1

            raw_category_str = as_str(row.get("category")).strip()
            raw_title_str = as_str(row.get("title")).strip()
            jid_str = as_str(row.get("jid")).strip()
            uid_str = as_str(row.get("uid")).strip()

            if raw_category_str:
                meta["category_counter_raw"][raw_category_str] += 1
            if raw_title_str:
                meta["title_counter_raw"][raw_title_str] += 1
            if jid_str:
                meta["jid_counter"][jid_str] += 1
            if uid_str and len(meta["uids_examples"]) < 10:
                meta["uids_examples"].append(uid_str)

            if len(meta["examples"]) < 5:
                meta["examples"].append({
                    "scene_uid": source_uid,
                    "uid": uid_str,
                    "jid": jid_str or None,
                    "category": raw_category_str or None,
                    "title": raw_title_str or None,
                    "valid": row.get("valid"),
                })

            for source_name, label in candidates:
                weight = 3 if source_name == "category" else 1
                label_counter_by_source_id[source_category_id][label] += weight

    stats["unique_source_category_ids"] = len(label_counter_by_source_id)

    resolved: Dict[str, Any] = {}
    unresolved: Dict[str, Any] = {}

    all_source_ids = set(meta_by_source_id.keys()) | set(label_counter_by_source_id.keys())

    for source_category_id in sorted(all_source_ids):
        counter = label_counter_by_source_id.get(source_category_id, Counter())
        meta = meta_by_source_id.get(source_category_id, {})

        if counter:
            best_label, best_score = counter.most_common(1)[0]
            resolved[source_category_id] = {
                "label": best_label,
                "score": best_score,
                "alternatives": [
                    {"label": label, "score": score}
                    for label, score in counter.most_common(10)
                ],
                "seen_count": meta.get("seen_count", 0),
                "valid_true_count": meta.get("valid_true_count", 0),
                "valid_false_count": meta.get("valid_false_count", 0),
                "top_raw_categories": [
                    {"value": label, "count": count}
                    for label, count in meta.get("category_counter_raw", Counter()).most_common(10)
                ],
                "top_raw_titles": [
                    {"value": label, "count": count}
                    for label, count in meta.get("title_counter_raw", Counter()).most_common(10)
                ],
                "top_jids": [
                    {"jid": jid, "count": count}
                    for jid, count in meta.get("jid_counter", Counter()).most_common(10)
                ],
                "examples": meta.get("examples", []),
            }
        else:
            unresolved[source_category_id] = {
                "seen_count": meta.get("seen_count", 0),
                "valid_true_count": meta.get("valid_true_count", 0),
                "valid_false_count": meta.get("valid_false_count", 0),
                "top_jids": [
                    {"jid": jid, "count": count}
                    for jid, count in meta.get("jid_counter", Counter()).most_common(10)
                ],
                "examples": meta.get("examples", []),
            }

    summary = {
        "stats": stats,
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "errors_count": len(errors),
    }

    return {
        "summary": summary,
        "resolved": resolved,
        "unresolved": unresolved,
        "errors": errors,
    }


# ============================================================
# Экспорт
# ============================================================

def write_csv_mapping(path: Path, resolved: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sourceCategoryId",
            "label",
            "score",
            "seen_count",
            "valid_true_count",
            "valid_false_count",
        ])
        for source_category_id in sorted(resolved.keys()):
            row = resolved[source_category_id]
            writer.writerow([
                source_category_id,
                row.get("label"),
                row.get("score"),
                row.get("seen_count"),
                row.get("valid_true_count"),
                row.get("valid_false_count"),
            ])


def export_results(out_dir: Path, result: Dict[str, Any]) -> None:
    out_dir = ensure_dir(out_dir)

    resolved = result["resolved"]
    unresolved = result["unresolved"]

    simple_mapping = {
        source_category_id: data["label"]
        for source_category_id, data in sorted(resolved.items())
    }

    save_json(out_dir / "source_category_id_to_label.json", simple_mapping)
    save_json(out_dir / "source_category_id_to_label_detailed.json", result)
    save_json(out_dir / "source_category_id_unresolved.json", unresolved)
    write_csv_mapping(out_dir / "source_category_id_to_label.csv", resolved)


# ============================================================
# CLI
# ============================================================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Строит mapping sourceCategoryId -> semantic label по полному 3D-FRONT"
    )
    p.add_argument(
        "--front-root",
        required=True,
        help="Папка с полными JSON 3D-FRONT, например data/sourse/3D-FRONT/3D-FRONT",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Куда сохранить mapping и подробные отчеты",
    )
    p.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Ограничить число обрабатываемых файлов для быстрого теста",
    )
    return p


def main() -> None:
    args = build_cli().parse_args()

    front_root = Path(args.front_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    result = build_source_category_index(
        front_root=front_root,
        limit_files=args.limit_files,
    )
    export_results(out_dir, result)

    summary = result["summary"]
    print("OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()