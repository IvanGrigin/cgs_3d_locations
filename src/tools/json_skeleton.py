#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from collections import defaultdict
from typing import Any, Dict, Set, Tuple

TypeSet = Set[str]

def type_name(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int) and not isinstance(x, bool):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    if isinstance(x, list):
        return "list"
    if isinstance(x, dict):
        return "object"
    return type(x).__name__

def merge_types(a: TypeSet, b: TypeSet) -> TypeSet:
    return set(a) | set(b)

def walk(value: Any,
         path: Tuple[str, ...],
         types_by_path: Dict[Tuple[str, ...], TypeSet],
         max_list_samples: int = 50) -> None:
    """
    Обход JSON и сбор:
      - types_by_path[path] = {types}
    Для массивов:
      - сам путь получает тип list
      - элементы обходятся как path + ("[]",)
    """
    t = type_name(value)
    types_by_path[path].add(t)

    if isinstance(value, dict):
        for k, v in value.items():
            walk(v, path + (k,), types_by_path, max_list_samples=max_list_samples)

    elif isinstance(value, list):
        # Сэмплируем элементы, чтобы не упираться в гигантские массивы
        for i, item in enumerate(value[:max_list_samples]):
            walk(item, path + ("[]",), types_by_path, max_list_samples=max_list_samples)
        # Если массив длиннее, структура элементов всё равно обычно повторяется;
        # при желании max_list_samples можно поднять.

def format_tree(types_by_path: Dict[Tuple[str, ...], TypeSet]) -> str:
    """
    Печать дерева путей с типами. Узлы сортируются.
    """
    # Построим вложенное дерево
    tree = {}
    for p, ts in types_by_path.items():
        node = tree
        for key in p:
            node = node.setdefault(key, {})
        node.setdefault("__types__", set()).update(ts)

    lines = []

    def rec(node: dict, indent: int, name: str = ""):
        if name:
            ts = node.get("__types__", set())
            tss = ", ".join(sorted(ts)) if ts else ""
            lines.append("  " * indent + f"- {name}" + (f" : {tss}" if tss else ""))
        # ключи кроме __types__
        for k in sorted([x for x in node.keys() if x != "__types__"]):
            rec(node[k], indent + (1 if name else 0), k)

    # корень
    for k in sorted(tree.keys()):
        rec(tree[k], 0, k)

    return "\n".join(lines)

def detect_type_conflicts(types_by_path: Dict[Tuple[str, ...], TypeSet]) -> Dict[str, Set[str]]:
    """
    Возвращает пути, где встречается более одного 'базового' типа.
    (int/float считаем разными; при желании можно схлопнуть в number)
    """
    conflicts = {}
    for p, ts in types_by_path.items():
        if len(ts) > 1:
            conflicts[".".join(p)] = set(ts)
    return conflicts

def main():
    if len(sys.argv) < 2:
        print("Usage: json_skeleton.py <file.json> [max_list_samples]")
        sys.exit(2)

    filename = sys.argv[1]
    max_list_samples = int(sys.argv[2]) if len(sys.argv) >= 3 else 50

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    types_by_path: Dict[Tuple[str, ...], TypeSet] = defaultdict(set)
    walk(data, tuple(), types_by_path, max_list_samples=max_list_samples)

    print("=== JSON FIELD TREE (paths with types) ===")
    print(format_tree(types_by_path))

    conflicts = detect_type_conflicts(types_by_path)
    if conflicts:
        print("\n=== TYPE CONFLICTS (same path, different types) ===")
        for p in sorted(conflicts.keys()):
            ts = ", ".join(sorted(conflicts[p]))
            print(f"- {p} : {ts}")

if __name__ == "__main__":
    main()