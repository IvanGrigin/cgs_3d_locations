#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select diverse mebel.ru cards for GLB generation smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "canonical_items", "products", "cards", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def has_image(card: dict[str, Any]) -> bool:
    images = card.get("images")
    return isinstance(images, list) and any(str(x).strip() for x in images)


def has_no_asset(card: dict[str, Any]) -> bool:
    if str(card.get("asset_status") or "") == "trellis2_generated_candidate":
        return False
    for key in ("asset_local_path", "glb_path", "fbx_path", "obj_path", "model_download_url"):
        if str(card.get(key) or "").strip():
            return False
    return True


def category(card: dict[str, Any]) -> str:
    return str(card.get("category_norm") or card.get("category") or card.get("category_raw") or "unknown")


def selected_payload(cards: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    return {
        "schema": "mebelru_diverse_glb_test_cards/v1",
        "source": str(source.resolve()),
        "item_count": len(cards),
        "categories": sorted({category(card) for card in cards}),
        "items": cards,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Select diverse mebel.ru cards for GLB smoke tests.")
    ap.add_argument("--catalog-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--source-site", default="mebel.ru")
    ap.add_argument("--min-per-category", type=int, default=1)
    args = ap.parse_args()

    source = Path(args.catalog_json).expanduser()
    cards = [
        card
        for card in extract_items(read_json(source))
        if str(card.get("source_site") or "") == args.source_site and has_image(card) and has_no_asset(card)
    ]

    by_category: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        by_category.setdefault(category(card), []).append(card)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cat in sorted(by_category):
        for card in by_category[cat][: max(1, int(args.min_per_category))]:
            key = str(card.get("unique_key") or card.get("external_id") or card.get("product_url"))
            if key and key not in seen:
                selected.append(card)
                seen.add(key)
                if len(selected) >= args.limit:
                    break
        if len(selected) >= args.limit:
            break

    if len(selected) < args.limit:
        for card in cards:
            key = str(card.get("unique_key") or card.get("external_id") or card.get("product_url"))
            if key and key in seen:
                continue
            selected.append(card)
            seen.add(key)
            if len(selected) >= args.limit:
                break

    write_json(args.out_json, selected_payload(selected[: args.limit], source))
    print(f"[out] {args.out_json}")
    print(f"[selected] {len(selected[: args.limit])}")
    for idx, card in enumerate(selected[: args.limit], 1):
        print(f"{idx}. {category(card)} | {card.get('unique_key')} | {card.get('title')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
