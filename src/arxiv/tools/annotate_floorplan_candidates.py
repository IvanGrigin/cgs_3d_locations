#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_floorplan_candidates.py

Интерактивная разметка crop-кандидатов планировок.

Клавиши:
  SPACE / ENTER / g -> good
  любая другая      -> bad
  n                 -> bad
  b / Backspace     -> previous
  u                 -> unlabel current
  q                 -> save and quit
  ?                 -> help

Вход:
  1) --input <dir>         папка ranked_all/top_k; порядок берётся из имени 000001__score_...
  2) --results-csv <csv>   results.csv ранжировщика; порядок по убыванию floorplan_score

Выход:
  out/annotation_manifest.json
  out/labels.jsonl
  out/labels.csv
  out/good/
  out/bad/

Пример:
  python3 src/tools/annotate_floorplan_candidates.py \
    --results-csv data/housesru/floorplans_ranked_v2/results.csv \
    --out data/housesru/floorplans_manual_labels_v1 \
    --mode symlink \
    --limit 1600
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Candidate:
    index: int
    src: str
    name: str
    score: float
    rank: int
    hard_reject: bool
    meta: dict[str, Any]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "image"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def infer_score_from_name(path: Path) -> float:
    m = re.search(r"score_([+-]?\d+(?:\.\d+)?)", path.name)
    if m:
        return parse_float(m.group(1), 0.0)
    return 0.0


def infer_rank_from_name(path: Path) -> int:
    m = re.match(r"^(\d{1,8})__", path.name)
    if m:
        return int(m.group(1))
    return 10**9


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_candidates_from_dir(input_dir: Path) -> list[Candidate]:
    images = find_images(input_dir)
    candidates: list[Candidate] = []

    for i, path in enumerate(images, start=1):
        candidates.append(
            Candidate(
                index=i,
                src=str(path.resolve()),
                name=path.name,
                score=infer_score_from_name(path),
                rank=infer_rank_from_name(path),
                hard_reject=False,
                meta={},
            )
        )

    candidates.sort(key=lambda c: (c.rank, -c.score, c.name))
    for i, c in enumerate(candidates, start=1):
        c.index = i
    return candidates


def resolve_candidate_source(row: dict[str, str], csv_path: Path) -> Optional[Path]:
    for key in ["top_k_path", "ranked_path", "src"]:
        value = row.get(key)
        if not value:
            continue
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = (csv_path.parent / p).resolve()
        if p.exists():
            return p
    return None


def load_candidates_from_csv(csv_path: Path, include_hard_rejects: bool) -> list[Candidate]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    candidates: list[Candidate] = []
    for row in rows:
        hard_reject = parse_bool(row.get("hard_reject"))
        if hard_reject and not include_hard_rejects:
            continue

        src_path = resolve_candidate_source(row, csv_path)
        if src_path is None:
            continue

        candidates.append(
            Candidate(
                index=len(candidates) + 1,
                src=str(src_path.resolve()),
                name=src_path.name,
                score=parse_float(row.get("floorplan_score"), 0.0),
                rank=parse_int(row.get("rank"), 10**9),
                hard_reject=hard_reject,
                meta=dict(row),
            )
        )

    candidates.sort(key=lambda c: (c.hard_reject, -c.score, c.rank, c.name))
    for i, c in enumerate(candidates, start=1):
        c.index = i
    return candidates


def read_existing_labels(labels_path: Path) -> dict[str, dict[str, Any]]:
    if not labels_path.exists():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            src = row.get("src")
            if src:
                labels[str(src)] = row
    return labels


def rewrite_labels(labels_path: Path, labels: dict[str, dict[str, Any]], candidates: list[Candidate]) -> None:
    ensure_dir(labels_path.parent)
    order = {c.src: i for i, c in enumerate(candidates)}
    rows = sorted(labels.values(), key=lambda r: order.get(str(r.get("src")), 10**9))
    with labels_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_labels_csv(csv_path: Path, labels: dict[str, dict[str, Any]], candidates: list[Candidate]) -> None:
    ensure_dir(csv_path.parent)
    order = {c.src: i for i, c in enumerate(candidates)}
    rows = sorted(labels.values(), key=lambda r: order.get(str(r.get("src")), 10**9))
    fieldnames = ["src", "name", "label", "is_good", "score", "rank", "hard_reject", "annotated_at_unix"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def materialize(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def rebuild_label_dirs(out_dir: Path, labels: dict[str, dict[str, Any]], candidates: list[Candidate], mode: str) -> None:
    good_dir = out_dir / "good"
    bad_dir = out_dir / "bad"
    if good_dir.exists():
        shutil.rmtree(good_dir)
    if bad_dir.exists():
        shutil.rmtree(bad_dir)
    ensure_dir(good_dir)
    ensure_dir(bad_dir)

    by_src = {c.src: c for c in candidates}
    for src, row in labels.items():
        c = by_src.get(src)
        if c is None:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        prefix = f"{c.index:06d}__score_{c.score:+08.3f}__"
        dst_name = prefix + safe_name(c.name)
        dst = (good_dir if row.get("is_good") else bad_dir) / dst_name
        materialize(src_path, dst, mode=mode)


def fit_image_to_screen(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 0.999:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_header(img: np.ndarray, text_lines: list[str]) -> np.ndarray:
    h, w = img.shape[:2]
    header_h = 28 + 22 * len(text_lines)
    out = np.full((h + header_h, w, 3), 245, dtype=np.uint8)
    out[header_h:, :, :] = img
    y = 28
    for line in text_lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y += 22
    return out


def load_display_image(path: Path, max_w: int, max_h: int, header_lines: list[str]) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = np.full((300, 900, 3), 255, dtype=np.uint8)
        cv2.putText(img, f"Cannot read image: {path}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    img = fit_image_to_screen(img, max_w=max_w, max_h=max_h)
    return add_header(img, header_lines)


def label_row(candidate: Candidate, label: str) -> dict[str, Any]:
    return {
        "src": candidate.src,
        "name": candidate.name,
        "label": label,
        "is_good": label == "good",
        "score": candidate.score,
        "rank": candidate.rank,
        "hard_reject": candidate.hard_reject,
        "annotated_at_unix": time.time(),
        "meta": candidate.meta,
    }


def print_help() -> None:
    print()
    print("Клавиши:")
    print("  SPACE / ENTER / g  -> good")
    print("  любая другая       -> bad")
    print("  n                  -> bad")
    print("  b / Backspace      -> previous")
    print("  u                  -> unlabel current")
    print("  q                  -> save and quit")
    print("  ?                  -> help")
    print()


def find_start_index(candidates: list[Candidate], labels: dict[str, dict[str, Any]]) -> int:
    for i, c in enumerate(candidates):
        if c.src not in labels:
            return i
    return max(0, len(candidates) - 1)


def write_manifest(out_dir: Path, candidates: list[Candidate], labels: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    good_count = sum(1 for row in labels.values() if row.get("is_good"))
    bad_count = sum(1 for row in labels.values() if not row.get("is_good"))
    manifest = {
        "schema": "manual_floorplan_annotation/v1",
        "input": args.input,
        "results_csv": args.results_csv,
        "out": str(out_dir),
        "mode": args.mode,
        "candidate_count": len(candidates),
        "labeled_count": len(labels),
        "good_count": good_count,
        "bad_count": bad_count,
        "remaining_count": max(0, len(candidates) - len(labels)),
        "labels_jsonl": str(out_dir / "labels.jsonl"),
        "labels_csv": str(out_dir / "labels.csv"),
        "good_dir": str(out_dir / "good"),
        "bad_dir": str(out_dir / "bad"),
    }
    (out_dir / "annotation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual annotation of floorplan candidates.")
    parser.add_argument("--input", help="Input directory with ranked images.")
    parser.add_argument("--results-csv", help="Ranker results.csv. If provided, order is taken from score.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-hard-rejects", action="store_true")
    parser.add_argument("--max-display-width", type=int, default=1400)
    parser.add_argument("--max-display-height", type=int, default=850)
    parser.add_argument("--window-name", default="floorplan annotation")
    parser.add_argument("--reset", action="store_true", help="Delete previous labels in out directory.")
    args = parser.parse_args()

    if not args.input and not args.results_csv:
        raise SystemExit("Need --input or --results-csv")

    out_dir = Path(args.out).expanduser().resolve()
    ensure_dir(out_dir)
    labels_path = out_dir / "labels.jsonl"
    labels_csv_path = out_dir / "labels.csv"

    if args.reset:
        if labels_path.exists():
            labels_path.unlink()
        if labels_csv_path.exists():
            labels_csv_path.unlink()
        for sub in ["good", "bad"]:
            d = out_dir / sub
            if d.exists():
                shutil.rmtree(d)

    if args.results_csv:
        candidates = load_candidates_from_csv(Path(args.results_csv).expanduser().resolve(), include_hard_rejects=args.include_hard_rejects)
    else:
        candidates = load_candidates_from_dir(Path(args.input).expanduser().resolve())

    if args.limit is not None:
        candidates = candidates[: args.limit]
        for i, c in enumerate(candidates, start=1):
            c.index = i

    if not candidates:
        raise SystemExit("No candidates found")

    labels = read_existing_labels(labels_path)
    current = find_start_index(candidates, labels)

    print("candidates:", len(candidates))
    print("already labeled:", len(labels))
    print("start index:", current + 1)
    print_help()

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        while 0 <= current < len(candidates):
            c = candidates[current]
            labeled = labels.get(c.src)
            label_text = labeled["label"] if labeled else "unlabeled"
            good_count = sum(1 for row in labels.values() if row.get("is_good"))
            bad_count = sum(1 for row in labels.values() if not row.get("is_good"))
            header = [
                f"{current + 1}/{len(candidates)} | score={c.score:+.4f} | rank={c.rank} | label={label_text}",
                f"good={good_count} bad={bad_count} remaining={len(candidates)-len(labels)}",
                f"{c.name}",
                "SPACE/ENTER/g=good | any other/n=bad | b=prev | u=unlabel | q=quit",
            ]
            img = load_display_image(Path(c.src), args.max_display_width, args.max_display_height, header)
            cv2.imshow(args.window_name, img)
            key = cv2.waitKey(0)
            key_low = key & 0xFF

            if key_low == ord("q"):
                break
            if key_low == ord("?"):
                print_help()
                continue
            if key_low in {ord("b"), 8, 127}:
                current = max(0, current - 1)
                continue
            if key_low == ord("u"):
                if c.src in labels:
                    del labels[c.src]
                    rewrite_labels(labels_path, labels, candidates)
                    write_labels_csv(labels_csv_path, labels, candidates)
                    write_manifest(out_dir, candidates, labels, args)
                    print(f"unlabeled: {current + 1}/{len(candidates)} {c.name}")
                continue

            if key_low in {32, 13, 10, ord("g")}:
                labels[c.src] = label_row(c, "good")
                print(f"GOOD {current + 1}/{len(candidates)} score={c.score:+.4f} {c.name}")
            else:
                labels[c.src] = label_row(c, "bad")
                print(f"BAD  {current + 1}/{len(candidates)} score={c.score:+.4f} {c.name}")

            rewrite_labels(labels_path, labels, candidates)
            write_labels_csv(labels_csv_path, labels, candidates)
            write_manifest(out_dir, candidates, labels, args)
            current += 1

    finally:
        cv2.destroyAllWindows()
        rewrite_labels(labels_path, labels, candidates)
        write_labels_csv(labels_csv_path, labels, candidates)
        write_manifest(out_dir, candidates, labels, args)
        rebuild_label_dirs(out_dir, labels, candidates, args.mode)

    print("saved:", out_dir)
    print("labels:", labels_path)
    print("csv:", labels_csv_path)
    print("good:", out_dir / "good")
    print("bad:", out_dir / "bad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
