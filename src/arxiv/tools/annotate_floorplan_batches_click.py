#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_floorplan_batches_click.py

Пакетная ручная разметка кандидатов планировок через клики мышью.

Логика
======
Показывается батч 3 x 4 = 12 изображений.

По умолчанию все картинки в батче считаются GOOD.
Если кликнуть по картинке:
    она помечается как BAD и подсвечивается красной рамкой 4 px.
Если кликнуть по ней второй раз:
    BAD снимается, картинка снова GOOD.

Переходы
========
SPACE / ENTER / ArrowRight / d / n
    сохранить текущий батч и перейти вперёд.

ArrowLeft / a / p / Backspace
    сохранить текущий батч и перейти назад.

q / ESC
    сохранить и выйти.

r
    сбросить выделения BAD в текущем батче: все GOOD.

x
    пометить все 12 как BAD.

Цель
====
Быстро разметить ~1400-1600 crop-кандидатов:
    кликами отмечаешь только плохие,
    Enter/Space — следующий батч.

Вход
====
Лучше использовать results.csv ранжировщика, чтобы порядок был по убыванию score:

python3 src/tools/annotate_floorplan_batches_click.py \
  --results-csv data/housesru/floorplans_ranked_v2/results.csv \
  --out data/housesru/floorplans_manual_click_v1 \
  --mode symlink \
  --limit 1600

Если надо размечать уже готовую папку ranked_all/top_k:

python3 src/tools/annotate_floorplan_batches_click.py \
  --input data/housesru/floorplans_ranked_v2/ranked_all \
  --out data/housesru/floorplans_manual_click_v1 \
  --mode symlink \
  --limit 1600

Выход
=====
out/
  labels.jsonl
  labels.csv
  annotation_manifest.json
  good/
  bad/

Можно прерываться через q/ESC и продолжать той же командой.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
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


@dataclass
class CellBox:
    position: int
    x0: int
    y0: int
    x1: int
    y1: int


class ClickState:
    def __init__(self) -> None:
        self.bad_positions: set[int] = set()
        self.cell_boxes: list[CellBox] = []
        self.dirty: bool = True
        self.action: Optional[str] = None


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
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "ok"}


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
        if value:
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

    fieldnames = [
        "src",
        "name",
        "label",
        "is_good",
        "score",
        "rank",
        "candidate_index",
        "batch_index",
        "position_in_batch",
        "hard_reject",
        "annotated_at_unix",
    ]

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


def write_manifest(out_dir: Path, candidates: list[Candidate], labels: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    good_count = sum(1 for row in labels.values() if row.get("is_good"))
    bad_count = sum(1 for row in labels.values() if not row.get("is_good"))

    manifest = {
        "schema": "manual_floorplan_click_batch_annotation/v1",
        "input": args.input,
        "results_csv": args.results_csv,
        "out": str(out_dir),
        "mode": args.mode,
        "batch_size": args.rows * args.cols,
        "rows": args.rows,
        "cols": args.cols,
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

    (out_dir / "annotation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def label_row(candidate: Candidate, label: str, batch_index: int, position_in_batch: int) -> dict[str, Any]:
    is_good = label == "good"
    return {
        "src": candidate.src,
        "name": candidate.name,
        "label": label,
        "is_good": is_good,
        "score": candidate.score,
        "rank": candidate.rank,
        "candidate_index": candidate.index,
        "batch_index": batch_index,
        "position_in_batch": position_in_batch,
        "hard_reject": candidate.hard_reject,
        "annotated_at_unix": time.time(),
        "meta": candidate.meta,
    }


def chunk_candidates(candidates: list[Candidate], batch_size: int) -> list[list[Candidate]]:
    return [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]


def find_start_batch(batches: list[list[Candidate]], labels: dict[str, dict[str, Any]]) -> int:
    for i, batch in enumerate(batches):
        if any(c.src not in labels for c in batch):
            return i
    return max(0, len(batches) - 1)


def save_all(out_dir: Path, labels: dict[str, dict[str, Any]], candidates: list[Candidate], args: argparse.Namespace) -> None:
    rewrite_labels(out_dir / "labels.jsonl", labels, candidates)
    write_labels_csv(out_dir / "labels.csv", labels, candidates)
    write_manifest(out_dir, candidates, labels, args)


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.putText(
            img,
            "cannot read",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def fit_to_cell(img: np.ndarray, cell_w: int, cell_h: int, pad: int = 10) -> np.ndarray:
    max_w = max(1, cell_w - 2 * pad)
    max_h = max(1, cell_h - 2 * pad)

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
    x0 = (cell_w - new_w) // 2
    y0 = (cell_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_label_box(cell: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, 36), color, thickness=-1)
    cv2.putText(
        cell,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_batch_canvas(
    batch: list[Candidate],
    labels: dict[str, dict[str, Any]],
    state: ClickState,
    *,
    batch_index: int,
    batch_count: int,
    rows: int,
    cols: int,
    cell_w: int,
    cell_h: int,
) -> np.ndarray:
    header_h = 98
    grid_h = rows * cell_h
    grid_w = cols * cell_w
    canvas = np.full((header_h + grid_h, grid_w, 3), 238, dtype=np.uint8)

    good_count = sum(1 for row in labels.values() if row.get("is_good"))
    bad_count = sum(1 for row in labels.values() if not row.get("is_good"))

    header_lines = [
        f"Batch {batch_index + 1}/{batch_count} | images {batch[0].index}-{batch[-1].index}",
        "Click image: toggle BAD red border | SPACE/ENTER/Right: next | Left/Backspace: prev | r: reset | x: all bad | q/ESC: quit",
        f"labeled={len(labels)} good={good_count} bad={bad_count} | current bad in batch={len(state.bad_positions)}",
    ]

    y = 26
    for line in header_lines:
        cv2.putText(
            canvas,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 26

    state.cell_boxes = []

    for pos in range(rows * cols):
        r = pos // cols
        c = pos % cols
        x0 = c * cell_w
        y0 = header_h + r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        cell = np.full((cell_h, cell_w, 3), 250, dtype=np.uint8)

        if pos < len(batch):
            cand = batch[pos]
            img = imread_unicode(Path(cand.src))
            cell_img = fit_to_cell(img, cell_w=cell_w, cell_h=cell_h, pad=14)
            cell[:] = cell_img

            position = pos + 1
            is_bad = position in state.bad_positions

            color = (30, 30, 180) if is_bad else (45, 120, 45)
            state_text = "BAD" if is_bad else "GOOD"
            title = f"{position} | #{cand.index} | {cand.score:+.2f} | {state_text}"
            draw_label_box(cell, title, color)

            short_name = cand.name
            if len(short_name) > 58:
                short_name = short_name[:55] + "..."

            cv2.rectangle(cell, (0, cell_h - 34), (cell_w - 1, cell_h - 1), (255, 255, 255), thickness=-1)
            cv2.putText(
                cell,
                short_name,
                (8, cell_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

            # Основная рамка.
            cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (80, 80, 80), 1)

            # Красная рамка 4 px для плохих.
            if is_bad:
                for t in range(4):
                    cv2.rectangle(
                        cell,
                        (t, t),
                        (cell_w - 1 - t, cell_h - 1 - t),
                        (0, 0, 255),
                        1,
                    )

            state.cell_boxes.append(CellBox(position=position, x0=x0, y0=y0, x1=x1, y1=y1))
        else:
            cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (160, 160, 160), 1)

        canvas[y0:y1, x0:x1] = cell

    return canvas


def apply_batch_labels(
    batch: list[Candidate],
    labels: dict[str, dict[str, Any]],
    state: ClickState,
    *,
    batch_index: int,
) -> None:
    for pos, cand in enumerate(batch, start=1):
        label = "bad" if pos in state.bad_positions else "good"
        labels[cand.src] = label_row(
            cand,
            label,
            batch_index=batch_index + 1,
            position_in_batch=pos,
        )


def load_batch_state_from_labels(batch: list[Candidate], labels: dict[str, dict[str, Any]]) -> ClickState:
    state = ClickState()

    # При первом показе неразмеченного батча все считаются хорошими.
    # Если батч уже был размечен, восстанавливаем BAD.
    for pos, cand in enumerate(batch, start=1):
        row = labels.get(cand.src)
        if row is not None and not row.get("is_good", False):
            state.bad_positions.add(pos)

    return state


def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
    state: ClickState = userdata

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for box in state.cell_boxes:
        if box.x0 <= x < box.x1 and box.y0 <= y < box.y1:
            if box.position in state.bad_positions:
                state.bad_positions.remove(box.position)
            else:
                state.bad_positions.add(box.position)
            state.dirty = True
            return


def print_help() -> None:
    print()
    print("Управление:")
    print("  click по картинке         -> toggle BAD / GOOD")
    print("  BAD = красная рамка 4 px")
    print("  SPACE / ENTER / Right / d -> сохранить батч и вперёд")
    print("  Left / a / p / Backspace  -> сохранить батч и назад")
    print("  r                         -> все GOOD в текущем батче")
    print("  x                         -> все BAD в текущем батче")
    print("  q / ESC                   -> сохранить и выйти")
    print()


def is_next_key(key: int) -> bool:
    low = key & 0xFF
    return low in {13, 10, 32, ord("d"), ord("n")} or key in {83, 65363, 2555904, 63235}


def is_prev_key(key: int) -> bool:
    low = key & 0xFF
    return low in {8, 127, ord("a"), ord("p"), ord("b")} or key in {81, 65361, 2424832, 63234}


def is_quit_key(key: int) -> bool:
    low = key & 0xFF
    return low in {27, ord("q")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Click-based batch annotation of floorplan candidates.")
    parser.add_argument("--input", help="Input directory with ranked images.")
    parser.add_argument("--results-csv", help="Ranker results.csv. If provided, order is taken from score.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-hard-rejects", action="store_true")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=360)
    parser.add_argument("--cell-height", type=int, default=260)
    parser.add_argument("--window-name", default="floorplan click annotation")
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

    batch_size = args.rows * args.cols
    batches = chunk_candidates(candidates, batch_size)

    labels = read_existing_labels(labels_path)
    current_batch = find_start_batch(batches, labels)

    print("candidates:", len(candidates))
    print("batch_size:", batch_size)
    print("batches:", len(batches))
    print("already labeled:", len(labels))
    print("start batch:", current_batch + 1)
    print_help()

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    try:
        while 0 <= current_batch < len(batches):
            batch = batches[current_batch]
            state = load_batch_state_from_labels(batch, labels)
            cv2.setMouseCallback(args.window_name, on_mouse, state)

            while True:
                if state.dirty:
                    canvas = make_batch_canvas(
                        batch,
                        labels,
                        state,
                        batch_index=current_batch,
                        batch_count=len(batches),
                        rows=args.rows,
                        cols=args.cols,
                        cell_w=args.cell_width,
                        cell_h=args.cell_height,
                    )
                    cv2.imshow(args.window_name, canvas)
                    state.dirty = False

                key = cv2.waitKey(40)

                if key == -1:
                    continue

                low = key & 0xFF

                if is_quit_key(key):
                    apply_batch_labels(batch, labels, state, batch_index=current_batch)
                    save_all(out_dir, labels, candidates, args)
                    current_batch = len(batches)
                    break

                if low == ord("?"):
                    print_help()
                    continue

                if low == ord("r"):
                    state.bad_positions.clear()
                    state.dirty = True
                    continue

                if low == ord("x"):
                    state.bad_positions = set(range(1, len(batch) + 1))
                    state.dirty = True
                    continue

                if is_next_key(key):
                    apply_batch_labels(batch, labels, state, batch_index=current_batch)
                    save_all(out_dir, labels, candidates, args)
                    print(
                        f"saved batch {current_batch + 1}/{len(batches)}: "
                        f"good={len(batch) - len(state.bad_positions)} bad={len(state.bad_positions)}"
                    )
                    current_batch += 1
                    break

                if is_prev_key(key):
                    apply_batch_labels(batch, labels, state, batch_index=current_batch)
                    save_all(out_dir, labels, candidates, args)
                    print(
                        f"saved batch {current_batch + 1}/{len(batches)} and moved back: "
                        f"good={len(batch) - len(state.bad_positions)} bad={len(state.bad_positions)}"
                    )
                    current_batch = max(0, current_batch - 1)
                    break

    finally:
        cv2.destroyAllWindows()
        save_all(out_dir, labels, candidates, args)
        print("rebuilding good/bad dirs...")
        rebuild_label_dirs(out_dir, labels, candidates, args.mode)

    print("saved:", out_dir)
    print("labels:", out_dir / "labels.jsonl")
    print("csv:", out_dir / "labels.csv")
    print("good:", out_dir / "good")
    print("bad:", out_dir / "bad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
