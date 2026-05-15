#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_floorplan_batches.py

Пакетная ручная разметка кандидатов планировок.

Сценарий
========
Показывает изображения батчами 3 x 4 = 12 штук.
Номера в батче: 1..12.

Правила разметки
================
ENTER / пустая строка / пробел + ENTER
    все 12 картинок в текущем батче хорошие.

Номера через пробел, например:
    1 3 8
    эти номера плохие, остальные хорошие.

b / back / prev / left / ←
    вернуться к предыдущему батчу.

q / quit / exit
    сохранить прогресс и выйти.

bad all / all bad / none
    все 12 плохие.

good all / all good
    все 12 хорошие.

Важно
=====
Разметка сохраняется после каждого батча:
    labels.jsonl
    labels.csv
    annotation_manifest.json
    good/
    bad/

Можно безопасно прерывать через q и потом запускать ту же команду:
скрипт продолжит с первого неразмеченного батча.

Вход
====
Лучше подавать results.csv ранжировщика, чтобы порядок был по убыванию score:

python3 src/tools/annotate_floorplan_batches.py \
  --results-csv data/housesru/floorplans_ranked_v2/results.csv \
  --out data/housesru/floorplans_manual_batches_v1 \
  --mode symlink \
  --limit 1600

Если надо размечать уже готовую папку ranked_all/top_k:

python3 src/tools/annotate_floorplan_batches.py \
  --input data/housesru/floorplans_ranked_v2/ranked_all \
  --out data/housesru/floorplans_manual_batches_v1 \
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
        score = infer_score_from_name(path)
        rank = infer_rank_from_name(path)
        candidates.append(
            Candidate(
                index=i,
                src=str(path.resolve()),
                name=path.name,
                score=score,
                rank=rank,
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

        score = parse_float(row.get("floorplan_score"), 0.0)
        rank = parse_int(row.get("rank"), 10**9)

        candidates.append(
            Candidate(
                index=len(candidates) + 1,
                src=str(src_path.resolve()),
                name=src_path.name,
                score=score,
                rank=rank,
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

        if row.get("is_good"):
            dst = good_dir / dst_name
        else:
            dst = bad_dir / dst_name

        materialize(src_path, dst, mode=mode)


def write_manifest(out_dir: Path, candidates: list[Candidate], labels: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    good_count = sum(1 for row in labels.values() if row.get("is_good"))
    bad_count = sum(1 for row in labels.values() if not row.get("is_good"))

    manifest = {
        "schema": "manual_floorplan_batch_annotation/v1",
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


def fit_to_cell(img: np.ndarray, cell_w: int, cell_h: int, pad: int = 8) -> np.ndarray:
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


def draw_label_box(cell: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, 34), color, thickness=-1)
    cv2.putText(
        cell,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_batch_canvas(
    batch: list[Candidate],
    labels: dict[str, dict[str, Any]],
    *,
    batch_index: int,
    batch_count: int,
    rows: int,
    cols: int,
    cell_w: int,
    cell_h: int,
) -> np.ndarray:
    header_h = 92
    grid_h = rows * cell_h
    grid_w = cols * cell_w
    canvas = np.full((header_h + grid_h, grid_w, 3), 238, dtype=np.uint8)

    good_count = sum(1 for row in labels.values() if row.get("is_good"))
    bad_count = sum(1 for row in labels.values() if not row.get("is_good"))

    header_lines = [
        f"Batch {batch_index + 1}/{batch_count} | images {batch[0].index}-{batch[-1].index}",
        f"ENTER/SPACE: all good | numbers: bad positions, e.g. 1 4 9 | b/left: previous | q: quit",
        f"labeled={len(labels)} good={good_count} bad={bad_count}",
    ]

    y = 26
    for line in header_lines:
        cv2.putText(
            canvas,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 25

    for pos in range(rows * cols):
        r = pos // cols
        c = pos % cols
        x0 = c * cell_w
        y0 = header_h + r * cell_h

        cell = np.full((cell_h, cell_w, 3), 250, dtype=np.uint8)

        if pos < len(batch):
            cand = batch[pos]
            img = imread_unicode(Path(cand.src))
            cell_img = fit_to_cell(img, cell_w=cell_w, cell_h=cell_h, pad=14)
            cell[:] = cell_img

            label = labels.get(cand.src)
            if label is None:
                color = (35, 35, 35)
                state = "?"
            elif label.get("is_good"):
                color = (40, 130, 40)
                state = "GOOD"
            else:
                color = (40, 40, 180)
                state = "BAD"

            title = f"{pos + 1} | #{cand.index} | {cand.score:+.2f} | {state}"
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
                0.38,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), (80, 80, 80), 1)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell

    return canvas


def parse_user_command(text: str, batch_len: int) -> tuple[str, set[int]]:
    raw = text
    text = text.strip().lower()

    # Empty line and whitespace-only input: all good.
    if text == "":
        return "all_good", set()

    if text in {"q", "quit", "exit", "stop"}:
        return "quit", set()

    if text in {"b", "back", "prev", "previous", "left", "arrowleft", "назад", "\x1b[d", "\x1b[D".lower()}:
        return "back", set()

    if text in {"bad all", "all bad", "none", "0", "все плохие"}:
        return "all_bad", set(range(1, batch_len + 1))

    if text in {"good all", "all good", "all", "все хорошие"}:
        return "all_good", set()

    # Extract numbers from any string: "1 3 8", "1,3,8", "bad 1 3".
    nums = [int(x) for x in re.findall(r"\d+", text)]
    nums = [x for x in nums if 1 <= x <= batch_len]

    if nums:
        return "bad_numbers", set(nums)

    # Any other command means: current batch all bad.
    # This is safer for batch mode than silently accepting all good.
    return "all_bad", set(range(1, batch_len + 1))


def save_all(
    out_dir: Path,
    labels: dict[str, dict[str, Any]],
    candidates: list[Candidate],
    args: argparse.Namespace,
) -> None:
    rewrite_labels(out_dir / "labels.jsonl", labels, candidates)
    write_labels_csv(out_dir / "labels.csv", labels, candidates)
    write_manifest(out_dir, candidates, labels, args)


def print_help() -> None:
    print()
    print("Команды батча:")
    print("  Enter / пусто / пробел+Enter -> все картинки батча хорошие")
    print("  1 3 8                       -> номера 1, 3, 8 плохие, остальные хорошие")
    print("  b / back / prev / left       -> предыдущий батч")
    print("  q / quit                     -> сохранить и выйти")
    print("  bad all / all bad / none     -> все плохие")
    print("  good all / all good          -> все хорошие")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch annotation of floorplan candidates.")
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
    parser.add_argument("--window-name", default="floorplan batch annotation")
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

            canvas = make_batch_canvas(
                batch,
                labels,
                batch_index=current_batch,
                batch_count=len(batches),
                rows=args.rows,
                cols=args.cols,
                cell_w=args.cell_width,
                cell_h=args.cell_height,
            )

            cv2.imshow(args.window_name, canvas)
            cv2.waitKey(1)

            prompt = f"batch {current_batch + 1}/{len(batches)} bad nums or Enter=all good > "
            try:
                command_text = input(prompt)
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break

            action, bad_positions = parse_user_command(command_text, batch_len=len(batch))

            if action == "quit":
                break

            if action == "back":
                current_batch = max(0, current_batch - 1)
                continue

            if action in {"all_good", "bad_numbers", "all_bad"}:
                for pos, cand in enumerate(batch, start=1):
                    label = "bad" if pos in bad_positions else "good"
                    labels[cand.src] = label_row(
                        cand,
                        label,
                        batch_index=current_batch + 1,
                        position_in_batch=pos,
                    )

                save_all(out_dir, labels, candidates, args)

                good_in_batch = len(batch) - len(bad_positions)
                bad_in_batch = len(bad_positions)
                print(
                    f"saved batch {current_batch + 1}/{len(batches)}: "
                    f"good={good_in_batch} bad={bad_in_batch}"
                )

                current_batch += 1
                continue

            print("unknown command")
            print_help()

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
