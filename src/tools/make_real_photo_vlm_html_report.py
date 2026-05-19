#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a privacy-preserving HTML report for real-photo VLM evaluation runs."""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter


SCORE_KEYS = [
    "total_score",
    "prompt_match_score",
    "layout_score",
    "collision_score",
    "style_score",
    "asset_quality_score",
    "camera_coverage_score",
    "confidence",
]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return statistics.mean(values) if values else None


def make_blurred_image(src: Path, dst: Path, *, max_side: int = 760, pixel_size: int = 28, blur_radius: float = 18.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        small_w = max(1, img.width // max(1, pixel_size))
        small_h = max(1, img.height // max(1, pixel_size))
        obscured = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
        obscured = obscured.resize(img.size, Image.Resampling.NEAREST)
        obscured = obscured.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        obscured.save(dst, quality=82, optimize=True)


def find_photo_path(photo_name: str, photos_dir: Path, meta_path: Path | None) -> Path:
    if meta_path and meta_path.exists():
        meta = load_json(meta_path)
        photo = meta.get("photo") if isinstance(meta, dict) else None
        if photo and Path(photo).exists():
            return Path(photo)
    return photos_dir / photo_name


def enrich_rows(summary_rows: list[dict[str, Any]], photos_dir: Path, report_dir: Path) -> list[dict[str, Any]]:
    blurred_dir = report_dir / "blurred"
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        item = dict(row)
        output = Path(str(row.get("output") or ""))
        eval_data = load_json(output) if output.exists() else {}
        room_type = str(row.get("room_type") or "")
        stem = Path(str(row.get("photo") or "")).stem
        meta_path = output.with_name(f"{room_type}.meta.json") if output.exists() else None
        photo_path = find_photo_path(str(row.get("photo") or ""), photos_dir, meta_path)
        blurred_path = blurred_dir / f"{stem}.blurred.jpg"
        if photo_path.exists():
            make_blurred_image(photo_path, blurred_path)
            item["blurred_rel"] = os.path.relpath(blurred_path, report_dir)
        else:
            item["blurred_rel"] = ""
        item["eval"] = eval_data if isinstance(eval_data, dict) else {}
        item["meta"] = load_json(meta_path) if meta_path and meta_path.exists() else {}
        out.append(item)
    return out


def conclusion_text(rows: list[dict[str, Any]]) -> list[str]:
    overall = mean(rows, "total_score")
    pass_rate = sum(1 for row in rows if row.get("passed")) / len(rows) if rows else 0.0
    by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_room[str(row.get("room_type") or "unknown")].append(row)
    room_means = {room: mean(items, "total_score") for room, items in by_room.items()}
    valid = {room: score for room, score in room_means.items() if score is not None}
    best = max(valid.items(), key=lambda x: x[1]) if valid else None
    worst = min(valid.items(), key=lambda x: x[1]) if valid else None

    bullets = [
        f"Overall average total score: {fmt(overall)}; pass rate: {pass_rate * 100:.1f}% ({sum(1 for row in rows if row.get('passed'))}/{len(rows)}).",
    ]
    if best:
        bullets.append(f"Best room type by average total score: {best[0]} ({fmt(best[1])}).")
    if worst:
        bullets.append(f"Weakest room type by average total score: {worst[0]} ({fmt(worst[1])}).")
    if mean(rows, "collision_score") and (mean(rows, "collision_score") or 0) >= 9:
        bullets.append("Collision/physical plausibility scores are consistently high, so most penalties come from prompt match, layout, style, or camera coverage.")
    if worst and worst[1] < 6:
        bullets.append(f"{worst[0]} examples should be inspected first: their average score is below the practical pass threshold.")
    return bullets


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = ["photo", "room_type", "total", "layout", "prompt", "style", "asset", "camera", "confidence", "passed"]
    lines = ["<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"]
    for row in rows:
        passed = "yes" if row.get("passed") else "no"
        cls = "pass" if row.get("passed") else "fail"
        lines.append(
            "<tr>"
            f"<td>{esc(row.get('photo'))}</td>"
            f"<td>{esc(row.get('room_type'))}</td>"
            f"<td>{fmt(row.get('total_score'), 1)}</td>"
            f"<td>{fmt(row.get('layout_score'), 1)}</td>"
            f"<td>{fmt(row.get('prompt_match_score'), 1)}</td>"
            f"<td>{fmt(row.get('style_score'), 1)}</td>"
            f"<td>{fmt(row.get('asset_quality_score'), 1)}</td>"
            f"<td>{fmt(row.get('camera_coverage_score'), 1)}</td>"
            f"<td>{fmt(row.get('confidence'), 2)}</td>"
            f"<td class='{cls}'>{passed}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def render_averages(rows: list[dict[str, Any]]) -> str:
    by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_room[str(row.get("room_type") or "unknown")].append(row)
    lines = ["<table><thead><tr><th>room type</th><th>count</th><th>pass rate</th><th>total avg</th><th>layout avg</th><th>prompt avg</th><th>style avg</th></tr></thead><tbody>"]
    for room, items in sorted(by_room.items()):
        pass_rate = sum(1 for row in items if row.get("passed")) / len(items)
        lines.append(
            "<tr>"
            f"<td>{esc(room)}</td><td>{len(items)}</td><td>{pass_rate * 100:.1f}%</td>"
            f"<td>{fmt(mean(items, 'total_score'))}</td>"
            f"<td>{fmt(mean(items, 'layout_score'))}</td>"
            f"<td>{fmt(mean(items, 'prompt_match_score'))}</td>"
            f"<td>{fmt(mean(items, 'style_score'))}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def render_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "<p class='muted'>n/a</p>"
    return "<ul>" + "".join(f"<li>{esc(value)}</li>" for value in values) + "</ul>"


def render_cards(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        ev = row.get("eval") if isinstance(row.get("eval"), dict) else {}
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        img = f"<img src='{esc(row.get('blurred_rel'))}' alt='blurred {esc(row.get('photo'))}'>" if row.get("blurred_rel") else "<div class='missing'>missing image</div>"
        cards.append(
            "<section class='photo-card'>"
            f"<div class='photo'>{img}</div>"
            "<div class='details'>"
            f"<h3>{esc(row.get('photo'))} <span>{esc(row.get('room_type'))}</span></h3>"
            f"<p class='score'>total {fmt(row.get('total_score'), 1)} / layout {fmt(row.get('layout_score'), 1)} / prompt {fmt(row.get('prompt_match_score'), 1)} / confidence {fmt(row.get('confidence'), 2)}</p>"
            f"<p><b>Prompt:</b> {esc(meta.get('prompt'))}</p>"
            f"<p><b>VLM notes:</b> {esc(ev.get('notes'))}</p>"
            "<div class='cols'>"
            "<div><h4>Strengths</h4>" + render_list(ev.get("strengths")) + "</div>"
            "<div><h4>Weaknesses</h4>" + render_list(ev.get("weaknesses")) + "</div>"
            "</div>"
            "<h4>Recommended fixes</h4>"
            + render_list(ev.get("recommended_fixes"))
            + "</div></section>"
        )
    return "\n".join(cards)


def build_html(rows: list[dict[str, Any]], title: str) -> str:
    conclusions = "".join(f"<li>{esc(text)}</li>" for text in conclusion_text(rows))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f6f7f9; }}
    header {{ padding: 28px 36px; background: #ffffff; border-bottom: 1px solid #dde2e8; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px 32px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 34px 0 14px; font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dde2e8; }}
    th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid #e7ebf0; font-size: 14px; }}
    th {{ background: #eef2f6; font-weight: 650; }}
    .pass {{ color: #0b7a3b; font-weight: 650; }}
    .fail {{ color: #b42318; font-weight: 650; }}
    .photo-card {{ display: grid; grid-template-columns: minmax(260px, 430px) 1fr; gap: 22px; background: #fff; border: 1px solid #dde2e8; margin: 18px 0; padding: 18px; }}
    .photo img {{ width: 100%; max-height: 360px; object-fit: contain; background: #e9edf2; }}
    .details h3 {{ margin: 0 0 8px; font-size: 18px; }}
    .details h3 span {{ color: #5d6b7a; font-weight: 500; margin-left: 8px; }}
    .score {{ font-weight: 650; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    h4 {{ margin-bottom: 6px; }}
    ul {{ margin-top: 6px; padding-left: 20px; }}
    .muted {{ color: #667587; }}
    @media (max-width: 860px) {{ .photo-card, .cols {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} }}
  </style>
</head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <div class="muted">Privacy-preserving report: source photos are shown only as heavily blurred thumbnails.</div>
</header>
<main>
  <h2>Summary Table</h2>
  {render_table(rows)}
  <h2>Averages</h2>
  {render_averages(rows)}
  <h2>Conclusions</h2>
  <ul>{conclusions}</ul>
  <h2>Photo Reviews</h2>
  {render_cards(rows)}
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HTML report with blurred photos for real-photo VLM eval.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--photos-dir", default="data/input/real_photos")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    photos_dir = Path(args.photos_dir).resolve()
    report_dir = run_dir / "html_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = load_json(run_dir / "real_photo_vlm_eval_summary.json")
    if not isinstance(rows, list):
        raise RuntimeError("summary JSON must be a list")
    enriched = enrich_rows(rows, photos_dir, report_dir)
    html_text = build_html(enriched, title=f"Real Photo VLM Evaluation - {run_dir.name}")
    out_path = Path(args.out).resolve() if args.out else report_dir / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
