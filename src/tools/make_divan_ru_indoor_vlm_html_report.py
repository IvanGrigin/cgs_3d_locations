#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build an offline HTML report for Divan.ru indoor VLM evaluation runs."""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


SCORE_KEYS = [
    "total_score",
    "prompt_match_score",
    "layout_score",
    "collision_score",
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


def find_image_path(row: dict[str, Any], images_dir: Path, meta: dict[str, Any]) -> Path:
    photo = meta.get("photo")
    if photo and Path(str(photo)).exists():
        return Path(str(photo))
    return images_dir / str(row.get("photo") or "")


def make_report_image(src: Path, dst: Path, *, max_side: int = 980, quality: int = 86) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        img.save(dst, format="JPEG", quality=int(quality), optimize=True)


def fallback_eval_path(row: dict[str, Any], run_dir: Path) -> Path:
    photo = str(row.get("photo") or "")
    return run_dir / Path(photo).stem / "eval.json"


def enrich_rows(
    summary_rows: list[dict[str, Any]],
    run_dir: Path,
    images_dir: Path,
    report_dir: Path,
    *,
    max_image_side: int,
) -> list[dict[str, Any]]:
    image_out_dir = report_dir / "images"
    enriched: list[dict[str, Any]] = []
    for row in summary_rows:
        item = dict(row)
        output = Path(str(row.get("output") or ""))
        if not output.exists():
            output = fallback_eval_path(row, run_dir)
        photo_dir = output.parent
        meta_path = photo_dir / "meta.json"
        payload_path = photo_dir / "payload.json"
        eval_data = load_json(output) if output.exists() else {}
        meta = load_json(meta_path) if meta_path.exists() else {}
        payload = load_json(payload_path) if payload_path.exists() else {}
        image_path = find_image_path(row, images_dir, meta if isinstance(meta, dict) else {})
        if image_path.exists():
            report_image = image_out_dir / f"{Path(str(row.get('photo') or image_path.name)).stem}.jpg"
            make_report_image(image_path, report_image, max_side=max_image_side)
            item["image_rel"] = os.path.relpath(report_image, report_dir)
            item["source_image"] = str(image_path)
        else:
            item["image_rel"] = ""
            item["source_image"] = ""
        item["eval"] = eval_data if isinstance(eval_data, dict) else {}
        item["meta"] = meta if isinstance(meta, dict) else {}
        item["payload"] = payload if isinstance(payload, dict) else {}
        enriched.append(item)
    return enriched


def pass_label(row: dict[str, Any]) -> str:
    return "yes" if row.get("passed") else "no"


def render_metric_cards(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    pass_rate = (passed / total * 100.0) if total else 0.0
    cards = [
        ("Rooms", str(total)),
        ("Passed", f"{passed}/{total} ({pass_rate:.1f}%)"),
        ("Avg total", fmt(mean(rows, "total_score"))),
        ("Avg layout", fmt(mean(rows, "layout_score"))),
        ("Avg prompt", fmt(mean(rows, "prompt_match_score"))),
        ("Avg camera", fmt(mean(rows, "camera_coverage_score"))),
    ]
    return "<div class='metrics'>" + "".join(
        f"<div class='metric'><span>{esc(label)}</span><b>{esc(value)}</b></div>" for label, value in cards
    ) + "</div>"


def render_table(rows: list[dict[str, Any]]) -> str:
    headers = ["photo", "category", "total", "prompt", "layout", "collision", "asset", "camera", "confidence", "passed"]
    lines = ["<table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>"]
    for row in rows:
        cls = "pass" if row.get("passed") else "fail"
        lines.append(
            "<tr>"
            f"<td>{esc(row.get('photo'))}</td>"
            f"<td>{esc(row.get('room_category_ru'))}</td>"
            f"<td>{fmt(row.get('total_score'), 1)}</td>"
            f"<td>{fmt(row.get('prompt_match_score'), 1)}</td>"
            f"<td>{fmt(row.get('layout_score'), 1)}</td>"
            f"<td>{fmt(row.get('collision_score'), 1)}</td>"
            f"<td>{fmt(row.get('asset_quality_score'), 1)}</td>"
            f"<td>{fmt(row.get('camera_coverage_score'), 1)}</td>"
            f"<td>{fmt(row.get('confidence'), 2)}</td>"
            f"<td class='{cls}'>{pass_label(row)}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def render_averages(rows: list[dict[str, Any]]) -> str:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("room_category_ru") or "unknown")].append(row)
    lines = [
        "<table><thead><tr><th>category</th><th>count</th><th>pass rate</th>"
        "<th>total avg</th><th>prompt avg</th><th>layout avg</th><th>asset avg</th><th>camera avg</th></tr></thead><tbody>"
    ]
    for category, items in sorted(by_category.items()):
        pass_rate = sum(1 for row in items if row.get("passed")) / len(items) * 100.0
        lines.append(
            "<tr>"
            f"<td>{esc(category)}</td>"
            f"<td>{len(items)}</td>"
            f"<td>{pass_rate:.1f}%</td>"
            f"<td>{fmt(mean(items, 'total_score'))}</td>"
            f"<td>{fmt(mean(items, 'prompt_match_score'))}</td>"
            f"<td>{fmt(mean(items, 'layout_score'))}</td>"
            f"<td>{fmt(mean(items, 'asset_quality_score'))}</td>"
            f"<td>{fmt(mean(items, 'camera_coverage_score'))}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def render_list(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "<p class='muted'>n/a</p>"
    return "<ul>" + "".join(f"<li>{esc(value)}</li>" for value in values) + "</ul>"


def render_card_scores(row: dict[str, Any]) -> str:
    return "".join(
        f"<div><span>{esc(key.replace('_score', '').replace('_', ' '))}</span><b>{fmt(row.get(key), 1)}</b></div>"
        for key in SCORE_KEYS
        if row.get(key) is not None
    )


def render_cards(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        ev = row.get("eval") if isinstance(row.get("eval"), dict) else {}
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        prompt = meta.get("prompt") or payload.get("original_prompt") or ""
        img = (
            f"<img src='{esc(row.get('image_rel'))}' alt='{esc(row.get('photo'))}'>"
            if row.get("image_rel")
            else "<div class='missing'>missing image</div>"
        )
        status_cls = "pass" if row.get("passed") else "fail"
        cards.append(
            "<section class='photo-card'>"
            f"<div class='photo'>{img}</div>"
            "<div class='details'>"
            f"<div class='title-row'><h3>{esc(row.get('photo'))}</h3><span class='{status_cls}'>{pass_label(row)}</span></div>"
            f"<p class='subtitle'>{esc(row.get('room_category_ru'))} | total {fmt(row.get('total_score'), 1)}</p>"
            f"<div class='score-grid'>{render_card_scores(row)}</div>"
            f"<p><b>Prompt:</b> {esc(prompt)}</p>"
            f"<p><b>VLM notes:</b> {esc(ev.get('notes'))}</p>"
            "<div class='cols'>"
            "<div><h4>Strengths</h4>" + render_list(ev.get("strengths")) + "</div>"
            "<div><h4>Weaknesses</h4>" + render_list(ev.get("weaknesses")) + "</div>"
            "</div>"
            "<div class='cols'>"
            "<div><h4>Visible problems</h4>" + render_list(ev.get("visible_problems")) + "</div>"
            "<div><h4>Recommended fixes</h4>" + render_list(ev.get("recommended_fixes")) + "</div>"
            "</div>"
            f"<p class='path'>source: {esc(row.get('source_image'))}</p>"
            "</div></section>"
        )
    return "\n".join(cards)


def conclusion_text(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No rows found."]
    total_avg = mean(rows, "total_score")
    pass_count = sum(1 for row in rows if row.get("passed"))
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("room_category_ru") or "unknown")].append(row)
    cat_scores = {
        category: score
        for category, items in by_category.items()
        if (score := mean(items, "total_score")) is not None
    }
    best = max(cat_scores.items(), key=lambda item: item[1]) if cat_scores else None
    worst = min(cat_scores.items(), key=lambda item: item[1]) if cat_scores else None
    bullets = [
        f"Overall average total score is {fmt(total_avg)} with {pass_count}/{len(rows)} passed.",
        f"Average collision score is {fmt(mean(rows, 'collision_score'))}, so most penalties are not physical-collision issues.",
    ]
    if best:
        bullets.append(f"Best category by total score: {best[0]} ({fmt(best[1])}).")
    if worst:
        bullets.append(f"Weakest category by total score: {worst[0]} ({fmt(worst[1])}).")
    return bullets


def build_html(rows: list[dict[str, Any]], title: str, source_dir: Path) -> str:
    conclusions = "".join(f"<li>{esc(text)}</li>" for text in conclusion_text(rows))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --text: #20262e;
      --muted: #687586;
      --line: #dfe5ec;
      --soft: #f5f7fa;
      --panel: #ffffff;
      --pass: #0b7a3b;
      --fail: #b42318;
      --accent: #375a7f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--text); background: var(--soft); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding: 28px 36px; background: var(--panel); border-bottom: 1px solid var(--line); }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px 32px 52px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 720; }}
    h2 {{ margin: 36px 0 14px; font-size: 20px; }}
    h3 {{ margin: 0; font-size: 18px; }}
    h4 {{ margin: 14px 0 6px; font-size: 14px; text-transform: uppercase; color: var(--muted); letter-spacing: 0; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid #e9edf2; font-size: 14px; vertical-align: top; }}
    th {{ background: #edf2f7; font-weight: 680; }}
    .muted {{ color: var(--muted); }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 12px; margin-top: 18px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); padding: 14px 16px; min-height: 82px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .metric b {{ display: block; font-size: 22px; color: var(--accent); }}
    .pass {{ color: var(--pass); font-weight: 700; }}
    .fail {{ color: var(--fail); font-weight: 700; }}
    .photo-card {{ display: grid; grid-template-columns: minmax(320px, 500px) 1fr; gap: 22px; background: var(--panel); border: 1px solid var(--line); margin: 18px 0; padding: 18px; }}
    .photo img {{ display: block; width: 100%; max-height: 430px; object-fit: contain; background: #e9edf2; border: 1px solid #e0e6ee; }}
    .title-row {{ display: flex; align-items: baseline; justify-content: space-between; gap: 14px; }}
    .subtitle {{ margin: 6px 0 14px; color: var(--muted); font-weight: 640; }}
    .score-grid {{ display: grid; grid-template-columns: repeat(4, minmax(100px, 1fr)); gap: 8px; margin: 10px 0 14px; }}
    .score-grid div {{ border: 1px solid #e4e9ef; background: #f9fafc; padding: 8px 10px; }}
    .score-grid span {{ display: block; color: var(--muted); font-size: 12px; }}
    .score-grid b {{ display: block; font-size: 18px; margin-top: 2px; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    ul {{ margin: 6px 0 0; padding-left: 20px; }}
    li {{ margin: 4px 0; }}
    .path {{ margin-top: 16px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .missing {{ min-height: 240px; display: grid; place-items: center; color: var(--muted); background: #e9edf2; }}
    @media (max-width: 980px) {{
      header {{ padding: 22px 18px; }}
      main {{ padding: 20px 18px 40px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .photo-card, .cols {{ grid-template-columns: 1fr; }}
      .score-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <div class="muted">Offline report generated from {esc(source_dir)}.</div>
</header>
<main>
  {render_metric_cards(rows)}
  <h2>Summary Table</h2>
  {render_table(rows)}
  <h2>Category Averages</h2>
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
    parser = argparse.ArgumentParser(description="Build HTML report for Divan.ru indoor VLM eval.")
    parser.add_argument("--run-dir", default="data/input/divan_ru_indoor/vlm_eval_20260516")
    parser.add_argument("--images-dir", default="data/input/divan_ru_indoor/images")
    parser.add_argument("--out", default="")
    parser.add_argument("--max-image-side", type=int, default=980)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise RuntimeError(f"run dir not found: {run_dir}")
    if not images_dir.is_dir():
        raise RuntimeError(f"images dir not found: {images_dir}")

    report_dir = run_dir / "html_report"
    out_path = Path(args.out).expanduser().resolve() if args.out else report_dir / "index.html"
    report_dir = out_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "divan_ru_indoor_vlm_summary.json"
    rows = load_json(summary_path)
    if not isinstance(rows, list):
        raise RuntimeError(f"summary JSON must be a list: {summary_path}")
    enriched = enrich_rows(rows, run_dir, images_dir, report_dir, max_image_side=int(args.max_image_side))
    html_text = build_html(enriched, title=f"Divan.ru Indoor VLM Evaluation - {run_dir.name}", source_dir=run_dir)
    out_path.write_text(html_text, encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
