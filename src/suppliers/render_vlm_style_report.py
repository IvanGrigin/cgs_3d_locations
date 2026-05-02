#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Render an offline HTML report for VLM supplier style enrichment results."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_top_samples(path: Path) -> list[dict[str, Any]]:
    json_path = path / "top30.json"
    if not json_path.is_file():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def rel_image_path(sample: dict[str, Any], top_samples_dir: Path, out_html: Path) -> str:
    raw = str(sample.get("image_path") or "")
    name = Path(raw).name
    local = top_samples_dir / "images" / name
    if not local.is_file():
        local = Path(raw)
    try:
        return local.resolve().relative_to(out_html.parent.resolve()).as_posix()
    except Exception:
        return local.resolve().as_posix()


def label_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value or "")


def compact_style(style: dict[str, Any]) -> str:
    fields = [
        ("object", style.get("object_category")),
        ("room", label_list(style.get("room_type"))),
        ("style", label_list(style.get("interior_style"))),
        ("form", label_list(style.get("form_style"))),
        ("finish", label_list(style.get("finish"))),
        ("color", label_list(style.get("color_family"))),
        ("hardware", style.get("hardware_style")),
        ("install", style.get("installation_type")),
        ("appliance", style.get("appliance_style")),
        ("conf", style.get("confidence")),
    ]
    return "<br>".join(f"<b>{esc(k)}</b>: {esc(v)}" for k, v in fields if v not in (None, "", []))


def compact_validation(validation: dict[str, Any]) -> str:
    if not validation:
        return ""
    scores = [
        ("overall", validation.get("overall_score")),
        ("detail", validation.get("detail_score")),
        ("color", validation.get("color_score")),
        ("category", validation.get("category_score")),
        ("style", validation.get("style_score")),
    ]
    issues = validation.get("issues") or []
    parts = [
        f"<b>matches</b>: {esc(validation.get('matches_image'))}",
        " ".join(f"<b>{esc(k)}</b>: {esc(v)}" for k, v in scores),
    ]
    if issues:
        parts.append("<b>issues</b>: " + esc("; ".join(str(x) for x in issues)))
    if validation.get("corrected_summary"):
        parts.append("<b>summary</b>: " + esc(validation.get("corrected_summary")))
    if validation.get("rationale"):
        parts.append("<b>rationale</b>: " + esc(validation.get("rationale")))
    return "<br>".join(parts)


def compact_description(description: dict[str, Any]) -> str:
    if not description:
        return ""
    fields = [
        ("summary", description.get("object_summary")),
        ("objects", label_list(description.get("visible_objects"))),
        ("materials", label_list(description.get("materials_visible"))),
        ("finish", label_list(description.get("finish_visible"))),
        ("colors", label_list(description.get("colors_visible"))),
        ("forms", label_list(description.get("forms_visible"))),
        ("hardware", description.get("hardware_visible")),
        ("support", description.get("installation_or_support_visible")),
        ("context", description.get("room_context_visible")),
        ("uncertainty", description.get("uncertainty")),
    ]
    parts = [f"<b>{esc(k)}</b>: {esc(v)}" for k, v in fields if v not in (None, "", [])]
    if description.get("detailed_description"):
        parts.append("<b>description</b>: " + esc(description.get("detailed_description")))
    return "<br>".join(parts)


def render_report(rows: list[dict[str, Any]], samples: list[dict[str, Any]], top_samples_dir: Path, out_html: Path, limit: int) -> str:
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    shown_rows = rows[:limit] if limit > 0 else rows

    cards: list[str] = []
    for sample in samples:
        style = sample.get("vlm_style") if isinstance(sample.get("vlm_style"), dict) else {}
        validation = sample.get("vlm_validation") if isinstance(sample.get("vlm_validation"), dict) else {}
        description = sample.get("vlm_description") if isinstance(sample.get("vlm_description"), dict) else {}
        img = rel_image_path(sample, top_samples_dir, out_html)
        left = compact_description(description) or compact_style(style)
        right = compact_validation(validation)
        cards.append(
            f"""
            <article class="card">
              <img src="{esc(img)}" alt="">
              <div class="body">
                <h3>{esc(sample.get("id"))}. {esc(sample.get("title"))}</h3>
                <p class="meta">rank={esc(sample.get("rank_score"))} source={esc(sample.get("source_site"))}</p>
                <div class="grid">
                  <div>{left}</div>
                  <div>{right}</div>
                </div>
              </div>
            </article>
            """
        )

    table_rows: list[str] = []
    for row in shown_rows:
        style = row.get("vlm_style") if isinstance(row.get("vlm_style"), dict) else {}
        validation = row.get("vlm_validation") if isinstance(row.get("vlm_validation"), dict) else {}
        description = row.get("vlm_description") if isinstance(row.get("vlm_description"), dict) else {}
        raw = "\n\n".join(x for x in [str(row.get("raw_vlm_text") or ""), str(row.get("raw_validation_text") or "")] if x)
        table_rows.append(
            f"""
            <tr>
              <td>{esc(row.get("id"))}</td>
              <td>{esc(row.get("status"))}<br><span class="err">{esc(row.get("error"))}</span></td>
              <td>{esc(row.get("title"))}</td>
              <td>{compact_description(description) or compact_style(style)}</td>
              <td>{compact_validation(validation)}</td>
              <td><details><summary>raw</summary><pre>{esc(raw)}</pre></details></td>
            </tr>
            """
        )

    error_items = "".join(
        f"<li><b>{esc(row.get('id'))}</b> {esc(row.get('title'))}: {esc(row.get('error'))}</li>"
        for row in error_rows[:100]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VLM Style Report</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 28px; }}
    .stat {{ border: 1px solid #d7dee8; border-radius: 8px; padding: 10px 12px; background: #f8fafc; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #d7dee8; border-radius: 8px; overflow: hidden; background: white; }}
    .card img {{ width: 100%; height: 260px; object-fit: contain; background: #f3f5f7; display: block; }}
    .body {{ padding: 12px; }}
    .meta, .err {{ color: #6b7280; font-size: 12px; }}
    .err {{ color: #b42318; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 13px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 12px; }}
    th, td {{ border: 1px solid #d7dee8; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f6; position: sticky; top: 0; }}
    pre {{ white-space: pre-wrap; max-width: 620px; max-height: 360px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>VLM Style Report</h1>
  <div class="stats">
    <div class="stat"><b>Total</b><br>{len(rows)}</div>
    <div class="stat"><b>OK</b><br>{len(ok_rows)}</div>
    <div class="stat"><b>Errors</b><br>{len(error_rows)}</div>
    <div class="stat"><b>Status</b><br>{esc(dict(status_counts))}</div>
  </div>
  <h2>Top Samples With Images</h2>
  <section class="cards">{"".join(cards)}</section>
  <h2>Errors</h2>
  <ul>{error_items}</ul>
  <h2>Rows</h2>
  <table>
    <thead><tr><th>ID</th><th>Status</th><th>Title</th><th>VLM description/style</th><th>Validation</th><th>Raw</th></tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--top-samples-dir", required=True)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    jsonl = Path(args.jsonl).expanduser()
    top_samples_dir = Path(args.top_samples_dir).expanduser()
    out_html = Path(args.out_html).expanduser()
    rows = load_jsonl(jsonl)
    samples = load_top_samples(top_samples_dir)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_report(rows, samples, top_samples_dir, out_html, args.limit), encoding="utf-8")
    print(f"Wrote {out_html} rows={len(rows)} top_samples={len(samples)}")


if __name__ == "__main__":
    main()
