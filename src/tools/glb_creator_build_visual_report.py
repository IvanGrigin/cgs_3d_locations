#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build a visual report for GLB Creator batch outputs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def stable_hash(value: str, n: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def rel(path: str | Path, base: Path) -> str:
    p = Path(path).expanduser()
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except Exception:
        return str(Path(path).expanduser())


def copy_asset(path: str, out_dir: Path, prefix: str) -> str:
    if not path:
        return ""
    src = Path(path).expanduser()
    if not src.is_file():
        return path
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".png"
    dst = assets_dir / f"{prefix}_{stable_hash(str(src.resolve()))}{suffix}"
    if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return str(dst.resolve())


def image_md(path: str, alt: str, base: Path) -> str:
    if not path:
        return ""
    return f"![{alt}]({rel(path, base)})"


def image_html(path: str, alt: str, base: Path) -> str:
    if not path:
        return ""
    return f'<img src="{html.escape(rel(path, base))}" alt="{html.escape(alt)}">'


def load_summary(item: dict[str, Any]) -> dict[str, Any]:
    summary_json = item.get("summary_json")
    if summary_json and Path(summary_json).is_file():
        return read_json(summary_json)
    return item


def load_card(summary: dict[str, Any]) -> dict[str, Any]:
    job_dir = Path(str(summary.get("local_job_dir") or "")).expanduser()
    for name in ("card.with_generated_glb.json", "card.normalized.json"):
        path = job_dir / name
        if path.is_file():
            payload = read_json(path)
            if isinstance(payload, dict):
                return payload
    return {}


def render_paths(summary: dict[str, Any]) -> list[str]:
    manifest = summary.get("render_manifest") if isinstance(summary.get("render_manifest"), dict) else {}
    renders = manifest.get("renders") if isinstance(manifest.get("renders"), list) else []
    paths = [str(r.get("path") or "") for r in renders if isinstance(r, dict)]
    return paths[:4] + [""] * max(0, 4 - len(paths))


def build_markdown(report: dict[str, Any], out_dir: Path) -> str:
    lines = [
        "# GLB Creator Visual Report",
        "",
        f"- Batch report: `{report.get('batch_report_path', '')}`",
        f"- Items: `{len(report.get('rows', []))}`",
        "",
        "| # | title | category | score | img_reference | img_from_glb_1 | img_from_glb_2 | img_from_glb_3 | img_from_glb_4 |",
        "|---:|---|---|---:|---|---|---|---|---|",
    ]
    for idx, row in enumerate(report.get("rows", []), 1):
        renders = row.get("renders") or ["", "", "", ""]
        cells = [
            str(idx),
            str(row.get("title") or ""),
            str(row.get("category") or ""),
            str(row.get("score") or ""),
            image_md(str(row.get("reference") or ""), "img_reference", out_dir),
            image_md(str(renders[0] or ""), "img_from_glb_1", out_dir),
            image_md(str(renders[1] or ""), "img_from_glb_2", out_dir),
            image_md(str(renders[2] or ""), "img_from_glb_3", out_dir),
            image_md(str(renders[3] or ""), "img_from_glb_4", out_dir),
        ]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def build_html(report: dict[str, Any], out_dir: Path) -> str:
    rows = []
    for idx, row in enumerate(report.get("rows", []), 1):
        renders = row.get("renders") or ["", "", "", ""]
        cells = [
            str(idx),
            html.escape(str(row.get("title") or "")),
            html.escape(str(row.get("category") or "")),
            html.escape(str(row.get("score") or "")),
            image_html(str(row.get("reference") or ""), "img_reference", out_dir),
            image_html(str(renders[0] or ""), "img_from_glb_1", out_dir),
            image_html(str(renders[1] or ""), "img_from_glb_2", out_dir),
            image_html(str(renders[2] or ""), "img_from_glb_3", out_dir),
            image_html(str(renders[3] or ""), "img_from_glb_4", out_dir),
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GLB Creator Visual Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    img {{ width: 180px; max-height: 180px; object-fit: contain; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>GLB Creator Visual Report</h1>
  <p>Items: {len(report.get("rows", []))}</p>
  <table>
    <thead>
      <tr><th>#</th><th>title</th><th>category</th><th>score</th><th>img_reference</th><th>img_from_glb_1</th><th>img_from_glb_2</th><th>img_from_glb_3</th><th>img_from_glb_4</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build visual report from GLB Creator batch report.")
    ap.add_argument("--batch-report", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    batch_path = Path(args.batch_report).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = read_json(batch_path)
    rows = []
    for item in batch.get("items", []):
        if not isinstance(item, dict):
            continue
        summary = load_summary(item)
        card = load_card(summary)
        reference = copy_asset(str(summary.get("selected_image") or ""), out_dir, "img_reference")
        renders = [copy_asset(path, out_dir, f"img_from_glb_{idx}") for idx, path in enumerate(render_paths(summary), 1)]
        rows.append(
            {
                "unique_key": summary.get("unique_key") or item.get("unique_key"),
                "title": summary.get("title") or card.get("title"),
                "category": card.get("category_norm") or card.get("category") or card.get("category_raw"),
                "score": summary.get("similarity_score_1_to_10"),
                "reference": reference,
                "renders": renders,
                "asset_glb": summary.get("asset_glb"),
                "ok": bool(summary.get("ok")),
            }
        )
    report = {"schema": "glb_creator_visual_report/v1", "batch_report_path": str(batch_path.resolve()), "rows": rows}
    write_text(out_dir / "visual_report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "visual_report.md", build_markdown(report, out_dir))
    write_text(out_dir / "visual_report.html", build_html(report, out_dir))
    print(f"[out] {out_dir / 'visual_report.md'}")
    print(f"[out] {out_dir / 'visual_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
