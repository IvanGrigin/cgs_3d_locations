# src/ml/metrics/report_runs.py

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import glob

import matplotlib.pyplot as plt


METRIC_KEYS = ["RMSE_xz", "MAE_xz", "BoundaryViolRate", "CollisionPairRate"]


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def run_label(run: dict, fallback_name: str) -> str:
    model = run.get("model") or fallback_name
    pp = run.get("postprocess")
    if pp and pp != "none":
        return f"{model}+{pp}"
    return f"{model}"


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.6f}"


def md_table(rows: list[dict]) -> str:
    # сортировка: сначала collision, потом RMSE (оба меньше = лучше)
    rows = sorted(rows, key=lambda r: (r["metrics"].get("CollisionPairRate", 1e9),
                                      r["metrics"].get("RMSE_xz", 1e9)))

    header = "| Run | RMSE_xz | MAE_xz | BoundaryViolRate | CollisionPairRate |\n"
    sep = "|---|---:|---:|---:|---:|\n"
    lines = [header, sep]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['label']} | {fmt(m.get('RMSE_xz'))} | {fmt(m.get('MAE_xz'))} | "
            f"{fmt(m.get('BoundaryViolRate'))} | {fmt(m.get('CollisionPairRate'))} |\n"
        )
    return "".join(lines)


def scatter(rows: list[dict], out_png: Path) -> None:
    # scatter: x = collision (лучше левее), y = rmse (лучше ниже)
    xs, ys, labels = [], [], []
    for r in rows:
        m = r["metrics"]
        x = m.get("CollisionPairRate")
        y = m.get("RMSE_xz")
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        labels.append(r["label"])

    if not xs:
        print("[report_runs] No valid points to plot (need CollisionPairRate and RMSE_xz).")
        return

    plt.figure()
    plt.scatter(xs, ys)

    # подписи точек
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, f" {lab}", fontsize=9)

    plt.xlabel("CollisionPairRate (↓ лучше)")
    plt.ylabel("RMSE_xz (↓ лучше)")
    plt.title("Baselines: trade-off RMSE vs Collisions")
    plt.grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def expand_inputs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        s = s.strip()
        if not s:
            continue
        # поддержка glob-паттернов
        if any(ch in s for ch in ["*", "?", "[", "]"]):
            for g in glob.glob(s):
                out.append(Path(g))
        else:
            out.append(Path(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="List of run JSON files (supports glob).")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    paths = expand_inputs(args.inputs)

    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"[report_runs] WARN: missing file: {p}")
            continue
        run = load_json(p)
        metrics = run.get("metrics", {})
        row = {
            "path": str(p),
            "label": run_label(run, fallback_name=p.stem),
            "metrics": {k: metrics.get(k) for k in METRIC_KEYS},
        }
        rows.append(row)

    if not rows:
        raise SystemExit("[report_runs] ERROR: no runs loaded. Check --inputs paths.")

    out_md = Path(args.out_md)
    out_png = Path(args.out_png)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# Baseline report\n\n")
    report.append("## Runs\n\n")
    for r in rows:
        report.append(f"- `{r['label']}` from `{r['path']}`\n")
    report.append("\n## Metrics table\n\n")
    report.append(md_table(rows))
    report.append("\n## Plot\n\n")
    report.append(f"- Scatter saved to `{out_png}`\n")

    out_md.write_text("".join(report), encoding="utf-8")
    scatter(rows, out_png)

    print(f"[report_runs] Saved markdown: {out_md}")
    print(f"[report_runs] Saved plot    : {out_png}")


if __name__ == "__main__":
    main()