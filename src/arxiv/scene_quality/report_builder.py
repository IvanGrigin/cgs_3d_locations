from __future__ import annotations

import json
from pathlib import Path


def _safe_load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_scene_report(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    prompt = (root / "prompt.txt").read_text(encoding="utf-8") if (root / "prompt.txt").is_file() else ""
    normalized_intent = _safe_load_json(root / "intent.normalized.json")
    compiled_policy = _safe_load_json(root / "compiled_policy.active.json")
    run_status = _safe_load_json(root / "run_status.json")
    screening_dir = root / "screening"
    candidate_rows: list[str] = []
    if screening_dir.is_dir():
        for candidate in sorted(screening_dir.iterdir()):
            if not candidate.is_dir():
                continue
            gate = _safe_load_json(candidate / "rule_gate.json")
            judge = _safe_load_json(candidate / "judge.json")
            candidate_rows.append(
                "| {name} | {rule:.2f} | {judge_score:.2f} | {passed} | {hard} |".format(
                    name=candidate.name,
                    rule=float(gate.get("rule_score", 0.0)),
                    judge_score=float(judge.get("total_score", 0.0)),
                    passed="yes" if gate.get("passed") else "no",
                    hard=", ".join(gate.get("hard_failures") or []),
                )
            )
    final_dir = root / "final"
    selected_name = (final_dir / "selected_candidate.txt").read_text(encoding="utf-8").strip() if (final_dir / "selected_candidate.txt").is_file() else ""
    report_lines = [
        "# Prompt Scene Report",
        "",
        "## Prompt",
        "",
        prompt,
        "",
        "## Normalized Intent",
        "",
        "```json",
        json.dumps(normalized_intent, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Compiled Policy Summary",
        "",
        "```json",
        json.dumps(
            {
                "room_type": (((compiled_policy.get("geometry") or {}).get("room_type"))),
                "area_sqm": (((compiled_policy.get("geometry") or {}).get("area_sqm"))),
                "area_bucket": (((compiled_policy.get("geometry") or {}).get("area_bucket"))),
                "style_label": (((compiled_policy.get("style_policy") or {}).get("style_label"))),
                "required_semantics": (((compiled_policy.get("program") or {}).get("required_semantics"))),
                "factory_whitelist": (((compiled_policy.get("program") or {}).get("factory_whitelist"))),
                "factory_blacklist": (((compiled_policy.get("program") or {}).get("factory_blacklist"))),
                "max_counts": (((compiled_policy.get("program") or {}).get("max_counts"))),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Screening Candidates",
        "",
        "| Candidate | Rule | Judge | Rule Pass | Hard Failures |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    report_lines.extend(candidate_rows or ["| none | 0.00 | 0.00 | no | no candidates |"])
    report_lines.extend(
        [
        "",
        "## Final Selection",
        "",
        f"Run status: `{run_status.get('status') or 'unknown'}`",
        "",
        f"Selected candidate: `{selected_name or 'none'}`",
        "",
        "Exact file paths:",
            f"- blend: `{(final_dir / 'scene.blend').resolve()}`",
            f"- render: `{(final_dir / 'render.png').resolve()}`",
            f"- placement: `{(final_dir / 'placement.json').resolve()}`",
            f"- inventory_summary: `{(final_dir / 'inventory_summary.json').resolve()}`",
            f"- rule_gate: `{(final_dir / 'rule_gate.json').resolve()}`",
            f"- judge: `{(final_dir / 'judge.json').resolve()}`",
        ]
    )
    report_path = final_dir / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path
