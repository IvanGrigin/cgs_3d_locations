#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/sourse/suppliers/supplier_catalog_one_table.json"
DEFAULT_OUT_JSONL = "data/sourse/suppliers/supplier_catalog_one_table.style_llm.jsonl"
DEFAULT_MERGED_JSON = "data/sourse/suppliers/supplier_catalog_one_table.with_style_llm.json"
DEFAULT_REPORT_JSON = "data/sourse/suppliers/supplier_catalog_one_table.style_llm.quality.json"
DEFAULT_REPORT_MD = "data/sourse/suppliers/supplier_catalog_one_table.style_llm.quality.md"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"

STYLE_DEFS: dict[str, str] = {
    "modern": "Modern / современный: простые формы, ровные линии, минимум декора, нейтральные цвета.",
    "contemporary": "Contemporary / контемпорари: актуальный современный стиль, мягче modern, может смешивать материалы.",
    "minimalism": "Minimalism / минимализм: максимально простые формы, отсутствие ручек, гладкие фасады, мало деталей.",
    "scandinavian": "Scandinavian / скандинавский: светлое дерево, белый/серый, простые формы, уют, натуральные ткани.",
    "japandi": "Japandi: смесь японского и скандинавского, низкая мебель, дерево, бежевые/серые тона.",
    "loft_industrial": "Loft / industrial: металл, тёмное дерево, бетон, грубые поверхности, чёрная фурнитура.",
    "high_tech": "High-tech: стекло, металл, глянец, подсветка, технологичный вид.",
    "eco_organic": "Eco / organic: натуральное дерево, ротанг, лен, округлые формы, природные оттенки.",
    "soft_minimalism": "Soft minimalism: минимализм, но с мягкими углами, тёплыми цветами и текстурами.",
    "mid_century_modern": "Mid-century modern: ножки под углом, дерево, простые геометрические формы, ретро 1950-1960-х.",
}

STYLE_LABELS = list(STYLE_DEFS.keys())


def load_catalog(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"schema": "supplier_catalog_style_llm_source/v1", "items": data}, [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        rows = data.get("items") or data.get("rows") or data.get("products") or []
        if not isinstance(rows, list):
            raise RuntimeError(f"Catalog object has no list items/rows/products: {path}")
        return data, [x for x in rows if isinstance(x, dict)]
    raise RuntimeError(f"Catalog must be a list or object: {path}")


def item_id(item: dict[str, Any], index: int) -> str:
    for key in ("unique_key", "id", "external_id", "source_url", "title"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row_{index}"


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        iid = str(obj.get("id") or "").strip()
        if iid:
            out[iid] = obj
    return out


def safe_json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def compact_context(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(item.get("title") or "")[:300],
        "description": str(item.get("description") or "")[:1800],
    }


def build_prompt(item: dict[str, Any]) -> str:
    style_lines = "\n".join(f"- {key}: {desc}" for key, desc in STYLE_DEFS.items())
    return (
        "/no_think\n"
        "Classify an interior catalog item into exactly one style from the allowed taxonomy. "
        "Use only the provided title and description text. Ignore any knowledge about brands, categories, prices, dimensions, sources, or previous style labels because they are not provided. "
        "Do not invent a new style.\n\n"
        "Allowed styles:\n"
        f"{style_lines}\n\n"
        "Return exactly one JSON object and nothing else. JSON schema:\n"
        "{"
        "\"style_llm\":\"modern\","
        "\"confidence\":0.0,"
        "\"secondary_styles\":[\"contemporary\"],"
        "\"quality_score\":1,"
        "\"quality_flags\":[\"ambiguous_metadata\"],"
        "\"evidence\":[\"short factual signal from catalog\"],"
        "\"rationale\":\"one short sentence\""
        "}\n\n"
        "Rules:\n"
        "- style_llm must be one of the allowed snake_case labels.\n"
        "- confidence is 0..1.\n"
        "- secondary_styles contains 0..3 allowed labels.\n"
        "- quality_score is 1..10 and estimates reliability of this classification from title and description text only.\n"
        "- The title may identify object type or explicit style words, but do not infer a confident style from a generic product title alone.\n"
        "- If the description mostly contains file formats, polygon counts, render engines, archive contents, dimensions, URLs, or generic visualization text, set confidence <= 0.45 and quality_score <= 4.\n"
        "- If the only style word is generic 'modern'/'современный' without materials, shape, color, or design details, do not set quality_score above 6.\n"
        "- If the item is not clearly furniture/interior decor/material/lighting from the description, set quality_score <= 3 and add the quality flag not_applicable_or_non_interior. This flag is not a style label.\n"
        "- Prefer modern over contemporary when the signal is strict simple lines/minimal decor; prefer contemporary only for softer current mixed-material design.\n"
        "- Use quality_flags from: strong_style_signal, weak_style_signal, ambiguous_metadata, description_too_technical, missing_design_details, generic_modern_signal, conflicting_signals, not_applicable_or_non_interior.\n"
        "- evidence should cite concise phrases present in the description, not prose.\n\n"
        "Title and description input:\n"
        f"{json.dumps(compact_context(item), ensure_ascii=False, indent=2)}"
    )


def post_ollama_chat(args: argparse.Namespace, prompt: str) -> str:
    payload = {
        "model": args.ollama_model,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": float(args.temperature),
            "num_ctx": int(args.num_ctx),
            "num_predict": int(args.num_predict),
        },
    }
    if args.ollama_format != "none":
        payload["format"] = args.ollama_format
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        args.ollama_url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=int(args.timeout)) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data.get("message") or {}
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"].strip()
    return str(data.get("response") or "").strip()


def parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        data = json.loads(raw, strict=False)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0), strict=False)
    if not isinstance(data, dict):
        raise RuntimeError("LLM response is not a JSON object")
    return data


def normalize_result(data: dict[str, Any], raw_text: str) -> dict[str, Any]:
    style = str(data.get("style_llm") or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "industrial": "loft_industrial",
        "loft": "loft_industrial",
        "eco": "eco_organic",
        "organic": "eco_organic",
        "minimalist": "minimalism",
        "midcentury_modern": "mid_century_modern",
        "mid_century": "mid_century_modern",
        "hightech": "high_tech",
    }
    style = aliases.get(style, style)
    if style not in STYLE_LABELS:
        style = "contemporary"

    def as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    confidence = max(0.0, min(as_float(data.get("confidence"), 0.0), 1.0))
    quality_score = int(round(max(1.0, min(as_float(data.get("quality_score"), 1.0), 10.0))))
    secondary_raw = data.get("secondary_styles")
    if isinstance(secondary_raw, str):
        secondary_raw = [x.strip() for x in re.split(r"[,;/]", secondary_raw) if x.strip()]
    secondary: list[str] = []
    if isinstance(secondary_raw, list):
        for value in secondary_raw:
            sv = str(value).strip().lower().replace("-", "_").replace(" ", "_")
            sv = aliases.get(sv, sv)
            if sv in STYLE_LABELS and sv != style and sv not in secondary:
                secondary.append(sv)
    flags = data.get("quality_flags")
    if isinstance(flags, str):
        flags = [x.strip() for x in re.split(r"[,;/]", flags) if x.strip()]
    if not isinstance(flags, list):
        flags = []
    evidence = data.get("evidence")
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    return {
        "style_llm": style,
        "style_llm_confidence": round(confidence, 4),
        "style_llm_secondary": secondary[:3],
        "style_llm_quality_score": quality_score,
        "style_llm_quality_flags": [str(x).strip() for x in flags if str(x).strip()][:6],
        "style_llm_evidence": [str(x).strip()[:180] for x in evidence if str(x).strip()][:5],
        "style_llm_rationale": str(data.get("rationale") or "").strip()[:500],
        "raw_llm_text": raw_text,
    }


def classify_one(args: argparse.Namespace, item: dict[str, Any]) -> tuple[dict[str, Any], str]:
    prompt = build_prompt(item)
    last_text = ""
    last_error: Exception | None = None
    for attempt in range(max(1, int(args.retries) + 1)):
        try:
            last_text = post_ollama_chat(args, prompt)
            parsed = parse_json_object(last_text)
            return normalize_result(parsed, last_text), last_text
        except Exception as exc:
            last_error = exc
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def merge_results(
    source_data: dict[str, Any],
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    out_path: Path,
    jsonl_path: Path,
) -> None:
    merged_items: list[dict[str, Any]] = []
    for idx, item in enumerate(rows):
        out = dict(item)
        result = results.get(item_id(item, idx))
        if result and result.get("status") == "ok":
            style_data = result.get("style_llm") or {}
            out.update({k: v for k, v in style_data.items() if k != "raw_llm_text"})
        merged_items.append(out)
    merged = dict(source_data)
    merged["items"] = merged_items
    meta = dict(merged.get("meta") or {})
    meta["style_llm_jsonl"] = str(jsonl_path)
    meta["style_llm_count"] = sum(1 for r in results.values() if r.get("status") == "ok")
    merged["meta"] = meta
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def build_quality_reports(results: dict[str, dict[str, Any]], report_json: Path, report_md: Path) -> None:
    ok = [r for r in results.values() if r.get("status") == "ok"]
    failed = [r for r in results.values() if r.get("status") != "ok"]
    style_counts = Counter((r.get("style_llm") or {}).get("style_llm") for r in ok)
    flags = Counter()
    quality_scores = []
    confidences = []
    for r in ok:
        s = r.get("style_llm") or {}
        quality_scores.append(float(s.get("style_llm_quality_score") or 0))
        confidences.append(float(s.get("style_llm_confidence") or 0))
        flags.update(s.get("style_llm_quality_flags") or [])
    summary = {
        "ok_count": len(ok),
        "failed_count": len(failed),
        "style_counts": dict(style_counts),
        "quality_flag_counts": dict(flags),
        "avg_quality_score": round(sum(quality_scores) / max(len(quality_scores), 1), 4),
        "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 4),
        "low_quality_count": sum(1 for x in quality_scores if x < 6),
        "low_confidence_count": sum(1 for x in confidences if x < 0.55),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# style_llm quality report",
        "",
        f"- ok: {summary['ok_count']}",
        f"- failed: {summary['failed_count']}",
        f"- avg confidence: {summary['avg_confidence']}",
        f"- avg quality score: {summary['avg_quality_score']}",
        f"- low confidence (<0.55): {summary['low_confidence_count']}",
        f"- low quality (<6): {summary['low_quality_count']}",
        "",
        "## Style counts",
        "",
    ]
    for style, count in style_counts.most_common():
        lines.append(f"- `{style}`: {count}")
    lines.extend(["", "## Quality flags", ""])
    for flag, count in flags.most_common():
        lines.append(f"- `{flag}`: {count}")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enrich supplier catalog with style_llm labels from text metadata.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL)
    p.add_argument("--merged-json", default=DEFAULT_MERGED_JSON)
    p.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    p.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--ollama-format", choices=["none", "json"], default="json")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--sleep-sec", type=float, default=0.0)
    return p


def main() -> None:
    args = build_cli().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_jsonl = Path(args.out_jsonl).expanduser().resolve()
    merged_json = Path(args.merged_json).expanduser().resolve()
    report_json = Path(args.report_json).expanduser().resolve()
    report_md = Path(args.report_md).expanduser().resolve()
    source_data, rows = load_catalog(input_path)
    if args.force and out_jsonl.exists():
        out_jsonl.unlink()
    existing = read_existing(out_jsonl)
    end = len(rows) if args.limit is None else min(len(rows), int(args.offset) + int(args.limit))

    for idx in range(max(0, int(args.offset)), end):
        item = rows[idx]
        iid = item_id(item, idx)
        if iid in existing and not args.force:
            print(f"[{idx + 1}/{len(rows)}] skip {iid}")
            continue
        try:
            style_data, _raw = classify_one(args, item)
            row = {
                "id": iid,
                "row_index": idx,
                "status": "ok",
                "unique_key": item.get("unique_key"),
                "title": item.get("title"),
                "category_norm": item.get("category_norm"),
                "source_site": item.get("source_site"),
                "style_llm": style_data,
                "processed_at_unix": time.time(),
                "model": args.ollama_model,
            }
            print(f"[{idx + 1}/{len(rows)}] ok {style_data['style_llm']} q={style_data['style_llm_quality_score']} c={style_data['style_llm_confidence']}")
        except Exception as exc:
            row = {
                "id": iid,
                "row_index": idx,
                "status": "error",
                "unique_key": item.get("unique_key"),
                "title": item.get("title"),
                "category_norm": item.get("category_norm"),
                "source_site": item.get("source_site"),
                "error": f"{type(exc).__name__}: {exc}",
                "processed_at_unix": time.time(),
                "model": args.ollama_model,
            }
            print(f"[{idx + 1}/{len(rows)}] error {row['error']}")
        append_jsonl(out_jsonl, row)
        existing[iid] = row
        if args.sleep_sec:
            time.sleep(float(args.sleep_sec))

    existing = read_existing(out_jsonl)
    merge_results(source_data, rows, existing, merged_json, out_jsonl)
    build_quality_reports(existing, report_json, report_md)
    print(f"jsonl = {out_jsonl}")
    print(f"merged = {merged_json}")
    print(f"report_json = {report_json}")
    print(f"report_md = {report_md}")


if __name__ == "__main__":
    main()
