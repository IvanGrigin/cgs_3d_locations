#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Describe Mosplitka material images with a local MLX VLM.

This is a second-pass enrichment step.  The text-derived fields in
mosplitka_surface_materials.jsonl stay unchanged; only the ``vlm`` block is
filled from image analysis.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Ты классифицируешь только видимый рисунок отделочного материала. "
    "Не пиши красивое описание. Не упоминай помещение, бренд, цену, прочность, назначение, пол или стену.\n"
    "Ответь ровно 6 строками в формате Ключ: значение.\n"
    "Visual type: marble|stone|concrete|wood|terrazzo|plain|geometric|ornament|floral|botanical|brick|fabric|other\n"
    "Details: veins|speckles|chips|grain|clouds|stripes|patches|solid|relief_pattern|small_shapes|large_shapes|mixed\n"
    "Contrast: low|medium|high\n"
    "Accent: low|medium|high\n"
    "Texture impression: matte|glossy|smooth|rough|grainy|relief|unknown\n"
    "Short ru: 3-8 русских слов только про видимый рисунок, например тонкие серые прожилки"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def record_key(record: dict[str, Any]) -> str:
    return str(record.get("url") or record.get("sku") or record.get("name") or "")


def resolve_image_path(record: dict[str, Any], root: Path) -> Path | None:
    image = record.get("material_image") or {}
    value = image.get("path") or image.get("source_path") or ""
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return root / path


def extract_generation(stdout: str) -> str:
    text = stdout.replace("\r\n", "\n")
    marker = "<|im_start|>assistant"
    if marker in text:
        text = text.split(marker, 1)[1]
    if "==========" in text:
        text = text.split("==========", 1)[0]
    text = re.sub(r"<\|im_end\|>", "", text)
    return text.strip().strip('"').strip()


def parse_labeled_output(text: str) -> dict[str, str]:
    parsed = {
        "color": "",
        "pattern": "",
        "pattern_details": "",
        "visual_texture": "",
        "style": "",
        "accent_level": "",
        "description_ru": "",
        "visual_type": "",
        "details": "",
        "contrast": "",
        "texture_impression": "",
        "short_ru": "",
    }
    label_map = {
        "цвет": "color",
        "тип рисунка": "pattern",
        "рисунок": "pattern",
        "детали рисунка": "pattern_details",
        "детали": "pattern_details",
        "визуальная фактура": "visual_texture",
        "фактура": "visual_texture",
        "стиль": "style",
        "акцентность": "accent_level",
        "описание": "description_ru",
        "visual type": "visual_type",
        "details": "details",
        "contrast": "contrast",
        "accent": "accent_level",
        "texture impression": "texture_impression",
        "short ru": "short_ru",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("-").strip()
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label_map.get(label.strip().lower())
        if key:
            parsed[key] = value.strip().strip(";").strip()
    if not parsed["description_ru"] and text:
        parsed["description_ru"] = " ".join(text.split())
    if not parsed["pattern"] and parsed["visual_type"]:
        parsed["pattern"] = parsed["visual_type"]
    if not parsed["pattern_details"] and parsed["details"]:
        parsed["pattern_details"] = parsed["details"]
    if not parsed["visual_texture"] and parsed["texture_impression"]:
        parsed["visual_texture"] = parsed["texture_impression"]
    if parsed["short_ru"]:
        parsed["description_ru"] = parsed["short_ru"]
    return parsed


RU_VISUAL_TYPES = {
    "marble": "под мрамор",
    "stone": "под камень",
    "concrete": "под бетон",
    "wood": "под дерево",
    "terrazzo": "терраццо",
    "plain": "однотонный",
    "geometric": "с геометрическим рисунком",
    "ornament": "с орнаментом",
    "floral": "с цветочным рисунком",
    "botanical": "с растительным рисунком",
    "brick": "под кирпич",
    "fabric": "под ткань",
}

RU_DETAILS = {
    "veins": "прожилками",
    "speckles": "мелкими вкраплениями",
    "chips": "каменной крошкой",
    "grain": "древесными волокнами",
    "clouds": "облачными разводами",
    "stripes": "полосами",
    "patches": "пятнистыми переходами",
    "solid": "ровным тоном",
    "relief_pattern": "рельефным узором",
    "small_shapes": "мелким повторяющимся рисунком",
    "large_shapes": "крупным рисунком",
    "mixed": "смешанным рисунком",
}

RU_SURFACES = {
    "матовая": "матовый",
    "матовый": "матовый",
    "matte": "матовый",
    "глянцевая": "глянцевый",
    "глянцевый": "глянцевый",
    "glossy": "глянцевый",
    "полированная": "полированный",
    "полированный": "полированный",
    "polished": "полированный",
    "лаппатированная": "лаппатированный",
    "lappato": "лаппатированный",
    "рельефная": "рельефный",
    "рельефный": "рельефный",
    "relief": "рельефный",
    "структурированная": "структурированный",
    "structured": "структурированный",
    "гладкая": "гладкий",
    "smooth": "гладкий",
    "шероховатая": "шероховатый",
    "rough": "шероховатый",
    "зернистая": "зернистый",
    "grainy": "зернистый",
}

RU_COLOR_FORMS = {
    "графит": "графитовый",
    "кремовый": "кремовый",
    "светлый": "светлый",
    "темный": "темный",
    "белый": "белый",
    "черный": "черный",
    "серый": "серый",
    "бежевый": "бежевый",
    "коричневый": "коричневый",
    "красный": "красный",
    "синий": "синий",
    "голубой": "голубой",
    "зеленый": "зеленый",
    "розовый": "розовый",
    "желтый": "желтый",
}


def first_token(value: str) -> str:
    return re.split(r"[,;/\s]+", value.strip().lower(), maxsplit=1)[0] if value else ""


def normalize_surface_ru(value: str) -> str:
    text = value.strip().lower()
    return RU_SURFACES.get(text, text)


def normalize_color_ru(value: str) -> str:
    text = value.strip().lower()
    return RU_COLOR_FORMS.get(text, text)


def normalize_style_ru(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if text.endswith("стиль"):
        return text
    return f"{text} стиль"


def build_final_description(record: dict[str, Any], parsed: dict[str, str]) -> str:
    facts = record.get("text_facts") or {}
    normalized = record.get("normalized") or {}
    color = normalize_color_ru(facts.get("precise_color") or facts.get("color") or normalized.get("precise_color_ru") or normalized.get("base_color") or "")
    surface = normalize_surface_ru(facts.get("surface") or normalized.get("surface_finish") or parsed.get("texture_impression") or "")
    site_pattern = facts.get("pattern") or ""
    visual_type = first_token(parsed.get("visual_type") or parsed.get("pattern") or "")
    pattern = RU_VISUAL_TYPES.get(visual_type) or site_pattern
    detail_tokens = [x.strip().lower() for x in re.split(r"[,;/\s]+", parsed.get("details") or "") if x.strip()]
    details = [RU_DETAILS[x] for x in detail_tokens if x in RU_DETAILS]
    if not details and parsed.get("short_ru"):
        details = [parsed["short_ru"]]
    style = normalize_style_ru(facts.get("style") or ", ".join(normalized.get("style_tags") or []))
    parts = [color, surface, "материал", pattern]
    phrase = " ".join(p for p in parts if p)
    if details:
        phrase = f"{phrase} с {', '.join(dict.fromkeys(details[:3]))}".strip()
    if style:
        phrase = f"{phrase}, {style}"
    return re.sub(r"\s+", " ", phrase).strip(" ,")


def build_prompt(base_prompt: str, record: dict[str, Any], include_text_facts: bool) -> str:
    if not include_text_facts:
        return base_prompt
    facts = record.get("text_facts") or {}
    fact_lines = [
        f"Цвет с сайта: {facts.get('precise_color') or facts.get('color') or ''}",
        f"Рисунок с сайта: {facts.get('pattern') or ''}",
        f"Поверхность с сайта: {facts.get('surface') or ''}",
        f"Стиль с сайта: {facts.get('style') or ''}",
    ]
    fact_text = "\n".join(line for line in fact_lines if not line.endswith(": "))
    if not fact_text:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "Подсказки из карточки сайта ниже. Используй их только для уточнения, "
        "но описание делай по изображению:\n"
        f"{fact_text}"
    )


def run_vlm(
    executable: str,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> tuple[dict[str, str], str]:
    cmd = [
        executable,
        "--model",
        model,
        "--image",
        str(image_path),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--prompt",
        prompt,
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip() or f"mlx_vlm failed with code {proc.returncode}")
    generated = extract_generation(proc.stdout)
    return parse_labeled_output(generated), generated


def is_fatal_vlm_error(error: str) -> bool:
    text = error.lower()
    if "timed out" in text or "timeout" in text:
        return False
    return any(
        needle in text
        for needle in [
            "no metal device available",
            "command not found",
            "no such file or directory",
            "mlx_vlm.generate",
        ]
    )


def should_process(record: dict[str, Any], only_selectable: bool, retry_errors: bool) -> bool:
    vlm = record.get("vlm") or {}
    if vlm.get("status") == "ok":
        return False
    if vlm.get("status") == "error" and not retry_errors:
        return False
    if only_selectable:
        n = record.get("normalized") or {}
        return bool(n.get("is_selectable_floor") or n.get("is_selectable_wall"))
    return True


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill VLM descriptions for Mosplitka surface materials.")
    parser.add_argument("--input", default="data/floor_materials/mosplitka/mosplitka_surface_materials.jsonl")
    parser.add_argument("--output", default="data/floor_materials/mosplitka/mosplitka_surface_materials_vlm.jsonl")
    parser.add_argument("--model", required=True, help="MLX model name or local snapshot path.")
    parser.add_argument("--images-root", default="data/floor_materials/mosplitka")
    parser.add_argument("--executable", default="mlx_vlm.generate")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-version", default="mosplitka_material_enum_v3")
    parser.add_argument("--include-text-facts", action="store_true", help="Add site color/pattern/style facts to each VLM prompt.")
    parser.add_argument("--overwrite-ok", action="store_true", help="Reprocess records that already have vlm.status=ok.")
    parser.add_argument("--print-vlm-fields", action="store_true", help="Print parsed VLM fields for interactive test runs.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep after each processed image.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N records; 0 means no limit.")
    parser.add_argument("--offset", type=int, default=0, help="Skip N processable records before starting.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--only-selectable", action="store_true", help="Process only selectable floor/wall records.")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    if not str(args.model).strip():
        print("ERROR: --model is empty. Set MODEL after activating the venv.", file=sys.stderr)
        return 2
    if not shutil.which(args.executable):
        print(f"ERROR: executable not found: {args.executable}. Activate .venv-vlm first.", file=sys.stderr)
        return 2
    model_path = Path(args.model)
    if model_path.exists() and not (model_path / "config.json").exists():
        print(f"ERROR: model path exists but has no config.json: {model_path}", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_path = Path(args.output)
    records = load_jsonl(input_path)
    if output_path.exists():
        existing = {record_key(r): r for r in load_jsonl(output_path)}
        records = [existing.get(record_key(r), r) for r in records]

    processed = 0
    seen_processable = 0
    changed_since_checkpoint = 0
    root = Path(args.images_root)
    def processable(record: dict[str, Any]) -> bool:
        if args.overwrite_ok and (record.get("vlm") or {}).get("status") == "ok":
            if args.only_selectable:
                n = record.get("normalized") or {}
                return bool(n.get("is_selectable_floor") or n.get("is_selectable_wall"))
            return True
        return should_process(record, args.only_selectable, args.retry_errors)

    total_processable = sum(1 for record in records if processable(record))
    if args.offset:
        total_processable = max(0, total_processable - args.offset)
    planned = min(args.limit, total_processable) if args.limit else total_processable
    started_at = time.monotonic()
    print(f"processable_remaining: {total_processable}")
    print(f"planned_this_run: {planned}")
    for index, record in enumerate(records, start=1):
        if not processable(record):
            continue
        image_path = resolve_image_path(record, root)
        if not image_path or not image_path.exists():
            record["vlm"] = {
                **(record.get("vlm") or {}),
                "status": "error",
                "model": args.model,
                "error": f"image not found: {image_path}",
            }
            changed_since_checkpoint += 1
            continue

        if seen_processable < args.offset:
            seen_processable += 1
            continue
        if args.limit and processed >= args.limit:
            break
        seen_processable += 1

        elapsed = time.monotonic() - started_at
        prompt = build_prompt(args.prompt, record, args.include_text_facts)
        print(
            f"[{processed + 1}/{planned or '?'}] record {index}/{len(records)} "
            f"elapsed={format_duration(elapsed)} {image_path}",
            flush=True,
        )
        try:
            parsed, raw = run_vlm(
                args.executable,
                args.model,
                image_path,
                prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
            record["vlm"] = {
                "status": "ok",
                "model": args.model,
                "prompt_version": args.prompt_version,
                "image_path": str(image_path),
                "raw_output": raw,
                "final_visual_description_ru": build_final_description(record, parsed),
                **parsed,
            }
            elapsed = time.monotonic() - started_at
            avg = elapsed / max(1, processed + 1)
            remaining = max(0, (planned - processed - 1) * avg) if planned else 0
            print(
                f"  {record['vlm'].get('final_visual_description_ru') or parsed.get('description_ru') or raw}\n"
                f"  elapsed={format_duration(elapsed)} avg={avg:.1f}s/item eta={format_duration(remaining)}",
                flush=True,
            )
            if args.print_vlm_fields:
                print(
                    "  fields: "
                    f"visual_type={parsed.get('visual_type') or ''}; "
                    f"details={parsed.get('details') or ''}; "
                    f"contrast={parsed.get('contrast') or ''}; "
                    f"accent={parsed.get('accent_level') or ''}; "
                    f"texture={parsed.get('texture_impression') or ''}; "
                    f"short_ru={parsed.get('short_ru') or ''}",
                    flush=True,
                )
                print(f"  raw: {raw}", flush=True)
        except Exception as exc:  # Keep the batch resumable.
            error_text = str(exc)
            if is_fatal_vlm_error(error_text):
                print(
                    f"  FATAL: {exc}\n"
                    "  VLM command/environment is not usable; stopping without marking this record as processed.",
                    file=sys.stderr,
                    flush=True,
                )
                break
            record["vlm"] = {
                **(record.get("vlm") or {}),
                "status": "error",
                "model": args.model,
                "image_path": str(image_path),
                "error": error_text[-2000:],
            }
            elapsed = time.monotonic() - started_at
            avg = elapsed / max(1, processed + 1)
            remaining = max(0, (planned - processed - 1) * avg) if planned else 0
            print(
                f"  ERROR: {exc}\n"
                f"  elapsed={format_duration(elapsed)} avg={avg:.1f}s/item eta={format_duration(remaining)}",
                file=sys.stderr,
                flush=True,
            )
        processed += 1
        changed_since_checkpoint += 1
        if changed_since_checkpoint >= args.checkpoint_every:
            write_jsonl(output_path, records)
            changed_since_checkpoint = 0
        if args.delay > 0:
            print(f"  sleeping={args.delay:g}s", flush=True)
            time.sleep(args.delay)

    write_jsonl(output_path, records)
    ok = sum(1 for r in records if (r.get("vlm") or {}).get("status") == "ok")
    errors = sum(1 for r in records if (r.get("vlm") or {}).get("status") == "error")
    print(f"processed_this_run: {processed}")
    print(f"vlm_ok_total: {ok}")
    print(f"vlm_errors_total: {errors}")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
