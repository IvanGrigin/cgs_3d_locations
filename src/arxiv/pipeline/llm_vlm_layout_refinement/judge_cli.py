#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отдельный запуск judge.

**Режим 1 — полный (как в пайплайне):** текстовый LLM + ``rule_gate.json`` + инвентарь
(пиксели в модель не передаются). Нужен каталог кандидата после ``evaluate_candidate``.

**Режим 2 — только изображение:** vision-модель Ollama (``/api/chat`` + images), тот же JSON-схема
``judge.json``, без ``rule_gate``.

Полные примеры команд см. в ``--help`` и в docstring модуля ниже.

Пример полного judge::

    python -m src.pipeline.llm_vlm_layout_refinement.judge_cli \\
        --candidate-dir /abs/path/out/run/refine_round_0/screening_i0/seed_0 \\
        --run-root /abs/path/out/run \\
        --llm-provider ollama \\
        --ollama-url http://127.0.0.1:11434 \\
        --ollama-model qwen2.5:14b

**Режим 3 — динамический пайплайн (промпт + картинка):** внутри делается compile как в ``run``,
VLM оценивает инвентарь по изображению → пишутся ``inventory_summary.json`` + ``rule_gate.json``,
затем обычный текстовый judge (если не ``--skip-text-judge``).

Пример динамического режима::

    python -m src.pipeline.llm_vlm_layout_refinement.judge_cli \\
        --dynamic-pipeline \\
        --prompt "Спальня 14 м², скандинавский стиль." \\
        --image /abs/path/render.png \\
        --out-dir /abs/path/dyn_out \\
        --policies /abs/path/src/pipeline/llm_vlm_scene_policies.yaml \\
        --llm-provider ollama \\
        --ollama-model qwen2.5:14b \\
        --vision-model llava \\
        --ollama-url http://127.0.0.1:11434
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from src.prompt_compiler.schemas import CompiledPolicy, JudgeResult, RoomType, StyleLabel

from src.pipeline.llm_vlm_screening import build_llm_client, default_policies_path, run_judge_llm_vlm

from . import evaluation as ev
from .dynamic_prompt_image_eval import run_dynamic_prompt_image_eval

_ROOM_CHOICES = [e.value for e in RoomType]
_STYLE_CHOICES = [e.value for e in StyleLabel]

_JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "total_score": {"type": "number"},
        "functionality_score": {"type": "number"},
        "prompt_match_score": {"type": "number"},
        "style_match_score": {"type": "number"},
        "composition_score": {"type": "number"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": [
        "passed",
        "total_score",
        "functionality_score",
        "prompt_match_score",
        "style_match_score",
        "composition_score",
        "strengths",
        "weaknesses",
        "notes",
    ],
    "additionalProperties": False,
}

_VISION_APPENDIX = """
VISION-ONLY MODE (standalone CLI):
- You are given the actual room render as an image attachment (not only metadata).
- There is NO engine rule_gate JSON and NO inventory_summary from the solver.
- Infer layout, furniture, density, and style match from pixels together with the fields in the user JSON.
- Return ONLY JSON matching the schema (same keys as the standard room judge).
- strengths and weaknesses MUST be non-empty arrays with concrete visual observations.
- If the image is empty or unreadable: all scores near 0, passed=false, explain in notes.
No markdown. JSON only.
"""


def _judge_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "llm_vlm_judge"


def _resolve_compiled_path(*, compiled_json: Path | None, run_root: Path | None) -> Path:
    if compiled_json is not None:
        p = Path(compiled_json).expanduser().resolve()
        if not p.is_file():
            sys.exit(f"нет файла compiled policy: {p}")
        return p
    if run_root is not None:
        p = Path(run_root).expanduser().resolve() / "compiled_policy.active.json"
        if p.is_file():
            return p
        sys.exit(f"нет {p}; укажите --compiled-json")
    sys.exit("нужен --compiled-json или --run-root с compiled_policy.active.json")


def _raw_to_judge_result(raw: dict[str, Any], *, candidate_dir_label: str) -> JudgeResult:
    return JudgeResult(
        passed=bool(raw.get("passed", False)),
        total_score=float(raw.get("total_score", 0.0)),
        functionality_score=float(raw.get("functionality_score", 0.0)),
        prompt_match_score=float(raw.get("prompt_match_score", 0.0)),
        style_match_score=float(raw.get("style_match_score", 0.0)),
        composition_score=float(raw.get("composition_score", 0.0)),
        strengths=list(raw.get("strengths") or []),
        weaknesses=list(raw.get("weaknesses") or []),
        notes=str(raw.get("notes") or ""),
        candidate_dir=candidate_dir_label,
        diagnostic_only=bool(raw.get("diagnostic_only", False)),
    )


def run_judge_image_only(
    *,
    image: Path,
    prompt_text: str,
    ollama_url: str,
    vision_model: str,
    out_dir: Path,
    compiled: CompiledPolicy | None,
    room_type: str,
    style_label: str,
    timeout_sec: int,
    temperature: float,
) -> Path:
    jdir = _judge_dir()
    prompt_path = jdir / "room_judge_prompt.md"
    rubric_path = jdir / "room_judge_rubric.yaml"
    base_prompt = prompt_path.read_text(encoding="utf-8")
    rubric = yaml.safe_load(rubric_path.read_text(encoding="utf-8"))
    system_prompt = base_prompt.strip() + "\n\n" + _VISION_APPENDIX.strip()

    if compiled is not None:
        ctx = {
            "original_prompt": compiled.prompt_text or prompt_text,
            "user_cli_prompt": prompt_text,
            "room_type": compiled.geometry.room_type.value,
            "area_sqm": compiled.geometry.area_sqm,
            "area_bucket": compiled.geometry.area_bucket.value,
            "style_label": compiled.style_policy.style_label,
            "required_semantics": compiled.program.required_semantics,
            "rubric": rubric,
            "vision_only": True,
        }
    else:
        ctx = {
            "original_prompt": prompt_text,
            "room_type": room_type,
            "style_label": style_label,
            "required_semantics": [],
            "rubric": rubric,
            "vision_only": True,
        }

    user_payload = json.dumps(ctx, ensure_ascii=False, indent=2)
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "llm_vlm_run.log"
    log_path.write_text("", encoding="utf-8")

    t0 = perf_counter()
    raw = ev._ollama_vision_json(
        base_url=ollama_url,
        model=vision_model,
        system_prompt=system_prompt,
        user_text=user_payload,
        image_paths=[image],
        timeout_sec=timeout_sec,
        temperature=temperature,
        response_json_schema=_JUDGE_JSON_SCHEMA,
    )
    result = _raw_to_judge_result(raw, candidate_dir_label=str(image.resolve()))
    out_judge = out_dir / "judge.json"
    result.save(out_judge)
    meta = {
        "mode": "image_only_vision",
        "image": str(image.resolve()),
        "vision_model": vision_model,
        "wall_sec": round(perf_counter() - t0, 3),
    }
    (out_dir / "judge_image_only_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "judge_user_payload.json").write_text(user_payload, encoding="utf-8")
    return out_judge


def build_judge_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Judge отдельно от пайплайна: (1) полный текстовый judge по rule_gate.json; "
            "(2) vision-judge только по изображению → judge.json; "
            "(3) динамически: compile + VLM-инвентарь + rule_gate + текстовый judge."
        ),
    )
    p.add_argument(
        "--dynamic-pipeline",
        action="store_true",
        help="Промпт + изображение: compile как в run, VLM→inventory_summary, evaluate_candidate→rule_gate, затем judge.",
    )
    p.add_argument(
        "--image-only",
        action="store_true",
        help="Режим только по картинке: Ollama vision, без candidate-dir и rule_gate.",
    )
    # --- image-only ---
    p.add_argument("--image", type=Path, default=None, help="[image-only] PNG/JPEG рендера")
    p.add_argument("--prompt", default="", help="[image-only] Текст ТЗ / промпт")
    p.add_argument("--prompt-file", type=Path, default=None, help="[image-only] Файл с промптом")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="[image-only / dynamic-pipeline] Каталог вывода (обязателен в этих режимах)",
    )
    p.add_argument(
        "--vision-model",
        default="llava",
        help="Vision-модель Ollama: для --image-only (judge) и для --dynamic-pipeline (инвентарь)",
    )
    p.add_argument("--timeout-sec", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--room-type",
        default="Bedroom",
        choices=_ROOM_CHOICES,
        help="[image-only без --compiled-json] room_type в контексте JSON для модели",
    )
    p.add_argument(
        "--style-label",
        default="scandinavian",
        choices=_STYLE_CHOICES,
        help="[image-only без --compiled-json] style_label в контексте JSON для модели",
    )
    # --- full gate ---
    p.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Каталог кандидата (seed_*), с rule_gate.json; не используется с --image-only",
    )
    p.add_argument(
        "--compiled-json",
        type=Path,
        default=None,
        help="compiled_policy.active.json (полный judge: из --run-root если не задан; image-only: опционально)",
    )
    p.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Корень прогона refine → compiled_policy.active.json для полного judge",
    )
    p.add_argument("--llm-provider", default="ollama", choices=["ollama", "none"])
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument(
        "--ollama-model",
        default="qwen2.5:14b",
        help="Полный judge: текстовая JSON-модель Ollama (не vision)",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Полный judge: каталог для llm_vlm_run.log. По умолчанию: --run-root или --candidate-dir",
    )
    p.add_argument(
        "--policies",
        type=str,
        default=None,
        help="[dynamic-pipeline] YAML политик (по умолчанию как у полного refine)",
    )
    p.add_argument(
        "--skip-text-judge",
        action="store_true",
        help="[dynamic-pipeline] Остановиться после rule_gate.json (не вызывать run_judge_llm_vlm)",
    )
    p.add_argument(
        "--copy-judge-json-to",
        type=Path,
        default=None,
        help="Полный judge: дополнительно скопировать judge.json в путь (файл или каталог)",
    )
    return p


def main() -> None:
    args = build_judge_cli().parse_args()

    if args.dynamic_pipeline and args.image_only:
        sys.exit("нельзя одновременно --dynamic-pipeline и --image-only")
    if args.dynamic_pipeline and args.candidate_dir is not None:
        sys.exit("с --dynamic-pipeline не используйте --candidate-dir")
    if args.image_only and args.candidate_dir is not None:
        sys.exit("с --image-only не используйте --candidate-dir")

    if args.dynamic_pipeline:
        if not args.image or not args.out_dir:
            sys.exit("[dynamic-pipeline] нужны --image и --out-dir")
        prompt_text = (args.prompt or "").strip()
        if args.prompt_file is not None:
            prompt_text = Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
        if not prompt_text:
            sys.exit("[dynamic-pipeline] задайте --prompt или --prompt-file")
        policies_path = args.policies or str(default_policies_path())
        paths = run_dynamic_prompt_image_eval(
            prompt_text=prompt_text,
            image=Path(args.image).expanduser().resolve(),
            out_dir=Path(args.out_dir).expanduser().resolve(),
            policies_path=policies_path,
            llm_provider=args.llm_provider,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            vision_model=str(args.vision_model).strip(),
            timeout_sec=int(args.timeout_sec),
            temperature=float(args.temperature),
            run_text_judge=not bool(args.skip_text_judge),
        )
        print(f"OK rule_gate={paths['rule_gate']}")
        if "judge" in paths:
            print(f"OK judge={paths['judge']}")
        return

    if args.image_only:
        if not args.image:
            sys.exit("[image-only] укажите --image /path/to/render.png")
        if not args.out_dir:
            sys.exit("[image-only] укажите --out-dir /path/to/output_dir/")
        image = Path(args.image).expanduser().resolve()
        if not image.is_file():
            sys.exit(f"нет файла: {image}")
        prompt_text = (args.prompt or "").strip()
        if args.prompt_file is not None:
            prompt_text = Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
        if not prompt_text:
            sys.exit("[image-only] задайте --prompt или --prompt-file")

        compiled: CompiledPolicy | None = None
        if args.compiled_json is not None:
            compiled = CompiledPolicy.load(Path(args.compiled_json).expanduser().resolve())
        elif args.run_root is not None:
            cp = Path(args.run_root).expanduser().resolve() / "compiled_policy.active.json"
            if cp.is_file():
                compiled = CompiledPolicy.load(cp)

        out_judge = run_judge_image_only(
            image=image,
            prompt_text=prompt_text,
            ollama_url=args.ollama_url,
            vision_model=str(args.vision_model).strip(),
            out_dir=Path(args.out_dir).expanduser().resolve(),
            compiled=compiled,
            room_type=str(args.room_type),
            style_label=str(args.style_label),
            timeout_sec=int(args.timeout_sec),
            temperature=float(args.temperature),
        )
        print(f"OK: {out_judge}")
        return

    # --- full judge (rule_gate) ---
    if args.candidate_dir is None:
        sys.exit("укажите --candidate-dir или используйте --image-only")
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    if not candidate_dir.is_dir():
        sys.exit(f"нет каталога кандидата: {candidate_dir}")
    gate_path = candidate_dir / "rule_gate.json"
    if not gate_path.is_file():
        sys.exit(
            f"нет {gate_path}. Нужен rule_gate после evaluate_candidate. "
            "Только по картинке без гейта: --image-only."
        )

    compiled_path = _resolve_compiled_path(
        compiled_json=args.compiled_json,
        run_root=Path(args.run_root).expanduser().resolve() if args.run_root else None,
    )
    compiled = CompiledPolicy.load(compiled_path)

    llm = build_llm_client(args.llm_provider, args.ollama_model, args.ollama_url)

    if args.log_dir is not None:
        log_root = Path(args.log_dir).expanduser().resolve()
    elif args.run_root is not None:
        log_root = Path(args.run_root).expanduser().resolve()
    else:
        log_root = candidate_dir

    result = run_judge_llm_vlm(compiled, candidate_dir, llm, log_run_root=log_root)
    out_judge = candidate_dir / "judge.json"
    print(f"OK: {out_judge} total_score={result.total_score} passed={result.passed}")

    if args.copy_judge_json_to is not None:
        dst = Path(args.copy_judge_json_to).expanduser().resolve()
        if dst.is_dir():
            dst = dst / "judge.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_judge, dst)
        print(f"copy: {dst}")


if __name__ == "__main__":
    main()
