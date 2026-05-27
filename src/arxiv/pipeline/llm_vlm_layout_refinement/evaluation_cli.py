#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отдельный запуск VLM-оценки (без полного refine-цикла).

Примеры::

    # Только картинка + текст «что хотели» (без infinigen_request / прогона)
    python -m src.pipeline.llm_vlm_layout_refinement.evaluation_cli \\
        --image ./render.png --prompt "спальня 12 м², скандинавский стиль" \\
        --vlm-model llava --out ./vlm_out

    python -m src.pipeline.llm_vlm_layout_refinement.evaluation_cli \\
        --run-root ./out/run1 --prompt "спальня 12 м²" --vlm-model llava \\
        --out ./out/run1/vlm_standalone

    python -m src.pipeline.llm_vlm_layout_refinement.evaluation_cli \\
        --request-json ./req.json --prompt "..." --image ./render.png \\
        --vlm-model llava --merge --out ./vlm_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from src.prompt_compiler.schemas import CompiledPolicy, RoomType, StyleLabel

from . import evaluation as ev

_ROOM_CHOICES = [e.value for e in RoomType]
_STYLE_CHOICES = [e.value for e in StyleLabel]


def _minimal_request_for_image_only(
    *,
    prompt_text: str,
    room_type: str,
    style_label: str,
) -> dict[str, Any]:
    """Короткий infinigen_request-заглушка: основной смысл в original_prompt и notes."""
    return {
        "room_type": room_type,
        "style_label": style_label,
        "style_raw": prompt_text[:800],
        "furniture": [
            {"semantic": "Chair", "count": 1, "priority": "desired"},
        ],
        "forbidden_objects": [],
        "favorite_colors": [],
        "avoid_colors": [],
        "material_family": [],
        "palette_hint": [],
        "wants_door": True,
        "wants_window": True,
        "notes": "[image-only] оценка по готовому рендеру; полное ТЗ см. original_prompt в JSON запроса к VLM.",
        "infinigen_runtime": {},
    }


def _required_semantics_from_request(req: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for row in req.get("furniture") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("priority", "")).lower() != "required":
            continue
        sem = str(row.get("semantic") or "").strip()
        if sem and sem not in out:
            out.append(sem)
    return out


def compiled_summary_for_vlm(
    compiled: CompiledPolicy | None,
    request: dict[str, Any],
) -> dict[str, Any]:
    if compiled is not None:
        return {
            "room_type": compiled.geometry.room_type.value,
            "style": compiled.style_policy.style_label,
            "required_semantics": compiled.program.required_semantics,
            "max_counts": compiled.program.max_counts,
            "monkeypatch_params": compiled.infinigen_policy.monkeypatch_params,
            "stage_flags": compiled.infinigen_policy.stage_flags,
            "solver_steps": compiled.infinigen_policy.solver_steps,
            "gin_overrides_head": (compiled.infinigen_policy.gin_overrides or [])[:24],
        }
    return {
        "room_type": str(request.get("room_type") or ""),
        "style": str(request.get("style_label") or ""),
        "required_semantics": _required_semantics_from_request(request),
        "max_counts": {},
        "monkeypatch_params": {},
        "stage_flags": {},
        "solver_steps": {},
        "gin_overrides_head": [],
    }


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"ожидался JSON-object в {path}")
    return data


def _resolve_image(*, run_root: Path | None, image: str | None) -> Path:
    if image:
        p = Path(image).expanduser().resolve()
        if not p.is_file():
            sys.exit(f"нет файла изображения: {p}")
        return p
    if run_root is None:
        sys.exit("укажите --image или --run-root с единственным render.png внутри")
    candidates = sorted(run_root.resolve().rglob("render.png"))
    if len(candidates) == 0:
        sys.exit(f"не найден render.png под {run_root}; укажите --image явно")
    if len(candidates) > 1:
        preview = "\n  ".join(str(c) for c in candidates[:12])
        more = f"\n  ... и ещё {len(candidates) - 12}" if len(candidates) > 12 else ""
        sys.exit(
            "несколько render.png под run-root; укажите --image:\n  " + preview + more
        )
    return candidates[0]


def _load_request_for_run_root(run_root: Path) -> dict[str, Any]:
    active = run_root / "infinigen_request.active.json"
    initial = run_root / "infinigen_request.initial.json"
    if active.is_file():
        return _load_json(active)
    if initial.is_file():
        return _load_json(initial)
    sys.exit(f"в {run_root} нет infinigen_request.active.json ни infinigen_request.initial.json")


def _load_compiled_optional(run_root: Path | None, path: str | None) -> CompiledPolicy | None:
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            sys.exit(f"нет compiled policy: {p}")
        return CompiledPolicy.load(p)
    if run_root is None:
        return None
    cand = run_root / "compiled_policy.active.json"
    if cand.is_file():
        return CompiledPolicy.load(cand)
    return None


def build_evaluation_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Отдельный вызов VLM (Ollama vision): готовое изображение + текст ТЗ, "
            "или полный контекст (request/run-root). Результат — JSON критики/патчей."
        ),
    )
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--vlm-model", required=True, help="Имя vision-модели в Ollama")
    p.add_argument("--timeout-sec", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--prompt",
        default="",
        help="Текст задания / исходный промпт (обязателен в режиме только --image).",
    )
    p.add_argument(
        "--prompt-file",
        default=None,
        help="Файл с текстом промпта (если не задан --prompt)",
    )
    p.add_argument(
        "--room-type",
        default="Bedroom",
        choices=_ROOM_CHOICES,
        help="Для режима только --image: room_type в заглушке infinigen_request (по умолчанию Bedroom).",
    )
    p.add_argument(
        "--style-label",
        default="scandinavian",
        choices=_STYLE_CHOICES,
        help="Для режима только --image: style_label в заглушке (по умолчанию scandinavian).",
    )
    p.add_argument("--run-root", type=Path, default=None, help="Каталог прогона refine: request + опционально compiled")
    p.add_argument(
        "--request-json",
        type=Path,
        default=None,
        help="Путь к infinigen_request JSON (альтернатива --run-root)",
    )
    p.add_argument(
        "--compiled-json",
        type=Path,
        default=None,
        help="compiled_policy.active.json (иначе из --run-root, если есть)",
    )
    p.add_argument(
        "--image",
        default=None,
        help="Путь к готовому изображению (PNG/JPEG). Обязателен без --run-root; с --run-root можно опустить, если render.png один.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Каталог для vlm_payload.json (и merged при --merge)",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="После VLM вызвать merge_infinigen_request_with_vlm → infinigen_request.merged.json",
    )
    return p


def main() -> None:
    args = build_evaluation_cli().parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else None

    image_only = args.request_json is None and run_root is None
    if image_only and not (args.image and str(args.image).strip()):
        sys.exit("режим «только изображение»: укажите --image и текст задания (--prompt или --prompt-file)")

    if args.request_json is not None:
        req = _load_json(Path(args.request_json).expanduser().resolve())
    elif run_root is not None:
        req = _load_request_for_run_root(run_root)
    else:
        req = None

    prompt_text = (args.prompt or "").strip()
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
    if not prompt_text and run_root is not None:
        up = run_root / "user_prompt.txt"
        if up.is_file():
            prompt_text = up.read_text(encoding="utf-8").strip()
    if not prompt_text:
        sys.exit("задайте --prompt или --prompt-file (текст ТЗ к изображению)")

    if req is None:
        req = _minimal_request_for_image_only(
            prompt_text=prompt_text,
            room_type=str(args.room_type),
            style_label=str(args.style_label),
        )

    compiled = _load_compiled_optional(
        run_root,
        str(args.compiled_json) if args.compiled_json else None,
    )
    summary = compiled_summary_for_vlm(compiled, req)
    image_path = _resolve_image(run_root=run_root, image=args.image)

    if image_only:
        (out_dir / "infinigen_request.stub_for_vlm.json").write_text(
            json.dumps(req, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    user_payload = json.dumps(
        {
            "original_prompt": prompt_text,
            "infinigen_request": req,
            "compiled_summary": summary,
        },
        ensure_ascii=False,
        indent=2,
    )

    t0 = perf_counter()
    try:
        vlm_payload = ev._ollama_vision_json(
            base_url=args.ollama_url,
            model=args.vlm_model,
            system_prompt=ev._vlm_system_prompt(),
            user_text=user_payload,
            image_paths=[image_path],
            timeout_sec=int(args.timeout_sec),
            temperature=float(args.temperature),
        )
    except Exception as exc:
        print(f"VLM error after {perf_counter() - t0:.2f}s: {exc!r}", file=sys.stderr)
        raise SystemExit(2) from exc

    (out_dir / "vlm_payload.json").write_text(
        json.dumps(vlm_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "vlm_user_payload.json").write_text(user_payload, encoding="utf-8")
    meta = {
        "image": str(image_path),
        "vlm_model": args.vlm_model,
        "wall_sec": round(perf_counter() - t0, 3),
        "image_only": bool(image_only),
    }
    (out_dir / "vlm_eval_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_dir / 'vlm_payload.json'} (wall {meta['wall_sec']}s)")

    if args.merge:
        if image_only:
            print(
                "предупреждение: --merge для заглушки infinigen_request даёт условный JSON; "
                "для пайплайна лучше --request-json или --run-root",
                file=sys.stderr,
            )
        merged = ev.merge_infinigen_request_with_vlm(req, vlm_payload)
        (out_dir / "infinigen_request.merged.json").write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"merged: {out_dir / 'infinigen_request.merged.json'}")


if __name__ == "__main__":
    main()
