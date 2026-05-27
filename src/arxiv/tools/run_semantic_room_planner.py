#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pipeline.semantic_room_planner.schemas import read_json
from pipeline.semantic_room_planner_stage import run_semantic_room_planner


def _read_prompt(args: argparse.Namespace, data: dict) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt
    return str(data.get("prompt") or "")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--room-json", default=None)
    p.add_argument("--input-json", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--provider", choices=["none", "ollama", "openrouter"], default="ollama")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--model", default=None)
    p.add_argument("--openrouter-model", default=None)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--apply-placement", action="store_true")
    p.add_argument("--max-repair-iterations", type=int, default=3)
    p.add_argument("--skip-catalog-queries", action="store_true")
    p.add_argument("--llm-catalog-queries", action="store_true")
    p.add_argument("--llm-catalog-max-objects", type=int, default=8)
    return p


def main() -> int:
    args = build_cli().parse_args()
    input_path = args.input_json or args.room_json
    if not input_path:
        raise SystemExit("--input-json or --room-json is required")
    data = read_json(input_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    settings = {
        "provider": args.provider,
        "model": args.model,
        "ollama_url": args.ollama_url,
        "openrouter_model": args.openrouter_model,
        "timeout": args.timeout,
        "temperature": args.temperature,
        "max_attempts": args.max_attempts,
        "debug_dir": str(out_dir / "llm_debug") if args.debug else None,
        "use_llm_catalog_queries": bool(args.llm_catalog_queries),
        "llm_catalog_max_objects": int(args.llm_catalog_max_objects),
    }
    info = run_semantic_room_planner(
        input_json=data,
        prompt=_read_prompt(args, data),
        out_dir=out_dir,
        llm_settings=settings,
        apply_placement=args.apply_placement,
        max_repair_iterations=args.max_repair_iterations,
        skip_catalog_queries=args.skip_catalog_queries,
    )
    hard_count = len(info.get("hard_errors") or [])
    warn_count = len(info.get("warnings") or [])
    print(f"final status: {info.get('status')}")
    print(f"out_dir: {info.get('out_dir')}")
    print(f"final_room_scene_plan: {info.get('final_room_scene_plan')}")
    print(f"scene.v1: {info.get('scene_v1')}")
    print(f"placement.v1: {info.get('placement_v1')}")
    print(f"validation score: {info.get('validation_score')}")
    print(f"warnings count: {warn_count}")
    print(f"hard errors count: {hard_count}")
    return 0 if info.get("status") in {"success", "partial_success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
